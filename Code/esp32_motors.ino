#include <Arduino.h>
#include <WiFi.h>
#include <esp_http_server.h>
#include <ArduinoJson.h>
#include <soc/gpio_reg.h>  // For direct GPIO register access

// WiFi credentials
#define SSID "TT"
#define PASSWORD "12345678"
#define PORT 82

// Number of motors (define early for scope)
const int NUM_MOTORS = 8;

// Control parameters (shared across all motors)
float kp = 0.9;
float ki = 0.001;
float kd = 0.3;
int DEAD_ZONE = 10;
int MAX_POWER = 255;
int POSITION_THRESHOLD = 5;

// Encoder constants
const int COUNTS_PER_REV = 1975;

// Timing
unsigned long lastTime = 0;
const unsigned long dt = 2;

// Software PWM variables for motors 4-7
volatile int dutyCycles[NUM_MOTORS][2] = {0};  // Duty cycles for IN1, IN2 per motor
hw_timer_t *timer = NULL;
volatile int pwmCounter = 0;
const int PWM_RESOLUTION = 255;  // Full 8-bit resolution for motor control
const int PWM_FREQ = 200;        // Reduced to 200 Hz to lower interrupt rate (200 * 255 = 51,000 Hz)

// HTTP server handle
httpd_handle_t server = NULL;

// Motor structure to hold per-motor data
struct Motor {
    int ENCODER_A;
    int ENCODER_B;
    int IN1;
    int IN2;
    volatile long encoderPos;
    long targetPos;
    long lastError;
    float integralError;
    bool controlEnabled;
};

Motor motors[NUM_MOTORS];

// Interrupt Handlers for Encoders
void IRAM_ATTR handleEncoderA(void* arg) {
    Motor* motor = (Motor*)arg;
    bool encAVal = digitalRead(motor->ENCODER_A);
    bool encBVal = digitalRead(motor->ENCODER_B);
    if (encAVal == encBVal) {
        motor->encoderPos--;
    } else {
        motor->encoderPos++;
    }
}

void IRAM_ATTR handleEncoderB(void* arg) {
    Motor* motor = (Motor*)arg;
    bool encAVal = digitalRead(motor->ENCODER_A);
    bool encBVal = digitalRead(motor->ENCODER_B);
    if (encAVal == encBVal) {
        motor->encoderPos++;
    } else {
        motor->encoderPos--;
    }
}

// Software PWM Interrupt Handler for Motors 4-7 (optimized with direct register access)
void IRAM_ATTR onTimer() {
    pwmCounter = (pwmCounter + 1) % PWM_RESOLUTION;
    for (int i = 4; i < NUM_MOTORS; i++) {
        Motor &motor = motors[i];
        if (motor.IN1 != -1) {
            if (pwmCounter < dutyCycles[i][0]) {
                REG_WRITE(GPIO_OUT_W1TS_REG, 1 << motor.IN1);  // Set HIGH
            } else {
                REG_WRITE(GPIO_OUT_W1TC_REG, 1 << motor.IN1);  // Set LOW
            }
        }
        if (motor.IN2 != -1) {
            if (pwmCounter < dutyCycles[i][1]) {
                REG_WRITE(GPIO_OUT_W1TS_REG, 1 << motor.IN2);  // Set HIGH
            } else {
                REG_WRITE(GPIO_OUT_W1TC_REG, 1 << motor.IN2);  // Set LOW
            }
        }
    }
}

// Motor Control Functions
void setMotor(int motorIndex, int power) {
    Motor &motor = motors[motorIndex];
    if (motorIndex < 4) {
        // Hardware PWM for motors 0-3 using LEDC
        int channel1 = motorIndex * 2;      // Channels 0, 2, 4, 6
        int channel2 = motorIndex * 2 + 1;  // Channels 1, 3, 5, 7

        if (power == 0) {
            ledcWrite(motor.IN1, 255);  // Brake
            ledcWrite(motor.IN2, 255);
        } else if (power > 0) {
            ledcWrite(motor.IN2, 0);
            ledcWrite(motor.IN1, power);
        } else {
            ledcWrite(motor.IN1, 0);
            ledcWrite(motor.IN2, -power);
        }
    } else {
        // Software PWM for motors 4-7
        if (power == 0) {
            dutyCycles[motorIndex][0] = PWM_RESOLUTION;  // Brake
            dutyCycles[motorIndex][1] = PWM_RESOLUTION;
        } else if (power > 0) {
            dutyCycles[motorIndex][0] = power;  // Already 0-255
            dutyCycles[motorIndex][1] = 0;
        } else {
            dutyCycles[motorIndex][0] = 0;
            dutyCycles[motorIndex][1] = -power;  // Already 0-255
        }
    }
}

int computePower(long error, long errorDelta) {
    if (abs(error) <= DEAD_ZONE) {
        return 0;
    }

    float scaled_error = constrain((float)error / POSITION_THRESHOLD, -1.0, 1.0);
    float dt_sec = dt / 1000.0;

    float p_term = kp * scaled_error * MAX_POWER;
    float d_term = kd * (errorDelta / dt_sec);

    if (abs(error) <= DEAD_ZONE * 5) {
        d_term *= 3.0;
    }

    int power = p_term + constrain(d_term, -MAX_POWER / 2, MAX_POWER / 2);
    return constrain(power, -MAX_POWER, MAX_POWER);
}

void controlMotor(int motorIndex) {
    Motor &motor = motors[motorIndex];
    long error = motor.targetPos - motor.encoderPos;
    long errorDelta = error - motor.lastError;
    motor.lastError = error;

    if (abs(error) < MAX_POWER / ki) {
        motor.integralError += error * (dt / 1000.0);
    } else {
        motor.integralError = constrain(motor.integralError, -MAX_POWER / ki, MAX_POWER / ki);
    }

    int power = computePower(error, errorDelta) + ki * motor.integralError;
    setMotor(motorIndex, power);
}

// HTTP URI Handlers
static esp_err_t set_control_params_handler(httpd_req_t *req) {
    char query[128];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char buf[20];
        if (httpd_query_key_value(query, "P", buf, sizeof(buf)) == ESP_OK) kp = atof(buf);
        if (httpd_query_key_value(query, "I", buf, sizeof(buf)) == ESP_OK) ki = atof(buf);
        if (httpd_query_key_value(query, "D", buf, sizeof(buf)) == ESP_OK) kd = atof(buf);
        if (httpd_query_key_value(query, "dead_zone", buf, sizeof(buf)) == ESP_OK) DEAD_ZONE = atoi(buf);
        if (httpd_query_key_value(query, "pos_thresh", buf, sizeof(buf)) == ESP_OK) POSITION_THRESHOLD = atoi(buf);
    }
    return httpd_resp_send(req, "K", 1);
}

static esp_err_t set_angle_handler(httpd_req_t *req) {
    char query[128];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char buf[20];
        int motorIndex;
        if (httpd_query_key_value(query, "motor", buf, sizeof(buf)) == ESP_OK) {
            motorIndex = atoi(buf);
            if (motorIndex < 0 || motorIndex >= NUM_MOTORS) {
                return httpd_resp_send(req, "Invalid motor index", 18);
            }
        } else {
            return httpd_resp_send(req, "Missing motor index", 19);
        }

        if (httpd_query_key_value(query, "a", buf, sizeof(buf)) == ESP_OK) {
            int angle = atoi(buf);
            motors[motorIndex].targetPos = angle * COUNTS_PER_REV / 360;
        }
    }
    return httpd_resp_send(req, "K", 1);
}

static esp_err_t set_angles_handler(httpd_req_t *req) {
    char query[256];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char buf[20];
        for (int i = 0; i < NUM_MOTORS; i++) {
            char param[3];
            snprintf(param, sizeof(param), "a%d", i);
            if (httpd_query_key_value(query, param, buf, sizeof(buf)) == ESP_OK) {
                int angle = atoi(buf);
                motors[i].targetPos = angle * COUNTS_PER_REV / 360;
            }
        }
    }
    return httpd_resp_send(req, "K", 1);
}

static esp_err_t set_all_pins_handler(httpd_req_t *req) {
    char query[1024];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char buf[20];
        for (int i = 0; i < NUM_MOTORS; i++) {
            Motor &motor = motors[i];
            int new_ENCODER_A = motor.ENCODER_A;
            int new_ENCODER_B = motor.ENCODER_B;
            int new_IN1 = motor.IN1;
            int new_IN2 = motor.IN2;

            char param[12];
            snprintf(param, sizeof(param), "ENCODER_A%d", i);
            if (httpd_query_key_value(query, param, buf, sizeof(buf)) == ESP_OK) new_ENCODER_A = atoi(buf);
            snprintf(param, sizeof(param), "ENCODER_B%d", i);
            if (httpd_query_key_value(query, param, buf, sizeof(buf)) == ESP_OK) new_ENCODER_B = atoi(buf);
            snprintf(param, sizeof(param), "IN1_%d", i);
            if (httpd_query_key_value(query, param, buf, sizeof(buf)) == ESP_OK) new_IN1 = atoi(buf);
            snprintf(param, sizeof(param), "IN2_%d", i);
            if (httpd_query_key_value(query, param, buf, sizeof(buf)) == ESP_OK) new_IN2 = atoi(buf);

            if (motor.ENCODER_A != -1) detachInterrupt(digitalPinToInterrupt(motor.ENCODER_A));
            if (motor.ENCODER_B != -1) detachInterrupt(digitalPinToInterrupt(motor.ENCODER_B));

            motor.ENCODER_A = new_ENCODER_A;
            motor.ENCODER_B = new_ENCODER_B;
            motor.IN1 = new_IN1;
            motor.IN2 = new_IN2;

            if (motor.ENCODER_A != -1) {
                pinMode(motor.ENCODER_A, INPUT_PULLUP);
                if (digitalPinToInterrupt(motor.ENCODER_A) != NOT_AN_INTERRUPT) {
                    attachInterruptArg(digitalPinToInterrupt(motor.ENCODER_A), handleEncoderA, &motor, CHANGE);
                }
            }
            if (motor.ENCODER_B != -1) {
                pinMode(motor.ENCODER_B, INPUT_PULLUP);
                if (digitalPinToInterrupt(motor.ENCODER_B) != NOT_AN_INTERRUPT) {
                    attachInterruptArg(digitalPinToInterrupt(motor.ENCODER_B), handleEncoderB, &motor, CHANGE);
                }
            }

            if (motor.IN1 != -1) {
                pinMode(motor.IN1, OUTPUT);
                digitalWrite(motor.IN1, LOW);
                if (i < 4) {  // Hardware PWM for motors 0-3
                    ledcAttachChannel(motor.IN1, 1000, 8, i * 2);  // Channels 0, 2, 4, 6
                    ledcWrite(motor.IN1, 255);
                }
            }
            if (motor.IN2 != -1) {
                pinMode(motor.IN2, OUTPUT);
                digitalWrite(motor.IN2, LOW);
                if (i < 4) {  // Hardware PWM for motors 0-3
                    ledcAttachChannel(motor.IN2, 1000, 8, i * 2 + 1);  // Channels 1, 3, 5, 7
                    ledcWrite(motor.IN2, 255);
                }
            }
        }
    }
    return httpd_resp_send(req, "K", 1);
}

static esp_err_t set_pins_handler(httpd_req_t *req) {
    char query[128];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char buf[20];
        int motorIndex;
        if (httpd_query_key_value(query, "motor", buf, sizeof(buf)) == ESP_OK) {
            motorIndex = atoi(buf);
            if (motorIndex < 0 || motorIndex >= NUM_MOTORS) {
                return httpd_resp_send(req, "Invalid motor index", 18);
            }
        } else {
            return httpd_resp_send(req, "Missing motor index", 19);
        }

        Motor &motor = motors[motorIndex];
        int new_ENCODER_A = motor.ENCODER_A;
        int new_ENCODER_B = motor.ENCODER_B;
        int new_IN1 = motor.IN1;
        int new_IN2 = motor.IN2;

        if (httpd_query_key_value(query, "ENCODER_A", buf, sizeof(buf)) == ESP_OK) new_ENCODER_A = atoi(buf);
        if (httpd_query_key_value(query, "ENCODER_B", buf, sizeof(buf)) == ESP_OK) new_ENCODER_B = atoi(buf);
        if (httpd_query_key_value(query, "IN1", buf, sizeof(buf)) == ESP_OK) new_IN1 = atoi(buf);
        if (httpd_query_key_value(query, "IN2", buf, sizeof(buf)) == ESP_OK) new_IN2 = atoi(buf);

        if (motor.ENCODER_A != -1) detachInterrupt(digitalPinToInterrupt(motor.ENCODER_A));
        if (motor.ENCODER_B != -1) detachInterrupt(digitalPinToInterrupt(motor.ENCODER_B));

        motor.ENCODER_A = new_ENCODER_A;
        motor.ENCODER_B = new_ENCODER_B;
        motor.IN1 = new_IN1;
        motor.IN2 = new_IN2;

        if (motor.ENCODER_A != -1) {
            pinMode(motor.ENCODER_A, INPUT_PULLUP);
            if (digitalPinToInterrupt(motor.ENCODER_A) != NOT_AN_INTERRUPT) {
                attachInterruptArg(digitalPinToInterrupt(motor.ENCODER_A), handleEncoderA, &motor, CHANGE);
            }
        }
        if (motor.ENCODER_B != -1) {
            pinMode(motor.ENCODER_B, INPUT_PULLUP);
            if (digitalPinToInterrupt(motor.ENCODER_B) != NOT_AN_INTERRUPT) {
                attachInterruptArg(digitalPinToInterrupt(motor.ENCODER_B), handleEncoderB, &motor, CHANGE);
            }
        }

        if (motor.IN1 != -1) {
            pinMode(motor.IN1, OUTPUT);
            digitalWrite(motor.IN1, LOW);
            if (motorIndex < 4) {  // Hardware PWM for motors 0-3
                ledcAttachChannel(motor.IN1, 1000, 8, motorIndex * 2);  // Channels 0, 2, 4, 6
                ledcWrite(motor.IN1, 255);
            }
        }
        if (motor.IN2 != -1) {
            pinMode(motor.IN2, OUTPUT);
            digitalWrite(motor.IN2, LOW);
            if (motorIndex < 4) {  // Hardware PWM for motors 0-3
                ledcAttachChannel(motor.IN2, 1000, 8, motorIndex * 2 + 1);  // Channels 1, 3, 5, 7
                ledcWrite(motor.IN2, 255);
            }
        }

        char response[64];
        snprintf(response, sizeof(response), "M%d: EA=%d, EB=%d, IN1=%d, IN2=%d", 
                 motorIndex, motor.ENCODER_A, motor.ENCODER_B, motor.IN1, motor.IN2);
        return httpd_resp_send(req, response, strlen(response));
    }
    return httpd_resp_send(req, "K", 1);
}

static esp_err_t get_angle_handler(httpd_req_t *req) {
    char query[128];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char buf[20];
        int motorIndex;
        if (httpd_query_key_value(query, "motor", buf, sizeof(buf)) == ESP_OK) {
            motorIndex = atoi(buf);
            if (motorIndex < 0 || motorIndex >= NUM_MOTORS) {
                return httpd_resp_send(req, "Invalid motor index", 18);
            }
        } else {
            return httpd_resp_send(req, "Missing motor index", 19);
        }

        float angle = (float)motors[motorIndex].encoderPos * 360 / COUNTS_PER_REV;
        char response[20];
        snprintf(response, sizeof(response), "%f", angle);
        return httpd_resp_send(req, response, strlen(response));
    }
    return httpd_resp_send(req, "Missing motor param", 19);
}

static esp_err_t get_encoderPos_handler(httpd_req_t *req) {
    char query[128];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char buf[20];
        int motorIndex;
        if (httpd_query_key_value(query, "motor", buf, sizeof(buf)) == ESP_OK) {
            motorIndex = atoi(buf);
            if (motorIndex < 0 || motorIndex >= NUM_MOTORS) {
                return httpd_resp_send(req, "Invalid motor index", 18);
            }
        } else {
            return httpd_resp_send(req, "Missing motor index", 19);
        }

        char response[20];
        snprintf(response, sizeof(response), "%ld", motors[motorIndex].encoderPos);
        return httpd_resp_send(req, response, strlen(response));
    }
    return httpd_resp_send(req, "Missing motor param", 19);
}

static esp_err_t set_control_status_handler(httpd_req_t *req) {
    char query[128];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char buf[20];
        int motorIndex;
        if (httpd_query_key_value(query, "motor", buf, sizeof(buf)) == ESP_OK) {
            motorIndex = atoi(buf);
            if (motorIndex < 0 || motorIndex >= NUM_MOTORS) {
                return httpd_resp_send(req, "Invalid motor index", 18);
            }
        } else {
            return httpd_resp_send(req, "Missing motor index", 19);
        }

        if (httpd_query_key_value(query, "status", buf, sizeof(buf)) == ESP_OK) {
            int status = atoi(buf);
            motors[motorIndex].controlEnabled = (status != 0);
            if (motors[motorIndex].controlEnabled) {
                // When enabling control, set targetPos to current encoderPos
                // to prevent the motor from moving to a previous position
                motors[motorIndex].targetPos = motors[motorIndex].encoderPos;
                motors[motorIndex].lastError = 0;  // Reset PID terms
                motors[motorIndex].integralError = 0;
            } else {
                // When disabling control, stop the motor immediately
                setMotor(motorIndex, 0);
            }
        }

        char response[32];
        snprintf(response, sizeof(response), "M%d: Control=%d", motorIndex, motors[motorIndex].controlEnabled);
        return httpd_resp_send(req, response, strlen(response));
    }
    return httpd_resp_send(req, "K", 1);
}

static esp_err_t reset_handler(httpd_req_t *req) {
    char query[128];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char buf[20];
        int motorIndex;
        if (httpd_query_key_value(query, "motor", buf, sizeof(buf)) == ESP_OK) {
            motorIndex = atoi(buf);
            if (motorIndex < 0 || motorIndex >= NUM_MOTORS) {
                return httpd_resp_send(req, "Invalid motor index", 18);
            }
            motors[motorIndex].encoderPos = 0;
            motors[motorIndex].targetPos = 0;
            motors[motorIndex].lastError = 0;  // Also reset lastError to avoid PID integral windup
            motors[motorIndex].integralError = 0;  // Reset integral error for PID
            // Debug output to confirm reset
            char response[64];
            snprintf(response, sizeof(response), "M%d: Pos=%ld, Target=%ld", 
                     motorIndex, motors[motorIndex].encoderPos, motors[motorIndex].targetPos);
            return httpd_resp_send(req, response, strlen(response));
        } else {
            return httpd_resp_send(req, "Missing motor index", 19);
        }
    }
    return httpd_resp_send(req, "K", 1);
}

static esp_err_t events_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/event-stream");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    httpd_resp_set_hdr(req, "Connection", "keep-alive");

    while (true) {
        StaticJsonDocument<256> doc;
        JsonArray angles = doc.createNestedArray("angles");
        JsonArray encoderPos = doc.createNestedArray("encoderPos");
        for (int i = 0; i < NUM_MOTORS; i++) {
            float angle = (float)motors[i].encoderPos * 360 / COUNTS_PER_REV;
            angles.add(angle);
            encoderPos.add(motors[i].encoderPos);
        }
        char json[256];
        size_t len = serializeJson(doc, json);
        char event[512];
        snprintf(event, sizeof(event), "data: %s\n\n", json);
        if (httpd_resp_send_chunk(req, event, strlen(event)) != ESP_OK) {
            break;
        }
        delay(2);
    }
    return ESP_OK;
}

void setup() {
    pinMode(40, OUTPUT);
    digitalWrite(40, LOW);

    pinMode(39, OUTPUT);
    digitalWrite(39, LOW);

    for (int pin = 35; pin <= 42; pin++) {
        pinMode(pin, OUTPUT);
        digitalWrite(pin, LOW);
    }

    // Initialize GPIO pins as OUTPUT and set to LOW
    for (int pin = 1; pin <= 2; pin++) {
        pinMode(pin, OUTPUT);
        digitalWrite(pin, LOW);
    }
    for (int pin = 4; pin <= 18; pin++) {
        pinMode(pin, OUTPUT);
        digitalWrite(pin, LOW);
    }
    pinMode(21, OUTPUT);
    digitalWrite(21, LOW);
    
    pinMode(47, OUTPUT);
    digitalWrite(47, LOW);
    pinMode(48, OUTPUT);
    digitalWrite(48, LOW);

    // Explicitly initialize motor control pins for FRONT RIGHT TURNING MOTOR
    pinMode(6, OUTPUT);
    digitalWrite(6, LOW);
    pinMode(7, OUTPUT);
    digitalWrite(7, LOW);

    // Initialize timer for software PWM (motors 4-7)
    timer = timerBegin(PWM_FREQ * PWM_RESOLUTION);  // 200 Hz * 255 steps = 51,000 Hz
    timerAttachInterrupt(timer, &onTimer);          // Attach interrupt
    timerAlarm(timer, 1, true, 0);                  // Trigger every tick, autoreload
    timerStart(timer);  

    // Initialize motor structure
    for (int i = 0; i < NUM_MOTORS; i++) {
        motors[i].ENCODER_A = -1;
        motors[i].ENCODER_B = -1;
        motors[i].IN1 = -1;
        motors[i].IN2 = -1;
        motors[i].encoderPos = 0;
        motors[i].targetPos = 0;
        motors[i].lastError = 0;
        motors[i].integralError = 0;
        motors[i].controlEnabled = false;
    }

    // Connect to WiFi
    WiFi.begin(SSID, PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
    }

    // Start HTTP server
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = PORT;
    config.core_id = 0;  // Run HTTP server on Core 0 to avoid interference with Core 1
    config.keep_alive_enable = true;
    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t set_all_pins_uri = { .uri = "/set_all_pins", .method = HTTP_GET, .handler = set_all_pins_handler, .user_ctx = NULL };
        httpd_register_uri_handler(server, &set_all_pins_uri);

        httpd_uri_t set_control_params_uri = { .uri = "/set_control_params", .method = HTTP_GET, .handler = set_control_params_handler, .user_ctx = NULL };
        httpd_register_uri_handler(server, &set_control_params_uri);

        httpd_uri_t set_angle_uri = { .uri = "/set_angle", .method = HTTP_GET, .handler = set_angle_handler, .user_ctx = NULL };
        httpd_register_uri_handler(server, &set_angle_uri);

        httpd_uri_t set_angles_uri = { .uri = "/set_angles", .method = HTTP_GET, .handler = set_angles_handler, .user_ctx = NULL };
        httpd_register_uri_handler(server, &set_angles_uri);

        httpd_uri_t set_pins_uri = { .uri = "/set_pins", .method = HTTP_GET, .handler = set_pins_handler, .user_ctx = NULL };
        httpd_register_uri_handler(server, &set_pins_uri);

        httpd_uri_t get_angle_uri = { .uri = "/get_angle", .method = HTTP_GET, .handler = get_angle_handler, .user_ctx = NULL };
        httpd_register_uri_handler(server, &get_angle_uri);

        httpd_uri_t get_encoderPos_uri = { .uri = "/get_encoderPos", .method = HTTP_GET, .handler = get_encoderPos_handler, .user_ctx = NULL };
        httpd_register_uri_handler(server, &get_encoderPos_uri);

        httpd_uri_t set_control_status_uri = { .uri = "/set_control_status", .method = HTTP_GET, .handler = set_control_status_handler, .user_ctx = NULL };
        httpd_register_uri_handler(server, &set_control_status_uri);

        httpd_uri_t reset_uri = { .uri = "/reset", .method = HTTP_GET, .handler = reset_handler, .user_ctx = NULL };
        httpd_register_uri_handler(server, &reset_uri);

        httpd_uri_t events_uri = { .uri = "/events", .method = HTTP_GET, .handler = events_handler, .user_ctx = NULL };
        httpd_register_uri_handler(server, &events_uri);
    }
}

void loop() {
    if (millis() - lastTime >= dt) {
        for (int i = 0; i < NUM_MOTORS; i++) {
            if (motors[i].controlEnabled) {
                controlMotor(i);
            } else {
                setMotor(i, 0); // Brake when control is off
            }
        }
        lastTime = millis();
    }
}