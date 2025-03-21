#include <Arduino.h>
#include <WiFi.h>
#include <esp_http_server.h>
#include <ArduinoJson.h>

// WiFi credentials
#define SSID "TT"
#define PASSWORD "12345678"
#define PORT 82

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

// Number of motors (adjust as needed)
const int NUM_MOTORS = 8;
Motor motors[NUM_MOTORS];

// Interrupt Handlers
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

// Motor Control Functions
void setMotor(int motorIndex, int power) {
    Motor &motor = motors[motorIndex];
    //Serial.printf("Motor %d: Power = %d, IN1 = %d, IN2 = %d\n", motorIndex, power, motor.IN1, motor.IN2);
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
    //Serial.printf("Motor %d: targetPos = %ld, encoderPos = %ld, error = %ld, power = %d\n", 
       //           motorIndex, motor.targetPos, motor.encoderPos, error, power);
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
            //Serial.printf("Motor %d: Set angle = %d, targetPos = %ld\n", motorIndex, angle, motors[motorIndex].targetPos);
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

        // Detach interrupts from old pins if they were previously set
        if (motor.ENCODER_A != -1) {
            detachInterrupt(digitalPinToInterrupt(motor.ENCODER_A));
        }
        if (motor.ENCODER_B != -1) {
            detachInterrupt(digitalPinToInterrupt(motor.ENCODER_B));
        }

        // Update pins
        motor.ENCODER_A = new_ENCODER_A;
        motor.ENCODER_B = new_ENCODER_B;
        motor.IN1 = new_IN1;
        motor.IN2 = new_IN2;

        // Configure encoder pins
        if (motor.ENCODER_A != -1) {
            pinMode(motor.ENCODER_A, INPUT_PULLUP);
            if (digitalPinToInterrupt(motor.ENCODER_A) != NOT_AN_INTERRUPT) {
                attachInterruptArg(digitalPinToInterrupt(motor.ENCODER_A), handleEncoderA, &motor, CHANGE);
            } else {
                //Serial.printf("ERROR: GPIO%d cannot be used for interrupts\n", motor.ENCODER_A);
            }
        }
        if (motor.ENCODER_B != -1) {
            pinMode(motor.ENCODER_B, INPUT_PULLUP);
            if (digitalPinToInterrupt(motor.ENCODER_B) != NOT_AN_INTERRUPT) {
                attachInterruptArg(digitalPinToInterrupt(motor.ENCODER_B), handleEncoderB, &motor, CHANGE);
            } else {
                //Serial.printf("ERROR: GPIO%d cannot be used for interrupts\n", motor.ENCODER_B);
            }
        }

        // Configure motor pins
        if (motor.IN1 != -1) {
            pinMode(motor.IN1, OUTPUT);
            digitalWrite(motor.IN1, LOW);  // Set to LOW before PWM
            ledcAttach(motor.IN1, 1000, 8);
            if (ledcRead(motor.IN1) == 0) {
                //Serial.printf("ERROR: Failed to attach PWM to GPIO%d (IN1)\n", motor.IN1);
            }
            ledcWrite(motor.IN1, 255); // Brake
        }
        if (motor.IN2 != -1) {
            pinMode(motor.IN2, OUTPUT);
            digitalWrite(motor.IN2, LOW);  // Set to LOW before PWM
            ledcAttach(motor.IN2, 1000, 8);
            if (ledcRead(motor.IN2) == 0) {
                //Serial.printf("ERROR: Failed to attach PWM to GPIO%d (IN2)\n", motor.IN2);
            }
            ledcWrite(motor.IN2, 255); // Brake
        }
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
        }
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
    Serial.begin(115200);

    // Initialize GPIO pins as OUTPUT and set to LOW
    for (int pin = 1; pin <= 2; pin++) {
        pinMode(pin, OUTPUT);
        digitalWrite(pin, LOW);
        //Serial.printf("Set GPIO%d to LOW\n", pin);
    }
    for (int pin = 4; pin <= 18; pin++) {
        pinMode(pin, OUTPUT);
        digitalWrite(pin, LOW);
        //Serial.printf("Set GPIO%d to LOW\n", pin);
    }
    pinMode(21, OUTPUT);
    digitalWrite(21, LOW);
    //Serial.println("Set GPIO21 to LOW");
    for (int pin = 35; pin <= 42; pin++) {
        pinMode(pin, OUTPUT);
        digitalWrite(pin, LOW);
        //Serial.printf("Set GPIO%d to LOW\n", pin);
    }
    pinMode(47, OUTPUT);
    digitalWrite(47, LOW);
    //Serial.println("Set GPIO47 to LOW");
    pinMode(48, OUTPUT);
    digitalWrite(48, LOW);
    //Serial.println("Set GPIO48 to LOW");

    // Explicitly initialize motor control pins for FRONT RIGHT TURNING MOTOR
    pinMode(6, OUTPUT);
    digitalWrite(6, LOW);
    //Serial.println("Set GPIO6 to LOW explicitly");
    pinMode(7, OUTPUT);
    digitalWrite(7, LOW);
    //Serial.println("Set GPIO7 to LOW explicitly");

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
        motors[i].controlEnabled = false; // Control OFF for all motors at start
    }

    // Connect to WiFi
    WiFi.begin(SSID, PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());

    // Start HTTP server
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = PORT;
    config.core_id = 0;
    config.keep_alive_enable = true;
    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t set_control_params_uri = { .uri = "/set_control_params", .method = HTTP_GET, .handler = set_control_params_handler, .user_ctx = NULL };
        httpd_register_uri_handler(server, &set_control_params_uri);

        httpd_uri_t set_angle_uri = { .uri = "/set_angle", .method = HTTP_GET, .handler = set_angle_handler, .user_ctx = NULL };
        httpd_register_uri_handler(server, &set_angle_uri);

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