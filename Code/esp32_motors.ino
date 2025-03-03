#include <Arduino.h>
#include <WiFi.h>
#include <esp_http_server.h>

// WiFi credentials
#define SSID "Wifi de sparky"
#define PASSWORD "$famcruz$"
#define PORT 82

// Pin variables (initial defaults)
int ENCODER_A = 8;
int ENCODER_B = 18;
int IN1 = 12;
int IN2 = 11;

// Control parameters
float kp = 0.9;
float ki = 0.001;
float kd = 0.3;
int DEAD_ZONE = 10;
int MAX_POWER = 255;
int POSITION_THRESHOLD = 5;

// Encoder constants
const int COUNTS_PER_REV = 1975;

// Variables
volatile long encoderPos = 0;
long targetPos = 0;
long lastError = 0;
float integralError = 0;

// Timing
unsigned long lastTime = 0;
const unsigned long dt = 2;


// Control status
bool control_enabled = true;

// HTTP server handle
httpd_handle_t server = NULL;

// **Interrupt Handlers**
void IRAM_ATTR handleEncoderA() {
    bool encAVal = digitalRead(ENCODER_A);
    bool encBVal = digitalRead(ENCODER_B);
    if (encAVal == encBVal) {
        encoderPos--;
    } else {
        encoderPos++;
    }
}

void IRAM_ATTR handleEncoderB() {
    bool encAVal = digitalRead(ENCODER_A);
    bool encBVal = digitalRead(ENCODER_B);
    if (encAVal == encBVal) {
        encoderPos++;
    } else {
        encoderPos--;
    }
}

// **Motor Control Functions**
void setMotor(int power) {
    if (power == 0) {
        ledcWrite(IN1, 255);  // IN1 HIGH (brake)
        ledcWrite(IN2, 255);  // IN2 HIGH
    } else if (power > 0) {
        ledcWrite(IN2, 0);    // IN2 LOW
        ledcWrite(IN1, power); // IN1 PWM
    } else {
        ledcWrite(IN1, 0);    // IN1 LOW
        ledcWrite(IN2, -power); // IN2 PWM
    }
}

int computePower(long error, long errorDelta) {
    if (abs(error) <= DEAD_ZONE) {
        integralError = 0;
        return 0;
    }

    float scaled_error = constrain((float)error / POSITION_THRESHOLD, -1.0, 1.0);
    float dt_sec = dt / 1000.0;

    float p_term = kp * scaled_error * MAX_POWER;
    float i_term = ki * integralError;
    float d_term = kd * (errorDelta / dt_sec);

    if (abs(error) <= DEAD_ZONE * 5) {
        d_term *= 3.0;
    }

    int power = p_term + i_term + constrain(d_term, -MAX_POWER/2, MAX_POWER/2);
    return constrain(power, -MAX_POWER, MAX_POWER);
}



// **HTTP URI Handlers**
static esp_err_t set_control_params_handler(httpd_req_t *req) {
    char query[128];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char buf[20];
        if (httpd_query_key_value(query, "P", buf, sizeof(buf)) == ESP_OK) {
            kp = atof(buf);
        }
        if (httpd_query_key_value(query, "I", buf, sizeof(buf)) == ESP_OK) {
            ki = atof(buf);
        }
        if (httpd_query_key_value(query, "D", buf, sizeof(buf)) == ESP_OK) {
            kd = atof(buf);
        }
        if (httpd_query_key_value(query, "dead_zone", buf, sizeof(buf)) == ESP_OK) {
            DEAD_ZONE = atoi(buf);
        }
        if (httpd_query_key_value(query, "pos_thresh", buf, sizeof(buf)) == ESP_OK) {
            POSITION_THRESHOLD = atoi(buf);
        }
    }
    return httpd_resp_send(req, "OK", 2);
}

static esp_err_t set_angle_handler(httpd_req_t *req) {
    char query[128];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char buf[20];
        if (httpd_query_key_value(query, "a", buf, sizeof(buf)) == ESP_OK) {
            int angle = atoi(buf);
            targetPos = angle * COUNTS_PER_REV / 360;
        }
    }
    return httpd_resp_send(req, "OK", 2);
}

static esp_err_t set_pins_handler(httpd_req_t *req) {
    char query[128];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char buf[20];
        int new_ENCODER_A = ENCODER_A;
        int new_ENCODER_B = ENCODER_B;
        int new_IN1 = IN1;
        int new_IN2 = IN2;

        if (httpd_query_key_value(query, "ENCODER_A", buf, sizeof(buf)) == ESP_OK) {
            new_ENCODER_A = atoi(buf);
        }
        if (httpd_query_key_value(query, "ENCODER_B", buf, sizeof(buf)) == ESP_OK) {
            new_ENCODER_B = atoi(buf);
        }
        if (httpd_query_key_value(query, "IN1", buf, sizeof(buf)) == ESP_OK) {
            new_IN1 = atoi(buf);
        }
        if (httpd_query_key_value(query, "IN2", buf, sizeof(buf)) == ESP_OK) {
            new_IN2 = atoi(buf);
        }

        // Update encoder pins
        if (new_ENCODER_A != ENCODER_A || new_ENCODER_B != ENCODER_B) {
            detachInterrupt(digitalPinToInterrupt(ENCODER_A));
            detachInterrupt(digitalPinToInterrupt(ENCODER_B));
            ENCODER_A = new_ENCODER_A;
            ENCODER_B = new_ENCODER_B;
            pinMode(ENCODER_A, INPUT_PULLUP);
            pinMode(ENCODER_B, INPUT_PULLUP);
            attachInterrupt(digitalPinToInterrupt(ENCODER_A), handleEncoderA, CHANGE);
            attachInterrupt(digitalPinToInterrupt(ENCODER_B), handleEncoderB, CHANGE);
        }

        // Update motor pins
        if (new_IN1 != IN1 || new_IN2 != IN2) {
            ledcDetach(IN1);
            ledcDetach(IN2);
            IN1 = new_IN1;
            IN2 = new_IN2;
            pinMode(IN1, OUTPUT);
            pinMode(IN2, OUTPUT);
            ledcAttach(IN1, 1000, 8);
            ledcAttach(IN2, 1000, 8);
        }
    }
    return httpd_resp_send(req, "OK", 2);
}

static esp_err_t get_angle_handler(httpd_req_t *req) {
    float angle = (float)encoderPos * 360 / COUNTS_PER_REV;
    char buf[20];
    snprintf(buf, sizeof(buf), "%f", angle);
    return httpd_resp_send(req, buf, strlen(buf));
}

static esp_err_t get_encoderPos_handler(httpd_req_t *req) {
    char buf[20];
    snprintf(buf, sizeof(buf), "%ld", encoderPos);
    return httpd_resp_send(req, buf, strlen(buf));
}

static esp_err_t set_control_status_handler(httpd_req_t *req) {
    char query[128];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char buf[20];
        if (httpd_query_key_value(query, "status", buf, sizeof(buf)) == ESP_OK) {
            control_enabled = atoi(buf) != 0;
        }
    }
    return httpd_resp_send(req, "OK", 2);
}

// **Setup Function**
void setup() {

    // Initialize pins
    pinMode(ENCODER_A, INPUT_PULLUP);
    pinMode(ENCODER_B, INPUT_PULLUP);
    pinMode(IN1, OUTPUT);
    pinMode(IN2, OUTPUT);

    // Configure PWM for IN1 and IN2
    ledcAttach(IN1, 1000, 8); // 1kHz, 8-bit resolution
    ledcAttach(IN2, 1000, 8); // 1kHz, 8-bit resolution

    // Initialize motor to brake
    ledcWrite(IN1, 255);
    ledcWrite(IN2, 255);

    // Attach interrupts
    attachInterrupt(digitalPinToInterrupt(ENCODER_A), handleEncoderA, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENCODER_B), handleEncoderB, CHANGE);

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
    config.stack_size = 8192; // Increase from default 4096

    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t set_control_params_uri = {
            .uri       = "/set_control_params",
            .method    = HTTP_GET,
            .handler   = set_control_params_handler,
            .user_ctx  = NULL
        };
        httpd_register_uri_handler(server, &set_control_params_uri);

        httpd_uri_t set_angle_uri = {
            .uri       = "/set_angle",
            .method    = HTTP_GET,
            .handler   = set_angle_handler,
            .user_ctx  = NULL
        };
        httpd_register_uri_handler(server, &set_angle_uri);

        httpd_uri_t set_pins_uri = {
            .uri       = "/set_pins",
            .method    = HTTP_GET,
            .handler   = set_pins_handler,
            .user_ctx  = NULL
        };
        httpd_register_uri_handler(server, &set_pins_uri);

        httpd_uri_t get_angle_uri = {
            .uri       = "/get_angle",
            .method    = HTTP_GET,
            .handler   = get_angle_handler,
            .user_ctx  = NULL
        };
        httpd_register_uri_handler(server, &get_angle_uri);

        httpd_uri_t get_encoderPos_uri = {
            .uri       = "/get_encoderPos",
            .method    = HTTP_GET,
            .handler   = get_encoderPos_handler,
            .user_ctx  = NULL
        };
        httpd_register_uri_handler(server, &get_encoderPos_uri);

        httpd_uri_t set_control_status_uri = {
            .uri       = "/set_control_status",
            .method    = HTTP_GET,
            .handler   = set_control_status_handler,
            .user_ctx  = NULL
        };
        httpd_register_uri_handler(server, &set_control_status_uri);
    }
}

// **Loop Function**
void loop() {
    if (millis() - lastTime >= dt) {
        long error = targetPos - encoderPos;
        long errorDelta = error - lastError;
        lastError = error;

        int power = 0;
        if (control_enabled) {
            if (abs(error) < MAX_POWER / ki) {
                integralError += error * (dt / 1000.0);
            } else {
                integralError = constrain(integralError, -MAX_POWER/ki, MAX_POWER/ki);
            }
            power = computePower(error, errorDelta);
        }
        setMotor(power);
        lastTime = millis();
    }
}