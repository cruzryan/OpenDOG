#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

// WiFi credentials
#define SSID "TT"
#define PASSWORD "12345678"
#define UDP_PORT 12345

// Set static IP configuration
IPAddress local_IP(192, 168, 137, 101); // 101 back, 100 front
IPAddress gateway(192, 168, 137, 1);
IPAddress subnet(255, 255, 255, 0);
IPAddress primaryDNS(8, 8, 8, 8);   // Google DNS
IPAddress secondaryDNS(8, 8, 4, 4); // Google DNS

// Number of motors
const int NUM_MOTORS = 4;

// Control parameters
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

// UDP instance
WiFiUDP udp;

// Synchronization flag for targetPos updates
volatile bool targetPosUpdated = false;

// Debug array for computed target positions
long debugTargetPos[4] = {0, 0, 0, 0};

// Motor structure
struct Motor {
    int ENCODER_A;
    int ENCODER_B;
    int IN1;
    int IN2;
    volatile long encoderPos;
    volatile long targetPos; 
    long lastError;
    float integralError;
    bool controlEnabled;
    unsigned long lastAChange;
    unsigned long lastBChange;
    bool lastAState;
    bool lastBState;
};

Motor motors[NUM_MOTORS];

// IMU Setup
#define I2C_SDA 19
#define I2C_SCL 20
Adafruit_MPU6050 mpu;
bool mpu_available = false;

// Interrupt Handlers for Encoders with Debouncing
void IRAM_ATTR handleEncoderA(void* arg) {
    Motor* motor = (Motor*)arg;
    unsigned long now = micros();
    if (now - motor->lastAChange < 1000) return; 
    motor->lastAChange = now;
    bool encAVal = digitalRead(motor->ENCODER_A);
    bool encBVal = digitalRead(motor->ENCODER_B);
    if (encAVal != motor->lastAState) {
        motor->lastAState = encAVal;
        if (encAVal == encBVal) {
            motor->encoderPos--;
        } else {
            motor->encoderPos++;
        }
    }
}

void IRAM_ATTR handleEncoderB(void* arg) {
    Motor* motor = (Motor*)arg;
    unsigned long now = micros();
    if (now - motor->lastBChange < 1000) return; 
    motor->lastBChange = now;
    bool encAVal = digitalRead(motor->ENCODER_A);
    bool encBVal = digitalRead(motor->ENCODER_B);
    if (encBVal != motor->lastBState) {
        motor->lastBState = encBVal;
        if (encAVal == encBVal) {
            motor->encoderPos++;
        } else {
            motor->encoderPos--;
        }
    }
}

// Motor Control Functions
void setMotor(int motorIndex, int power) {
    Motor &motor = motors[motorIndex];
    int channel1 = motorIndex * 2;      
    int channel2 = motorIndex * 2 + 1;  
    if (power == 0) {
        ledcWrite(motor.IN1, 255); 
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
    if (!motor.controlEnabled) {
        setMotor(motorIndex, 0);
        return;
    }
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

// UDP Command Handlers
void handle_set_control_params(const JsonDocument& doc) {
    if (doc.containsKey("P")) kp = doc["P"].as<float>();
    if (doc.containsKey("I")) ki = doc["I"].as<float>();
    if (doc.containsKey("D")) kd = doc["D"].as<float>();
    if (doc.containsKey("dead_zone")) DEAD_ZONE = doc["dead_zone"].as<int>();
    if (doc.containsKey("pos_thresh")) POSITION_THRESHOLD = doc["pos_thresh"].as<int>();
}

void handle_set_angles(int angles[], size_t arraySize) {
    for (size_t i = 0; i < arraySize && i < NUM_MOTORS; i++) {
        int angleDeg = angles[i];
        long computedTargetPos = static_cast<long>(angleDeg * static_cast<float>(COUNTS_PER_REV) / 360.0);
        motors[i].targetPos = computedTargetPos;
        debugTargetPos[i] = computedTargetPos; 
    }
    targetPosUpdated = true; 
}

void handle_set_all_pins(const JsonDocument& doc) {
    for (int i = 0; i < NUM_MOTORS; i++) {
        Motor &motor = motors[i];
        int new_ENCODER_A = motor.ENCODER_A;
        int new_ENCODER_B = motor.ENCODER_B;
        int new_IN1 = motor.IN1;
        int new_IN2 = motor.IN2;
        char key[12];
        snprintf(key, sizeof(key), "ENCODER_A%d", i);
        if (doc.containsKey(key)) new_ENCODER_A = doc[key].as<int>();
        snprintf(key, sizeof(key), "ENCODER_B%d", i);
        if (doc.containsKey(key)) new_ENCODER_B = doc[key].as<int>();
        snprintf(key, sizeof(key), "IN1_%d", i);
        if (doc.containsKey(key)) new_IN1 = doc[key].as<int>();
        snprintf(key, sizeof(key), "IN2_%d", i);
        if (doc.containsKey(key)) new_IN2 = doc[key].as<int>();
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
            ledcAttachChannel(motor.IN1, 1000, 8, i * 2); 
            ledcWrite(motor.IN1, 255);
        }
        if (motor.IN2 != -1) {
            pinMode(motor.IN2, OUTPUT);
            digitalWrite(motor.IN2, LOW);
            ledcAttachChannel(motor.IN2, 1000, 8, i * 2 + 1); 
            ledcWrite(motor.IN2, 255);
        }
    }
}

void handle_set_control_status(const JsonDocument& doc) {
    if (doc.containsKey("motor") && doc.containsKey("status")) {
        int motorIndex = doc["motor"].as<int>();
        if (motorIndex >= 0 && motorIndex < NUM_MOTORS) {
            int status = doc["status"].as<int>();
            motors[motorIndex].controlEnabled = (status != 0);
            if (motors[motorIndex].controlEnabled) {
                motors[motorIndex].targetPos = motors[motorIndex].encoderPos;
                motors[motorIndex].lastError = 0;
                motors[motorIndex].integralError = 0;
            } else {
                setMotor(motorIndex, 0);
            }
        }
    }
}

void handle_reset_all() {
    for (int i = 0; i < NUM_MOTORS; i++) {
        motors[i].encoderPos = 0;
        motors[i].targetPos = 0;
        motors[i].lastError = 0;
        motors[i].integralError = 0;
        debugTargetPos[i] = 0; 
    }
}

void handle_get_imu_data(IPAddress senderIP, int senderPort) {
    if (mpu_available) {
        sensors_event_t a, g, temp; 
        mpu.getEvent(&a, &g, &temp);
        StaticJsonDocument<256> doc; 
        JsonObject imu_data = doc.createNestedObject("imu"); 
        imu_data["accel_x"] = a.acceleration.x;
        imu_data["accel_y"] = a.acceleration.y;
        imu_data["accel_z"] = a.acceleration.z;
        imu_data["gyro_x"] = g.gyro.x;
        imu_data["gyro_y"] = g.gyro.y;
        imu_data["gyro_z"] = g.gyro.z;
        imu_data["temp"] = temp.temperature;
        char json[256];
        size_t len = serializeJson(doc, json);
        udp.beginPacket(senderIP, senderPort); // Use senderPort for reply
        udp.write((uint8_t*)json, len);
        udp.endPacket();
    } else {
        StaticJsonDocument<64> doc; 
        doc["error"] = "MPU6050 not initialized";
        char json[64];
        size_t len = serializeJson(doc, json);
        udp.beginPacket(senderIP, senderPort); // Use senderPort for reply
        udp.write((uint8_t*)json, len);
        udp.endPacket();
    }
}

// UDP Task for Sending Angles and Receiving Commands (Core 0)
void udpTask(void *pvParameters) {
    unsigned long lastSendTime = 0;
    unsigned long sendInterval = 50; 
    IPAddress broadcastIP(255, 255, 255, 255); 
    int debugAngles[4] = {0, 0, 0, 0}; 

    while (true) {
        if (WiFi.status() != WL_CONNECTED) {
            WiFi.disconnect();
            WiFi.begin(SSID, PASSWORD);
            while (WiFi.status() != WL_CONNECTED) {
                vTaskDelay(500 / portTICK_PERIOD_MS);
            }
            udp.begin(UDP_PORT); 
        }

        int packetSize = udp.parsePacket();
        if (packetSize) {
            char packetBuffer[1024];
            int len = udp.read(packetBuffer, sizeof(packetBuffer) - 1);
            packetBuffer[len] = 0; 

            StaticJsonDocument<1024> doc;
            DeserializationError error = deserializeJson(doc, packetBuffer);
            if (!error && doc.containsKey("command")) {
                String command = doc["command"].as<String>();
                IPAddress senderIP = udp.remoteIP();
                int senderPort = udp.remotePort(); // ****** GET SENDER'S PORT ******

                if (command == "set_control_params") {
                    handle_set_control_params(doc);
                } else if (command == "set_angles") {
                    if (doc.containsKey("angles") && doc["angles"].is<JsonArray>()) {
                        size_t arraySize = doc["angles"].size();
                        for (size_t i = 0; i < arraySize && i < NUM_MOTORS; i++) {
                            debugAngles[i] = static_cast<int>(doc["angles"][i].as<float>());
                        }
                        handle_set_angles(debugAngles, arraySize);
                    }
                } else if (command == "set_all_pins") {
                    handle_set_all_pins(doc);
                } else if (command == "set_control_status") {
                    handle_set_control_status(doc);
                } else if (command == "reset_all") {
                    handle_reset_all();
                } else if (command == "get_imu_data") {
                    handle_get_imu_data(senderIP, senderPort); // Pass senderPort here too
                } else if (command == "set_send_interval") {
                    if (doc.containsKey("interval") && doc["interval"].is<int>()) {
                        int newInterval = doc["interval"].as<int>();
                        if (newInterval > 0) {
                            sendInterval = newInterval;
                        }
                    }
                }
                StaticJsonDocument<64> responseDoc;
                responseDoc["status"] = "OK";
                char response[64];
                size_t responseLen = serializeJson(responseDoc, response);

                // ****** MODIFICATION: Reply to senderIP AND senderPort ******
                udp.beginPacket(senderIP, senderPort); 
                udp.write((uint8_t*)response, responseLen);
                udp.endPacket();
                // ****** END MODIFICATION ******
                vTaskDelay(2 / portTICK_PERIOD_MS); 
            }
        }

        if (millis() - lastSendTime >= sendInterval) {
            StaticJsonDocument<512 + 64> doc; 
            JsonArray angles = doc.createNestedArray("angles");
            JsonArray encoderPos_arr = doc.createNestedArray("encoderPos"); 
            JsonArray targetPos_arr = doc.createNestedArray("targetPos");   
            JsonArray debug_arr = doc.createNestedArray("debug");          
            JsonArray debugComputed_arr = doc.createNestedArray("debugComputed"); 
            bool allMotorsCurrentlyEnabled = true; 
            for (int i = 0; i < NUM_MOTORS; i++) {
                float angle = (float)motors[i].encoderPos * 360.0f / COUNTS_PER_REV;
                angles.add(angle);
                encoderPos_arr.add(motors[i].encoderPos);
                targetPos_arr.add(motors[i].targetPos);
                debug_arr.add(debugAngles[i]); 
                debugComputed_arr.add(debugTargetPos[i]);
                if (!motors[i].controlEnabled) { 
                    allMotorsCurrentlyEnabled = false;
                }
            }
            doc["esp_control_fully_enabled"] = allMotorsCurrentlyEnabled; 
            doc["mpu_available"] = mpu_available;
            if (mpu_available) {
                sensors_event_t a, g, temp; 
                mpu.getEvent(&a, &g, &temp); 
                JsonObject imu_data_out = doc.createNestedObject("imu");
                imu_data_out["accel_x"] = a.acceleration.x;
                imu_data_out["accel_y"] = a.acceleration.y;
                imu_data_out["accel_z"] = a.acceleration.z;
                imu_data_out["gyro_x"] = g.gyro.x;
                imu_data_out["gyro_y"] = g.gyro.y;
                imu_data_out["gyro_z"] = g.gyro.z;
                imu_data_out["temp"] = temp.temperature;
            }
            char json[512 + 64]; 
            size_t len = serializeJson(doc, json);
            if (len > 0) { 
                udp.beginPacket(broadcastIP, UDP_PORT);
                udp.write((uint8_t*)json, len);
                udp.endPacket();
            }
            lastSendTime = millis();
        }
        vTaskDelay(1 / portTICK_PERIOD_MS); 
    }
}

void setup() {
    Serial.begin(115200);
    Wire.begin(I2C_SDA, I2C_SCL, 100000); 
    int attempts = 0;
    while (!mpu.begin(0x68, &Wire)) { 
        Serial.println("Failed to find MPU6050 chip, retrying...");
        delay(100);
        attempts++;
        if (attempts > 3) { 
            Serial.println("MPU6050 not found after multiple attempts. Continuing without it.");
            mpu_available = false; 
            break;
        }
    }
    if (mpu.begin(0x68, &Wire)) { 
         Serial.println("MPU6050 Found!");
         mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
         mpu.setGyroRange(MPU6050_RANGE_500_DEG);
         mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);
         mpu_available = true;
    } else if (!mpu_available) { 
        Serial.println("MPU6050 not found. Continuing without it.");
    }

    for (int pin = 35; pin <= 42; pin++) {
        pinMode(pin, OUTPUT);
        digitalWrite(pin, LOW);
    }
    for (int pin = 1; pin <= 2; pin++) {
        pinMode(pin, OUTPUT);
        digitalWrite(pin, LOW);
    }
    for (int pin = 4; pin <= 18; pin++) {
        pinMode(pin, OUTPUT);
        digitalWrite(pin, LOW);
    }
    pinMode(6, OUTPUT); digitalWrite(6, LOW);
    pinMode(7, OUTPUT); digitalWrite(7, LOW);

    for (int i = 0; i < NUM_MOTORS; i++) {
        motors[i].ENCODER_A = -1; motors[i].ENCODER_B = -1;
        motors[i].IN1 = -1; motors[i].IN2 = -1;
        motors[i].encoderPos = 0; motors[i].targetPos = 0;
        motors[i].lastError = 0; motors[i].integralError = 0;
        motors[i].controlEnabled = false;
        motors[i].lastAChange = 0; motors[i].lastBChange = 0;
        motors[i].lastAState = false; motors[i].lastBState = false;
    }

    WiFi.config(local_IP, gateway, subnet, primaryDNS, secondaryDNS);
    WiFi.begin(SSID, PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500); Serial.print(".");
    }
    Serial.println("\nWiFi connected");
    Serial.print("IP address: "); Serial.println(WiFi.localIP());
    WiFi.setSleep(false); 
    udp.begin(UDP_PORT);
    xTaskCreatePinnedToCore( udpTask, "UDPTask", 8192, NULL, 1, NULL, 0 );
    Serial.print("UDP Broadcast on port: "); Serial.println(UDP_PORT);
}

void loop() {
    if (millis() - lastTime >= dt) {
        if (targetPosUpdated) {
            targetPosUpdated = false; 
        }
        for (int i = 0; i < NUM_MOTORS; i++) {
            if (motors[i].controlEnabled) {
                controlMotor(i);
            } else {
                setMotor(i, 0); 
            }
        }
        lastTime = millis();
    }
}