#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include <Wire.h>

// MPU6050 DMP Libraries
#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"

// WiFi credentials
#define SSID "TT"
#define PASSWORD "12345678"
#define UDP_PORT 12345

// Set static IP configuration
IPAddress local_IP(192, 168, 137, 101);
IPAddress gateway(192, 168, 137, 1);
IPAddress subnet(255, 255, 255, 0);
IPAddress primaryDNS(8, 8, 8, 8);
IPAddress secondaryDNS(8, 8, 4, 4);

const int NUM_MOTORS = 4;

float kp = 0.9;
float ki = 0.001;
float kd = 0.3;
int DEAD_ZONE = 10;
int MAX_POWER = 255;
int POSITION_THRESHOLD = 5;

const int COUNTS_PER_REV = 1975;

unsigned long lastTime = 0;
const unsigned long dt = 2;

WiFiUDP udp;
volatile bool targetPosUpdated = false;
long debugTargetPos[4] = {0, 0, 0, 0};

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

#define I2C_SDA 19
#define I2C_SCL 20

MPU6050 mpu;

#define EARTH_GRAVITY_MS2 9.80665
#define DEG_TO_RAD        0.017453292519943295769236907684886
#define RAD_TO_DEG        57.295779513082320876798154814105

bool dmpReady = false;
uint8_t mpuIntStatus;
uint8_t devStatus;
uint16_t packetSize;
uint8_t fifoBuffer[64];

Quaternion q_mpu; // Renamed from q to q_mpu
VectorInt16 aa_raw_fifo;      
VectorInt16 aa_linear_sensor; 
VectorInt16 aaWorld;          
VectorFloat gravity;
float ypr[3];

float currentQuaternion[4];
float currentWorldAccel[3];
float currentYPR[3];

void IRAM_ATTR handleEncoderA(void* arg) {
    Motor* motor = (Motor*)arg;
    unsigned long now = micros();
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

void setMotor(int motorIndex, int power) {
    Motor &motor = motors[motorIndex];
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
    if (ki != 0 && abs(error) < MAX_POWER / abs(ki)) { // Check ki is not zero and use abs(ki)
        motor.integralError += error * (dt / 1000.0);
    } else if (ki != 0) {
        motor.integralError = constrain(motor.integralError, -MAX_POWER / abs(ki), MAX_POWER / abs(ki));
    } else {
        motor.integralError = 0;
    }
    int power = computePower(error, errorDelta) + ki * motor.integralError;
    setMotor(motorIndex, power);
}

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

        if (motor.ENCODER_A != -1 && digitalPinToInterrupt(motor.ENCODER_A) != NOT_AN_INTERRUPT) detachInterrupt(digitalPinToInterrupt(motor.ENCODER_A));
        if (motor.ENCODER_B != -1 && digitalPinToInterrupt(motor.ENCODER_B) != NOT_AN_INTERRUPT) detachInterrupt(digitalPinToInterrupt(motor.ENCODER_B));
        
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
    StaticJsonDocument<512> doc;
    if (dmpReady) {
        JsonObject imu_data = doc.createNestedObject("dmp_data");
        JsonObject quat_data = imu_data.createNestedObject("quaternion");
        quat_data["w"] = currentQuaternion[0];
        quat_data["x"] = currentQuaternion[1];
        quat_data["y"] = currentQuaternion[2];
        quat_data["z"] = currentQuaternion[3];
        JsonObject accel_world_data = imu_data.createNestedObject("world_accel_mps2");
        accel_world_data["ax"] = currentWorldAccel[0];
        accel_world_data["ay"] = currentWorldAccel[1];
        accel_world_data["az"] = currentWorldAccel[2];
        JsonObject ypr_data = imu_data.createNestedObject("ypr_deg");
        ypr_data["yaw"] = currentYPR[0];
        ypr_data["pitch"] = currentYPR[1];
        ypr_data["roll"] = currentYPR[2];
        doc["dmp_status"] = "ready";
    } else {
        doc["dmp_status"] = "not_ready";
        doc["error"] = "MPU6050 DMP not initialized or error";
    }
    char json[512];
    size_t len = serializeJson(doc, json);
    udp.beginPacket(senderIP, senderPort);
    udp.write((uint8_t*)json, len);
    udp.endPacket();
}

// Helper function for manual quaternion rotation
void rotateVectorByQuaternion(VectorInt16* v_sensor, Quaternion* q_rot, VectorInt16* v_world) {
    float vx = v_sensor->x;
    float vy = v_sensor->y;
    float vz = v_sensor->z;

    float qw = q_rot->w;
    float qx = q_rot->x;
    float qy = q_rot->y;
    float qz = q_rot->z;

    // Pre-calculate products for efficiency
    float _2qx = 2.0f * qx;
    float _2qy = 2.0f * qy;
    float _2qz = 2.0f * qz;
    // float _2qw = 2.0f * qw; // Not directly used in this form of the matrix

    float _2qwqx = _2qx * qw; // 2qwqx
    float _2qwqy = _2qy * qw; // 2qwqy
    float _2qwqz = _2qz * qw; // 2qwqz
    float _2qxqy = _2qx * qy; // 2qxqy
    float _2qxqz = _2qx * qz; // 2qxqz
    float _2qyqz = _2qy * qz; // 2qyqz
    
    float qx_sq = qx * qx;
    float qy_sq = qy * qy;
    float qz_sq = qz * qz;
    // float qw_sq = qw * qw; // Not directly used in this form of the matrix

    // Rotation matrix elements
    // Rxx = 1 - 2*qy^2 - 2*qz^2  (qw^2 + qx^2 - qy^2 - qz^2 if normalized)
    // Rxy = 2*qx*qy - 2*qw*qz
    // Rxz = 2*qx*qz + 2*qw*qy
    // Ryx = 2*qx*qy + 2*qw*qz
    // Ryy = 1 - 2*qx^2 - 2*qz^2  (qw^2 - qx^2 + qy^2 - qz^2 if normalized)
    // Ryz = 2*qy*qz - 2*qw*qx
    // Rzx = 2*qx*qz - 2*qw*qy
    // Rzy = 2*qy*qz + 2*qw*qx
    // Rzz = 1 - 2*qx^2 - 2*qy^2  (qw^2 - qx^2 - qy^2 + qz^2 if normalized)

    // Assuming q is a unit quaternion (qw^2 + qx^2 + qy^2 + qz^2 = 1)
    // If not, it should be normalized before this function.
    // The DMP output quaternion should be a unit quaternion.

    v_world->x = vx * (1.0f - 2.0f*qy_sq - 2.0f*qz_sq) + vy * (_2qxqy - _2qwqz)         + vz * (_2qxqz + _2qwqy);
    v_world->y = vx * (_2qxqy + _2qwqz)         + vy * (1.0f - 2.0f*qx_sq - 2.0f*qz_sq) + vz * (_2qyqz - _2qwqx);
    v_world->z = vx * (_2qxqz - _2qwqy)         + vy * (_2qyqz + _2qwqx)         + vz * (1.0f - 2.0f*qx_sq - 2.0f*qy_sq);
}

void updateDMPData() {
    if (!dmpReady) return;
    if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
        mpu.dmpGetQuaternion(&q_mpu, fifoBuffer); 
        currentQuaternion[0] = q_mpu.w;
        currentQuaternion[1] = q_mpu.x;
        currentQuaternion[2] = q_mpu.y;
        currentQuaternion[3] = q_mpu.z;

        mpu.dmpGetGravity(&gravity, &q_mpu);
        mpu.dmpGetAccel(&aa_raw_fifo, fifoBuffer);
        mpu.dmpGetLinearAccel(&aa_linear_sensor, &aa_raw_fifo, &gravity);
        rotateVectorByQuaternion(&aa_linear_sensor, &q_mpu, &aaWorld);

        currentWorldAccel[0] = aaWorld.x * mpu.get_acce_resolution() * EARTH_GRAVITY_MS2;
        currentWorldAccel[1] = aaWorld.y * mpu.get_acce_resolution() * EARTH_GRAVITY_MS2;
        currentWorldAccel[2] = aaWorld.z * mpu.get_acce_resolution() * EARTH_GRAVITY_MS2;

        mpu.dmpGetYawPitchRoll(ypr, &q_mpu, &gravity);
        currentYPR[0] = ypr[0] * RAD_TO_DEG;
        currentYPR[1] = ypr[1] * RAD_TO_DEG;
        currentYPR[2] = ypr[2] * RAD_TO_DEG;
    }
}

void udpTask(void *pvParameters) {
    unsigned long lastSendTime = 0;
    unsigned long sendInterval = 50;
    IPAddress broadcastIP(255, 255, 255, 255);

    while (true) {
        if (WiFi.status() != WL_CONNECTED) {
            WiFi.disconnect(true, true);
            WiFi.mode(WIFI_STA);
            WiFi.config(local_IP, gateway, subnet, primaryDNS, secondaryDNS);
            WiFi.begin(SSID, PASSWORD);
            while (WiFi.status() != WL_CONNECTED) {
                vTaskDelay(500 / portTICK_PERIOD_MS);
            }
            udp.begin(UDP_PORT);
        }

        int packetSizeUDP = udp.parsePacket();
        if (packetSizeUDP) {
            char packetBuffer[1024];
            int len = udp.read(packetBuffer, sizeof(packetBuffer) - 1);
            packetBuffer[len] = 0;
            StaticJsonDocument<1024> doc;
            DeserializationError error = deserializeJson(doc, packetBuffer);
            if (!error && doc.containsKey("command")) {
                String command = doc["command"].as<String>();
                IPAddress senderIP = udp.remoteIP();
                int senderPort = udp.remotePort();
                if (command == "set_control_params") {
                    handle_set_control_params(doc);
                } else if (command == "set_angles") {
                    if (doc.containsKey("angles") && doc["angles"].is<JsonArray>()) {
                        size_t arraySize = doc["angles"].size();
                        int tempAngles[NUM_MOTORS]; 
                        for (size_t i = 0; i < arraySize && i < NUM_MOTORS; i++) {
                            tempAngles[i] = static_cast<int>(doc["angles"][i].as<float>());
                        }
                        handle_set_angles(tempAngles, arraySize);
                    }
                } else if (command == "set_all_pins") {
                    handle_set_all_pins(doc);
                } else if (command == "set_control_status") {
                    handle_set_control_status(doc);
                } else if (command == "reset_all") {
                    handle_reset_all();
                } else if (command == "get_imu_data") {
                    handle_get_imu_data(senderIP, senderPort);
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
                udp.beginPacket(senderIP, senderPort);
                udp.write((uint8_t*)response, responseLen);
                udp.endPacket();
                vTaskDelay(2 / portTICK_PERIOD_MS);
            }
        }

        updateDMPData();

        if (millis() - lastSendTime >= sendInterval) {
            StaticJsonDocument<1024> doc;
            JsonArray angles_json = doc.createNestedArray("angles");
            JsonArray encoderPos_arr = doc.createNestedArray("encoderPos");
            JsonArray targetPos_arr = doc.createNestedArray("targetPos");
            bool allMotorsCurrentlyEnabled = true;
            for (int i = 0; i < NUM_MOTORS; i++) {
                float angle = (float)motors[i].encoderPos * 360.0f / COUNTS_PER_REV;
                angles_json.add(angle);
                encoderPos_arr.add(motors[i].encoderPos);
                targetPos_arr.add(motors[i].targetPos);
                if (!motors[i].controlEnabled) {
                    allMotorsCurrentlyEnabled = false;
                }
            }
            doc["esp_control_fully_enabled"] = allMotorsCurrentlyEnabled;
            doc["dmp_ready"] = dmpReady;

            if (dmpReady) {
                JsonObject dmp_data_out = doc.createNestedObject("dmp_data");
                JsonObject quat_data = dmp_data_out.createNestedObject("quaternion");
                quat_data["w"] = currentQuaternion[0];
                quat_data["x"] = currentQuaternion[1];
                quat_data["y"] = currentQuaternion[2];
                quat_data["z"] = currentQuaternion[3];
                JsonObject accel_world_data = dmp_data_out.createNestedObject("world_accel_mps2");
                accel_world_data["ax"] = currentWorldAccel[0];
                accel_world_data["ay"] = currentWorldAccel[1];
                accel_world_data["az"] = currentWorldAccel[2];
                JsonObject ypr_data = dmp_data_out.createNestedObject("ypr_deg");
                ypr_data["yaw"] = currentYPR[0];
                ypr_data["pitch"] = currentYPR[1];
                ypr_data["roll"] = currentYPR[2];
            }

            char json[1024];
            size_t len = serializeJson(doc, json);
            if (len > 0 && len < sizeof(json)) {
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
    #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
        Wire.begin(I2C_SDA, I2C_SCL);
        Wire.setClock(400000);
    #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
        Fastwire::setup(400, true);
    #endif

    mpu.initialize();
    pinMode(LED_BUILTIN, OUTPUT); 

    if (mpu.testConnection()) {
        devStatus = mpu.dmpInitialize();
        mpu.setXGyroOffset(0); 
        mpu.setYGyroOffset(0);
        mpu.setZGyroOffset(0);
        mpu.setXAccelOffset(0);
        mpu.setYAccelOffset(0);
        mpu.setZAccelOffset(0);

        if (devStatus == 0) {
            mpu.CalibrateAccel(10);
            mpu.CalibrateGyro(10);
            mpu.setDMPEnabled(true);
            dmpReady = true;
            packetSize = mpu.dmpGetFIFOPacketSize();
            digitalWrite(LED_BUILTIN, HIGH); 
        } else {
            dmpReady = false; 
            digitalWrite(LED_BUILTIN, LOW); 
        }
    } else {
        dmpReady = false; 
        digitalWrite(LED_BUILTIN, LOW); 
    }

    for (int i = 0; i < NUM_MOTORS; i++) {
        motors[i].ENCODER_A = -1; motors[i].ENCODER_B = -1;
        motors[i].IN1 = -1; motors[i].IN2 = -1;
        motors[i].encoderPos = 0; motors[i].targetPos = 0;
        motors[i].lastError = 0; motors[i].integralError = 0;
        motors[i].controlEnabled = false;
        motors[i].lastAChange = 0; motors[i].lastBChange = 0;
        motors[i].lastAState = false; motors[i].lastBState = false;
    }

    WiFi.mode(WIFI_STA);
    WiFi.config(local_IP, gateway, subnet, primaryDNS, secondaryDNS);
    WiFi.begin(SSID, PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500); 
    }
    WiFi.setSleep(false);
    
    udp.begin(UDP_PORT);

    xTaskCreatePinnedToCore( udpTask, "UDPTask", 1024 * 12, NULL, 1, NULL, 0 );
}

void loop() {
    if (millis() - lastTime >= dt) {
        for (int i = 0; i < NUM_MOTORS; i++) {
            if (motors[i].controlEnabled) {
                controlMotor(i);
            }
        }
        lastTime = millis();
    }
    vTaskDelay(1 / portTICK_PERIOD_MS);
}