#include <Arduino.h>

#define ENCODER_A 8  
#define ENCODER_B 18
#define IN1 12
#define IN2 11

// Control parameters (unchanged)
float kp = 0.9;
float ki = 0.001;
float kd = 0.3;
const int DEAD_ZONE = 10;
const int MAX_POWER = 255;
const int POSITION_THRESHOLD = 5;
const int BRAKE_THRESHOLD = 3;

// Encoder constants (unchanged)
const int COUNTS_PER_REV = 1975;

// Variables (unchanged)
volatile long encoderPos = 0;
long targetPos = 0;
long lastError = 0;
float integralError = 0;

// Timing (unchanged)A70
unsigned long lastTime = 0;
const unsigned long dt = 2;

// Logging (unchanged)
unsigned long lastLogTime = 0;
const unsigned long LOG_INTERVAL = 100;

void setup() {
  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // Configure PWM for IN1 and IN2 using new API
  ledcAttach(IN1, 1000, 8);  // Attach IN1 with 1kHz, 8-bit resolution, auto channel
  ledcAttach(IN2, 1000, 8);  // Attach IN2 with 1kHz, 8-bit resolution, auto channel

  // Initialize motor to brake
  ledcWrite(IN1, 255);  // IN1 HIGH
  ledcWrite(IN2, 255);  // IN2 HIGH

  Serial.begin(115200);
  Serial.println("Time,pos,target,error,power,dir");

  attachInterrupt(digitalPinToInterrupt(ENCODER_A), handleEncoderA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_B), handleEncoderB, CHANGE);
}

// Interrupt Service Routines (unchanged)
void handleEncoderA() {
  bool encAVal = digitalRead(ENCODER_A);
  bool encBVal = digitalRead(ENCODER_B);
  if (encAVal == encBVal) {
    encoderPos--;
  } else {
    encoderPos++;
  }
}

void handleEncoderB() {
  bool encAVal = digitalRead(ENCODER_A);
  bool encBVal = digitalRead(ENCODER_B);
  if (encAVal == encBVal) {
    encoderPos++;
  } else {
    encoderPos--;
  }
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    parseCommand(input);
  }

  if (millis() - lastTime >= dt) {
    long error = targetPos - encoderPos;
    long errorDelta = error - lastError;
    lastError = error;

    if (abs(error) < MAX_POWER / ki) {
      integralError += error * (dt / 1000.0);
    } else {
      integralError = constrain(integralError, -MAX_POWER/ki, MAX_POWER/ki);
    }

    int power = computePower(error, errorDelta);
    setMotor(power);

    logData(error, power);
    lastTime = millis();
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

void setMotor(int power) {
  if (power == 0) {
    ledcWrite(IN1, 255);  // IN1 HIGH
    ledcWrite(IN2, 255);  // IN2 HIGH
  } else if (power > 0) {
    ledcWrite(IN2, 0);    // IN2 LOW
    ledcWrite(IN1, power); // IN1 PWM
  } else {
    ledcWrite(IN1, 0);    // IN1 LOW
    ledcWrite(IN2, -power); // IN2 PWM
  }
}

void logData(long error, int power) {
  if (millis() - lastLogTime >= LOG_INTERVAL) {
    Serial.print(millis());
    Serial.print(",");
    Serial.print(encoderPos);
    Serial.print(",");
    Serial.print(targetPos);
    Serial.print(",");
    Serial.print(error);
    Serial.print(",");
    Serial.print(power);
    Serial.print(",");
    Serial.println(power == 0 ? 0 : (power > 0 ? 1 : -1));
    lastLogTime = millis();
  }
}

void parseCommand(String input) {
  if (input.length() < 2) return;

  char cmd = input.charAt(0);
  String val = input.substring(1);

  if (cmd == 'P') {
    kp = val.toFloat();
    Serial.print("KP set to: ");
    Serial.println(kp);
  } else if (cmd == 'I') {
    ki = val.toFloat();
    Serial.print("KI set to: ");
    Serial.println(ki);
  } else if (cmd == 'D') {
    kd = val.toFloat();
    Serial.print("KD set to: ");
    Serial.println(kd);
  } else if (cmd == 'A') {
    int angle = val.toInt();
    if (angle >= -360 && angle <= 360) {
      targetPos = angle * COUNTS_PER_REV / 360;
      Serial.print("Target: ");
      Serial.print(angle);
      Serial.println("°");
    }
  }
}