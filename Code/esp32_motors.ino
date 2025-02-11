#define ENCODER_A 5
#define ENCODER_B 4
#define IN1 6
#define IN2 7
#define ENA 46
#define ENCODER_A2 17
#define ENCODER_B2 16
#define IN3 18
#define IN4 8
#define ENB 9

// Control parameters
const float kp = 0.5;            // Proportional gain
const float kd = 0.005;            // Derivative gain for damping ~~~~ things that work: 0.005
const int DEAD_ZONE = 2;         // Dead zone in encoder counts
const int MAX_POWER = 3;       // Maximum motor power
const int POSITION_THRESHOLD = 70; // Position threshold for power scaling ~~ prev: 70

// Encoder constants
const int COUNTS_PER_REV = 1975;  // Encoder counts per revolution

// Variables for position tracking - Motor 1
long encoderPos = 0;
long targetPos = 0;
byte lastState = 0;
long lastError = 0;

// Variables for position tracking - Motor 2
long encoderPos2 = 0;
long targetPos2 = 0;
byte lastState2 = 0;
long lastError2 = 0;

// Timing
unsigned long lastTime = 0;
unsigned long dt = 20;

// Logging
unsigned long lastLogTime = 0;
const unsigned long LOG_INTERVAL = 100;

void setup() {
  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  pinMode(ENCODER_A2, INPUT_PULLUP);
  pinMode(ENCODER_B2, INPUT_PULLUP);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, 0);


  Serial.begin(115200);
  Serial.println("time,pos1,target1,error1,power1,dir1,pos2,target2,error2,power2,dir2");
}

void loop() {
  // Update encoder position Motor 1
  byte currentState = (digitalRead(ENCODER_A) << 1) | digitalRead(ENCODER_B);

  if (currentState != lastState) {
    switch (lastState) {
      case 0:
        if (currentState == 2) encoderPos++;
        if (currentState == 1) encoderPos--;
        break;
      case 1:
        if (currentState == 0) encoderPos++;
        if (currentState == 3) encoderPos--;
        break;
      case 2:
        if (currentState == 3) encoderPos++;
        if (currentState == 0) encoderPos--;
        break;
      case 3:
        if (currentState == 1) encoderPos++;
        if (currentState == 2) encoderPos--;
        break;
    }
    lastState = currentState;
  }

  // Update encoder position Motor 2
  byte currentState2 = (digitalRead(ENCODER_A2) << 1) | digitalRead(ENCODER_B2);

  if (currentState2 != lastState2) {
    switch (lastState2) {
      case 0:
        if (currentState2 == 2) encoderPos2--; // INVERTED: Changed ++ to --
        if (currentState2 == 1) encoderPos2++; // INVERTED: Changed -- to ++
        break;
      case 1:
        if (currentState2 == 0) encoderPos2--; // INVERTED: Changed ++ to --
        if (currentState2 == 3) encoderPos2++; // INVERTED: Changed -- to ++
        break;
      case 2:
        if (currentState2 == 3) encoderPos2--; // INVERTED: Changed ++ to --
        if (currentState2 == 0) encoderPos2++; // INVERTED: Changed -- to ++
        break;
      case 3:
        if (currentState2 == 1) encoderPos2--; // INVERTED: Changed ++ to --
        if (currentState2 == 2) encoderPos2++; // INVERTED: Changed -- to ++
        break;
    }
    lastState2 = currentState2;
  }

  // Check for new serial commands
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    if (input.startsWith("a")) {
      int angle = input.substring(1).toInt();
      if (angle >= 0 && angle <= 360) {
        targetPos = (long)angle * COUNTS_PER_REV / 360; // Convert angle to encoder counts
        Serial.print("Motor A - New target angle: ");
        Serial.print(angle);
        Serial.print(" degrees -> Target position: ");
        Serial.println(targetPos);
      }
    } else if (input.startsWith("b")) {
      int angle = input.substring(1).toInt();
      if (angle >= 0 && angle <= 360) {
        targetPos2 = (long)angle * COUNTS_PER_REV / 360; // Convert angle to encoder counts
        Serial.print("Motor B - New target angle: ");
        Serial.print(angle);
        Serial.print(" degrees -> Target position: ");
        Serial.println(targetPos2);
      }
    }
  }

  // Update motor control every dt milliseconds
  if (millis() - lastTime >= dt) {
    // Calculate error Motor 1
    long error = targetPos - encoderPos;
    long errorDelta = error - lastError; // Change in error
    lastError = error;

    // Calculate power Motor 1
    int power = 0;
    if (abs(error) > DEAD_ZONE) {
      // Scale error for power calculation
      float scaled_error = (float)error / POSITION_THRESHOLD;
      if (scaled_error > 1.0) scaled_error = 1.0;
      if (scaled_error < -1.0) scaled_error = -1.0;

      // Apply proportional and derivative control
      power = kp * scaled_error * MAX_POWER + kd * errorDelta;

      // Clamp power
      if (power > MAX_POWER) power = MAX_POWER;
      if (power < -MAX_POWER) power = -MAX_POWER;
    }

    // Set motor power Motor 1
    setMotor(power);

    // Calculate error Motor 2
    long error2 = targetPos2 - encoderPos2;
    long errorDelta2 = error2 - lastError2; // Change in error
    lastError2 = error2;

    // Calculate power Motor 2
    int power2 = 0;
    if (abs(error2) > DEAD_ZONE) {
      // Scale error for power calculation
      float scaled_error2 = (float)error2 / POSITION_THRESHOLD;
      if (scaled_error2 > 1.0) scaled_error2 = 1.0;
      if (scaled_error2 < -1.0) scaled_error2 = -1.0;

      // Apply proportional and derivative control
      power2 = kp * scaled_error2 * MAX_POWER + kd * errorDelta2;

      // Clamp power
      if (power2 > MAX_POWER) power2 = MAX_POWER;
      if (power2 < -MAX_POWER) power2 = -MAX_POWER;
    }

    // Set motor power Motor 2
    setMotor2(power2);


    // Log data
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
      Serial.print(power == 0 ? 0 : (power > 0 ? 1 : -1));
      Serial.print(",");
      Serial.print(encoderPos2);
      Serial.print(",");
      Serial.print(targetPos2);
      Serial.print(",");
      Serial.print(error2);
      Serial.print(",");
      Serial.print(power2);
      Serial.print(",");
      Serial.println(power2 == 0 ? 0 : (power2 > 0 ? 1 : -1));


      lastLogTime = millis();
    }

    lastTime = millis();
  }
}

void setMotor(int power) {
  if (power == 0) {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, 0);
    return;
  }

  if (power > 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
  }

  analogWrite(ENA, abs(power));
}

void setMotor2(int power) {
  if (power == 0) {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
    analogWrite(ENB, 0);
    return;
  }

  if (power > 0) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
  }

  analogWrite(ENB, abs(power));
}