#include "esp_camera.h"
#include <WiFi.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "fb_gfx.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "esp_http_server.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>              // Using standard Wire.h
#include <ArduinoJson.h>       // For creating JSON responses
#include <Adafruit_ADS1X15.h>  // Added for ADS1115 support

// Define the MIN macro
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Replace with your network credentials
const char *ssid = "Wifi de sparky";
const char *password = "$famcruz$";

#define PART_BOUNDARY "123456789000000000000987654321"

// Define the camera model
#define CAMERA_MODEL_AI_THINKER

#if defined(CAMERA_MODEL_AI_THINKER)
  #define PWDN_GPIO_NUM     32
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM      0
  #define SIOD_GPIO_NUM     26
  #define SIOC_GPIO_NUM     27
  #define Y9_GPIO_NUM       35
  #define Y8_GPIO_NUM       34
  #define Y7_GPIO_NUM       39
  #define Y6_GPIO_NUM       36
  #define Y5_GPIO_NUM       21
  #define Y4_GPIO_NUM       19
  #define Y3_GPIO_NUM       18
  #define Y2_GPIO_NUM        5
  #define VSYNC_GPIO_NUM    25
  #define HREF_GPIO_NUM     23
  #define PCLK_GPIO_NUM     22
#else
  #error "Camera model not selected"
#endif

static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t stream_httpd = NULL;
framesize_t current_framesize = FRAMESIZE_VGA;
bool frame_size_changed = false;

// --- IMU Setup ---
#define I2C_SDA 14  // SDA Connected to GPIO 14
#define I2C_SCL 15  // SCL Connected to GPIO 15
Adafruit_MPU6050 mpu;

// --- ADS1115 Setup ---
Adafruit_ADS1115 ads;  // ADS1115 instance with address 0x48

// --- JSON Buffers ---
StaticJsonDocument<200> imu_doc;  // For IMU data
char imu_json_buffer[200];
StaticJsonDocument<200> ads_doc;  // For ADS data
char ads_json_buffer[200];

// --- Stream Handler ---
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t *_jpg_buf = NULL;
  char *part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) {
    return res;
  }

  while (true) {
    if (frame_size_changed) {
      sensor_t *s = esp_camera_sensor_get();
      s->set_framesize(s, current_framesize);
      frame_size_changed = false;
    }

    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
    } else {
      if (fb->width > 400) {
        if (fb->format != PIXFORMAT_JPEG) {
          bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
          esp_camera_fb_return(fb);
          fb = NULL;
          if (!jpeg_converted) {
            Serial.println("JPEG compression failed");
            res = ESP_FAIL;
          }
        } else {
          _jpg_buf_len = fb->len;
          _jpg_buf = fb->buf;
        }
      }
    }
    if (res == ESP_OK) {
      size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }
    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if (_jpg_buf) {
      free(_jpg_buf);
      _jpg_buf = NULL;
    }
    if (res != ESP_OK) {
      break;
    }
  }
  return res;
}

// --- Command Handler ---
static esp_err_t cmd_handler(httpd_req_t *req) {
  char buf[100];
  int ret, remaining = req->content_len;

  while (remaining > 0) {
    if ((ret = httpd_req_recv(req, buf, MIN(remaining, sizeof(buf)))) <= 0) {
      if (ret == HTTPD_SOCK_ERR_TIMEOUT) {
        continue;
      }
      return ESP_FAIL;
    }
    buf[ret] = 0; // Null-terminate
    String command = String(buf);
    if (command.startsWith("framesize:")) {
      String framesize_str = command.substring(10);
      Serial.print("Changing framesize to: ");
      Serial.println(framesize_str);

      if (framesize_str == "96X96") current_framesize = FRAMESIZE_96X96;
      else if (framesize_str == "QQVGA") current_framesize = FRAMESIZE_QQVGA;
      else if (framesize_str == "128X128") current_framesize = FRAMESIZE_128X128;
      else if (framesize_str == "QCIF") current_framesize = FRAMESIZE_QCIF;
      else if (framesize_str == "HQVGA") current_framesize = FRAMESIZE_HQVGA;
      else if (framesize_str == "240X240") current_framesize = FRAMESIZE_240X240;
      else if (framesize_str == "QVGA") current_framesize = FRAMESIZE_QVGA;
      else if (framesize_str == "320X320") current_framesize = FRAMESIZE_320X320;
      else if (framesize_str == "CIF") current_framesize = FRAMESIZE_CIF;
      else if (framesize_str == "HVGA") current_framesize = FRAMESIZE_HVGA;
      else if (framesize_str == "VGA") current_framesize = FRAMESIZE_VGA;
      else if (framesize_str == "SVGA") current_framesize = FRAMESIZE_SVGA;
      else if (framesize_str == "XGA") current_framesize = FRAMESIZE_XGA;
      else if (framesize_str == "HD") current_framesize = FRAMESIZE_HD;
      else if (framesize_str == "SXGA") current_framesize = FRAMESIZE_SXGA;
      else if (framesize_str == "UXGA") current_framesize = FRAMESIZE_UXGA;

      frame_size_changed = true;
      Serial.print("Frame size set to: ");
      Serial.println(framesize_str);
    }
    remaining -= ret;
  }

  httpd_resp_send(req, NULL, 0);
  return ESP_OK;
}

// --- IMU Data Handler ---
static esp_err_t imu_data_handler(httpd_req_t *req) {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  imu_doc.clear();
  imu_doc["accel_x"] = a.acceleration.x;
  imu_doc["accel_y"] = a.acceleration.y;
  imu_doc["accel_z"] = a.acceleration.z;
  imu_doc["gyro_x"] = g.gyro.x;
  imu_doc["gyro_y"] = g.gyro.y;
  imu_doc["gyro_z"] = g.gyro.z;
  imu_doc["temp"] = temp.temperature;

  serializeJson(imu_doc, imu_json_buffer);

  httpd_resp_set_type(req, "application/json");
  httpd_resp_send(req, imu_json_buffer, strlen(imu_json_buffer));
  return ESP_OK;
}

// --- ADS Data Handler ---
static esp_err_t ads_data_handler(httpd_req_t *req) {
  int16_t a0 = ads.readADC_SingleEnded(0);  // Read A0
  int16_t a1 = ads.readADC_SingleEnded(1);  // Read A1
  int16_t a2 = ads.readADC_SingleEnded(2);  // Read A2
  int16_t a3 = ads.readADC_SingleEnded(3);  // Read A3

  ads_doc.clear();
  ads_doc["A0"] = a0;
  ads_doc["A1"] = a1;
  ads_doc["A2"] = a2;
  ads_doc["A3"] = a3;

  serializeJson(ads_doc, ads_json_buffer);

  httpd_resp_set_type(req, "application/json");
  httpd_resp_send(req, ads_json_buffer, strlen(ads_json_buffer));
  return ESP_OK;
}

// --- Start HTTP Server ---
void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  httpd_uri_t cmd_uri = {
    .uri       = "/control",
    .method    = HTTP_POST,
    .handler   = cmd_handler,
    .user_ctx  = NULL
  };

  httpd_uri_t imu_data_uri = {
    .uri       = "/imu_data",
    .method    = HTTP_GET,
    .handler   = imu_data_handler,
    .user_ctx  = NULL
  };

  httpd_uri_t ads_data_uri = {
    .uri       = "/ads_data",
    .method    = HTTP_GET,
    .handler   = ads_data_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    httpd_register_uri_handler(stream_httpd, &cmd_uri);
    httpd_register_uri_handler(stream_httpd, &imu_data_uri);
    httpd_register_uri_handler(stream_httpd, &ads_data_uri);  // Register ADS data handler
  }
}

// --- Setup Function ---
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  Serial.setDebugOutput(false);

  // --- I2C and Sensor Initialization ---
  Wire.begin(I2C_SDA, I2C_SCL, 100000);  // Initialize I2C on pins 14 and 15

  // MPU6050 Initialization
  while (!mpu.begin(0x68, &Wire)) {
    Serial.println("Failed to find MPU6050 chip, waiting...");
    delay(1000);
  }
  Serial.println("MPU6050 Found!");
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_5_HZ);

  // ADS1115 Initialization
  while (!ads.begin(0x48, &Wire)) {
    Serial.println("Failed to initialize ADS1115, waiting...");
    delay(1000);

  }
  Serial.println("ADS1115 Initialized");

  // --- Camera Initialization ---
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_XGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  // --- WiFi Connection ---
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  // --- Print URLs ---
  Serial.print("Camera Stream Ready! Go to: http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");
  Serial.print("IMU Data Ready! Go to: http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/imu_data");
  Serial.print("ADS Data Ready! Go to: http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/ads_data");

  startCameraServer();
}

// --- Loop Function ---
void loop() {
  delay(1);  // Keep loop minimal for HTTP server responsiveness
}