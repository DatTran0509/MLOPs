# Image Classification with Hyperparameter Tuning and MLflow Tracking

This project focuses on image classification using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It includes **automated hyperparameter tuning with Optuna** and **training experiment tracking using MLflow**.
---
### 🧑‍💻 **Thông tin Nhóm**

| Họ và Tên              | MSSV      |
|------------------------|-----------|
| **Trần Ngọc Thiện**    | 21521465  |
| **Trịnh Thị Lan Anh**  | 22520083  |
| **Trần Quang Đạt**     | 22520236  |
| **Vương Dương Thái Hà** | 22520375  |
---


## DEMO LAB2
[🔗 VIDEO DEMO →](https://drive.google.com/file/d/1WrXHEIFrORCimSjhhIZRO7D9iyzhxiO4/view?usp=sharing)

# Kiến trúc 
* **Monitoring**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API App       │    │   Node Exporter │    │   Other Apps    │
│   (Port 8000)   │    │   (Port 9100)   │    │   (Various)     │
│                 │    │                 │    │                 │
│ /metrics        │    │ /metrics        │    │ /metrics        │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Prometheus          │
                    │     (Port 9090)         │
                    │                         │
                    │ - Scrapes metrics       │
                    │ - Stores time series    │
                    │ - Evaluates rules       │
                    │ - Sends alerts          │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      Grafana            │
                    │      (Port 3000)        │
                    │                         │
                    │ - Visualizes data       │
                    │ - Creates dashboards    │
                    │ - Manages alerts        │
                    └─────────────────────────┘

```
* **Logging**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Docker    │───▶│   Fluentd   │───▶│    Loki     │
│ Containers  │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌─────────────┐           │
│ Prometheus  │───▶│   Grafana   │◀──────────┘
│             │    │ (Dashboard) │
└─────────────┘    └─────────────┘
       │
┌─────────────┐    ┌─────────────┐
│Alertmanager │───▶│ Webhook     │
│             │    │ Logger      │
└─────────────┘    └─────────────┘

```
* **Cấu hình** 
1. [Install Python3.12](https://www.python.org/downloads/release/python-3120/)
2. [Install Docker](https://docs.docker.com/engine/install/)
3. Cài đặt thư viện
```
pip install -r requirement.txt
```
1. Chạy docker compose
```
docker compose up -d
``` 
5.Cài đặt Dashboard Monitoring với Grafana
```
Để thiết lập dashboard monitoring, thực hiện các bước sau:

Truy cập Grafana: Mở trình duyệt và truy cập vào Grafana tại địa chỉ http://localhost:3000

Cấu hình Data Source:

Điều hướng đến tab Data Sources trong menu cài đặt

Chọn Add data source và chọn Prometheus

Nhập địa chỉ Prometheus: http://prometheus:9090/

Nhấn Save & Test để xác nhận kết nối thành công

Thêm Dashboard:

Vào phần Dashboards từ menu chính

Nhấn New và chọn Import

Bạn có hai lựa chọn:

Custom Dashboard: Copy và paste nội dung từ các file dashboard có sẵn trong thư mục dashboard/

Pre-built Dashboard: Sử dụng dashboard ID 1860 (Node Exporter Full) để có dashboard monitoring cơ bản

Hoàn tất: Sau khi import thành công, dashboard sẽ hiển thị các metrics từ Prometheus và bạn có thể bắt đầu theo dõi hệ thống.

Grafana sẽ tự động kết nối với Prometheus để hiển thị các metrics và cảnh báo trong thời gian thực, hỗ trợ việc trực quan hóa dữ liệu log từ Fluentd trong hệ thống MLOps của bạn
```
6. Logging với Loki
```
Cấu hình Data Source cho Loki:

Điều hướng đến Connections trong menu bên trái

Chọn Add new connection và tìm kiếm Loki

Chọn Loki data source và nhấn Create a Loki data source

Nhập địa chỉ Loki: http://loki:3100 (hoặc http://localhost:3100 nếu chạy local)

Cấu hình authentication nếu cần thiết (Basic Auth, TLS, etc.)

Nhấn Save & Test để xác nhận kết nối thành công

Cấu hình Data Source cho Prometheus (nếu cần metrics):

Thêm data source mới và chọn Prometheus

Nhập địa chỉ Prometheus: http://prometheus:9090/

Nhấn Save & Test

Thêm Dashboard:

Vào phần Dashboards từ menu chính

Nhấn New và chọn Import

Bạn có các lựa chọn:

Custom Dashboard: Copy và paste nội dung từ các file dashboard có sẵn trong thư mục dashboard/

Pre-built Dashboard: Sử dụng dashboard ID phù hợp với Loki logs

Loki-specific Dashboard: Tạo dashboard mới với LogQL queries để truy vấn logs từ Loki
```