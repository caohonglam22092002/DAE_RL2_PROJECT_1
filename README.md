DAE_RL2_PROJECT_1
DAE_RL2_PROJECT_1 là dự án xây dựng pipeline xử lý dữ liệu IoT phục vụ bài toán phát hiện bất thường (Anomaly Detection).
Pipeline được triển khai bằng Astro + Apache Airflow, tự động hóa các bước:
Kiểm tra dữ liệu đầu vào
Tiền xử lý và chuẩn hóa dữ liệu IoT
Lưu trữ dữ liệu đã xử lý để phục vụ huấn luyện mô hình phát hiện bất thường

Cấu trúc dự án
├── dags/                     # DAG của Airflow
│   └── dae_rl2_data_pipeline.py
├── include/
│   └── DAE_RL2.py            # Module tiền xử lý & phát hiện bất thường
├── tests/                    # Unit test cho pipeline
├── .gitignore                # Bỏ qua dataset và file tạm
├── Dockerfile                # Định nghĩa image chạy Airflow
├── compose.yaml              # Docker Compose khởi chạy Airflow
├── requirements.txt          # Thư viện Python cần thiết
└── README.md                 # Hướng dẫn sử dụng
Lưu ý: Dataset không nằm trong repo.
Bạn cần tải dataset từ nguồn ngoài và đặt tại include/datasets/.

Yêu cầu hệ thống
Python 3.9 trở lên
Docker và Docker Compose
Astro CLI (xem hướng dẫn cài đặt tại trang chủ Astronomer)

Cài đặt và chạy pipeline
1. Clone repo
git clone git@github.com:caohonglam22092002/DAE_RL2_PROJECT_1.git
cd DAE_RL2_PROJECT_1
2. Chuẩn bị dataset
Tải dataset IoT từ nguồn ngoài (ví dụ Google Drive, HuggingFace)
Đặt dataset vào:
include/datasets/
Thư mục này đã được .gitignore để tránh đẩy file lớn lên GitHub.
3. Cài đặt Python packages (nếu chạy local)
pip install -r requirements.txt
4. Khởi động Airflow bằng Astro CLI
astro dev start
Truy cập Airflow UI tại: http://localhost:8080
Đăng nhập: admin / admin (mặc định)
5. Kích hoạt DAG
Bật DAG dae_rl2_data_pipeline trên Airflow UI
Trigger DAG để thực hiện các bước:
check_file_exists – Kiểm tra dataset
load_and_process_data – Tiền xử lý dữ liệu
save_processed_data – Lưu dữ liệu đã chuẩn hóa
Kết quả pipeline
Dữ liệu IoT đã chuẩn hóa được lưu tại:
include/datasets/dataset.csv
include/datasets/processed_y_binary.csv