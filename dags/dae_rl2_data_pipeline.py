# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime, timedelta
# import pandas as pd
# import os

# from include.DAE_RL2 import OptimizedAnomalyDetector

# RAW_STREAM_PATH = "/usr/local/airflow/include/datasets/streaming.csv"
# PROCESSED_DATA_PATH = "/usr/local/airflow/include/datasets/dataset.csv"
# PROCESSED_LABEL_PATH = "/usr/local/airflow/include/datasets/processed_y_binary.csv"

# default_args = {
#     "owner": "airflow",
#     "retries": 1,
#     "retry_delay": timedelta(minutes=1),
# }

# dag = DAG(
#     dag_id="dae_rl2_preprocessing_dag",
#     default_args=default_args,
#     description="Tiền xử lý dữ liệu streaming cho mô hình DAE-RL2",
#     start_date=datetime(2025, 7, 17),
#     schedule_interval=None,
#     catchup=False,
# )

# def preprocess_streaming_data():
#     if not os.path.exists(RAW_STREAM_PATH):
#         raise FileNotFoundError(f"Không tìm thấy file: {RAW_STREAM_PATH}")

#     detector = OptimizedAnomalyDetector()
#     X, y_binary, _ = detector.load_and_preprocess_data(RAW_STREAM_PATH)

#     # Lưu dữ liệu đã xử lý
#     pd.DataFrame(X).to_csv(PROCESSED_DATA_PATH, index=False)
#     pd.DataFrame(y_binary, columns=["label"]).to_csv(PROCESSED_LABEL_PATH, index=False)
#     print(f"Đã lưu dữ liệu tiền xử lý tại: {PROCESSED_DATA_PATH} và {PROCESSED_LABEL_PATH}")

# preprocess_task = PythonOperator(
#     task_id="preprocess_streaming_data",
#     python_callable=preprocess_streaming_data,
#     dag=dag,
# )
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import pickle

from include.DAE_RL2 import OptimizedAnomalyDetector

RAW_STREAM_PATH = "/usr/local/airflow/include/datasets/streaming.csv"
TEMP_PICKLE_PATH = "/usr/local/airflow/include/datasets/temp_processed.pkl"
PROCESSED_DATA_PATH = "/usr/local/airflow/include/datasets/dataset.csv"
PROCESSED_LABEL_PATH = "/usr/local/airflow/include/datasets/processed_y_binary.csv"

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    dag_id="dae_rl2_preprocessing_dag_v2",
    default_args=default_args,
    description="Tiền xử lý dữ liệu streaming cho mô hình DAE-RL2 (phiên bản chia task)",
    start_date=datetime(2025, 7, 17),
    schedule_interval=None,
    catchup=False,
)

def check_file_exists():
    if not os.path.exists(RAW_STREAM_PATH):
        raise FileNotFoundError(f"Không tìm thấy file: {RAW_STREAM_PATH}")
    print(f"File tồn tại: {RAW_STREAM_PATH}")

def load_and_process_data():
    detector = OptimizedAnomalyDetector()
    X, y_binary, _ = detector.load_and_preprocess_data(RAW_STREAM_PATH)

    # Lưu tạm ra file pickle
    with open(TEMP_PICKLE_PATH, "wb") as f:
        pickle.dump((X, y_binary), f)
    print("Dữ liệu đã được xử lý và lưu tạm")

def save_processed_data():
    with open(TEMP_PICKLE_PATH, "rb") as f:
        X, y_binary = pickle.load(f)

    pd.DataFrame(X).to_csv(PROCESSED_DATA_PATH, index=False)
    pd.DataFrame(y_binary, columns=["label"]).to_csv(PROCESSED_LABEL_PATH, index=False)
    print(f"Đã lưu dữ liệu tại: {PROCESSED_DATA_PATH} và {PROCESSED_LABEL_PATH}")
    os.remove(TEMP_PICKLE_PATH)

check_file_task = PythonOperator(
    task_id="check_file_exists",
    python_callable=check_file_exists,
    dag=dag,
)

process_data_task = PythonOperator(
    task_id="load_and_process_data",
    python_callable=load_and_process_data,
    dag=dag,
)

save_data_task = PythonOperator(
    task_id="save_processed_data",
    python_callable=save_processed_data,
    dag=dag,
)

check_file_task >> process_data_task >> save_data_task
