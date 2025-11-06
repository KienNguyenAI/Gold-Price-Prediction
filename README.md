# Dự án Dự đoán Giá vàng (Gold Price Prediction)

Dự án này sử dụng mô hình LSTM (Long Short-Term Memory) để dự đoán giá vàng dựa trên dữ liệu lịch sử 10 năm (2013-2023).

## 🚀 Cấu trúc thư mục

/
├── data/
│   ├── raw/
│   │   └── Gold Price (2013-2023).csv
│   └── processed/
│   └── final/
├── models/
│   ├── lstm_model.h5
│   └── price_scaler.pkl
├── notebooks/
│   └── gold_price_exploration.ipynb
├── src/
│   ├── data_processing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── reports/
│   └── figures/
└── requirements.txt

## 🛠️ Cài đặt

1.  Clone repository này:
    `git clone ...`
2.  Tạo môi trường ảo (khuyến nghị):
    `python -m venv venv`
    `source venv/bin/activate` (hoặc `.\venv\Scripts\activate` trên Windows)
3.  Cài đặt các thư viện cần thiết:
    `pip install -r requirements.txt`

## ⚙️ Cách chạy

Bạn phải chạy các script theo thứ tự sau:

**1. Xử lý dữ liệu:**
Script này sẽ tải dữ liệu thô, làm sạch, chuẩn hóa và tạo các cửa sổ trượt (windows).
`python src/data_processing.py`

**2. Huấn luyện mô hình:**
Script này tải dữ liệu đã xử lý, xây dựng mô hình LSTM và huấn luyện.
`python src/train.py`

**3. Đánh giá mô hình:**
Script này tải mô hình đã huấn luyện, dự đoán trên tập test và lưu biểu đồ kết quả.
`python src/evaluate.py`

**4. Dự đoán mới (Tùy chọn):**
Sử dụng mô hình đã lưu để dự đoán giá của ngày tiếp theo.
`python src/predict.py`