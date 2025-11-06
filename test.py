import yfinance as yf
import pandas as pd
import os

# --- Cấu hình ---
ticker_symbol = "GC=F"  # Mã ticker cho Gold Dec 25 (GC=F)
csv_file_name = "data/raw/gold_price.csv"  # Tên file CSV để lưu

# --- Tải dữ liệu ---
print(f"Đang tiến hành tải dữ liệu cho mã: {ticker_symbol}...")

try:
    # 1. Tạo đối tượng ticker
    ticker_data = yf.Ticker(ticker_symbol)

    # 2. Tải toàn bộ lịch sử dữ liệu ("max")
    # Bạn cũng có thể thay "max" bằng khoảng thời gian cụ thể
    # Ví dụ: period="10y" (10 năm)
    # Hoặc: start="2013-01-01", end="2023-12-31" (giống dự án của bạn)
    hist_data = ticker_data.history(period="max")

    if hist_data.empty:
        print(f"Không tìm thấy dữ liệu cho mã {ticker_symbol}. Vui lòng kiểm tra lại mã ticker.")
    else:
        # 3. Lưu dữ liệu ra file CSV
        # index=True để giữ lại cột 'Date' làm chỉ mục
        hist_data.to_csv(csv_file_name, index=True)

        print("-" * 30)
        print(f"THÀNH CÔNG!")
        print(f"Đã lưu dữ liệu vào file: {os.path.abspath(csv_file_name)}")

        # 4. In ra 5 dòng đầu tiên để xem trước
        print("\nXem trước 5 dòng dữ liệu đầu tiên:")
        print(hist_data.head())

except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")