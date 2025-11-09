import pandas as pd
import numpy as np

# Tên file đầu vào và đầu ra
file_moi = "gold_price_2013_2025.csv"
output_filename = "../../data/raw/gold_2013_2025.csv"

print(f"Đang xử lý file: {file_moi}")

try:
    # 1. Tải file mới
    df = pd.read_csv(file_moi)

    # 2. Xử lý cột Date và Sắp xếp
    # Chuyển đổi 'Date' sang datetime, xử lý múi giờ bằng utc=True
    # và .dt.normalize() để xóa thông tin giờ, chỉ giữ lại ngày
    df['Date_dt'] = pd.to_datetime(df['Date'], utc=True).dt.normalize()

    # Sắp xếp TĂNG DẦN (cũ nhất trước) để tính toán 'Change %'
    df = df.sort_values(by='Date_dt', ascending=True)

    # 3. Tính toán cột 'Change %'
    # pct_change() tính % thay đổi so với hàng TRƯỚC ĐÓ
    df['Change %_calc'] = df['Close'].pct_change()

    # 4. Sắp xếp lại GIẢM DẦN (mới nhất trước) để giống file mẫu
    df = df.sort_values(by='Date_dt', ascending=False).reset_index(drop=True)

    # --- BẮT ĐẦU ĐỊNH DẠNG CHUỖI (string formatting) ---

    # 5. Định dạng Cột 'Date' (MM/DD/YYYY)
    df['Date'] = df['Date_dt'].dt.strftime('%m/%d/%Y')

    # 6. Đổi tên và định dạng Cột 'Price' (từ 'Close')
    # Định dạng: 2 số thập phân, có dấu phẩy hàng nghìn
    df['Price'] = df['Close'].apply(lambda x: f"{x:,.2f}")

    # 7. Định dạng Cột 'Open', 'High', 'Low'
    df['Open'] = df['Open'].apply(lambda x: f"{x:,.2f}")
    df['High'] = df['High'].apply(lambda x: f"{x:,.2f}")
    df['Low'] = df['Low'].apply(lambda x: f"{x:,.2f}")


    # 8. Đổi tên và định dạng Cột 'Vol.' (từ 'Volume')
    # Định dạng: chia 1000, 2 số thập phân, thêm "K"
    def format_volume(v):
        if pd.isna(v) or v == 0:
            return np.nan  # Giống file mẫu có NaN
        return f"{(v / 1000):.2f}K"


    df['Vol.'] = df['Volume'].apply(format_volume)


    # 9. Định dạng Cột 'Change %'
    # Định dạng: nhân 100, 2 số thập phân, thêm "%"
    def format_change(c):
        if pd.isna(c):
            return np.nan  # Hàng đầu tiên (ngày mới nhất) sẽ là NaN
        return f"{(c * 100):.2f}%"


    df['Change %'] = df['Change %_calc'].apply(format_change)

    # 10. Chọn và sắp xếp lại các cột cuối cùng
    # Thứ tự file mẫu: ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
    final_columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
    df_final = df[final_columns]

    # 11. Lưu file và hiển thị kết quả
    df_final.to_csv(output_filename, index=False)

    print("-" * 30)
    print(f"THÀNH CÔNG!")
    print(f"Đã biến đổi và lưu vào file: {output_filename}")
    print("\nXem trước 5 dòng đầu của file đã định dạng:")
    print(df_final.head())

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file {file_moi}")
except Exception as e:
    print(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
    import traceback

    traceback.print_exc()  # In chi tiết lỗi để gỡ rối