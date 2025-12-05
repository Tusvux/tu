#!/usr/bin/env python3
"""
MASTER SCRIPT - Chạy toàn bộ quy trình dự đoán giấc mơ
Từ tạo dữ liệu → huấn luyện → dự đoán
"""

import subprocess
import sys
import os

def print_header(text):
    """In header đẹp"""
    print("\n" + "=" * 70)
    print(f"🌙 {text}")
    print("=" * 70 + "\n")

def run_script(script_name, description):
    """Chạy một script Python"""
    print_header(description)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✅ {description} - HOÀN THÀNH!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - THẤT BẠI!")
        print(f"Lỗi: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Không tìm thấy file: {script_name}")
        return False

def check_data_exists():
    """Kiểm tra dữ liệu có sẵn chưa"""
    if os.path.exists('dream_data_real.csv'):
        return 'dream_data_real.csv', 'DỮ LIỆU THẬT'
    elif os.path.exists('dream_data.csv'):
        return 'dream_data.csv', 'DỮ LIỆU GIẢ ĐỊNH'
    else:
        return None, None

def main():
    """Hàm chính - chạy toàn bộ pipeline"""
    
    print("=" * 70)
    print("🌙 DREAM PREDICTION - MASTER SCRIPT 🌙")
    print("Chạy toàn bộ quy trình từ đầu đến cuối")
    print("=" * 70)
    
    # Kiểm tra dữ liệu
    data_file, data_type = check_data_exists()
    
    if data_file:
        print(f"\n✅ Đã tìm thấy dữ liệu: {data_file} ({data_type})")
        response = input("\n❓ Bạn có muốn tạo lại dữ liệu không? (y/n): ").strip().lower()
        
        if response == 'y':
            data_file = None
    
    # BƯỚC 1: Tạo hoặc tải dữ liệu
    if not data_file:
        print("\n" + "=" * 70)
        print("📊 CHỌN NGUỒN DỮ LIỆU")
        print("=" * 70)
        print("\n1. Dữ liệu giả định (nhanh, demo)")
        print("2. Dữ liệu thật từ Kaggle (khuyến nghị, cần setup)")
        
        choice = input("\nLựa chọn (1/2): ").strip()
        
        if choice == '2':
            print("\n💡 Để sử dụng dữ liệu thật:")
            print("  1. Chạy: python load_real_data.py")
            print("  2. Tải dataset từ Kaggle theo hướng dẫn")
            print("  3. Chạy lại master script này")
            print("\n⚠️  Tiếp tục với dữ liệu giả định...")
            
            if not run_script('generate_data.py', 'BƯỚC 1: TẠO DỮ LIỆU GIẢ ĐỊNH'):
                print("\n❌ Không thể tiếp tục. Thoát.")
                sys.exit(1)
        else:
            if not run_script('generate_data.py', 'BƯỚC 1: TẠO DỮ LIỆU GIẢ ĐỊNH'):
                print("\n❌ Không thể tiếp tục. Thoát.")
                sys.exit(1)
    
    # BƯỚC 2: Huấn luyện mô hình
    if not run_script('train_model.py', 'BƯỚC 2: HUẤN LUYỆN MÔ HÌNH'):
        print("\n❌ Không thể tiếp tục. Thoát.")
        sys.exit(1)
    
    # BƯỚC 3: Dự đoán demo
    print_header('BƯỚC 3: DỰ ĐOÁN DEMO')
    
    print("📊 Chạy dự đoán với dữ liệu test mẫu...")
    
    # Tạo dữ liệu test nhanh
    test_code = """
import joblib
import pandas as pd

# Load model
model = joblib.load('best_dream_model.pkl')
scaler = joblib.load('scaler.pkl')

# Dữ liệu test
test_data = pd.DataFrame({
    'age': [25, 30, 35, 28, 40],
    'sleep_hours': [7.5, 6.0, 8.5, 5.0, 8.0],
    'stress_level': [6.0, 8.5, 3.0, 9.0, 2.0],
    'caffeine_intake': [2, 3, 1, 4, 0],
    'exercise_minutes': [45, 20, 60, 10, 90],
    'sleep_quality': [7.0, 5.0, 9.0, 3.0, 8.5],
    'screen_time': [2.0, 4.0, 1.0, 6.0, 0.5]
})

# Predict
X_scaled = scaler.transform(test_data)
predictions = model.predict(X_scaled)

dream_labels = {0: 'Ác mộng 😱', 1: 'Mơ đẹp 😊', 2: 'Ngủ sâu 😴'}

print('\\n🔮 KẾT QUẢ DỰ ĐOÁN DEMO:')
print('=' * 60)
for i, pred in enumerate(predictions):
    print(f'Người {i+1}: {dream_labels[pred]}')
print('=' * 60)
"""
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', test_code],
            check=True,
            capture_output=False,
            text=True
        )
        print("\n✅ BƯỚC 3: DỰ ĐOÁN DEMO - HOÀN THÀNH!")
    except Exception as e:
        print(f"\n⚠️  Lỗi khi chạy demo: {e}")
    
    # HOÀN THÀNH
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH TOÀN BỘ QUY TRÌNH!")
    print("=" * 70)
    
    print("\n📊 KẾT QUẢ:")
    print("  ✅ Dữ liệu đã được tạo/tải")
    print("  ✅ Mô hình đã được huấn luyện")
    print("  ✅ Dự đoán demo đã chạy thành công")
    
    print("\n📁 FILES ĐÃ TẠO:")
    files_to_check = [
        ('dream_data.csv', 'Dữ liệu giả định'),
        ('dream_data_real.csv', 'Dữ liệu thật'),
        ('best_dream_model.pkl', 'Mô hình tốt nhất'),
        ('scaler.pkl', 'Scaler'),
        ('model_comparison.png', 'Biểu đồ so sánh'),
        ('confusion_matrix.png', 'Confusion matrix'),
        ('feature_importance.png', 'Feature importance')
    ]
    
    for filename, description in files_to_check:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  ✅ {filename} ({description}) - {size:,} bytes")
    
    print("\n🚀 BƯỚC TIẾP THEO:")
    print("  1. Xem kết quả trong các file .png")
    print("  2. Chạy dự đoán tương tác: python predict.py")
    print("  3. Để dùng dữ liệu thật: python load_real_data.py")
    
    print("\n" + "=" * 70)
    print("Cảm ơn bạn đã sử dụng Dream Prediction System! 🌙✨")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
