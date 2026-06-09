"""
Script tải và xử lý dữ liệu thật về giấc ngủ từ Kaggle
Dataset: Sleep Health and Lifestyle Dataset
Nguồn: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from paths import DATA_DIR, ensure_project_dirs

def download_instructions():
    """Hướng dẫn tải dữ liệu"""
    print("=" * 70)
    print("📥 HƯỚNG DẪN TẢI DỮ LIỆU THẬT TỪ KAGGLE")
    print("=" * 70)
    print("\n🔗 Nguồn dữ liệu đáng tin cậy:")
    print("\n1. Sleep Health and Lifestyle Dataset (Khuyến nghị)")
    print("   URL: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset")
    print("   - 400 mẫu dữ liệu thật")
    print("   - 13 cột: age, gender, occupation, sleep_duration, quality_of_sleep,")
    print("     physical_activity_level, stress_level, bmi_category, blood_pressure,")
    print("     heart_rate, daily_steps, sleep_disorder")
    
    print("\n2. Sleep Efficiency Dataset")
    print("   URL: https://www.kaggle.com/datasets/equilibriumm/sleep-efficiency")
    print("   - Dữ liệu về hiệu suất giấc ngủ")
    print("   - Bao gồm: REM sleep, deep sleep, caffeine, exercise")
    
    print("\n" + "=" * 70)
    print("📋 CÁCH TẢI:")
    print("=" * 70)
    print("\nCách 1: Tải thủ công")
    print("  1. Truy cập link trên")
    print("  2. Đăng nhập Kaggle (tạo tài khoản miễn phí nếu chưa có)")
    print("  3. Click 'Download' để tải file CSV")
    print("  4. Đặt file vào thư mục: dream_prediction/")
    print("     Đổi tên thành: sleep_health_lifestyle.csv")
    
    print("\nCách 2: Sử dụng Kaggle API (Nâng cao)")
    print("  1. Cài đặt: pip install kaggle")
    print("  2. Tải API key từ Kaggle Account Settings")
    print("  3. Chạy lệnh:")
    print("     kaggle datasets download -d uom190346a/sleep-health-and-lifestyle-dataset")
    
    print("\n" + "=" * 70)
    print("⚠️  LƯU Ý:")
    print("=" * 70)
    print("  - Dữ liệu này KHÔNG có cột 'dream_type' trực tiếp")
    print("  - Script sẽ tự động tạo nhãn dựa trên:")
    print("    • Stress level (cao → ác mộng)")
    print("    • Sleep quality (tốt → mơ đẹp)")
    print("    • Sleep duration (đủ giấc → ngủ sâu)")
    print("=" * 70)

def load_kaggle_dataset(filepath='sleep_health_lifestyle.csv'):
    """Tải và xử lý dữ liệu từ Kaggle"""
    candidate = Path(filepath)
    if not candidate.exists():
        candidate = DATA_DIR / filepath
    
    if not candidate.exists():
        print(f"\n❌ Không tìm thấy file: {filepath}")
        print("\n💡 Vui lòng tải dữ liệu theo hướng dẫn trên.")
        download_instructions()
        return None
    
    print(f"\n✅ Đã tìm thấy file: {candidate}")
    print("📂 Đang tải dữ liệu...")
    
    # Đọc dữ liệu
    df = pd.read_csv(candidate)
    
    print(f"\n📊 Thông tin dataset:")
    print(f"  - Số mẫu: {len(df)}")
    print(f"  - Số cột: {len(df.columns)}")
    print(f"\n📋 Các cột có sẵn:")
    for col in df.columns:
        print(f"  • {col}")
    
    return df

def process_real_data(df):
    """Xử lý và chuyển đổi dữ liệu thật"""
    
    print("\n🔄 Đang xử lý dữ liệu...")
    
    # Tạo DataFrame mới với các cột cần thiết
    processed_data = pd.DataFrame()
    
    # Mapping các cột
    if 'Age' in df.columns:
        processed_data['age'] = df['Age']
    elif 'age' in df.columns:
        processed_data['age'] = df['age']
    
    if 'Sleep Duration' in df.columns:
        processed_data['sleep_hours'] = df['Sleep Duration']
    elif 'sleep_duration' in df.columns:
        processed_data['sleep_hours'] = df['sleep_duration']
    
    if 'Stress Level' in df.columns:
        processed_data['stress_level'] = df['Stress Level']
    elif 'stress_level' in df.columns:
        processed_data['stress_level'] = df['stress_level']
    
    # Xử lý các cột khác
    # Caffeine intake - ước tính từ stress level hoặc set mặc định
    if 'Caffeine Consumption' in df.columns:
        processed_data['caffeine_intake'] = df['Caffeine Consumption']
    else:
        # Ước tính: stress cao → nhiều caffeine
        processed_data['caffeine_intake'] = (processed_data['stress_level'] / 2.5).astype(int).clip(0, 4)
    
    # Exercise minutes
    if 'Physical Activity Level' in df.columns:
        # Chuyển đổi từ level sang minutes (ví dụ: level 60 = 60 phút)
        processed_data['exercise_minutes'] = df['Physical Activity Level']
    elif 'physical_activity_level' in df.columns:
        processed_data['exercise_minutes'] = df['physical_activity_level']
    elif 'Daily Steps' in df.columns:
        # Ước tính từ số bước: 10000 bước ≈ 60 phút
        processed_data['exercise_minutes'] = (df['Daily Steps'] / 10000 * 60).clip(0, 120).astype(int)
    else:
        processed_data['exercise_minutes'] = np.random.randint(20, 80, len(df))
    
    # Sleep quality
    if 'Quality of Sleep' in df.columns:
        processed_data['sleep_quality'] = df['Quality of Sleep']
    elif 'quality_of_sleep' in df.columns:
        processed_data['sleep_quality'] = df['quality_of_sleep']
    
    # Screen time - ước tính từ các yếu tố khác
    if 'Screen Time' in df.columns:
        processed_data['screen_time'] = df['Screen Time']
    else:
        # Ước tính: stress cao và sleep quality thấp → nhiều screen time
        processed_data['screen_time'] = (
            (processed_data['stress_level'] / 2) + 
            ((10 - processed_data['sleep_quality']) / 3)
        ).clip(0, 8)
    
    # Tạo nhãn dream_type dựa trên logic
    processed_data['dream_type'] = create_dream_labels(processed_data)
    
    # Xử lý missing values
    processed_data = processed_data.fillna(processed_data.mean())
    
    print("\n✅ Xử lý hoàn tất!")
    print(f"\n📊 Dữ liệu sau xử lý:")
    print(f"  - Số mẫu: {len(processed_data)}")
    print(f"  - Số đặc trưng: {len(processed_data.columns) - 1}")
    
    return processed_data

def create_dream_labels(df):
    """
    Tạo nhãn dream_type từ các đặc trưng
    Logic:
    - Ác mộng (0): Stress cao, sleep quality thấp
    - Mơ đẹp (1): Stress thấp, sleep quality cao, exercise tốt
    - Ngủ sâu (2): Sleep duration đủ, sleep quality cao
    - Không mơ (3): Các chỉ số gần trung bình, ít dấu hiệu nổi bật
    """
    
    labels = []
    
    for idx, row in df.iterrows():
        stress = row['stress_level']
        sleep_quality = row['sleep_quality']
        sleep_hours = row['sleep_hours']
        exercise = row['exercise_minutes']
        caffeine = row['caffeine_intake']
        screen_time = row['screen_time']
        
        # Tính điểm cho mỗi loại
        nightmare_score = (stress * 0.35) + ((10 - sleep_quality) * 0.25) + (screen_time * 0.20) + (caffeine * 0.20)
        
        good_dream_score = ((10 - stress) * 0.3) + (sleep_quality * 0.3) + (exercise / 120 * 10 * 0.4)
        
        deep_sleep_score = (sleep_hours * 0.35) + (sleep_quality * 0.35) + ((5 - caffeine) * 0.20) + (exercise / 120 * 10 * 0.10)

        no_dream_score = (
            (10 - abs(sleep_hours - 7)) * 0.25 +
            (10 - abs(stress - 5)) * 0.20 +
            (5 - abs(caffeine - 2)) * 0.15 +
            ((120 - exercise) / 120 * 10) * 0.20 +
            ((8 - screen_time) / 8 * 10) * 0.10 -
            abs(sleep_quality - 5) * 0.10
        )
        
        # Chọn nhãn có điểm cao nhất
        scores = [nightmare_score, good_dream_score, deep_sleep_score, no_dream_score]
        label = scores.index(max(scores))
        labels.append(label)
    
    return labels

def main():
    """Hàm chính"""
    ensure_project_dirs()
    
    print("=" * 70)
    print("🌙 TẢI VÀ XỬ LÝ DỮ LIỆU THẬT VỀ GIẤC NGỦ 🌙")
    print("=" * 70)
    
    # Hiển thị hướng dẫn
    download_instructions()
    
    print("\n" + "=" * 70)
    print("🔍 ĐANG TÌM KIẾM DỮ LIỆU...")
    print("=" * 70)
    
    # Thử tải dữ liệu
    df = load_kaggle_dataset('sleep_health_lifestyle.csv')
    
    if df is None:
        # Thử tên file khác
        alternative_files = [
            'Sleep_health_and_lifestyle_dataset.csv',
            'Sleep_Health_and_Lifestyle_Dataset.csv',
            'sleep-health-and-lifestyle-dataset.csv'
        ]
        
        for filename in alternative_files:
            if os.path.exists(filename) or (DATA_DIR / filename).exists():
                df = load_kaggle_dataset(filename)
                break
    
    if df is not None:
        # Xử lý dữ liệu
        processed_df = process_real_data(df)
        
        # Lưu dữ liệu đã xử lý
        output_file = DATA_DIR / 'dream_data_real.csv'
        processed_df.to_csv(output_file, index=False)
        
        print(f"\n💾 Đã lưu dữ liệu đã xử lý: data/{output_file.name}")
        
        # Thống kê
        dream_labels = {0: 'Ác mộng', 1: 'Mơ đẹp', 2: 'Ngủ sâu', 3: 'Không mơ'}
        print("\n📊 PHÂN BỐ NHÃN:")
        print("=" * 70)
        for dream_type, count in processed_df['dream_type'].value_counts().sort_index().items():
            percentage = (count / len(processed_df)) * 100
            print(f"  {dream_labels[dream_type]}: {count} người ({percentage:.1f}%)")
        
        print("\n📈 THỐNG KÊ MÔ TẢ:")
        print("=" * 70)
        print(processed_df.describe())
        
        print("\n✅ HOÀN THÀNH!")
        print("=" * 70)
        print("\n💡 Bước tiếp theo:")
        print("  1. Chạy: python src/train_model_vn.py")
        print("     (Script sẽ tự động sử dụng dream_data_real.csv nếu có)")
        print("  2. So sánh kết quả với dữ liệu giả định")
        
    else:
        print("\n⚠️  Chưa tìm thấy dữ liệu!")
        print("\n💡 Vui lòng:")
        print("  1. Tải dataset từ Kaggle theo hướng dẫn trên")
        print("  2. Đặt file CSV vào thư mục dream_prediction/")
        print("  3. Chạy lại script này")

if __name__ == "__main__":
    main()
