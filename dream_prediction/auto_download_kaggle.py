#!/usr/bin/env python3
"""
Script tự động tải dữ liệu từ Kaggle với setup hướng dẫn
"""

import os
import sys
import subprocess

def check_kaggle_credentials():
    """Kiểm tra Kaggle credentials"""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    return os.path.exists(kaggle_json)

def setup_kaggle_credentials():
    """Hướng dẫn setup Kaggle credentials"""
    
    print("=" * 70)
    print("🔑 SETUP KAGGLE API CREDENTIALS")
    print("=" * 70)
    
    print("\n📋 Bước 1: Lấy API Token từ Kaggle")
    print("  1. Truy cập: https://www.kaggle.com/settings/account")
    print("  2. Scroll xuống phần 'API'")
    print("  3. Click 'Create New Token'")
    print("  4. File 'kaggle.json' sẽ được tải về")
    
    print("\n📋 Bước 2: Cài đặt credentials")
    kaggle_dir = os.path.expanduser('~/.kaggle')
    
    print(f"\n  Tạo thư mục: {kaggle_dir}")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    print(f"\n  Di chuyển file kaggle.json vào: {kaggle_dir}/")
    print(f"  Lệnh: mv ~/Downloads/kaggle.json {kaggle_dir}/")
    
    print("\n📋 Bước 3: Set permissions")
    print(f"  chmod 600 {kaggle_dir}/kaggle.json")
    
    print("\n" + "=" * 70)
    
    # Hỏi người dùng có muốn setup ngay không
    response = input("\n❓ Bạn đã tải file kaggle.json chưa? (y/n): ").strip().lower()
    
    if response == 'y':
        # Tìm file kaggle.json trong Downloads
        downloads_path = os.path.expanduser('~/Downloads/kaggle.json')
        
        if os.path.exists(downloads_path):
            import shutil
            dest_path = os.path.join(kaggle_dir, 'kaggle.json')
            shutil.copy(downloads_path, dest_path)
            os.chmod(dest_path, 0o600)
            print(f"\n✅ Đã setup credentials tại: {dest_path}")
            return True
        else:
            print(f"\n❌ Không tìm thấy file tại: {downloads_path}")
            print("💡 Vui lòng di chuyển thủ công:")
            print(f"   mv ~/Downloads/kaggle.json {kaggle_dir}/")
            print(f"   chmod 600 {kaggle_dir}/kaggle.json")
            return False
    else:
        print("\n💡 Vui lòng hoàn thành các bước trên rồi chạy lại script này!")
        return False

def download_dataset():
    """Tải dataset từ Kaggle"""
    
    print("\n" + "=" * 70)
    print("📥 ĐANG TẢI DATASET TỪ KAGGLE")
    print("=" * 70)
    
    dataset_name = "uom190346a/sleep-health-and-lifestyle-dataset"
    
    try:
        print(f"\n⏳ Đang tải: {dataset_name}")
        
        # Tải dataset
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', dataset_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Tải thành công!")
        
        # Giải nén
        import zipfile
        
        zip_file = 'sleep-health-and-lifestyle-dataset.zip'
        
        if os.path.exists(zip_file):
            print(f"\n📦 Đang giải nén: {zip_file}")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall('.')
            
            print("✅ Giải nén thành công!")
            
            # Tìm file CSV
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'sleep' in f.lower()]
            
            if csv_files:
                original_file = csv_files[0]
                target_file = 'sleep_health_lifestyle.csv'
                
                # Đổi tên
                if original_file != target_file:
                    os.rename(original_file, target_file)
                    print(f"\n✅ Đã đổi tên: {original_file} -> {target_file}")
                
                # Xóa file zip
                os.remove(zip_file)
                print(f"🗑️  Đã xóa file zip")
                
                return True
            else:
                print("❌ Không tìm thấy file CSV trong archive")
                return False
        else:
            print(f"❌ Không tìm thấy file: {zip_file}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Lỗi khi tải dataset:")
        print(f"   {e.stderr}")
        return False
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        return False

def main():
    """Hàm chính"""
    
    print("=" * 70)
    print("🌙 TỰ ĐỘNG TẢI DỮ LIỆU THẬT TỪ KAGGLE 🌙")
    print("=" * 70)
    
    # Kiểm tra credentials
    if not check_kaggle_credentials():
        print("\n⚠️  Chưa tìm thấy Kaggle API credentials")
        
        if not setup_kaggle_credentials():
            print("\n" + "=" * 70)
            print("❌ SETUP CHƯA HOÀN TẤT")
            print("=" * 70)
            print("\n💡 Sau khi setup xong, chạy lại:")
            print("   python auto_download_kaggle.py")
            sys.exit(1)
    else:
        print("\n✅ Đã tìm thấy Kaggle credentials")
    
    # Tải dataset
    if download_dataset():
        print("\n" + "=" * 70)
        print("✅ HOÀN THÀNH!")
        print("=" * 70)
        
        print("\n📊 Bước tiếp theo:")
        print("  1. Chạy: python load_real_data.py")
        print("     (Xử lý dữ liệu và tạo dream_data_real.csv)")
        print("\n  2. Chạy: python train_model.py")
        print("     (Huấn luyện với dữ liệu thật)")
        
    else:
        print("\n" + "=" * 70)
        print("❌ TẢI DATASET THẤT BẠI")
        print("=" * 70)
        
        print("\n💡 Tùy chọn thay thế:")
        print("  1. Tải thủ công từ:")
        print("     https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset")
        print("\n  2. Đặt file CSV vào thư mục này với tên:")
        print("     sleep_health_lifestyle.csv")
        print("\n  3. Chạy: python load_real_data.py")

if __name__ == "__main__":
    main()
