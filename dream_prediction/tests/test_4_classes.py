"""
Script test nhanh để kiểm tra dữ liệu 4 loại giấc mơ
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import DATA_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Đọc dữ liệu
print("=" * 60)
print("KIỂM TRA DỮ LIỆU 4 LOẠI GIẤC MƠ")
print("=" * 60)

df = pd.read_csv(DATA_DIR / 'dream_data_vn.csv')

print(f"\n✓ Đã tải dữ liệu: {len(df)} mẫu")
print(f"✓ Số đặc trưng: {len(df.columns) - 1}")

# Kiểm tra các loại giấc mơ
print("\n📊 Phân bố nhãn:")
dream_labels = {0: 'Ác mộng', 1: 'Mơ đẹp', 2: 'Ngủ sâu', 3: 'Không mơ'}
for label, count in df['loai_giac_mo'].value_counts().sort_index().items():
    pct = count / len(df) * 100
    print(f"  {label} - {dream_labels[label]}: {count} ({pct:.1f}%)")

# Chia dữ liệu
X = df.drop('loai_giac_mo', axis=1)
y = df['loai_giac_mo']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n✓ Đã chia dữ liệu: Train={len(X_train)}, Test={len(X_test)}")

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Đã chuẩn hóa dữ liệu")

# Huấn luyện Random Forest
print("\n🤖 Đang huấn luyện Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)

# Đánh giá
y_pred = rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Độ chính xác: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 Báo cáo phân loại:")
print(classification_report(y_test, y_pred, 
                           target_names=list(dream_labels.values()),
                           digits=4))

print("\n" + "=" * 60)
print("✅ KIỂM TRA HOÀN TẤT - HỆ THỐNG HOẠT ĐỘNG TỐT VỚI 4 LOẠI!")
print("=" * 60)
