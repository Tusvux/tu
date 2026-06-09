# 🌙 Hướng Dẫn Sử Dụng - Dự Đoán Giấc Mơ bằng Machine Learning

**Version:** 2.0  
**Cập nhật:** 2025-12-05

---

## 📋 Mục Lục

1. [Giới Thiệu](#giới-thiệu)
2. [Cài Đặt](#cài-đặt)
3. [Sử Dụng Nhanh](#sử-dụng-nhanh)
4. [Hướng Dẫn Chi Tiết](#hướng-dẫn-chi-tiết)
5. [Tải Dữ Liệu Thật](#tải-dữ-liệu-thật)
6. [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
7. [Tùy Chỉnh & Nâng Cao](#tùy-chỉnh--nâng-cao)
8. [Troubleshooting](#troubleshooting)
9. [Kết Quả & So Sánh](#kết-quả--so-sánh)

---

## 📖 Giới Thiệu

### Dự Án Là Gì?

Dự án sử dụng Machine Learning để dự đoán loại giấc mơ của một người dựa trên các yếu tố như tuổi, giờ ngủ, mức độ stress, và các thói quen sinh hoạt.

### Phân Loại Giấc Mơ

Chương trình phân loại giấc mơ thành 3 loại:
- **Ác mộng** 😱: Giấc mơ không thoải mái, thường do stress cao
- **Mơ đẹp** 😊: Giấc mơ dễ chịu, tích cực
- **Ngủ sâu** 😴: Ngủ ngon không có giấc mơ đáng kể

### Các Đặc Trưng Sử Dụng

1. **Tuổi** (age): Tuổi của người tham gia
2. **Số giờ ngủ** (sleep_hours): Số giờ ngủ trung bình mỗi đêm (4-10)
3. **Mức độ stress** (stress_level): Thang đo 0-10 (0=không stress, 10=rất stress)
4. **Lượng caffeine** (caffeine_intake): Số cốc cà phê/ngày (0-5)
5. **Thời gian tập thể dục** (exercise_minutes): Phút tập thể dục mỗi ngày (0-120)
6. **Chất lượng giấc ngủ** (sleep_quality): Tự đánh giá 0-10 (0=rất tệ, 10=rất tốt)
7. **Thời gian dùng màn hình** (screen_time): Giờ dùng màn hình trước khi ngủ (0-8)

### Các Thuật Toán Machine Learning

Dự án so sánh 9 thuật toán ML:

1. **Random Forest** - Ensemble learning với decision trees
2. **Gradient Boosting** - Boosting algorithm mạnh mẽ
3. **Decision Tree** - Cây quyết định đơn giản
4. **SVM (Support Vector Machine)** - Phân loại với kernel RBF
5. **K-Nearest Neighbors** - Phân loại dựa trên láng giềng gần nhất
6. **Naive Bayes** - Phân loại xác suất
7. **Logistic Regression** - Hồi quy logistic
8. **Neural Network (MLP)** - Mạng neural nhiều lớp
9. **AdaBoost** - Adaptive boosting

---

## 🔧 Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- pip (Python package manager)

### Bước 1: Clone hoặc Tải Dự Án

```bash
cd dream_prediction
```

### Bước 2: Tạo Virtual Environment (Khuyến nghị)

```bash
python3 -m venv venv
source venv/bin/activate  # Trên macOS/Linux
# hoặc
venv\Scripts\activate  # Trên Windows
```

### Bước 3: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies bao gồm:**
- pandas>=1.5.0
- numpy>=1.23.0
- scikit-learn>=1.2.0
- matplotlib>=3.6.0
- seaborn>=0.12.0
- joblib>=1.2.0
- rich>=13.0.0 (cho giao diện đẹp)
- colorama>=0.4.6 (cho màu sắc)

---

## 🚀 Sử Dụng Nhanh

### Option 1: Chạy Tất Cả Tự Động (Khuyến nghị)

```bash
python run_all.py
```

Script này tự động:
1. ✅ Tạo/tải dữ liệu
2. ✅ Huấn luyện 9 mô hình ML
3. ✅ Chạy dự đoán demo
4. ✅ Tạo visualizations

### Option 2: Chạy Từng Bước

#### Với Dữ Liệu Giả Định (Demo/Learning)

```bash
# Bước 1: Tạo dữ liệu
python src/generate_data_4_classes.py

# Bước 2: Huấn luyện mô hình
python src/train_model_vn.py

# Bước 3: Dự đoán
python src/predict_vn.py
```

#### Với Dữ Liệu Thật (Production)

```bash
# Bước 1: Tải dữ liệu từ Kaggle (xem phần "Tải Dữ Liệu Thật")
python src/load_real_data.py

# Bước 2: Huấn luyện (tự động dùng dữ liệu thật)
python src/train_model_vn.py

# Bước 3: Dự đoán
python src/predict_vn.py
```

---

## 📚 Hướng Dẫn Chi Tiết

### 1. Tạo Dữ Liệu (generate_data_4_classes.py)

**Mục đích:** Tạo dữ liệu giả định để demo và học tập

**Cách chạy:**
```bash
python src/generate_data_4_classes.py
```

**Kết quả:**
- Tạo file `dream_data.csv` với 1000 mẫu
- Hiển thị thống kê phân bố và mô tả
- Giao diện đẹp với progress bar và tables

**Thông tin:**
- Số mẫu: 1000
- Phân bố: Ác mộng (~21%), Mơ đẹp (~47%), Ngủ sâu (~31%)

### 2. Huấn Luyện Mô Hình (train_model_vn.py)

**Mục đích:** Huấn luyện và so sánh 9 thuật toán ML

**Cách chạy:**
```bash
python src/train_model_vn.py
```

**Tính năng:**
- Tự động phát hiện dữ liệu tốt nhất (ưu tiên dữ liệu thật)
- Huấn luyện 9 mô hình với progress bar
- So sánh hiệu suất trong table đẹp
- Hiển thị mô hình tốt nhất
- Classification report chi tiết
- Tạo visualizations:
  - `model_comparison.png` - So sánh độ chính xác
  - `confusion_matrix.png` - Ma trận nhầm lẫn
  - `feature_importance.png` - Tầm quan trọng đặc trưng (nếu mô hình hỗ trợ)

**Output:**
- `mo_hinh_tot_nhat_vn.pkl` - Mô hình tốt nhất
- `scaler_vn.pkl` - Scaler để chuẩn hóa dữ liệu

**Kết quả mẫu:**
- SVM: 82.5% accuracy (với dữ liệu giả định)
- Logistic Regression: 97.5% accuracy (với dữ liệu thật)

### 3. Dự Đoán (predict_vn.py)

**Mục đích:** Dự đoán loại giấc mơ cho người dùng mới

**Cách chạy:**
```bash
python src/predict_vn.py
```

**Chế độ 1: Dự đoán đơn lẻ**
- Nhập thông tin cá nhân tương tác
- Nhận dự đoán với xác suất chi tiết
- Nhận khuyến nghị cá nhân hóa
- Giao diện đẹp với colors và progress bars

**Chế độ 2: Dự đoán hàng loạt**
- Chuẩn bị file CSV với các cột:
  - `age`, `sleep_hours`, `stress_level`, `caffeine_intake`
  - `exercise_minutes`, `sleep_quality`, `screen_time`
- Chạy dự đoán cho nhiều người cùng lúc
- Nhận file `predictions_result.csv` với kết quả

**Ví dụ file CSV:**
```csv
age,sleep_hours,stress_level,caffeine_intake,exercise_minutes,sleep_quality,screen_time
25,7.5,6,2,45,7,2
30,6.0,8,3,20,5,4
35,8.5,3,1,60,9,1
```

---

## 📥 Tải Dữ Liệu Thật

### Tại Sao Cần Dữ Liệu Thật?

| Tiêu chí | Dữ Liệu Thật | Dữ Liệu Giả Định |
|----------|--------------|------------------|
| Nguồn | Kaggle Research | Synthetic |
| Số mẫu | 400+ | 1000 |
| Độ tin cậy | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Phù hợp | Production | Demo/Learning |
| Accuracy | ~97% | ~82% |

### Cách 1: Tải Thủ Công (Đơn giản nhất)

#### Bước 1: Chạy script hướng dẫn
```bash
python src/load_real_data.py
```

#### Bước 2: Tải dataset từ Kaggle
1. Truy cập: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
2. Đăng nhập Kaggle (tạo tài khoản miễn phí nếu chưa có)
3. Click nút **"Download"** (file ~15KB)
4. Giải nén file ZIP (nếu có)

#### Bước 3: Đặt file vào thư mục
```bash
# Copy file CSV vào thư mục project
# Đổi tên thành: sleep_health_lifestyle.csv
mv ~/Downloads/Sleep_health_and_lifestyle_dataset.csv sleep_health_lifestyle.csv
```

#### Bước 4: Xử lý dữ liệu
```bash
python src/load_real_data.py
```

**Kết quả:** File `dream_data_real.csv` được tạo với ~400 mẫu dữ liệu thật!

### Cách 2: Tải Tự Động với Kaggle API (Nâng cao)

#### Bước 1: Lấy Kaggle API Token
1. Truy cập: https://www.kaggle.com/settings/account
2. Scroll xuống phần "API"
3. Click "Create New API Token"
4. File `kaggle.json` sẽ tự động tải về

#### Bước 2: Cài Đặt Credentials
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### Bước 3: Tải Dataset Tự Động
```bash
python auto_download_kaggle.py
```

Script sẽ:
- ✅ Kiểm tra credentials
- ✅ Tải dataset từ Kaggle (400+ mẫu)
- ✅ Giải nén và đổi tên file
- ✅ Sẵn sàng để xử lý

#### Bước 4: Xử Lý Dữ Liệu
```bash
python src/load_real_data.py
```

### Các Nguồn Dữ Liệu Khác

1. **Sleep Efficiency Dataset**
   - URL: https://www.kaggle.com/datasets/equilibriumm/sleep-efficiency
   - Đặc điểm: REM sleep, deep sleep, caffeine, exercise

2. **National Sleep Research Resource (NSRR)**
   - URL: https://sleepdata.org/
   - Đặc điểm: 26,000+ subjects, polysomnography data
   - Lưu ý: Cần đăng ký và approval

3. **DREAM Database (EEG + Dream Reports)**
   - URL: https://bridges.monash.edu/articles/dataset/The_DREAM_EEG_and_Mentation_database/22126149
   - Đặc điểm: EEG data + dream content reports
   - Lưu ý: Dữ liệu phức tạp, cần xử lý chuyên sâu

---

## 📁 Cấu Trúc Dự Án

```
dream_prediction/
├── generate_data_4_classes.py          # Tạo dữ liệu giả định
├── load_real_data.py         # Tải và xử lý dữ liệu thật
├── auto_download_kaggle.py   # Tự động tải từ Kaggle API
├── train_model_vn.py            # Huấn luyện mô hình
├── predict_vn.py                # Dự đoán
├── run_all.py                # Master script (chạy tất cả)
├── test_system.py            # Script test toàn diện
├── create_real_data_sample.py # Tạo dữ liệu thật mẫu
├── requirements.txt          # Dependencies
├── GUIDE.md                  # Hướng dẫn này (file duy nhất)
│
├── dream_data.csv           # Dữ liệu giả định (sau khi chạy generate_data_4_classes.py)
├── dream_data_real.csv      # Dữ liệu thật (sau khi chạy load_real_data.py)
├── sleep_health_lifestyle.csv # Dữ liệu gốc từ Kaggle
│
├── mo_hinh_tot_nhat_vn.pkl     # Mô hình tốt nhất (sau khi train)
├── scaler_vn.pkl               # Scaler (sau khi train)
│
├── model_comparison.png     # Biểu đồ so sánh (sau khi train)
├── confusion_matrix.png     # Confusion matrix (sau khi train)
├── feature_importance.png   # Feature importance (nếu mô hình hỗ trợ)
│
├── predictions_result.csv   # Kết quả dự đoán hàng loạt
├── test_data.csv            # File test mẫu
│
└── venv/                    # Virtual environment
```

---

## ⚙️ Tùy Chỉnh & Nâng Cao

### Thay Đổi Số Lượng Dữ Liệu

Trong `generate_data_4_classes.py`:
```python
df = generate_dream_data(2000)  # Thay vì 1000
```

### Thay Đổi Tham Số Mô Hình

Trong `train_model_vn.py`, sửa các tham số:
```python
'Random Forest': RandomForestClassifier(
    n_estimators=200,  # Tăng số cây
    max_depth=10,      # Giới hạn độ sâu
    random_state=42
)
```

### Cải Thiện Mô Hình

Để cải thiện độ chính xác:

1. **Thu thập dữ liệu thực**: Thay thế dữ liệu giả lập bằng dữ liệu thực
2. **Feature Engineering**: Thêm các đặc trưng mới (thời tiết, chế độ ăn uống, v.v.)
3. **Hyperparameter Tuning**: Sử dụng GridSearchCV hoặc RandomizedSearchCV
4. **Ensemble Methods**: Kết hợp nhiều mô hình
5. **Deep Learning**: Thử các mô hình neural network phức tạp hơn

### Test Hệ Thống

Chạy script test toàn diện:
```bash
python tests/test_system.py
```

Script sẽ kiểm tra:
- ✅ Data generation
- ✅ Data consistency
- ✅ Model loading
- ✅ Prediction edge cases
- ✅ Batch prediction
- ✅ File outputs

---

## 🔧 Troubleshooting

### Lỗi: "ModuleNotFoundError: No module named 'rich'"

**Giải pháp:**
```bash
pip install rich colorama
```

### Lỗi: "File not found: mo_hinh_tot_nhat_vn.pkl"

**Giải pháp:**
```bash
# Chạy train_model_vn.py trước
python src/train_model_vn.py
```

### Lỗi: "Kaggle credentials not found"

**Giải pháp:**
```bash
# Kiểm tra file có tồn tại không
ls -la ~/.kaggle/kaggle.json

# Nếu không có, làm lại các bước setup Kaggle API
```

### Lỗi: "Permission denied" (Kaggle API)

**Giải pháp:**
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Lỗi: "Missing columns" khi load dữ liệu thật

Dataset có thể có tên cột khác. Chỉnh sửa `load_real_data.py`:
```python
# Thay đổi mapping tại hàm process_real_data()
```

### Accuracy thấp hơn dữ liệu giả

Đây là bình thường! Dữ liệu thật phức tạp hơn. Thử:
- Tăng số lượng estimators
- Hyperparameter tuning
- Feature engineering

### Neural Network không hội tụ

**Giải pháp:**
- Tăng `max_iter` trong MLPClassifier
- Điều chỉnh learning rate
- Thử architecture khác

---

## 📊 Kết Quả & So Sánh

### Hiệu Suất Mô Hình

#### Với Dữ Liệu Giả Định (1000 mẫu)
- **SVM**: 82.5% accuracy ⭐ (Best)
- **Logistic Regression**: 82.5% (tie)
- **Random Forest**: 80.5%
- **Gradient Boosting**: 80.0%
- **Naive Bayes**: 79.5%
- **Neural Network**: 79.5%
- **K-Nearest Neighbors**: 75.5%
- **AdaBoost**: 77.5%
- **Decision Tree**: 64.0%

#### Với Dữ Liệu Thật (400 mẫu)
- **Logistic Regression**: 97.5% accuracy ⭐ (Best)
- **Neural Network**: 97.5% (tie)
- **SVM**: 93.75%
- **Gradient Boosting**: 88.75%
- **Random Forest**: 83.75%
- **K-Nearest Neighbors**: 82.5%
- **AdaBoost**: 82.5%
- **Decision Tree**: 80.0%
- **Naive Bayes**: 80.0%

### So Sánh Dữ Liệu Thật vs Giả Định

| Metric | Dữ Liệu Giả | Dữ Liệu Thật | Cải Thiện |
|--------|-------------|--------------|-----------|
| **Best Accuracy** | 82.5% | **97.5%** | **+15%** |
| **Top 3 Models** | 79.5% - 82.5% | 93.75% - 97.5% | +11-18% |
| **CV Score** | ~75% | ~88% | +13% |

**Kết luận:** Dữ liệu thật cho kết quả tốt hơn đáng kể!

### Classification Report Mẫu (SVM - 82.5%)

```
              precision    recall  f1-score   support
     Ác mộng       0.84      0.86      0.85        43
      Mơ đẹp       0.89      0.83      0.86        94
     Ngủ sâu       0.74      0.79      0.76        63
    accuracy                           0.82       200
```

### Classification Report Mẫu (Logistic Regression - 97.5%)

```
              precision    recall  f1-score   support
     Ác mộng       1.00      0.96      0.98        25
      Mơ đẹp       1.00      0.93      0.96        14
     Ngủ sâu       0.95      1.00      0.98        41
    accuracy                           0.97        80
```

---

## 💡 Ví Dụ Sử Dụng

### Ví Dụ 1: Dự Đoán Cho 1 Người

```bash
python src/predict_vn.py
# Chọn option 1
# Nhập:
Tuổi: 25
Số giờ ngủ: 7.5
Mức độ stress: 6
Số cốc cà phê: 2
Phút tập thể dục: 45
Chất lượng giấc ngủ: 7
Giờ dùng màn hình: 2

# Kết quả:
Dự đoán: Mơ đẹp 😊
Xác suất:
  Ác mộng: 15.23%
  Mơ đẹp: 68.45%
  Ngủ sâu: 16.32%
```

### Ví Dụ 2: Dự Đoán Hàng Loạt

Tạo file `test_data.csv`:
```csv
age,sleep_hours,stress_level,caffeine_intake,exercise_minutes,sleep_quality,screen_time
25,7.5,6,2,45,7,2
30,6.0,8,3,20,5,4
35,8.5,3,1,60,9,1
```

Chạy:
```bash
python src/predict_vn.py
# Chọn option 2
# Nhập: test_data.csv
```

Kết quả được lưu trong `predictions_result.csv`

---

## ⚠️ Lưu Ý Quan Trọng

- Dữ liệu giả định chỉ phù hợp cho **demo và học tập**
- Để sử dụng thực tế, cần thu thập **dữ liệu thật** từ người dùng
- Kết quả dự đoán chỉ mang tính **tham khảo**
- **Không thay thế** tư vấn y tế chuyên nghiệp
- Luôn **backup dữ liệu gốc** trước khi xử lý
- **Kiểm tra phân bố nhãn** sau khi tạo dream_type

---

## 🎨 Tính Năng Giao Diện

Dự án sử dụng **Rich library** để tạo giao diện đẹp và chuyên nghiệp:

- ✅ **Progress bars** - Hiển thị tiến trình
- ✅ **Tables** - Dữ liệu dễ đọc
- ✅ **Panels** - Thông báo rõ ràng
- ✅ **Colors** - Phân biệt loại thông báo
- ✅ **Status indicators** - Loading states
- ✅ **Better prompts** - Input có gợi ý

---

## 📈 Visualization

Dự án tạo ra các biểu đồ đẹp mắt:

1. **Model Comparison** (`model_comparison.png`)
   - So sánh độ chính xác và cross-validation scores
   - Bar charts với màu sắc

2. **Confusion Matrix** (`confusion_matrix.png`)
   - Ma trận nhầm lẫn với heatmap
   - Hiển thị số lượng và tỷ lệ

3. **Feature Importance** (`feature_importance.png`)
   - Tầm quan trọng của từng đặc trưng
   - Chỉ có với một số mô hình (Random Forest, Gradient Boosting)

---

## 🚀 Next Steps

### Cho Người Dùng Mới
1. ✅ Cài đặt dependencies
2. ✅ Chạy `generate_data_4_classes.py` để tạo dữ liệu demo
3. ✅ Chạy `train_model_vn.py` để huấn luyện
4. ✅ Chạy `predict_vn.py` để dự đoán
5. 📥 Tải dữ liệu thật từ Kaggle
6. 🔄 Huấn luyện lại với dữ liệu thật
7. 📊 So sánh kết quả

### Cải Tiến Tiềm Năng
- 🔧 Hyperparameter tuning
- 📊 Thêm features mới
- 🌐 Web app deployment
- 📱 Mobile app integration
- ⌚ Tích hợp với smartwatch
- 🤖 API service

---

## 🤝 Đóng Góp

Mọi đóng góp đều được hoan nghênh! Hãy:
- Tạo pull request
- Mở issue để báo lỗi
- Đề xuất tính năng mới
- Cải thiện documentation

---

## 📝 License

MIT License - Tự do sử dụng và chỉnh sửa

---

## 👨‍💻 Tác Giả

Dự án Machine Learning - Dự đoán Giấc Mơ

**Version:** 2.0  
**Cập nhật:** 2025-12-05  
**Tính năng:** Giao diện đẹp với Rich, hỗ trợ dữ liệu thật, test toàn diện

---

**Chúc bạn có những giấc mơ đẹp! 🌙✨**


