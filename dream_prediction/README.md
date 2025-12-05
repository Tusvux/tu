# 🌙 Dự Đoán Giấc Mơ - Machine Learning

Dự án sử dụng Machine Learning để dự đoán loại giấc mơ của một người dựa trên các yếu tố như tuổi, giờ ngủ, mức độ stress, và các thói quen sinh hoạt.

## 📖 Hướng Dẫn Sử Dụng

**👉 Xem file [GUIDE.md](GUIDE.md) để có hướng dẫn đầy đủ và chi tiết!**

File GUIDE.md bao gồm:
- ✅ Hướng dẫn cài đặt chi tiết
- ✅ Sử dụng nhanh và từng bước
- ✅ Tải dữ liệu thật từ Kaggle
- ✅ Troubleshooting
- ✅ So sánh kết quả
- ✅ Tùy chỉnh và nâng cao

## 📋 Mô tả

Chương trình phân loại giấc mơ thành 3 loại:
- **Ác mộng** 😱: Giấc mơ không thoải mái, thường do stress cao
- **Mơ đẹp** 😊: Giấc mơ dễ chịu, tích cực
- **Ngủ sâu** 😴: Ngủ ngon không có giấc mơ đáng kể

## 🎯 Các đặc trưng sử dụng

1. **Tuổi** (age): Tuổi của người tham gia
2. **Số giờ ngủ** (sleep_hours): Số giờ ngủ trung bình mỗi đêm
3. **Mức độ stress** (stress_level): Thang đo 0-10
4. **Lượng caffeine** (caffeine_intake): Số cốc cà phê/ngày
5. **Thời gian tập thể dục** (exercise_minutes): Phút tập thể dục mỗi ngày
6. **Chất lượng giấc ngủ** (sleep_quality): Tự đánh giá 0-10
7. **Thời gian dùng màn hình** (screen_time): Giờ dùng màn hình trước khi ngủ

## 🤖 Các thuật toán Machine Learning

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

## 📦 Cài đặt

### 1. Clone hoặc tải dự án

```bash
cd dream_prediction
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Sử dụng

### 🎯 Tùy chọn 1: Sử dụng Dữ Liệu THẬT (Khuyến nghị)

#### Bước 1: Tải dữ liệu từ Kaggle

```bash
python load_real_data.py
```

Script sẽ hiển thị hướng dẫn chi tiết để tải **Sleep Health and Lifestyle Dataset** từ Kaggle:

1. Truy cập: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
2. Đăng nhập Kaggle (miễn phí)
3. Download file CSV
4. Đặt file vào thư mục `dream_prediction/` với tên `sleep_health_lifestyle.csv`
5. Chạy lại `python load_real_data.py`

**Dữ liệu thật bao gồm:**
- 400+ mẫu từ nghiên cứu thực tế
- Các đặc trưng: age, sleep_duration, stress_level, physical_activity, sleep_quality
- Script tự động tạo nhãn dream_type dựa trên logic khoa học

---

### 🧪 Tùy chọn 2: Sử dụng Dữ Liệu Giả Định (Demo)

#### Bước 1: Tạo dữ liệu

```bash
python generate_data.py
```

Script này sẽ:
- Tạo 1000 mẫu dữ liệu giả lập
- Lưu vào file `dream_data.csv`
- Hiển thị thống kê phân bố

---

### So Sánh Dữ Liệu Thật vs Giả Định

| Tiêu chí | Dữ Liệu Thật | Dữ Liệu Giả Định |
|----------|--------------|------------------|
| Nguồn | Kaggle Research | Synthetic |
| Số mẫu | 400+ | 1000 |
| Độ tin cậy | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Phù hợp | Production | Demo/Learning |

---

### Bước 2: Huấn luyện mô hình

```bash
python train_model.py
```

Script này sẽ:
- Tải dữ liệu từ `dream_data.csv`
- Huấn luyện 9 mô hình ML khác nhau
- So sánh hiệu suất các mô hình
- Lưu mô hình tốt nhất vào `best_dream_model.pkl`
- Tạo các biểu đồ visualization:
  - `model_comparison.png` - So sánh độ chính xác
  - `confusion_matrix.png` - Ma trận nhầm lẫn
  - `feature_importance.png` - Tầm quan trọng đặc trưng

### Bước 3: Dự đoán

```bash
python predict.py
```

Script này cung cấp 2 chế độ:

#### Chế độ 1: Dự đoán đơn lẻ
- Nhập thông tin cá nhân
- Nhận dự đoán và xác suất
- Nhận khuyến nghị cá nhân hóa

#### Chế độ 2: Dự đoán hàng loạt
- Chuẩn bị file CSV với các cột cần thiết
- Chạy dự đoán cho nhiều người cùng lúc
- Nhận file kết quả `predictions_result.csv`

## 📊 Kết quả mẫu

### Độ chính xác các mô hình (ví dụ):

```
Random Forest:        0.9450
Gradient Boosting:    0.9400
Neural Network:       0.9250
SVM:                  0.9100
AdaBoost:            0.9050
...
```

### Phân bố dữ liệu mẫu:

```
Ác mộng:  320 người (32.0%)
Mơ đẹp:   350 người (35.0%)
Ngủ sâu:  330 người (33.0%)
```

## 📁 Cấu trúc thư mục

```
dream_prediction/
├── generate_data.py          # Tạo dữ liệu mẫu
├── train_model.py            # Huấn luyện mô hình
├── predict.py                # Dự đoán
├── requirements.txt          # Dependencies
├── README.md                 # Tài liệu này
├── dream_data.csv           # Dữ liệu (sau khi chạy generate_data.py)
├── best_dream_model.pkl     # Mô hình tốt nhất (sau khi train)
├── scaler.pkl               # Scaler (sau khi train)
└── *.png                    # Các biểu đồ visualization
```

## 🎨 Visualization

Dự án tạo ra các biểu đồ đẹp mắt:

1. **Model Comparison**: So sánh độ chính xác và cross-validation scores
2. **Confusion Matrix**: Ma trận nhầm lẫn với heatmap
3. **Feature Importance**: Tầm quan trọng của từng đặc trưng

## 💡 Ví dụ sử dụng

### Dự đoán cho 1 người:

```python
# Chạy predict.py và nhập:
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

### Dự đoán hàng loạt:

Tạo file `test_data.csv`:
```csv
age,sleep_hours,stress_level,caffeine_intake,exercise_minutes,sleep_quality,screen_time
25,7.5,6,2,45,7,2
30,6.0,8,3,20,5,4
35,8.0,3,1,60,9,1
```

Chạy predict.py chế độ 2 và nhập đường dẫn file.

## 🔧 Tùy chỉnh

### Thay đổi số lượng dữ liệu:

Trong `generate_data.py`:
```python
df = generate_dream_data(2000)  # Thay vì 1000
```

### Thay đổi tham số mô hình:

Trong `train_model.py`, sửa các tham số:
```python
'Random Forest': RandomForestClassifier(
    n_estimators=200,  # Tăng số cây
    max_depth=10,      # Giới hạn độ sâu
    random_state=42
)
```

## 📈 Cải thiện mô hình

Để cải thiện độ chính xác:

1. **Thu thập dữ liệu thực**: Thay thế dữ liệu giả lập bằng dữ liệu thực
2. **Feature Engineering**: Thêm các đặc trưng mới (thời tiết, chế độ ăn uống, v.v.)
3. **Hyperparameter Tuning**: Sử dụng GridSearchCV hoặc RandomizedSearchCV
4. **Ensemble Methods**: Kết hợp nhiều mô hình
5. **Deep Learning**: Thử các mô hình neural network phức tạp hơn

## ⚠️ Lưu ý

- Dữ liệu hiện tại là **giả lập** cho mục đích demo
- Để sử dụng thực tế, cần thu thập dữ liệu thật từ người dùng
- Kết quả dự đoán chỉ mang tính tham khảo
- Không thay thế tư vấn y tế chuyên nghiệp

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Hãy tạo pull request hoặc mở issue.

## 📝 License

MIT License - Tự do sử dụng và chỉnh sửa

## 👨‍💻 Tác giả

Dự án Machine Learning - Dự đoán Giấc Mơ

---

**Chúc bạn có những giấc mơ đẹp! 🌙✨**
