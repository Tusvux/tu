# HỆ THỐNG DỰ ĐOÁN GIẤC MƠ - 4 LOẠI

## 📋 Tổng Quan

Hệ thống Machine Learning dự đoán loại giấc mơ dựa trên các yếu tố sinh hoạt của người dùng. Phiên bản này hỗ trợ **4 loại giấc mơ**:

- **0️⃣ Ác mộng (😱)**: Stress cao, ngủ ít, caffeine nhiều, chất lượng ngủ kém
- **1️⃣ Mơ đẹp (😊)**: Stress thấp, ngủ đủ, tập thể dục tốt, chất lượng ngủ cao
- **2️⃣ Ngủ sâu (😴)**: Ngủ nhiều giờ, ít caffeine, chất lượng ngủ rất tốt
- **3️⃣ Không mơ (🌙)**: Ngủ vừa phải, stress trung bình, ít vận động

## 🎯 Độ Chính Xác

Với dữ liệu 6000 mẫu, hệ thống đạt:
- **Độ chính xác**: ~88%
- **F1-Score trung bình**: ~0.85

### Kết quả chi tiết theo từng loại:

| Loại Giấc Mơ | Precision | Recall | F1-Score | Số mẫu |
|-------------|-----------|--------|----------|---------|
| Ác mộng     | 81.3%     | 64.9%  | 72.2%    | 168     |
| Mơ đẹp      | 91.5%     | 91.8%  | 91.6%    | 584     |
| Ngủ sâu     | 88.3%     | 82.9%  | 85.5%    | 363     |
| Không mơ    | 86.2%     | 93.0%  | 89.5%    | 685     |

## 📊 Dữ Liệu

### Các đặc trưng đầu vào:

1. **tuoi**: Tuổi (18-70)
2. **gio_ngu**: Số giờ ngủ mỗi đêm (4-10 giờ)
3. **muc_stress**: Mức độ stress (0-10)
4. **caffeine**: Số lượng caffeine/ngày (0-4 cups)
5. **phut_tap_luyen**: Số phút tập thể dục/ngày (0-120 phút)
6. **chat_luong_ngu**: Chất lượng giấc ngủ tự đánh giá (0-10)
7. **thoi_gian_man_hinh**: Thời gian xem màn hình trước khi ngủ (0-8 giờ)

### Phân bố dữ liệu:

```
Ác mộng:   9.3%  (559 mẫu)
Mơ đẹp:   32.5% (1948 mẫu)
Ngủ sâu:  20.2% (1209 mẫu)
Không mơ: 38.1% (2284 mẫu)
```

## 🚀 Cách Sử Dụng

### 1. Tạo dữ liệu mới (6000 mẫu với 4 loại)

```bash
python src/generate_data_4_classes.py
```

Tạo ra 2 file:
- `dream_data.csv` (tiếng Anh)
- `dream_data_vn.csv` (tiếng Việt)

### 2. Huấn luyện mô hình đầy đủ

```bash
python src/train_model_vn.py
```

Tạo ra:
- `mo_hinh_tot_nhat_vn.pkl` - Mô hình đã huấn luyện
- `scaler_vn.pkl` - Scaler để chuẩn hóa
- `ten_dac_trung_vn.pkl` - Tên các đặc trưng
- Các biểu đồ phân tích (PNG files)

### 3. Test nhanh

```bash
python tests/test_4_classes.py
```

Kiểm tra nhanh độ chính xác của mô hình với Random Forest.

### 4. Dự đoán với dữ liệu mới

```bash
python src/predict_vn.py
```

Nhập thông tin cá nhân và nhận dự đoán loại giấc mơ.

## 📁 Cấu Trúc File

```
dream_prediction/
├── generate_data_4_classes.py    # Tạo dữ liệu 4 loại
├── train_model_vn.py             # Huấn luyện mô hình (tiếng Việt)
├── predict_vn.py                 # Dự đoán (tiếng Việt)
├── test_4_classes.py             # Test nhanh
├── dream_data_vn.csv             # Dữ liệu tiếng Việt
├── mo_hinh_tot_nhat_vn.pkl       # Mô hình đã train
├── scaler_vn.pkl                 # Scaler
└── ten_dac_trung_vn.pkl          # Tên features
```

## 🔬 Chi Tiết Kỹ Thuật

### Các mô hình được thử nghiệm:

1. Random Forest ⭐
2. Gradient Boosting
3. AdaBoost
4. Decision Tree
5. SVM (RBF & Linear)
6. K-Nearest Neighbors
7. Naive Bayes
8. Logistic Regression
9. Ridge Classifier
10. Neural Network (MLP)
11. Linear Discriminant Analysis
12. Quadratic Discriminant Analysis
13. Voting Classifiers (Hard & Soft)
14. Bagging Classifier

### Các kỹ thuật tiền xử lý:

- **Standardization**: StandardScaler, RobustScaler, MinMaxScaler
- **Feature Selection**: F-score, Mutual Information
- **Noise Addition**: Để tăng tính robust (3% samples)
- **Cross-Validation**: StratifiedKFold (5 folds)

### Đặc trưng quan trọng nhất (theo thứ tự):

1. 🥇 Mức stress (F-Score: 1833.34)
2. 🥈 Phút tập luyện (F-Score: 990.84)
3. 🥉 Chất lượng ngủ (F-Score: 733.91)
4. Giờ ngủ
5. Caffeine
6. Thời gian màn hình
7. Tuổi

## 📈 Cải Tiến So Với Phiên Bản 3 Loại

### Thêm loại "Không mơ":

**Đặc điểm nhận dạng**:
- Ngủ khoảng 7 giờ (gần mức trung bình)
- Stress ở mức trung bình (5/10)
- Caffeine vừa phải (2 cups)
- Ít vận động (< 50 phút/ngày)
- Ít thời gian màn hình

**Ý nghĩa**: Loại này đại diện cho giấc ngủ bình thường mà không có giấc mơ đáng kể.

## 🎨 Visualization

Các biểu đồ được tạo tự động:

1. **Ma trận tương quan**: Mối quan hệ giữa các đặc trưng
2. **Feature importance**: Tầm quan trọng của từng đặc trưng
3. **Confusion matrix**: Ma trận nhầm lẫn 4x4
4. **ROC curves**: Đường cong ROC cho 4 lớp
5. **Learning curves**: Đường cong học tập
6. **PCA visualization**: Phân tích thành phần chính

## 💡 Tips Sử Dụng

### Để có kết quả tốt nhất:

1. **Mơ đẹp**: Giảm stress, ngủ đủ 7-8 giờ, tập thể dục đều đặn
2. **Tránh ác mộng**: Giảm caffeine buổi tối, hạn chế màn hình trước ngủ
3. **Ngủ sâu**: Tạo môi trường yên tĩnh, tối, mát mẻ
4. **Kiểm soát giấc mơ**: Ghi nhật ký giấc mơ, thiền định

## 🔧 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
rich
```

Cài đặt:
```bash
pip install -r requirements.txt
```

## 📞 Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra dữ liệu có đầy đủ 4 loại (0, 1, 2, 3)
2. Đảm bảo encoding UTF-8 khi chạy trên Windows
3. Sử dụng `test_4_classes.py` để kiểm tra nhanh

## 📜 License

MIT License - Tự do sử dụng và chỉnh sửa

---

**Phiên bản**: 2.0 (4 Loại Giấc Mơ)  
**Ngày cập nhật**: December 22, 2025  
**Tác giả**: Dream Prediction Team

