# 🌙 DỰ ĐOÁN GIẤC MƠ - PHIÊN BẢN TIẾNG VIỆT

## 📝 Giới thiệu

Hệ thống dự đoán loại giấc mơ sử dụng Machine Learning với **dữ liệu tiếng Việt** và **các phương pháp ML nâng cao**.

### 🎯 Mục tiêu
- Dự đoán loại giấc mơ: **Ác mộng** 😱, **Mơ đẹp** 😊, **Ngủ sâu** 😴
- Phân tích các yếu tố ảnh hưởng đến giấc ngủ
- Đưa ra lời khuyên cải thiện chất lượng giấc ngủ

---

## 🚀 Cài đặt

### Yêu cầu
- Python 3.8+
- pip

### Các thư viện cần thiết
```bash
pip install pandas numpy scikit-learn matplotlib seaborn rich joblib scipy
```

---

## 📊 Cấu trúc dữ liệu

### Dữ liệu đầu vào (tiếng Việt)

| Cột | Mô tả | Đơn vị | Phạm vi |
|-----|-------|--------|---------|
| `tuoi` | Tuổi của người dùng | tuổi | 18-80 |
| `gio_ngu` | Số giờ ngủ mỗi đêm | giờ | 4-12 |
| `muc_stress` | Mức độ căng thẳng | điểm | 0-10 |
| `caffeine` | Lượng caffeine tiêu thụ | tách | 0-5 |
| `phut_tap_luyen` | Thời gian tập thể dục | phút | 0-180 |
| `chat_luong_ngu` | Chất lượng giấc ngủ (càng cao càng kém) | điểm | 0-10 |
| `thoi_gian_man_hinh` | Thời gian sử dụng màn hình trước khi ngủ | giờ | 0-12 |

### Dữ liệu đầu ra

| Giá trị | Loại giấc mơ | Mô tả |
|---------|--------------|-------|
| 0 | Ác mộng 😱 | Giấc ngủ không tốt, có nhiễu loạn |
| 1 | Mơ đẹp 😊 | Giấc ngủ tốt với những giấc mơ dễ chịu |
| 2 | Ngủ sâu 😴 | Giấc ngủ sâu, nghỉ ngơi hiệu quả |

---

## 🎓 Các phương pháp Machine Learning được sử dụng

### 1. **Tiền xử lý dữ liệu**
- ✅ **Chuyển đổi sang tiếng Việt**: Tất cả tên cột và nhãn
- ✅ **StandardScaler**: Chuẩn hóa dữ liệu theo phân phối chuẩn
- ✅ **RobustScaler**: Chuẩn hóa chống nhiễu (dựa trên median và IQR)
- ✅ **MinMaxScaler**: Chuẩn hóa về khoảng [0, 1]
- ✅ **Thêm nhiễu**: Gaussian noise và label noise để tăng tính robust

### 2. **Lựa chọn đặc trưng (Feature Selection)**
- 🎯 **F-score (ANOVA)**: Đo mức độ khác biệt giữa các nhóm
- 🎯 **Mutual Information**: Đo lượng thông tin chung giữa feature và target
- 📊 Kết quả: `muc_stress` và `chat_luong_ngu` là 2 đặc trưng quan trọng nhất

### 3. **Các thuật toán phân loại**

#### Thuật toán cơ bản
- 🌳 **Decision Tree**: Cây quyết định đơn giản
- 🎲 **K-Nearest Neighbors (KNN)**: Phân loại dựa trên k điểm gần nhất
- 📊 **Naive Bayes**: Phân loại xác suất dựa trên định lý Bayes
- 📈 **Logistic Regression**: Hồi quy logistic đa lớp

#### Thuật toán nâng cao
- 🌲 **Random Forest**: Tổ hợp nhiều cây quyết định
  - Hyperparameters: n_estimators, max_depth, min_samples_split, min_samples_leaf
- 🚀 **Gradient Boosting**: Boosting tuần tự với gradient descent
  - Hyperparameters: learning_rate, n_estimators, max_depth, subsample
- 🎯 **AdaBoost**: Adaptive Boosting
- 🔮 **Support Vector Machine (SVM)**: Phân loại dựa trên siêu phẳng
  - Kernels: RBF, Polynomial
  - Hyperparameters: C, gamma
- 🧠 **Neural Network (MLP)**: Mạng nơ-ron nhiều lớp
- 📉 **Linear/Quadratic Discriminant Analysis**: Phân tích phân biệt tuyến tính

### 4. **Ensemble Learning (Học tổ hợp)**
- 🎭 **Voting Classifier (Hard)**: Bỏ phiếu theo đa số
- 🎭 **Voting Classifier (Soft)**: Bỏ phiếu theo xác suất trung bình
- 🎒 **Bagging**: Bootstrap Aggregating
- 🔗 **Stacking**: Kết hợp dự đoán từ nhiều mô hình

### 5. **Tối ưu hóa siêu tham số**
- ⚙️ **GridSearchCV**: Tìm kiếm lưới với cross-validation
- 🔄 **StratifiedKFold**: Cross-validation 5-fold có tầng
- 🎯 **Scoring**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 6. **Phân tích và đánh giá**

#### Metrics (Chỉ số đánh giá)
- ✅ **Accuracy**: Độ chính xác tổng thể
- ✅ **Precision**: Độ chính xác dương
- ✅ **Recall**: Độ nhạy
- ✅ **F1-Score**: Trung bình điều hòa của Precision và Recall
- ✅ **ROC-AUC**: Diện tích dưới đường cong ROC
- ✅ **Confusion Matrix**: Ma trận nhầm lẫn

#### Visualization (Trực quan hóa)
- 📊 **Ma trận tương quan**: Heatmap correlation matrix
- 📈 **Đường cong ROC**: ROC curves cho từng lớp
- 🎯 **Feature Importance**: Tầm quan trọng của đặc trưng
- 📉 **Learning Curve**: Đường cong học tập
- 🔍 **PCA 2D**: Phân tích thành phần chính
- 📊 **Residual Plot**: Biểu đồ phần dư
- 🎨 **Distribution plots**: Phân bố xác suất dự đoán

---

## 🔥 Sử dụng

### 1. Huấn luyện mô hình (Training)

```bash
python src/train_model_vn.py
```

**Quá trình huấn luyện bao gồm:**
1. ✅ Chuyển đổi dữ liệu sang tiếng Việt
2. 📊 Phân tích thống kê chi tiết
3. 📈 Ma trận tương quan
4. 🎯 Phân tích lựa chọn đặc trưng (F-score, Mutual Information)
5. 🔄 So sánh các phương pháp chuẩn hóa (Standard, Robust, MinMax)
6. ⚙️ Tối ưu hóa siêu tham số với GridSearchCV
7. 🎭 Huấn luyện mô hình Ensemble
8. 📊 Tạo biểu đồ phân tích toàn diện
9. 💾 Lưu mô hình tốt nhất

**Output:**
- `dream_data_vn.csv` - Dữ liệu tiếng Việt
- `mo_hinh_tot_nhat_vn.pkl` - Mô hình tốt nhất
- `scaler_vn.pkl` - Scaler đã fit
- `ten_dac_trung_vn.pkl` - Tên các đặc trưng
- `ma_tran_tuong_quan.png` - Ma trận tương quan
- `phan_tich_dac_trung.png` - Phân tích đặc trưng
- `phan_tich_toan_dien.png` - Biểu đồ tổng hợp

### 2. Dự đoán (Prediction)

```bash
python src/predict_vn.py
```

**Chức năng:**
- 🎮 5 chế độ dự đoán:
  1. Nhập dữ liệu thủ công
  2. Dữ liệu mẫu: Stress cao
  3. Dữ liệu mẫu: Ngủ ngon
  4. Dữ liệu mẫu: Ngủ sâu
  5. Dữ liệu ngẫu nhiên
- 📊 Hiển thị xác suất cho từng loại giấc mơ
- 💡 Đưa ra lời khuyên cải thiện giấc ngủ
- 🔄 Dự đoán nhiều lần liên tiếp

---

## 📈 Kết quả

### So sánh các thuật toán (Độ chính xác trên test set)

| Thuật toán | Accuracy | Đặc điểm |
|-----------|----------|----------|
| **Neural Network (MLP)** | ~94% | ⭐ Tốt nhất trên lần train hiện tại |
| SVM (RBF) | ~92% | Mạnh với quan hệ phi tuyến |
| Gradient Boosting | ~91% | Tốt, có feature importance |
| Random Forest | ~88% | Tốt, dễ giải thích |
| Logistic Regression | ~70-72% | Nhanh, đơn giản |
| Logistic Regression | baseline | Nhanh, đơn giản |
| KNN | ~65-70% | Đơn giản nhưng chậm |

### Đặc trưng quan trọng (Feature Importance)

```
1. Mức stress         ████████████████████ 100.0%
2. Chất lượng ngủ     ████████░░░░░░░░░░░░  40.5%
3. Giờ ngủ            ███░░░░░░░░░░░░░░░░░  15.2%
4. Caffeine           ██░░░░░░░░░░░░░░░░░░  10.8%
5. Phút tập luyện     █░░░░░░░░░░░░░░░░░░░   8.3%
6. Thời gian màn hình █░░░░░░░░░░░░░░░░░░░   6.1%
7. Tuổi               ░░░░░░░░░░░░░░░░░░░░   2.4%
```

---

## 🔬 Phân tích chuyên sâu

### 1. Cross-validation
- Sử dụng **StratifiedKFold** với 5 folds
- Đảm bảo phân bố nhãn đồng đều trong mỗi fold
- Tránh overfitting

### 2. Regularization
- **L2 regularization** trong Logistic Regression (C parameter)
- **max_depth, min_samples_split** trong tree-based models
- **alpha** trong Neural Networks

### 3. Handling Imbalanced Data
- Dữ liệu không cân bằng: Mơ đẹp (47%) > Ngủ sâu (32%) > Ác mộng (20%)
- Giải pháp: StratifiedKFold, class_weight='balanced'

### 4. Noise Injection
- Thêm **Gaussian noise** (σ=0.08) vào features
- Thêm **label noise** (10% probability) để tăng robustness
- Giúp mô hình không overfit và tổng quát hóa tốt hơn

---

## 💡 Lời khuyên để có giấc ngủ tốt

### ✅ Nên làm
- 😴 Ngủ 7-9 giờ mỗi đêm
- 🧘 Giảm stress bằng thiền, yoga
- 🏃 Tập thể dục ít nhất 30 phút/ngày
- ☕ Hạn chế caffeine sau 3 giờ chiều
- 📵 Tắt màn hình trước khi ngủ 1-2 giờ
- 🌡️ Giữ phòng ngủ mát mẻ, tối, yên tĩnh

### ❌ Không nên làm
- 😰 Stress quá nhiều
- ☕ Uống quá nhiều caffeine
- 📱 Xem điện thoại trước khi ngủ
- 🍔 Ăn no trước khi ngủ
- 🌞 Ngủ không đều giờ

---

## 📚 Tài liệu tham khảo

### Papers & Books
- Breiman, L. (2001). Random Forests. Machine Learning.
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
- Cortes, C., & Vapnik, V. (1995). Support-Vector Networks.

### Libraries
- [scikit-learn](https://scikit-learn.org/) - Machine Learning
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) - Visualization
- [rich](https://rich.readthedocs.io/) - Terminal formatting

---

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Hãy tạo pull request hoặc mở issue.

---

## 📜 License

MIT License

---

## 👨‍💻 Tác giả

**Dream Prediction Team**
- Machine Learning Engineer
- Data Scientist

---

## 🌟 Features (Tính năng)

- ✅ Dữ liệu tiếng Việt hoàn toàn
- ✅ 9+ thuật toán Machine Learning
- ✅ Tối ưu hóa siêu tham số tự động
- ✅ Ensemble Learning
- ✅ Phân tích đặc trưng chi tiết
- ✅ Visualization đẹp mắt với Rich
- ✅ Interactive prediction
- ✅ Lời khuyên cá nhân hóa
- ✅ Cross-validation nghiêm ngặt
- ✅ Multiple scaling methods
- ✅ ROC-AUC analysis
- ✅ Learning curve analysis
- ✅ Residual analysis

---

## 🔮 Tương lai

- [ ] Deep Learning (CNN, LSTM)
- [ ] Web API với Flask/FastAPI
- [ ] Mobile App
- [ ] Real-time prediction
- [ ] Thêm nhiều features (nhiệt độ, độ ẩm, ...)
- [ ] Personalized recommendations
- [ ] Integration với thiết bị đeo tay

---

**Chúc bạn có những giấc ngủ ngon! 😴🌙✨**

