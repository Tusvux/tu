"""
So sánh các phương pháp Machine Learning - TIẾNG VIỆT
Phân tích chi tiết ưu nhược điểm của từng thuật toán
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# Thiết lập matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_ml_comparison_guide():
    """Tạo bảng so sánh chi tiết các thuật toán ML"""
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]📚 SO SÁNH CÁC THUẬT TOÁN MACHINE LEARNING[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # 1. Bảng tổng quan
    overview_table = Table(
        title="📊 Tổng Quan Các Thuật Toán",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    
    overview_table.add_column("Thuật toán", style="yellow", width=20)
    overview_table.add_column("Loại", style="green", width=15)
    overview_table.add_column("Độ chính xác", style="cyan", width=15)
    overview_table.add_column("Tốc độ", style="magenta", width=12)
    overview_table.add_column("Giải thích", style="blue", width=12)
    
    algorithms = [
        ("Random Forest", "Ensemble", "Cao (75-77%)", "Trung bình", "Tốt"),
        ("Gradient Boosting", "Ensemble", "Cao (75-77%)", "Chậm", "Trung bình"),
        ("SVM (RBF)", "Kernel", "Rất cao (76-78%)", "Chậm", "Khó"),
        ("Logistic Regression", "Linear", "Trung bình (70-72%)", "Rất nhanh", "Rất tốt"),
        ("Neural Network", "Deep Learning", "Trung bình (70-72%)", "Chậm", "Khó"),
        ("K-Nearest Neighbors", "Instance-based", "Thấp (65-70%)", "Rất chậm", "Tốt"),
        ("Naive Bayes", "Probabilistic", "Thấp (60-65%)", "Rất nhanh", "Tốt"),
        ("Decision Tree", "Tree", "Thấp (60-65%)", "Nhanh", "Rất tốt"),
        ("Voting Ensemble", "Ensemble", "Cao (75-77%)", "Chậm", "Trung bình"),
        ("AdaBoost", "Boosting", "Trung bình (70-75%)", "Trung bình", "Trung bình"),
    ]
    
    for algo in algorithms:
        overview_table.add_row(*algo)
    
    console.print("\n")
    console.print(overview_table)
    
    # 2. Phân tích chi tiết từng thuật toán
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]🔬 PHÂN TÍCH CHI TIẾT[/bold green]",
        border_style="green"
    ))
    
    algorithms_detail = {
        "Random Forest": {
            "Nguyên lý": "Tổ hợp nhiều cây quyết định, mỗi cây được huấn luyện trên subset ngẫu nhiên của dữ liệu",
            "Ưu điểm": [
                "✅ Độ chính xác cao",
                "✅ Xử lý tốt dữ liệu nhiều chiều",
                "✅ Feature importance rõ ràng",
                "✅ Ít bị overfitting",
                "✅ Hoạt động tốt với dữ liệu nhiễu"
            ],
            "Nhược điểm": [
                "❌ Tốn bộ nhớ",
                "❌ Khó giải thích từng dự đoán",
                "❌ Chậm hơn các mô hình đơn giản"
            ],
            "Hyperparameters": [
                "• n_estimators: Số lượng cây (50-200)",
                "• max_depth: Độ sâu tối đa (3-10 hoặc None)",
                "• min_samples_split: Số mẫu tối thiểu để split (2-10)",
                "• min_samples_leaf: Số mẫu tối thiểu ở lá (1-4)"
            ],
            "Khi nào dùng": "Dữ liệu lớn, cần độ chính xác cao, không cần giải thích chi tiết"
        },
        
        "Gradient Boosting": {
            "Nguyên lý": "Xây dựng các cây tuần tự, mỗi cây học từ lỗi của cây trước bằng gradient descent",
            "Ưu điểm": [
                "✅ Độ chính xác rất cao",
                "✅ Xử lý tốt các mối quan hệ phức tạp",
                "✅ Feature importance chi tiết",
                "✅ Hoạt động tốt với dữ liệu không cân bằng"
            ],
            "Nhược điểm": [
                "❌ Dễ bị overfitting nếu không tune tốt",
                "❌ Chậm trong training",
                "❌ Nhạy cảm với hyperparameters",
                "❌ Tốn bộ nhớ"
            ],
            "Hyperparameters": [
                "• learning_rate: Tốc độ học (0.01-0.1)",
                "• n_estimators: Số lượng cây (50-200)",
                "• max_depth: Độ sâu cây (2-5)",
                "• subsample: Tỷ lệ sampling (0.8-1.0)"
            ],
            "Khi nào dùng": "Cần độ chính xác tối đa, có thời gian để tune parameters"
        },
        
        "SVM (Support Vector Machine)": {
            "Nguyên lý": "Tìm siêu phẳng tối ưu phân tách các lớp với margin lớn nhất, sử dụng kernel trick",
            "Ưu điểm": [
                "✅ Hiệu quả với dữ liệu nhiều chiều",
                "✅ Sử dụng subset của training data (support vectors)",
                "✅ Hiệu quả với các kernel khác nhau (RBF, polynomial)",
                "✅ Robust với dữ liệu nhiễu",
                "✅ Hoạt động tốt với dữ liệu nhỏ"
            ],
            "Nhược điểm": [
                "❌ Rất chậm với dữ liệu lớn",
                "❌ Nhạy cảm với việc chọn kernel và parameters",
                "❌ Khó giải thích",
                "❌ Không cho feature importance trực tiếp",
                "❌ Tốn bộ nhớ với dữ liệu lớn"
            ],
            "Hyperparameters": [
                "• C: Regularization (0.1-100)",
                "• gamma: Kernel coefficient (0.001-1 hoặc 'scale')",
                "• kernel: Loại kernel ('rbf', 'poly', 'linear')"
            ],
            "Khi nào dùng": "Dữ liệu nhỏ-trung bình, cần độ chính xác cao, không quan tâm tốc độ"
        },
        
        "Logistic Regression": {
            "Nguyên lý": "Mô hình tuyến tính sử dụng hàm sigmoid/softmax để dự đoán xác suất",
            "Ưu điểm": [
                "✅ Rất nhanh",
                "✅ Dễ giải thích (coefficients)",
                "✅ Không cần nhiều hyperparameter tuning",
                "✅ Hoạt động tốt với dữ liệu tuyến tính",
                "✅ Ít bộ nhớ",
                "✅ Cho xác suất dự đoán"
            ],
            "Nhược điểm": [
                "❌ Giả định quan hệ tuyến tính",
                "❌ Độ chính xác thấp hơn với dữ liệu phức tạp",
                "❌ Nhạy cảm với outliers",
                "❌ Không xử lý tốt feature interaction"
            ],
            "Hyperparameters": [
                "• C: Inverse regularization (1-100)",
                "• penalty: L1 hoặc L2 ('l1', 'l2')",
                "• solver: Thuật toán tối ưu ('lbfgs', 'saga')",
                "• max_iter: Số iterations (100-5000)"
            ],
            "Khi nào dùng": "Baseline model, cần tốc độ cao, cần giải thích, dữ liệu gần tuyến tính"
        },
        
        "Neural Network (MLP)": {
            "Nguyên lý": "Mạng nơ-ron nhiều lớp với các kết nối phi tuyến",
            "Ưu điểm": [
                "✅ Học được các mối quan hệ phức tạp",
                "✅ Linh hoạt về kiến trúc",
                "✅ Hoạt động tốt với dữ liệu lớn",
                "✅ Có thể mở rộng thành deep learning"
            ],
            "Nhược điểm": [
                "❌ Cần nhiều dữ liệu",
                "❌ Khó tune hyperparameters",
                "❌ Dễ overfit",
                "❌ Black box (khó giải thích)",
                "❌ Chậm trong training",
                "❌ Nhạy cảm với khởi tạo"
            ],
            "Hyperparameters": [
                "• hidden_layer_sizes: Kích thước các lớp ẩn (20, 50, 100)",
                "• activation: Hàm kích hoạt ('relu', 'tanh')",
                "• alpha: Regularization (0.0001-0.1)",
                "• learning_rate_init: Tốc độ học ban đầu (0.001-0.01)"
            ],
            "Khi nào dùng": "Dữ liệu lớn, quan hệ phức tạp, có GPU, không cần giải thích"
        },
        
        "K-Nearest Neighbors (KNN)": {
            "Nguyên lý": "Phân loại dựa trên nhãn của k điểm gần nhất trong không gian đặc trưng",
            "Ưu điểm": [
                "✅ Rất đơn giản, dễ hiểu",
                "✅ Không cần training",
                "✅ Hoạt động tốt với dữ liệu phân tách rõ ràng",
                "✅ Không giả định về phân phối dữ liệu"
            ],
            "Nhược điểm": [
                "❌ Rất chậm trong prediction",
                "❌ Nhạy cảm với scale của features",
                "❌ Không hoạt động tốt với dữ liệu nhiều chiều (curse of dimensionality)",
                "❌ Cần nhiều bộ nhớ để lưu toàn bộ training data",
                "❌ Nhạy cảm với k và metric"
            ],
            "Hyperparameters": [
                "• n_neighbors: Số láng giềng k (3-20)",
                "• weights: Trọng số ('uniform', 'distance')",
                "• metric: Độ đo khoảng cách ('euclidean', 'manhattan')"
            ],
            "Khi nào dùng": "Dữ liệu nhỏ, không nhiều chiều, cần baseline đơn giản"
        },
        
        "Voting Ensemble": {
            "Nguyên lý": "Kết hợp dự đoán từ nhiều mô hình khác nhau bằng voting (hard hoặc soft)",
            "Ưu điểm": [
                "✅ Độ chính xác cao và ổn định",
                "✅ Giảm variance",
                "✅ Kết hợp điểm mạnh của nhiều mô hình",
                "✅ Ít bị overfitting hơn từng mô hình đơn"
            ],
            "Nhược điểm": [
                "❌ Chậm (phải chạy nhiều mô hình)",
                "❌ Tốn bộ nhớ",
                "❌ Khó giải thích",
                "❌ Phức tạp trong deployment"
            ],
            "Hyperparameters": [
                "• voting: 'hard' (đa số) hoặc 'soft' (xác suất TB)",
                "• estimators: Danh sách các mô hình base",
                "• weights: Trọng số cho từng mô hình"
            ],
            "Khi nào dùng": "Cần độ chính xác tối đa, có tài nguyên, không quan tâm tốc độ"
        }
    }
    
    for algo_name, details in algorithms_detail.items():
        console.print("\n")
        console.print(f"[bold yellow]{'='*80}[/bold yellow]")
        console.print(f"[bold cyan]📌 {algo_name.upper()}[/bold cyan]")
        console.print(f"[bold yellow]{'='*80}[/bold yellow]")
        
        console.print(f"\n[bold]Nguyên lý:[/bold]")
        console.print(f"[dim]{details['Nguyên lý']}[/dim]")
        
        console.print(f"\n[bold green]Ưu điểm:[/bold green]")
        for pro in details['Ưu điểm']:
            console.print(f"  {pro}")
        
        console.print(f"\n[bold red]Nhược điểm:[/bold red]")
        for con in details['Nhược điểm']:
            console.print(f"  {con}")
        
        console.print(f"\n[bold blue]Hyperparameters chính:[/bold blue]")
        for param in details['Hyperparameters']:
            console.print(f"  {param}")
        
        console.print(f"\n[bold magenta]💡 Khi nào nên dùng:[/bold magenta]")
        console.print(f"  {details['Khi nào dùng']}")
    
    # 3. Bảng so sánh các phương pháp chuẩn hóa
    console.print("\n\n")
    console.print(Panel.fit(
        "[bold cyan]📏 SO SÁNH CÁC PHƯƠNG PHÁP CHUẨN HÓA[/bold cyan]",
        border_style="cyan"
    ))
    
    scaler_table = Table(
        title="Các Phương Pháp Scaling",
        box=box.ROUNDED,
        show_header=True
    )
    
    scaler_table.add_column("Phương pháp", style="cyan")
    scaler_table.add_column("Công thức", style="yellow")
    scaler_table.add_column("Ưu điểm", style="green")
    scaler_table.add_column("Nhược điểm", style="red")
    scaler_table.add_column("Khi nào dùng", style="magenta")
    
    scalers = [
        (
            "StandardScaler",
            "(x - μ) / σ",
            "Đơn giản, phổ biến\nHoạt động tốt với\nphân phối chuẩn",
            "Nhạy cảm với\noutliers",
            "Dữ liệu gần\nphân phối chuẩn"
        ),
        (
            "RobustScaler",
            "(x - median) / IQR",
            "Chống nhiễu tốt\nKhông bị ảnh hưởng\nbởi outliers",
            "Phức tạp hơn",
            "Dữ liệu có\nnhiều outliers"
        ),
        (
            "MinMaxScaler",
            "(x - min) / (max - min)",
            "Dữ liệu trong [0,1]\nDễ giải thích",
            "Rất nhạy cảm\nvới outliers",
            "Neural networks\nDữ liệu bounded"
        ),
    ]
    
    for scaler in scalers:
        scaler_table.add_row(*scaler)
    
    console.print("\n")
    console.print(scaler_table)
    
    # 4. Bảng so sánh các kỹ thuật feature selection
    console.print("\n\n")
    console.print(Panel.fit(
        "[bold cyan]🎯 SO SÁNH CÁC PHƯƠNG PHÁP LỰA CHỌN ĐẶC TRƯNG[/bold cyan]",
        border_style="cyan"
    ))
    
    fs_table = Table(
        title="Feature Selection Methods",
        box=box.ROUNDED,
        show_header=True
    )
    
    fs_table.add_column("Phương pháp", style="cyan")
    fs_table.add_column("Loại", style="yellow")
    fs_table.add_column("Mô tả", style="green")
    fs_table.add_column("Ưu điểm", style="blue")
    
    fs_methods = [
        (
            "F-score (ANOVA)",
            "Filter",
            "Đo mức độ khác biệt\ngiữa các nhóm bằng\npphương sai",
            "Nhanh, đơn giản\nĐộc lập với model"
        ),
        (
            "Mutual Information",
            "Filter",
            "Đo lượng thông tin\nchung giữa feature\nvà target",
            "Phát hiện quan hệ\nphi tuyến"
        ),
        (
            "Feature Importance",
            "Embedded",
            "Importance từ\ntree-based models",
            "Tự động, chính xác\nXem xét interaction"
        ),
        (
            "L1 Regularization",
            "Embedded",
            "LASSO - đưa\ncoefficients về 0",
            "Tự động loại bỏ\nfeatures không quan trọng"
        ),
        (
            "RFE",
            "Wrapper",
            "Recursive Feature\nElimination",
            "Tối ưu cho\nmô hình cụ thể"
        ),
    ]
    
    for method in fs_methods:
        fs_table.add_row(*method)
    
    console.print("\n")
    console.print(fs_table)
    
    # 5. Hướng dẫn chọn thuật toán
    console.print("\n\n")
    console.print(Panel(
        "[bold cyan]📚 HƯỚNG DẪN CHỌN THUẬT TOÁN[/bold cyan]\n\n"
        
        "[bold yellow]1. Nếu bạn mới bắt đầu:[/bold yellow]\n"
        "   • Thử Logistic Regression trước (baseline)\n"
        "   • Sau đó thử Random Forest\n\n"
        
        "[bold yellow]2. Nếu cần độ chính xác cao nhất:[/bold yellow]\n"
        "   • SVM với RBF kernel\n"
        "   • Gradient Boosting\n"
        "   • Voting Ensemble\n\n"
        
        "[bold yellow]3. Nếu cần tốc độ nhanh:[/bold yellow]\n"
        "   • Logistic Regression\n"
        "   • Naive Bayes\n"
        "   • Decision Tree đơn giản\n\n"
        
        "[bold yellow]4. Nếu cần giải thích kết quả:[/bold yellow]\n"
        "   • Logistic Regression (coefficients)\n"
        "   • Decision Tree (cây quyết định)\n"
        "   • Random Forest (feature importance)\n\n"
        
        "[bold yellow]5. Nếu dữ liệu nhỏ (<1000 mẫu):[/bold yellow]\n"
        "   • SVM\n"
        "   • Logistic Regression\n"
        "   • Naive Bayes\n\n"
        
        "[bold yellow]6. Nếu dữ liệu lớn (>100k mẫu):[/bold yellow]\n"
        "   • Logistic Regression\n"
        "   • SGDClassifier\n"
        "   • Neural Network\n\n"
        
        "[bold yellow]7. Nếu dữ liệu có nhiều outliers:[/bold yellow]\n"
        "   • Random Forest\n"
        "   • Gradient Boosting\n"
        "   • SVM với RBF kernel\n\n"
        
        "[bold yellow]8. Nếu muốn tự động hóa hoàn toàn:[/bold yellow]\n"
        "   • AutoML (Auto-sklearn, TPOT)\n"
        "   • GridSearchCV với nhiều models\n"
        "   • Ensemble với voting",
        
        border_style="green",
        box=box.ROUNDED
    ))
    
    # 6. Tips & Tricks
    console.print("\n\n")
    console.print(Panel(
        "[bold cyan]💡 TIPS & TRICKS[/bold cyan]\n\n"
        
        "[bold green]✅ Best Practices:[/bold green]\n"
        "1. Luôn bắt đầu với baseline model đơn giản\n"
        "2. Sử dụng cross-validation (ít nhất 5-fold)\n"
        "3. Chuẩn hóa dữ liệu trước khi training\n"
        "4. Thử nhiều scalers khác nhau\n"
        "5. Tune hyperparameters với GridSearchCV/RandomizedSearchCV\n"
        "6. Ensemble nhiều mô hình tốt để cải thiện accuracy\n"
        "7. Phân tích feature importance để hiểu dữ liệu\n"
        "8. Kiểm tra confusion matrix để hiểu lỗi\n"
        "9. Sử dụng stratified sampling với dữ liệu không cân bằng\n"
        "10. Lưu lại mô hình và scaler để deploy\n\n"
        
        "[bold red]❌ Những sai lầm thường gặp:[/bold red]\n"
        "1. Không chia train/test set\n"
        "2. Không chuẩn hóa dữ liệu\n"
        "3. Data leakage (fit scaler trên toàn bộ data)\n"
        "4. Overfitting (accuracy quá cao trên train set)\n"
        "5. Không tune hyperparameters\n"
        "6. Chỉ nhìn vào accuracy (bỏ qua precision, recall)\n"
        "7. Không xử lý missing values\n"
        "8. Không kiểm tra phân bố dữ liệu\n"
        "9. Sử dụng mô hình phức tạp cho dữ liệu đơn giản\n"
        "10. Không validation trước khi deploy",
        
        border_style="yellow",
        box=box.ROUNDED
    ))

def main():
    """Hàm chính"""
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🎓 HƯỚNG DẪN MACHINE LEARNING CHO DỰ ĐOÁN GIẤC MƠ[/bold cyan]\n"
        "[dim]Tài liệu tham khảo về các thuật toán và phương pháp[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    create_ml_comparison_guide()
    
    console.print("\n\n")
    console.print(Panel.fit(
        "[bold green]✅ HOÀN THÀNH![/bold green]\n"
        "[dim]Đã hiển thị toàn bộ hướng dẫn về Machine Learning[/dim]",
        border_style="green",
        box=box.DOUBLE
    ))

if __name__ == "__main__":
    main()
