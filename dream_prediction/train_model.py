"""
Script huấn luyện mô hình dự đoán giấc mơ
Sử dụng nhiều thuật toán Machine Learning và so sánh hiệu suất
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Import các thuật toán ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Rich imports
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.panel import Panel
from rich import box
from rich.text import Text

# Khởi tạo console
console = Console()

# Thiết lập style cho biểu đồ
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_prepare_data(filepath=None):
    """Tải và chuẩn bị dữ liệu - ưu tiên dữ liệu thật"""
    
    # Tự động chọn dữ liệu tốt nhất có sẵn
    if filepath is None:
        if os.path.exists('dream_data_real.csv'):
            filepath = 'dream_data_real.csv'
            with console.status("[bold cyan]Đang tải dữ liệu..."):
                df = pd.read_csv(filepath)
            console.print(Panel(
                "[bold green]✓ Sử dụng DỮ LIỆU THẬT từ Kaggle Sleep Health Dataset[/bold green]",
                border_style="green"
            ))
        else:
            filepath = 'dream_data.csv'
            with console.status("[bold cyan]Đang tải dữ liệu..."):
                df = pd.read_csv(filepath)
            console.print(Panel(
                "[bold yellow]⚠ Sử dụng dữ liệu giả định[/bold yellow]\n"
                "[dim]💡 Để sử dụng dữ liệu thật: chạy 'python load_real_data.py'[/dim]",
                border_style="yellow"
            ))
    else:
        with console.status(f"[bold cyan]Đang tải dữ liệu từ: {filepath}..."):
    df = pd.read_csv(filepath)
    
    # Tách features và target
    X = df.drop('dream_type', axis=1)
    y = df['dream_type']
    
    return X, y, df

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Huấn luyện và đánh giá nhiều mô hình"""
    
    # Định nghĩa các mô hình
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🤖 Đang huấn luyện các mô hình...[/bold cyan]",
        border_style="cyan"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Huấn luyện mô hình...", total=len(models))
        
        for name, model in models.items():
            progress.update(task, description=f"[cyan]Đang huấn luyện {name}...")
        
        # Huấn luyện
        model.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = model.predict(X_test)
        
        # Đánh giá
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
            progress.advance(task)
    
    # Hiển thị kết quả trong table
    console.print("\n")
    results_table = Table(title="📊 Kết Quả Huấn Luyện", box=box.ROUNDED, show_header=True)
    results_table.add_column("Mô Hình", style="cyan", no_wrap=True)
    results_table.add_column("Accuracy", justify="right", style="green")
    results_table.add_column("CV Mean", justify="right", style="yellow")
    results_table.add_column("CV Std", justify="right", style="blue")
    
    for name in models.keys():
        acc = results[name]['accuracy']
        cv_mean = results[name]['cv_mean']
        cv_std = results[name]['cv_std']
        
        # Highlight best accuracy
        acc_style = "bold green" if acc == max(r['accuracy'] for r in results.values()) else "green"
        
        results_table.add_row(
            name,
            f"[{acc_style}]{acc:.4f}[/{acc_style}]",
            f"{cv_mean:.4f}",
            f"±{cv_std:.4f}"
        )
    
    console.print(results_table)
    
    return results

def plot_model_comparison(results):
    """Vẽ biểu đồ so sánh các mô hình"""
    
    # Chuẩn bị dữ liệu
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    cv_means = [results[name]['cv_mean'] for name in model_names]
    
    # Tạo biểu đồ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Biểu đồ 1: Độ chính xác
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars1 = ax1.barh(model_names, accuracies, color=colors)
    ax1.set_xlabel('Độ chính xác', fontsize=12, fontweight='bold')
    ax1.set_title('So sánh độ chính xác các mô hình', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1])
    
    # Thêm giá trị lên thanh
    for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
        ax1.text(acc + 0.01, i, f'{acc:.4f}', va='center', fontsize=10)
    
    # Biểu đồ 2: Cross-validation scores
    bars2 = ax2.barh(model_names, cv_means, color=colors)
    ax2.set_xlabel('Cross-validation Score', fontsize=12, fontweight='bold')
    ax2.set_title('Cross-validation Performance', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    
    for i, (bar, cv) in enumerate(zip(bars2, cv_means)):
        ax2.text(cv + 0.01, i, f'{cv:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    console.print(f"[green]✓[/green] Biểu đồ so sánh đã được lưu: [cyan]model_comparison.png[/cyan]")
    
def plot_confusion_matrix(y_test, y_pred, model_name):
    """Vẽ confusion matrix"""
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ác mộng', 'Mơ đẹp', 'Ngủ sâu'],
                yticklabels=['Ác mộng', 'Mơ đẹp', 'Ngủ sâu'],
                cbar_kws={'label': 'Số lượng'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Thực tế', fontsize=12, fontweight='bold')
    plt.xlabel('Dự đoán', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    console.print(f"[green]✓[/green] Confusion matrix đã được lưu: [cyan]confusion_matrix.png[/cyan]")

def plot_feature_importance(model, feature_names):
    """Vẽ biểu đồ tầm quan trọng của các đặc trưng"""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        colors = plt.cm.plasma(np.linspace(0, 1, len(feature_names)))
        
        plt.bar(range(len(importances)), importances[indices], color=colors)
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Đặc trưng', fontsize=12, fontweight='bold')
        plt.ylabel('Tầm quan trọng', fontsize=12, fontweight='bold')
        plt.title('Tầm quan trọng của các đặc trưng', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        console.print(f"[green]✓[/green] Biểu đồ tầm quan trọng đặc trưng đã được lưu: [cyan]feature_importance.png[/cyan]")
    else:
        console.print(Panel(
            "[bold yellow]⚠ Mô hình này không hỗ trợ feature importance[/bold yellow]\n"
            "[dim]💡 Để xem feature importance, hãy sử dụng mô hình như Random Forest hoặc Gradient Boosting[/dim]",
            border_style="yellow"
        ))

def main():
    """Hàm chính"""
    
    # Header
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]DREAM PREDICTION - MACHINE LEARNING TRAINING[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # 1. Tải dữ liệu
    X, y, df = load_and_prepare_data()
    
    # Thông tin dữ liệu
    info_table = Table(title="📊 Thông Tin Dữ Liệu", box=box.ROUNDED, show_header=False)
    info_table.add_column("Metric", style="cyan")
    info_table.add_column("Value", style="green", justify="right")
    info_table.add_row("Tổng số mẫu", str(len(df)))
    info_table.add_row("Số đặc trưng", str(X.shape[1]))
    console.print("\n")
    console.print(info_table)
    
    # Phân bố nhãn
    dream_labels = {0: 'Ác mộng', 1: 'Mơ đẹp', 2: 'Ngủ sâu'}
    dist_table = Table(title="📊 Phân Bố Dữ Liệu", box=box.ROUNDED, show_header=True)
    dist_table.add_column("Loại Giấc Mơ", style="cyan")
    dist_table.add_column("Số Lượng", justify="right", style="magenta")
    dist_table.add_column("Tỷ Lệ", justify="right", style="green")
    
    for dream_type, count in y.value_counts().sort_index().items():
        percentage = (count / len(y)) * 100
        dist_table.add_row(
            dream_labels[dream_type],
            str(count),
            f"{percentage:.1f}%"
        )
    
    console.print("\n")
    console.print(dist_table)
    
    # 2. Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Huấn luyện và đánh giá các mô hình
    results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 5. Tìm mô hình tốt nhất
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']
    
    # Hiển thị mô hình tốt nhất
    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]🏆 MÔ HÌNH TỐT NHẤT: {best_model_name}[/bold green]\n"
        f"[bold yellow]🎯 Độ chính xác: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)[/bold yellow]",
        border_style="green",
        box=box.DOUBLE
    ))
    
    # 6. Báo cáo chi tiết cho mô hình tốt nhất
    y_pred_best = results[best_model_name]['predictions']
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]📋 BÁO CÁO PHÂN LOẠI CHI TIẾT[/bold cyan]",
        border_style="cyan"
    ))
    
    report = classification_report(y_test, y_pred_best, 
                                   target_names=['Ác mộng', 'Mơ đẹp', 'Ngủ sâu'],
                                   output_dict=True)
    
    # Tạo table cho classification report
    report_table = Table(box=box.ROUNDED, show_header=True)
    report_table.add_column("Class", style="cyan")
    report_table.add_column("Precision", justify="right", style="green")
    report_table.add_column("Recall", justify="right", style="yellow")
    report_table.add_column("F1-Score", justify="right", style="magenta")
    report_table.add_column("Support", justify="right", style="blue")
    
    for class_name in ['Ác mộng', 'Mơ đẹp', 'Ngủ sâu']:
        report_table.add_row(
            class_name,
            f"{report[class_name]['precision']:.4f}",
            f"{report[class_name]['recall']:.4f}",
            f"{report[class_name]['f1-score']:.4f}",
            str(int(report[class_name]['support']))
        )
    
    report_table.add_row(
        "[bold]Accuracy[/bold]",
        "",
        "",
        f"[bold]{report['accuracy']:.4f}[/bold]",
        str(int(report['macro avg']['support']))
    )
    
    console.print(report_table)
    
    # 7. Vẽ các biểu đồ
    console.print("\n")
    with console.status("[bold cyan]Đang tạo các biểu đồ..."):
    plot_model_comparison(results)
    plot_confusion_matrix(y_test, y_pred_best, best_model_name)
    plot_feature_importance(best_model, X.columns.tolist())
    
    # 8. Lưu mô hình và scaler
    with console.status("[bold cyan]Đang lưu mô hình..."):
    joblib.dump(best_model, 'best_dream_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    console.print("\n")
    console.print(Panel(
        f"[bold green]✓ Mô hình đã được lưu:[/bold green] [cyan]best_dream_model.pkl[/cyan]\n"
        f"[bold green]✓ Scaler đã được lưu:[/bold green] [cyan]scaler.pkl[/cyan]",
        border_style="green",
        box=box.ROUNDED
    ))
    
    # Footer
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]✅ HOÀN THÀNH![/bold green]",
        border_style="green",
        box=box.DOUBLE
    ))

if __name__ == "__main__":
    main()
