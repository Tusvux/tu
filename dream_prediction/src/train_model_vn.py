# -*- coding: utf-8 -*-
"""
Script huấn luyện mô hình dự đoán giấc mơ - PHIÊN BẢN TIẾNG VIỆT
Sử dụng các phương pháp Machine Learning nâng cao
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, log_loss
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich import box
from paths import DATA_DIR, FIGURES_DIR, MODELS_DIR, ensure_project_dirs

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

console = Console()

# Thiết lập matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Mapping tiếng Anh -> tiếng Việt
FEATURE_MAPPING = {
    'age': 'tuoi',
    'sleep_hours': 'gio_ngu',
    'stress_level': 'muc_stress',
    'caffeine_intake': 'caffeine',
    'exercise_minutes': 'phut_tap_luyen',
    'sleep_quality': 'chat_luong_ngu',
    'screen_time': 'thoi_gian_man_hinh',
    'dream_type': 'loai_giac_mo'
}

FEATURE_NAMES_VN = {
    'tuoi': 'Tuổi',
    'gio_ngu': 'Giờ ngủ',
    'muc_stress': 'Mức stress',
    'caffeine': 'Caffeine',
    'phut_tap_luyen': 'Phút tập luyện',
    'chat_luong_ngu': 'Chất lượng ngủ',
    'thoi_gian_man_hinh': 'Thời gian màn hình'
}

DREAM_LABELS = {0: 'Ác mộng', 1: 'Mơ đẹp', 2: 'Ngủ sâu', 3: 'Không mơ'}

def get_present_labels(*arrays):
    """Lấy danh sách nhãn có mặt trong dữ liệu theo đúng mapping hiện tại."""
    labels = sorted(set(np.concatenate([np.asarray(array) for array in arrays]).astype(int)))
    return labels, [DREAM_LABELS.get(label, f"Không xác định ({label})") for label in labels]

def load_and_convert_data_to_vietnamese():
    """Tải và chuyển đổi dữ liệu sang tiếng Việt"""
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]📂 TẢI VÀ CHUYỂN ĐỔI DỮ LIỆU[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    data_file = DATA_DIR / 'dream_data_real.csv' if (DATA_DIR / 'dream_data_real.csv').exists() else DATA_DIR / 'dream_data.csv'

    try:
        df = pd.read_csv(data_file)
        console.print(f"[green]✓[/green] Đã tải file: [cyan]{data_file.name}[/cyan]")
    except FileNotFoundError:
        console.print("[red]✗[/red] Không tìm thấy file dream_data.csv hoặc dream_data_real.csv")
        console.print("[yellow]💡 Hãy chạy generate_data_4_classes.py hoặc load_real_data.py trước![/yellow]")
        exit(1)
    
    # Chuyển đổi tên cột
    df_vn = df.rename(columns=FEATURE_MAPPING)
    
    # Lưu file tiếng Việt
    df_vn.to_csv(DATA_DIR / 'dream_data_vn.csv', index=False, encoding='utf-8-sig')
    console.print(f"[green]✓[/green] Đã lưu file huấn luyện tiếng Việt: [cyan]data/dream_data_vn.csv[/cyan]")
    
    return df_vn

def analyze_data_statistics(df):
    """Phân tích thống kê dữ liệu"""
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]📊 PHÂN TÍCH THỐNG KÊ DỮ LIỆU[/bold cyan]",
        border_style="cyan"
    ))
    
    # Thống kê mô tả
    stats_table = Table(title="Thống Kê Mô Tả", box=box.ROUNDED)
    stats_table.add_column("Đặc trưng", style="cyan")
    stats_table.add_column("Trung bình", justify="right", style="yellow")
    stats_table.add_column("Độ lệch chuẩn", justify="right", style="green")
    stats_table.add_column("Min", justify="right", style="red")
    stats_table.add_column("Max", justify="right", style="magenta")
    
    for col in df.columns:
        if col != 'loai_giac_mo':
            stats_table.add_row(
                FEATURE_NAMES_VN.get(col, col),
                f"{df[col].mean():.2f}",
                f"{df[col].std():.2f}",
                f"{df[col].min():.2f}",
                f"{df[col].max():.2f}"
            )
    
    console.print("\n")
    console.print(stats_table)
    
    # Ma trận tương quan
    X = df.drop('loai_giac_mo', axis=1)
    corr_matrix = X.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    
    labels = [FEATURE_NAMES_VN.get(col, col) for col in corr_matrix.columns]
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.yticks(range(len(labels)), labels, rotation=0)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ma_tran_tuong_quan.png', dpi=300, bbox_inches='tight')
    console.print(f"[green]✓[/green] Đã lưu ma trận tương quan: [cyan]ma_tran_tuong_quan.png[/cyan]")
    plt.close()

def plot_distribution_histograms(df):
    """Vẽ biểu đồ histogram phân bố độ tuổi và thời lượng ngủ"""
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]📊 BIỂU ĐỒ PHÂN BỐ DỮ LIỆU[/bold cyan]",
        border_style="cyan"
    ))
    
    # Hình 4.1: Biểu đồ phân bố độ tuổi
    plt.figure(figsize=(10, 6))
    plt.hist(df['tuoi'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Độ tuổi (năm)', fontsize=12, fontweight='bold')
    plt.ylabel('Tần suất', fontsize=12, fontweight='bold')
    plt.title('Hình 4.1: Biểu đồ phân bố độ tuổi của tập dữ liệu khảo sát', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Thêm thống kê
    mean_age = df['tuoi'].mean()
    median_age = df['tuoi'].median()
    plt.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Trung bình: {mean_age:.1f}')
    plt.axvline(median_age, color='green', linestyle='--', linewidth=2, label=f'Trung vị: {median_age:.1f}')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'hinh_4_1_phan_bo_do_tuoi.png', dpi=300, bbox_inches='tight')
    console.print(f"[green]✓[/green] Đã lưu: [cyan]hinh_4_1_phan_bo_do_tuoi.png[/cyan]")
    plt.close()
    
    # Hình 4.2: Biểu đồ phân bố thời lượng ngủ
    plt.figure(figsize=(10, 6))
    plt.hist(df['gio_ngu'], bins=25, color='mediumseagreen', edgecolor='black', alpha=0.7)
    plt.xlabel('Thời lượng ngủ (giờ/đêm)', fontsize=12, fontweight='bold')
    plt.ylabel('Tần suất', fontsize=12, fontweight='bold')
    plt.title('Hình 4.2: Biểu đồ phân bố thời lượng ngủ trung bình mỗi đêm', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Thêm thống kê
    mean_sleep = df['gio_ngu'].mean()
    median_sleep = df['gio_ngu'].median()
    plt.axvline(mean_sleep, color='red', linestyle='--', linewidth=2, label=f'Trung bình: {mean_sleep:.1f}h')
    plt.axvline(median_sleep, color='blue', linestyle='--', linewidth=2, label=f'Trung vị: {median_sleep:.1f}h')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'hinh_4_2_phan_bo_thoi_luong_ngu.png', dpi=300, bbox_inches='tight')
    console.print(f"[green]✓[/green] Đã lưu: [cyan]hinh_4_2_phan_bo_thoi_luong_ngu.png[/cyan]")
    plt.close()
    
    # Hiển thị thống kê tóm tắt
    stats_table = Table(title="Thống Kê Chi Tiết", box=box.ROUNDED)
    stats_table.add_column("Đặc trưng", style="cyan")
    stats_table.add_column("Trung bình", justify="right", style="yellow")
    stats_table.add_column("Trung vị", justify="right", style="green")
    stats_table.add_column("Độ lệch chuẩn", justify="right", style="magenta")
    stats_table.add_column("Min", justify="right", style="red")
    stats_table.add_column("Max", justify="right", style="blue")
    
    stats_table.add_row(
        "Độ tuổi (năm)",
        f"{df['tuoi'].mean():.2f}",
        f"{df['tuoi'].median():.2f}",
        f"{df['tuoi'].std():.2f}",
        f"{df['tuoi'].min():.0f}",
        f"{df['tuoi'].max():.0f}"
    )
    
    stats_table.add_row(
        "Thời lượng ngủ (giờ)",
        f"{df['gio_ngu'].mean():.2f}",
        f"{df['gio_ngu'].median():.2f}",
        f"{df['gio_ngu'].std():.2f}",
        f"{df['gio_ngu'].min():.1f}",
        f"{df['gio_ngu'].max():.1f}"
    )
    
    console.print("\n")
    console.print(stats_table)

def plot_gradient_boosting_analysis(model, X_train, y_train, X_test, y_test, feature_names):
    """Phân tích chi tiết cho Gradient Boosting"""
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]📊 PHÂN TÍCH GRADIENT BOOSTING[/bold cyan]",
        border_style="cyan"
    ))
    
    # 1. Feature Importance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names_vn = [FEATURE_NAMES_VN.get(feature_names[i], feature_names[i]) for i in range(len(feature_names))]
    sorted_features = [feature_names_vn[i] for i in indices]
    sorted_importances = importances[indices]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))
    bars = axes[0, 0].barh(range(len(sorted_features)), sorted_importances, color=colors, edgecolor='black')
    axes[0, 0].set_yticks(range(len(sorted_features)))
    axes[0, 0].set_yticklabels(sorted_features)
    axes[0, 0].set_xlabel('Độ quan trọng', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Hình 4.3a: Độ quan trọng các đặc trưng\ntrong Gradient Boosting', 
                         fontsize=12, fontweight='bold', pad=15)
    axes[0, 0].grid(axis='x', alpha=0.3, linestyle='--')
    
    # Thêm giá trị lên cột
    for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
        axes[0, 0].text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # Subplot 2: Training vs Test Accuracy over iterations
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Tính accuracy theo từng iteration
    train_scores = []
    test_scores = []
    for i, y_pred in enumerate(model.staged_predict(X_train)):
        train_scores.append(accuracy_score(y_train, y_pred))
    for i, y_pred in enumerate(model.staged_predict(X_test)):
        test_scores.append(accuracy_score(y_test, y_pred))
    
    iterations = range(1, len(train_scores) + 1)
    axes[0, 1].plot(iterations, train_scores, label='Tập huấn luyện', 
                   color='#2E7D32', linewidth=2.5, marker='o', markersize=3, markevery=10)
    axes[0, 1].plot(iterations, test_scores, label='Tập kiểm thử', 
                   color='#D32F2F', linewidth=2.5, marker='s', markersize=3, markevery=10)
    axes[0, 1].set_xlabel('Số vòng lặp (iterations)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Độ chính xác', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Hình 4.3b: Đường cong học tập\nGradient Boosting', 
                        fontsize=12, fontweight='bold', pad=15)
    axes[0, 1].legend(fontsize=10, loc='lower right')
    axes[0, 1].grid(alpha=0.3, linestyle='--')
    axes[0, 1].set_ylim([0.5, 1.0])
    
    # Subplot 3: Confusion Matrix
    present_labels, present_label_names = get_present_labels(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred, labels=present_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                cbar_kws={'label': 'Tỷ lệ'}, ax=axes[1, 0],
                xticklabels=present_label_names,
                yticklabels=present_label_names,
                linewidths=1, linecolor='white', square=True)
    axes[1, 0].set_xlabel('Nhãn dự đoán', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Nhãn thực tế', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Hình 4.3c: Ma trận nhầm lẫn\nGradient Boosting', 
                        fontsize=12, fontweight='bold', pad=15)
    
    # Subplot 4: Loss curve
    train_deviance = []
    for y_proba in model.staged_predict_proba(X_train):
        train_deviance.append(log_loss(y_train, y_proba, labels=model.classes_))
    
    test_deviance = []
    for y_proba in model.staged_predict_proba(X_test):
        test_deviance.append(log_loss(y_test, y_proba, labels=model.classes_))
    
    axes[1, 1].plot(iterations, train_deviance, label='Tập huấn luyện', 
                   color='#1976D2', linewidth=2.5, alpha=0.8)
    axes[1, 1].plot(iterations, test_deviance, label='Tập kiểm thử', 
                   color='#F57C00', linewidth=2.5, alpha=0.8)
    axes[1, 1].set_xlabel('Số vòng lặp (iterations)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Deviance (Loss)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Hình 4.3d: Đường cong Loss\nGradient Boosting', 
                        fontsize=12, fontweight='bold', pad=15)
    axes[1, 1].legend(fontsize=10, loc='upper right')
    axes[1, 1].grid(alpha=0.3, linestyle='--')
    
    plt.suptitle('Hình 4.3: Phân tích chi tiết thuật toán Gradient Boosting', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(FIGURES_DIR / 'hinh_4_3_gradient_boosting_analysis.png', dpi=300, bbox_inches='tight')
    console.print(f"[green]✓[/green] Đã lưu: [cyan]hinh_4_3_gradient_boosting_analysis.png[/cyan]")
    plt.close()

def plot_neural_network_analysis(model, X_train, y_train, X_test, y_test, feature_names):
    """Phân tích chi tiết cho Neural Network"""
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🧠 PHÂN TÍCH NEURAL NETWORK[/bold cyan]",
        border_style="cyan"
    ))
    
    # Huấn luyện lại model để lấy loss curve
    from sklearn.neural_network import MLPClassifier
    mlp_detailed = MLPClassifier(
        hidden_layer_sizes=(100, 50), 
        max_iter=1000, 
        random_state=42, 
        early_stopping=True,
        validation_fraction=0.2,
        verbose=False
    )
    mlp_detailed.fit(X_train, y_train)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Loss Curve
    loss_curve = mlp_detailed.loss_curve_
    validation_scores = mlp_detailed.validation_scores_ if hasattr(mlp_detailed, 'validation_scores_') else None
    
    epochs = range(1, len(loss_curve) + 1)
    axes[0, 0].plot(epochs, loss_curve, label='Loss (Huấn luyện)', 
                   color='#E91E63', linewidth=2.5, marker='o', markersize=2, markevery=20)
    
    if validation_scores:
        # Convert validation scores to loss-like metric for consistency
        axes[0, 0].plot(range(1, len(validation_scores) + 1), 
                       [1 - score for score in validation_scores], 
                       label='Loss (Validation)', 
                       color='#9C27B0', linewidth=2.5, marker='s', markersize=2, markevery=20)
    
    axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Hình 4.4a: Đường cong Loss\nNeural Network (MLP)', 
                        fontsize=12, fontweight='bold', pad=15)
    axes[0, 0].legend(fontsize=10, loc='upper right')
    axes[0, 0].grid(alpha=0.3, linestyle='--')
    
    # Subplot 2: Accuracy over epochs
    train_scores_nn = []
    test_scores_nn = []
    
    # Tính accuracy cho từng epoch bằng cách train lại với số epoch khác nhau
    for n_iter in range(10, min(len(loss_curve) + 1, 500), 20):
        temp_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=n_iter,
            random_state=42,
            warm_start=False,
            verbose=False
        )
        temp_model.fit(X_train, y_train)
        train_scores_nn.append(temp_model.score(X_train, y_train))
        test_scores_nn.append(temp_model.score(X_test, y_test))
    
    epoch_points = range(10, min(len(loss_curve) + 1, 500), 20)
    axes[0, 1].plot(epoch_points, train_scores_nn, label='Tập huấn luyện', 
                   color='#4CAF50', linewidth=2.5, marker='o', markersize=4)
    axes[0, 1].plot(epoch_points, test_scores_nn, label='Tập kiểm thử', 
                   color='#FF5722', linewidth=2.5, marker='s', markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Độ chính xác', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Hình 4.4b: Đường cong học tập\nNeural Network', 
                        fontsize=12, fontweight='bold', pad=15)
    axes[0, 1].legend(fontsize=10, loc='lower right')
    axes[0, 1].grid(alpha=0.3, linestyle='--')
    axes[0, 1].set_ylim([0.5, 1.0])
    
    # Subplot 3: Confusion Matrix
    y_pred_nn = mlp_detailed.predict(X_test)
    present_labels, present_label_names = get_present_labels(y_test, y_pred_nn)
    cm_nn = confusion_matrix(y_test, y_pred_nn, labels=present_labels)
    cm_nn_normalized = cm_nn.astype('float') / cm_nn.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_nn_normalized, annot=True, fmt='.2%', cmap='Purples', 
                cbar_kws={'label': 'Tỷ lệ'}, ax=axes[1, 0],
                xticklabels=present_label_names,
                yticklabels=present_label_names,
                linewidths=1, linecolor='white', square=True)
    axes[1, 0].set_xlabel('Nhãn dự đoán', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Nhãn thực tế', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Hình 4.4c: Ma trận nhầm lẫn\nNeural Network', 
                        fontsize=12, fontweight='bold', pad=15)
    
    # Subplot 4: Network Architecture Visualization
    axes[1, 1].axis('off')
    
    # Vẽ kiến trúc mạng
    layer_sizes = [X_train.shape[1]] + list(mlp_detailed.hidden_layer_sizes) + [len(np.unique(y_train))]
    n_layers = len(layer_sizes)
    
    # Tính vị trí các layer
    layer_positions = np.linspace(0.1, 0.9, n_layers)
    max_neurons = max(layer_sizes)
    
    # Vẽ từng layer
    for i, (n_neurons, x_pos) in enumerate(zip(layer_sizes, layer_positions)):
        y_positions = np.linspace(0.1, 0.9, n_neurons)
        
        # Màu sắc cho từng layer
        if i == 0:
            color = '#2196F3'  # Input layer
            label = f'Input\n({n_neurons})'
        elif i == n_layers - 1:
            color = '#4CAF50'  # Output layer
            label = f'Output\n({n_neurons})'
        else:
            color = '#FF9800'  # Hidden layers
            label = f'Hidden {i}\n({n_neurons})'
        
        # Vẽ neurons
        for y_pos in y_positions:
            circle = plt.Circle((x_pos, y_pos), 0.02, color=color, ec='black', linewidth=1.5, zorder=3)
            axes[1, 1].add_patch(circle)
        
        # Thêm label
        axes[1, 1].text(x_pos, 0.05, label, ha='center', va='top', 
                       fontsize=10, fontweight='bold', color=color)
        
        # Vẽ connections (chỉ vẽ một vài để đơn giản hóa)
        if i < n_layers - 1:
            next_y_positions = np.linspace(0.1, 0.9, layer_sizes[i + 1])
            # Vẽ một số connections ngẫu nhiên
            for y1 in y_positions[::max(1, len(y_positions)//3)]:
                for y2 in next_y_positions[::max(1, len(next_y_positions)//3)]:
                    axes[1, 1].plot([x_pos + 0.02, layer_positions[i+1] - 0.02], 
                                   [y1, y2], 'k-', alpha=0.2, linewidth=0.5, zorder=1)
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Hình 4.4d: Kiến trúc mạng Neural Network\n(Multi-Layer Perceptron)', 
                        fontsize=12, fontweight='bold', pad=15)
    
    plt.suptitle('Hình 4.4: Phân tích chi tiết thuật toán Neural Network (MLP)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(FIGURES_DIR / 'hinh_4_4_neural_network_analysis.png', dpi=300, bbox_inches='tight')
    console.print(f"[green]✓[/green] Đã lưu: [cyan]hinh_4_4_neural_network_analysis.png[/cyan]")
    plt.close()

def train_simple_models(X_train, y_train, X_test, y_test):
    """Huấn luyện các mô hình với tham số mặc định"""
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🤖 HUẤN LUYỆN CÁC MÔ HÌNH[/bold cyan]",
        border_style="cyan"
    ))
    
    # Định nghĩa các mô hình
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'SVM (RBF Kernel)': SVC(probability=True, C=10, gamma='scale', kernel='rbf', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, early_stopping=True),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Đang huấn luyện mô hình...", total=len(models))
        
        for name, model in models.items():
            progress.update(task, description=f"[cyan]Đang huấn luyện {name}...")
            
            try:
                # Huấn luyện mô hình
                model.fit(X_train, y_train)
                
                # Đánh giá với cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Tính điểm trên tập train
                train_score = model.score(X_train, y_train)
                
                # Tính điểm trên tập test
                test_score = model.score(X_test, y_test)
                
                # Dự đoán trên tập test để tính các metrics
                y_pred = model.predict(X_test)
                
                # Tính precision, recall, f1 macro average
                from sklearn.metrics import precision_score, recall_score, f1_score
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                
                trained_models[name] = {
                    'model': model,
                    'cv_score': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'train_score': train_score,
                    'test_score': test_score,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'train_test_gap': abs(train_score - test_score)
                }
            except Exception as e:
                console.print(f"[yellow]⚠️  {name} gặp lỗi: {str(e)}[/yellow]")
            
            progress.advance(task)
    
    # Sắp xếp kết quả theo độ chính xác test giảm dần
    sorted_models = dict(sorted(trained_models.items(), 
                                key=lambda x: x[1]['test_score'], 
                                reverse=True))
    
    # Hiển thị kết quả
    results_table = Table(title="📊 Kết Quả Huấn Luyện", box=box.ROUNDED)
    results_table.add_column("Hạng", justify="center", style="bold magenta")
    results_table.add_column("Mô hình", style="cyan")
    results_table.add_column("Điểm CV", justify="right", style="green")
    results_table.add_column("CV Std", justify="right", style="dim")
    results_table.add_column("Điểm Test", justify="right", style="yellow")
    
    for rank, (name, result) in enumerate(sorted_models.items(), 1):
        rank_str = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        
        results_table.add_row(
            rank_str,
            name,
            f"{result['cv_score']:.4f}",
            f"±{result['cv_std']:.4f}",
            f"{result['test_score']:.4f}"
        )
    
    console.print("\n")
    console.print(results_table)
    
    best_model_name = list(sorted_models.keys())[0]
    best_score = sorted_models[best_model_name]['test_score']
    
    console.print(f"\n[bold green]🏆 Mô hình tốt nhất:[/bold green] [cyan]{best_model_name}[/cyan]")
    console.print(f"[bold yellow]📈 Độ chính xác:[/bold yellow] [green]{best_score:.4f} ({best_score*100:.2f}%)[/green]")
    
    return trained_models

def create_comparison_table(trained_models):
    """Tạo Bảng 4.1: So sánh độ chính xác của các thuật toán"""
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]📊 BẢNG 4.1: SO SÁNH THUẬT TOÁN[/bold cyan]",
        border_style="cyan"
    ))
    
    # Sắp xếp theo độ chính xác test giảm dần
    sorted_models = dict(sorted(trained_models.items(), 
                                key=lambda x: x[1]['test_score'], 
                                reverse=True))
    
    # Tạo figure với kích thước phù hợp
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Tạo dữ liệu bảng
    table_data = []
    table_data.append(['Xếp hạng', 'Tên Thuật toán', 'Accuracy (%)', 
                       'Precision (%)', 'Recall (%)', 'F1-Score (%)', 
                       'Độ lệch\n(Train-Test)'])
    
    for rank, (name, result) in enumerate(sorted_models.items(), 1):
        table_data.append([
            str(rank),
            name,
            f"{result['test_score']*100:.2f}%",
            f"{result['precision']*100:.1f}%",
            f"{result['recall']*100:.1f}%",
            f"{result['f1_score']*100:.1f}%",
            f"{result['train_test_gap']:.3f}"
        ])
    
    # Tạo bảng
    table = ax.table(cellText=table_data, 
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Định dạng bảng
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Định dạng header (hàng đầu tiên)
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
        cell.set_edgecolor('white')
        cell.set_linewidth(2)
    
    # Định dạng các hàng dữ liệu
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            # Màu xen kẽ
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#F2F2F2')
            
            # Highlight top 3
            if i == 1:  # Hạng 1
                cell.set_facecolor('#FFD700')  # Vàng gold
                cell.set_text_props(weight='bold')
            elif i == 2:  # Hạng 2
                cell.set_facecolor('#C0C0C0')  # Bạc
                cell.set_text_props(weight='bold')
            elif i == 3:  # Hạng 3
                cell.set_facecolor('#CD7F32')  # Đồng
                cell.set_text_props(weight='bold', color='white')
            
            cell.set_edgecolor('#999999')
            cell.set_linewidth(1)
    
    # Tiêu đề bảng
    plt.title('Bảng 4.1: So sánh độ chính xác của các thuật toán trên tập kiểm thử',
              fontsize=14, fontweight='bold', pad=20, loc='center')
    
    # Thêm ghi chú
    fig.text(0.5, 0.02, 
             'Ghi chú: Các chỉ số Precision, Recall, F1-Score được tính theo macro average. Độ lệch là giá trị tuyệt đối giữa Train và Test accuracy.',
             ha='center', fontsize=9, style='italic', color='gray', wrap=True)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bang_4_1_so_sanh_thuat_toan.png', dpi=300, bbox_inches='tight', facecolor='white')
    console.print(f"[green]✓[/green] Đã lưu: [cyan]bang_4_1_so_sanh_thuat_toan.png[/cyan]")
    plt.close()
    
    # Hiển thị bảng chi tiết trong console
    detailed_table = Table(title="📊 Bảng 4.1: So sánh các thuật toán", box=box.ROUNDED)
    detailed_table.add_column("Hạng", justify="center", style="bold magenta")
    detailed_table.add_column("Thuật toán", style="cyan")
    detailed_table.add_column("Accuracy", justify="right", style="green")
    detailed_table.add_column("Precision", justify="right", style="yellow")
    detailed_table.add_column("Recall", justify="right", style="blue")
    detailed_table.add_column("F1-Score", justify="right", style="magenta")
    detailed_table.add_column("Độ lệch", justify="right", style="red")
    
    for rank, (name, result) in enumerate(sorted_models.items(), 1):
        rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        detailed_table.add_row(
            rank_emoji,
            name,
            f"{result['test_score']*100:.2f}%",
            f"{result['precision']*100:.1f}%",
            f"{result['recall']*100:.1f}%",
            f"{result['f1_score']*100:.1f}%",
            f"{result['train_test_gap']:.3f}"
        )
    
    console.print("\n")
    console.print(detailed_table)
    
    # Tạo biểu đồ cột so sánh
    fig, ax = plt.subplots(figsize=(14, 7))
    
    algorithms = list(sorted_models.keys())
    accuracies = [result['test_score']*100 for result in sorted_models.values()]
    precisions = [result['precision']*100 for result in sorted_models.values()]
    recalls = [result['recall']*100 for result in sorted_models.values()]
    f1_scores = [result['f1_score']*100 for result in sorted_models.values()]
    
    x = np.arange(len(algorithms))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', 
                   color='#4472C4', alpha=0.9, edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, precisions, width, label='Precision', 
                   color='#ED7D31', alpha=0.9, edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, recalls, width, label='Recall', 
                   color='#A5A5A5', alpha=0.9, edgecolor='black')
    bars4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', 
                   color='#FFC000', alpha=0.9, edgecolor='black')
    
    # Thêm giá trị lên cột
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Thuật toán', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tỷ lệ (%)', fontsize=12, fontweight='bold')
    ax.set_title('Biểu đồ so sánh hiệu suất các thuật toán Machine Learning', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=20, ha='right')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bieu_do_so_sanh_thuat_toan.png', dpi=300, bbox_inches='tight')
    console.print(f"[green]✓[/green] Đã lưu: [cyan]bieu_do_so_sanh_thuat_toan.png[/cyan]")
    plt.close()

def main():
    """Hàm chính"""
    ensure_project_dirs()
    
    # Header
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]DỰ ĐOÁN GIẤC MƠ - MACHINE LEARNING[/bold cyan]\n"
        "[bold yellow]Phiên bản Tiếng Việt[/bold yellow]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # 1. Chuyển đổi dữ liệu sang tiếng Việt
    df_vn = load_and_convert_data_to_vietnamese()
    
    # 2. Kiểm tra và lọc dữ liệu hợp lệ
    valid_classes = list(DREAM_LABELS.keys())
    invalid_mask = ~df_vn['loai_giac_mo'].isin(valid_classes)
    
    if invalid_mask.any():
        invalid_count = invalid_mask.sum()
        invalid_classes = df_vn.loc[invalid_mask, 'loai_giac_mo'].unique()
        console.print(f"\n[yellow]⚠️  Phát hiện {invalid_count} mẫu có nhãn không hợp lệ: {list(invalid_classes)}[/yellow]")
        console.print(f"[yellow]📌 Chỉ giữ lại các nhãn hợp lệ: {valid_classes}[/yellow]")
        df_vn = df_vn[~invalid_mask].copy()
        console.print(f"[green]✓[/green] Đã lọc dữ liệu, còn lại {len(df_vn)} mẫu hợp lệ")
    
    # 3. Phân tích thống kê
    analyze_data_statistics(df_vn)
    
    # 3.5. Vẽ biểu đồ phân bố
    plot_distribution_histograms(df_vn)
    
    # 4. Tách features và target
    X = df_vn.drop('loai_giac_mo', axis=1)
    y = df_vn['loai_giac_mo']
    
    # Thông tin dữ liệu
    console.print("\n")
    info_table = Table(title="📊 Thông Tin Dữ Liệu", box=box.ROUNDED)
    info_table.add_column("Chỉ số", style="cyan")
    info_table.add_column("Giá trị", style="green", justify="right")
    info_table.add_row("Tổng số mẫu", str(len(df_vn)))
    info_table.add_row("Số đặc trưng", str(X.shape[1]))
    info_table.add_row("Số lớp", str(len(y.unique())))
    console.print(info_table)
    
    # Phân bố nhãn
    console.print("\n")
    dist_table = Table(title="📊 Phân Bố Nhãn", box=box.ROUNDED)
    dist_table.add_column("Loại giấc mơ", style="cyan")
    dist_table.add_column("Số lượng", justify="right", style="magenta")
    dist_table.add_column("Tỷ lệ (%)", justify="right", style="green")
    
    for dream_type in sorted(y.unique()):
        count = (y == dream_type).sum()
        percentage = (count / len(y)) * 100
        dream_label = DREAM_LABELS.get(dream_type, f"Không xác định ({dream_type})")
        dist_table.add_row(
            dream_label,
            str(count),
            f"{percentage:.2f}%"
        )
    
    console.print(dist_table)
    
    # 5. Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    console.print(f"\n[green]✓[/green] Đã chia dữ liệu: Train={len(X_train)}, Test={len(X_test)}")
    
    # 5. Chuẩn hóa dữ liệu
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    console.print(f"[green]✓[/green] Đã chuẩn hóa dữ liệu với RobustScaler")
    
    # 6. Huấn luyện các mô hình
    trained_models = train_simple_models(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # 6.5. Tạo bảng so sánh
    create_comparison_table(trained_models)
    
    # 6.6. Phân tích chi tiết Gradient Boosting
    if 'Gradient Boosting' in trained_models:
        gb_model = trained_models['Gradient Boosting']['model']
        plot_gradient_boosting_analysis(gb_model, X_train_scaled, y_train, 
                                       X_test_scaled, y_test, list(X.columns))
    
    # 6.7. Phân tích chi tiết Neural Network
    if 'Neural Network (MLP)' in trained_models:
        nn_model = trained_models['Neural Network (MLP)']['model']
        plot_neural_network_analysis(nn_model, X_train_scaled, y_train, 
                                    X_test_scaled, y_test, list(X.columns))
    
    # 7. Tìm mô hình tốt nhất
    best_model_name = max(trained_models, key=lambda x: trained_models[x]['test_score'])
    best_model = trained_models[best_model_name]['model']
    best_score = trained_models[best_model_name]['test_score']
    
    # 8. Báo cáo chi tiết
    y_pred = best_model.predict(X_test_scaled)
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]📋 BÁO CÁO PHÂN LOẠI CHI TIẾT[/bold cyan]",
        border_style="cyan"
    ))
    
    present_labels, present_label_names = get_present_labels(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=present_labels,
        target_names=present_label_names,
        output_dict=True,
        zero_division=0
    )
    
    report_table = Table(box=box.ROUNDED)
    report_table.add_column("Lớp", style="cyan")
    report_table.add_column("Precision", justify="right", style="green")
    report_table.add_column("Recall", justify="right", style="yellow")
    report_table.add_column("F1-Score", justify="right", style="magenta")
    report_table.add_column("Support", justify="right", style="blue")
    
    for label_name in present_label_names:
        report_table.add_row(
            label_name,
            f"{report[label_name]['precision']:.4f}",
            f"{report[label_name]['recall']:.4f}",
            f"{report[label_name]['f1-score']:.4f}",
            str(int(report[label_name]['support']))
        )
    
    console.print(report_table)
    
    # 9. Lưu mô hình
    console.print("\n")
    with console.status("[bold cyan]Đang lưu mô hình..."):
        joblib.dump(best_model, MODELS_DIR / 'mo_hinh_tot_nhat_vn.pkl')
        joblib.dump(scaler, MODELS_DIR / 'scaler_vn.pkl')
        joblib.dump(list(X.columns), MODELS_DIR / 'ten_dac_trung_vn.pkl')
    
    console.print(Panel(
        f"[bold green]✓ Mô hình đã được lưu:[/bold green] [cyan]models/mo_hinh_tot_nhat_vn.pkl[/cyan]\n"
        f"[bold green]✓ Scaler đã được lưu:[/bold green] [cyan]models/scaler_vn.pkl[/cyan]\n"
        f"[bold green]✓ Tên đặc trưng đã được lưu:[/bold green] [cyan]models/ten_dac_trung_vn.pkl[/cyan]",
        border_style="green"
    ))
    
    # Footer
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]✅ HOÀN THÀNH![/bold green]\n"
        "[dim]Mô hình đã được huấn luyện và lưu thành công[/dim]",
        border_style="green",
        box=box.DOUBLE
    ))

if __name__ == "__main__":
    main()

def add_advanced_noise(X, y, noise_level=0.03, feature_noise=0.08):
    X_noisy = X.copy()
    y_noisy = y.copy()
    

    n_samples = len(X)
    n_noise = int(n_samples * noise_level)
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)
    
    for idx in noise_indices:
        noise = np.random.normal(0, feature_noise, size=X.shape[1])
        X_noisy.iloc[idx] = (X_noisy.iloc[idx] + noise).clip(lower=0)
        if np.random.random() < 0.1:
            current_label = int(y_noisy.iloc[idx])
            possible_labels = [0, 1, 2, 3]
            possible_labels.remove(current_label)
            y_noisy.iloc[idx] = np.random.choice(possible_labels)
            
    return X_noisy, y_noisy
