"""
Script tạo dữ liệu mẫu cho dự đoán giấc mơ
Các đặc trưng: tuổi, giờ ngủ, stress level, caffeine intake, exercise, sleep quality
Nhãn: 0 = Ác mộng, 1 = Mơ đẹp, 2 = Ngủ sâu
"""

import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import box

# Khởi tạo console
console = Console()

# Thiết lập seed để tái tạo kết quả
np.random.seed(42)

def generate_dream_data(n_samples=1000):
    """
    Tạo dữ liệu giấc mơ với các đặc trưng có ý nghĩa
    """
    data = []
    
    for _ in range(n_samples):
        # Các đặc trưng
        age = np.random.randint(18, 70)
        sleep_hours = np.random.uniform(4, 10)
        stress_level = np.random.uniform(0, 10)
        caffeine_intake = np.random.randint(0, 5)  # số cốc cà phê/ngày
        exercise_minutes = np.random.randint(0, 120)  # phút tập thể dục
        sleep_quality = np.random.uniform(0, 10)  # chất lượng giấc ngủ tự đánh giá
        screen_time = np.random.uniform(0, 8)  # giờ dùng màn hình trước khi ngủ
        
        # Logic tạo nhãn dựa trên đặc trưng
        # Ác mộng (0): stress cao, ít ngủ, nhiều caffeine
        nightmare_score = (stress_level * 0.3 + 
                          (10 - sleep_hours) * 0.2 + 
                          caffeine_intake * 0.2 +
                          screen_time * 0.15 +
                          (10 - sleep_quality) * 0.15)
        
        # Mơ đẹp (1): stress thấp, ngủ đủ, tập thể dục
        good_dream_score = ((10 - stress_level) * 0.3 + 
                           sleep_hours * 0.2 + 
                           (exercise_minutes / 120) * 10 * 0.25 +
                           sleep_quality * 0.25)
        
        # Ngủ sâu (2): ngủ đủ giấc, ít caffeine, chất lượng giấc ngủ tốt
        deep_sleep_score = (sleep_hours * 0.3 + 
                           (5 - caffeine_intake) * 0.2 +
                           sleep_quality * 0.3 +
                           (exercise_minutes / 120) * 10 * 0.2)
        
        # Thêm một chút nhiễu ngẫu nhiên
        nightmare_score += np.random.uniform(-1, 1)
        good_dream_score += np.random.uniform(-1, 1)
        deep_sleep_score += np.random.uniform(-1, 1)
        
        # Chọn nhãn dựa trên điểm cao nhất
        scores = [nightmare_score, good_dream_score, deep_sleep_score]
        dream_type = scores.index(max(scores))
        
        data.append({
            'age': age,
            'sleep_hours': round(sleep_hours, 2),
            'stress_level': round(stress_level, 2),
            'caffeine_intake': caffeine_intake,
            'exercise_minutes': exercise_minutes,
            'sleep_quality': round(sleep_quality, 2),
            'screen_time': round(screen_time, 2),
            'dream_type': dream_type
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Header
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]DREAM PREDICTION DATA GENERATOR[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # Tạo dữ liệu với progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Đang tạo dữ liệu giấc mơ...", total=1000)
        
    # Tạo dữ liệu
        data = []
        for i in range(1000):
            # Các đặc trưng
            age = np.random.randint(18, 70)
            sleep_hours = np.random.uniform(4, 10)
            stress_level = np.random.uniform(0, 10)
            caffeine_intake = np.random.randint(0, 5)
            exercise_minutes = np.random.randint(0, 120)
            sleep_quality = np.random.uniform(0, 10)
            screen_time = np.random.uniform(0, 8)
            
            # Logic tạo nhãn
            nightmare_score = (stress_level * 0.3 + 
                              (10 - sleep_hours) * 0.2 + 
                              caffeine_intake * 0.2 +
                              screen_time * 0.15 +
                              (10 - sleep_quality) * 0.15)
            
            good_dream_score = ((10 - stress_level) * 0.3 + 
                               sleep_hours * 0.2 + 
                               (exercise_minutes / 120) * 10 * 0.25 +
                               sleep_quality * 0.25)
            
            deep_sleep_score = (sleep_hours * 0.3 + 
                               (5 - caffeine_intake) * 0.2 +
                               sleep_quality * 0.3 +
                               (exercise_minutes / 120) * 10 * 0.2)
            
            nightmare_score += np.random.uniform(-1, 1)
            good_dream_score += np.random.uniform(-1, 1)
            deep_sleep_score += np.random.uniform(-1, 1)
            
            scores = [nightmare_score, good_dream_score, deep_sleep_score]
            dream_type = scores.index(max(scores))
            
            data.append({
                'age': age,
                'sleep_hours': round(sleep_hours, 2),
                'stress_level': round(stress_level, 2),
                'caffeine_intake': caffeine_intake,
                'exercise_minutes': exercise_minutes,
                'sleep_quality': round(sleep_quality, 2),
                'screen_time': round(screen_time, 2),
                'dream_type': dream_type
            })
            
            progress.update(task, advance=1)
        
        df = pd.DataFrame(data)
    
    # Lưu vào file CSV
    with console.status("[bold green]Đang lưu dữ liệu..."):
    df.to_csv('dream_data.csv', index=False)
    
    # Thống kê với table đẹp
    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]✓ Đã tạo thành công {len(df)} mẫu dữ liệu[/bold green]",
        border_style="green"
    ))
    
    # Phân bố loại giấc mơ
    dream_counts = df['dream_type'].value_counts().sort_index()
    dream_labels = {0: 'Ác mộng', 1: 'Mơ đẹp', 2: 'Ngủ sâu'}
    
    table = Table(title="📊 Phân Bố Loại Giấc Mơ", box=box.ROUNDED, show_header=True)
    table.add_column("Loại Giấc Mơ", style="cyan", no_wrap=True)
    table.add_column("Số Lượng", justify="right", style="magenta")
    table.add_column("Tỷ Lệ", justify="right", style="green")
    
    for dream_type, count in dream_counts.items():
        percentage = (count / len(df)) * 100
        table.add_row(
            dream_labels[dream_type],
            str(count),
            f"{percentage:.1f}%"
        )
    
    console.print(table)
    
    # Thống kê mô tả
    console.print("\n")
    stats_table = Table(title="📈 Thống Kê Mô Tả", box=box.ROUNDED, show_header=True)
    stats_table.add_column("Đặc Trưng", style="cyan")
    stats_table.add_column("Trung Bình", justify="right", style="yellow")
    stats_table.add_column("Min", justify="right", style="red")
    stats_table.add_column("Max", justify="right", style="green")
    stats_table.add_column("Std", justify="right", style="blue")
    
    desc = df.describe()
    for col in ['age', 'sleep_hours', 'stress_level', 'sleep_quality']:
        if col in desc.columns:
            stats_table.add_row(
                col.replace('_', ' ').title(),
                f"{desc[col]['mean']:.2f}",
                f"{desc[col]['min']:.2f}",
                f"{desc[col]['max']:.2f}",
                f"{desc[col]['std']:.2f}"
            )
    
    console.print(stats_table)
    
    # Footer
    console.print("\n")
    console.print(Panel(
        f"[bold green]✓ Dữ liệu đã được lưu vào:[/bold green] [cyan]dream_data.csv[/cyan]",
        border_style="green",
        box=box.ROUNDED
    ))
