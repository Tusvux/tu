"""
Script tạo dữ liệu mẫu cho dự đoán giấc mơ - 4 LOẠI
Các đặc trưng: tuổi, giờ ngủ, stress level, caffeine intake, exercise, sleep quality, screen time
Nhãn: 0 = Ác mộng, 1 = Mơ đẹp, 2 = Ngủ sâu, 3 = Không mơ
"""

import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import box

console = Console()
np.random.seed(42)

def generate_dream_data_4_classes(n_samples=5000):
    """
    Tạo dữ liệu giấc mơ với 4 loại bao gồm "Không mơ"
    
    Loại 0 - Ác mộng: Stress cao, ngủ ít, caffeine nhiều, chất lượng ngủ kém
    Loại 1 - Mơ đẹp: Stress thấp, ngủ đủ, tập thể dục, chất lượng ngủ tốt
    Loại 2 - Ngủ sâu: Ngủ nhiều, không caffeine, chất lượng ngủ rất tốt
    Loại 3 - Không mơ: Ngủ vừa phải, caffeine vừa, stress trung bình, nhưng có các yếu tố đặc biệt
    """
    data = []
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🔧 TẠO DỮ LIỆU 4 LOẠI GIẤC MƠ[/bold cyan]\n"
        "[dim]0: Ác mộng | 1: Mơ đẹp | 2: Ngủ sâu | 3: Không mơ[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Đang tạo dữ liệu...", total=n_samples)
        
        for _ in range(n_samples):
            # Sinh các đặc trưng cơ bản
            age = np.random.randint(18, 70)
            sleep_hours = np.random.uniform(4, 10)
            stress_level = np.random.uniform(0, 10)
            caffeine_intake = np.random.randint(0, 5)
            exercise_minutes = np.random.randint(0, 120)
            sleep_quality = np.random.uniform(0, 10)
            screen_time = np.random.uniform(0, 8)
            
            # Tính điểm cho mỗi loại giấc mơ
            
            # Ác mộng (0): Stress cao, ngủ kém, caffeine nhiều, màn hình nhiều
            nightmare_score = (
                stress_level * 0.35 + 
                (10 - sleep_hours) * 0.20 + 
                caffeine_intake * 0.20 +
                screen_time * 0.15 +
                (10 - sleep_quality) * 0.10
            )
            
            # Mơ đẹp (1): Stress thấp, ngủ đủ, tập luyện tốt, chất lượng ngủ tốt
            good_dream_score = (
                (10 - stress_level) * 0.30 + 
                sleep_hours * 0.20 + 
                (exercise_minutes / 120) * 10 * 0.25 +
                sleep_quality * 0.25
            )
            
            # Ngủ sâu (2): Ngủ nhiều giờ, ít caffeine, chất lượng ngủ cao
            deep_sleep_score = (
                sleep_hours * 0.35 + 
                (5 - caffeine_intake) * 0.20 +
                sleep_quality * 0.35 +
                (exercise_minutes / 120) * 10 * 0.10
            )
            
            # Không mơ (3): Đặc trưng riêng - ngủ vừa phải, stress trung bình, 
            # caffeine vừa, ít vận động, màn hình ít, chất lượng ngủ trung bình
            # Điểm càng cao khi các giá trị càng gần trung bình và ít vận động
            no_dream_score = (
                (10 - abs(sleep_hours - 7)) * 0.25 +  # Gần 7 giờ thì điểm cao
                (10 - abs(stress_level - 5)) * 0.20 +  # Stress trung bình
                (5 - abs(caffeine_intake - 2)) * 0.15 +  # Caffeine vừa phải (2 cups)
                (120 - exercise_minutes) / 120 * 10 * 0.20 +  # Ít vận động thì điểm cao
                (8 - screen_time) / 8 * 10 * 0.10 +  # Ít màn hình thì điểm cao
                abs(sleep_quality - 5) * (-0.10)  # Chất lượng ngủ trung bình
            )
            
            # Thêm nhiễu ngẫu nhiên - điều chỉnh để cân bằng phân bố
            noise = np.random.uniform(-2.0, 2.0)
            nightmare_score += noise * 1.2
            good_dream_score += noise * 1.1
            deep_sleep_score += noise * 1.0
            no_dream_score += noise * 0.95
            
            # Chọn loại giấc mơ có điểm cao nhất
            scores = [nightmare_score, good_dream_score, deep_sleep_score, no_dream_score]
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
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]DREAM PREDICTION DATA GENERATOR - 4 CLASSES[/bold cyan]\n"
        "[bold yellow]Phiên bản mở rộng với loại 'Không mơ'[/bold yellow]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # Tạo dữ liệu
    df = generate_dream_data_4_classes(n_samples=6000)
    
    # Lưu dữ liệu
    with console.status("[bold green]Đang lưu dữ liệu..."):
        df.to_csv('dream_data.csv', index=False)
        console.print(f"[green]✓[/green] Đã lưu: [cyan]dream_data.csv[/cyan]")
    
    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]✓ Đã tạo thành công {len(df)} mẫu dữ liệu[/bold green]",
        border_style="green"
    ))
    
    # Phân bố loại giấc mơ
    dream_counts = df['dream_type'].value_counts().sort_index()
    dream_labels = {0: 'Ác mộng 😱', 1: 'Mơ đẹp 😊', 2: 'Ngủ sâu 😴', 3: 'Không mơ 🌙'}
    
    table = Table(title="📊 Phân Bố Loại Giấc Mơ (4 Lớp)", box=box.ROUNDED, show_header=True)
    table.add_column("Loại Giấc Mơ", style="cyan", no_wrap=True)
    table.add_column("Số Lượng", justify="right", style="magenta")
    table.add_column("Tỷ Lệ", justify="right", style="green")
    
    for dream_type in sorted(dream_counts.index):
        count = dream_counts[dream_type]
        percentage = (count / len(df)) * 100
        table.add_row(
            dream_labels[dream_type],
            str(count),
            f"{percentage:.1f}%"
        )
    
    console.print(table)
    
    # Thống kê mô tả theo từng loại
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]📈 THỐNG KÊ ĐẶC TRƯNG THEO TỪNG LOẠI GIẤC MƠ[/bold cyan]",
        border_style="cyan"
    ))
    
    for dream_type in sorted(df['dream_type'].unique()):
        subset = df[df['dream_type'] == dream_type]
        
        stats_table = Table(
            title=f"🎯 {dream_labels[dream_type]} (n={len(subset)})", 
            box=box.SIMPLE, 
            show_header=True
        )
        stats_table.add_column("Đặc Trưng", style="cyan")
        stats_table.add_column("Trung Bình", justify="right", style="yellow")
        stats_table.add_column("Min", justify="right", style="red")
        stats_table.add_column("Max", justify="right", style="green")
        
        for col in ['sleep_hours', 'stress_level', 'caffeine_intake', 'exercise_minutes', 'sleep_quality', 'screen_time']:
            stats_table.add_row(
                col.replace('_', ' ').title(),
                f"{subset[col].mean():.2f}",
                f"{subset[col].min():.2f}",
                f"{subset[col].max():.2f}"
            )
        
        console.print(stats_table)
        console.print()
    
    # Lưu file tiếng Việt
    console.print(Panel.fit(
        "[bold cyan]🔄 TẠO PHIÊN BẢN TIẾNG VIỆT[/bold cyan]",
        border_style="cyan"
    ))
    
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
    
    df_vn = df.rename(columns=FEATURE_MAPPING)
    df_vn.to_csv('dream_data_vn.csv', index=False, encoding='utf-8-sig')
    console.print(f"[green]✓[/green] Đã lưu phiên bản tiếng Việt: [cyan]dream_data_vn.csv[/cyan]")
    
    # Footer
    console.print("\n")
    console.print(Panel(
        f"[bold green]✅ HOÀN TẤT![/bold green]\n\n"
        f"[cyan]📁 Đã tạo 2 file:[/cyan]\n"
        f"  • dream_data.csv (tiếng Anh)\n"
        f"  • dream_data_vn.csv (tiếng Việt)\n\n"
        f"[yellow]📊 Tổng số mẫu:[/yellow] {len(df)}\n"
        f"[yellow]🎯 Số loại giấc mơ:[/yellow] 4\n"
        f"[yellow]🔢 Số đặc trưng:[/yellow] {len(df.columns) - 1}",
        border_style="green",
        box=box.DOUBLE
    ))
