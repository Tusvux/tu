"""
Script dự đoán giấc mơ cho người dùng mới
Sử dụng mô hình đã được huấn luyện
"""

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.prompt import Prompt, Confirm

# Khởi tạo console
console = Console()

def load_model():
    """Tải mô hình và scaler đã lưu"""
    try:
        with console.status("[bold cyan]Đang tải mô hình..."):
        model = joblib.load('best_dream_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        console.print(Panel(
            "[bold red]❌ Lỗi: Không tìm thấy mô hình[/bold red]\n"
            "[dim]Vui lòng chạy 'python train_model.py' trước.[/dim]",
            border_style="red"
        ))
        return None, None

def get_user_input():
    """Nhập thông tin từ người dùng"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]NHẬP THÔNG TIN CỦA BẠN[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    try:
        age = int(Prompt.ask("\n[cyan]👤 Tuổi của bạn[/cyan]", default="25"))
        sleep_hours = float(Prompt.ask("[cyan]😴 Số giờ ngủ trung bình (4-10)[/cyan]", default="7.5"))
        stress_level = float(Prompt.ask("[cyan]😰 Mức độ stress (0-10, 0=không stress, 10=rất stress)[/cyan]", default="5.0"))
        caffeine_intake = int(Prompt.ask("[cyan]☕ Số cốc cà phê/ngày[/cyan]", default="2"))
        exercise_minutes = int(Prompt.ask("[cyan]🏃 Số phút tập thể dục/ngày[/cyan]", default="30"))
        sleep_quality = float(Prompt.ask("[cyan]💤 Chất lượng giấc ngủ tự đánh giá (0-10, 0=rất tệ, 10=rất tốt)[/cyan]", default="7.0"))
        screen_time = float(Prompt.ask("[cyan]📱 Số giờ dùng màn hình trước khi ngủ (0-8)[/cyan]", default="2.0"))
        
        return {
            'age': age,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'caffeine_intake': caffeine_intake,
            'exercise_minutes': exercise_minutes,
            'sleep_quality': sleep_quality,
            'screen_time': screen_time
        }
    except ValueError:
        console.print(Panel(
            "[bold red]❌ Lỗi: Vui lòng nhập đúng định dạng số![/bold red]",
            border_style="red"
        ))
        return None

def predict_dream(model, scaler, user_data):
    """Dự đoán loại giấc mơ"""
    
    # Chuyển đổi dữ liệu thành DataFrame
    df = pd.DataFrame([user_data])
    
    # Chuẩn hóa
    X_scaled = scaler.transform(df)
    
    # Dự đoán
    prediction = model.predict(X_scaled)[0]
    
    # Lấy xác suất nếu mô hình hỗ trợ
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)[0]
    else:
        probabilities = None
    
    return prediction, probabilities

def display_prediction(prediction, probabilities):
    """Hiển thị kết quả dự đoán"""
    
    dream_labels = {
        0: 'Ác mộng',
        1: 'Mơ đẹp',
        2: 'Ngủ sâu'
    }
    
    dream_icons = {
        0: '😱',
        1: '😊',
        2: '😴'
    }
    
    dream_colors = {
        0: 'red',
        1: 'green',
        2: 'blue'
    }
    
    dream_descriptions = {
        0: "Bạn có khả năng gặp ác mộng. Hãy thử giảm stress và tránh caffeine trước khi ngủ.",
        1: "Bạn có khả năng có những giấc mơ đẹp. Tiếp tục duy trì lối sống lành mạnh!",
        2: "Bạn có khả năng ngủ sâu và ngon giấc. Tuyệt vời!"
    }
    
    console.print("\n")
    console.print(Panel.fit(
        f"[bold {dream_colors[prediction]}]🔮 KẾT QUẢ DỰ ĐOÁN 🔮[/bold {dream_colors[prediction]}]",
        border_style=dream_colors[prediction],
        box=box.DOUBLE
    ))
    
    # Hiển thị dự đoán chính
    console.print("\n")
    console.print(Panel(
        f"[bold {dream_colors[prediction]}]🎯 Dự đoán: {dream_labels[prediction]} {dream_icons[prediction]}[/bold {dream_colors[prediction]}]\n"
        f"[dim]{dream_descriptions[prediction]}[/dim]",
        border_style=dream_colors[prediction],
        box=box.ROUNDED
    ))
    
    # Xác suất chi tiết
    if probabilities is not None:
        prob_table = Table(title="📊 Xác Suất Chi Tiết", box=box.ROUNDED, show_header=True)
        prob_table.add_column("Loại Giấc Mơ", style="cyan")
        prob_table.add_column("Xác Suất", justify="right", style="magenta")
        prob_table.add_column("Thanh", style="green")
        
        for i, prob in enumerate(probabilities):
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            prob_table.add_row(
                f"{dream_labels[i]} {dream_icons[i]}",
                f"{prob*100:.2f}%",
                f"[{dream_colors[i]}]{bar}[/{dream_colors[i]}]"
            )
    
        console.print("\n")
        console.print(prob_table)

def batch_predict_from_file(model, scaler, filepath):
    """Dự đoán hàng loạt từ file CSV"""
    
    try:
        with console.status(f"[bold cyan]Đang đọc file {filepath}..."):
        df = pd.read_csv(filepath)
        
        # Kiểm tra các cột cần thiết
        required_columns = ['age', 'sleep_hours', 'stress_level', 'caffeine_intake', 
                          'exercise_minutes', 'sleep_quality', 'screen_time']
        
        if not all(col in df.columns for col in required_columns):
            console.print(Panel(
                f"[bold red]❌ Lỗi: File CSV thiếu các cột cần thiết![/bold red]\n"
                f"[dim]Các cột cần có: {', '.join(required_columns)}[/dim]",
                border_style="red"
            ))
            return
        
        # Chuẩn hóa và dự đoán
        with console.status("[bold cyan]Đang dự đoán..."):
        X = df[required_columns]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Thêm kết quả vào DataFrame
        dream_labels = {0: 'Ác mộng', 1: 'Mơ đẹp', 2: 'Ngủ sâu'}
        df['prediction'] = [dream_labels[p] for p in predictions]
        
        # Lưu kết quả
        output_file = 'predictions_result.csv'
        with console.status("[bold cyan]Đang lưu kết quả..."):
        df.to_csv(output_file, index=False)
        
        console.print(Panel(
            f"[bold green]✓ Đã dự đoán cho {len(df)} người[/bold green]\n"
            f"[bold green]✓ Kết quả đã được lưu vào:[/bold green] [cyan]{output_file}[/cyan]",
            border_style="green"
        ))
        
        # Thống kê
        stats_table = Table(title="📊 Thống Kê Kết Quả", box=box.ROUNDED, show_header=True)
        stats_table.add_column("Loại Giấc Mơ", style="cyan")
        stats_table.add_column("Số Lượng", justify="right", style="magenta")
        stats_table.add_column("Tỷ Lệ", justify="right", style="green")
        
        for dream_type, count in pd.Series(predictions).value_counts().sort_index().items():
            percentage = (count / len(predictions)) * 100
            stats_table.add_row(
                dream_labels[dream_type],
                str(count),
                f"{percentage:.1f}%"
            )
        
        console.print("\n")
        console.print(stats_table)
        
    except FileNotFoundError:
        console.print(Panel(
            f"[bold red]❌ Lỗi: Không tìm thấy file '{filepath}'[/bold red]",
            border_style="red"
        ))
    except Exception as e:
        console.print(Panel(
            f"[bold red]❌ Lỗi: {str(e)}[/bold red]",
            border_style="red"
        ))

def provide_recommendations(user_data, prediction):
    """Đưa ra khuyến nghị dựa trên dự đoán"""
    
    recommendations = []
    
    if prediction == 0:  # Ác mộng
        recommendations.append("Giảm mức độ stress bằng cách thiền định hoặc yoga")
        if user_data['caffeine_intake'] > 2:
            recommendations.append("Giảm lượng caffeine, đặc biệt sau 2 giờ chiều")
        if user_data['sleep_hours'] < 7:
            recommendations.append("Tăng thời gian ngủ lên ít nhất 7-8 giờ")
        if user_data['screen_time'] > 2:
            recommendations.append("Giảm thời gian dùng màn hình trước khi ngủ")
        if user_data['exercise_minutes'] < 30:
            recommendations.append("Tăng cường tập thể dục, ít nhất 30 phút/ngày")
    
    elif prediction == 1:  # Mơ đẹp
        recommendations.append("Tiếp tục duy trì lối sống lành mạnh hiện tại")
        recommendations.append("Giữ thói quen ngủ đều đặn")
        recommendations.append("Tiếp tục quản lý stress tốt")
    
    else:  # Ngủ sâu
        recommendations.append("Bạn đang có thói quen ngủ rất tốt!")
        recommendations.append("Duy trì lịch trình ngủ đều đặn")
        recommendations.append("Chia sẻ bí quyết với người khác")
    
    if recommendations:
        rec_text = "\n".join([f"  • {rec}" for rec in recommendations])
        console.print("\n")
        console.print(Panel(
            f"[bold yellow]💡 KHUYẾN NGHỊ:[/bold yellow]\n\n{rec_text}",
            border_style="yellow",
            box=box.ROUNDED
        ))

def main():
    """Hàm chính"""
    
    # Header
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]DREAM PREDICTION SYSTEM[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # Tải mô hình
    model, scaler = load_model()
    
    if model is None or scaler is None:
        return
    
    console.print(Panel(
        "[bold green]✓ Mô hình đã được tải thành công![/bold green]",
        border_style="green"
    ))
    
    while True:
        console.print("\n")
        menu_table = Table(box=box.ROUNDED, show_header=False)
        menu_table.add_column("Option", style="cyan", width=5)
        menu_table.add_column("Description", style="white")
        menu_table.add_row("1", "Dự đoán cho 1 người (nhập thông tin)")
        menu_table.add_row("2", "Dự đoán hàng loạt từ file CSV")
        menu_table.add_row("3", "Thoát")
        
        console.print(menu_table)
        
        choice = Prompt.ask("\n[cyan]Lựa chọn của bạn[/cyan]", choices=["1", "2", "3"], default="1")
        
        if choice == '1':
            # Dự đoán đơn lẻ
            user_data = get_user_input()
            
            if user_data is not None:
                with console.status("[bold cyan]Đang dự đoán..."):
                prediction, probabilities = predict_dream(model, scaler, user_data)
                display_prediction(prediction, probabilities)
                provide_recommendations(user_data, prediction)
        
        elif choice == '2':
            # Dự đoán hàng loạt
            filepath = Prompt.ask("\n[cyan]Nhập đường dẫn file CSV[/cyan]")
            batch_predict_from_file(model, scaler, filepath)
        
        elif choice == '3':
            console.print("\n")
            console.print(Panel.fit(
                "[bold green]👋 Tạm biệt! Chúc bạn có giấc ngủ ngon! 🌙[/bold green]",
                border_style="green"
            ))
            break

if __name__ == "__main__":
    main()
