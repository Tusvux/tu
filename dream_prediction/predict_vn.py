"""
Script dự đoán giấc mơ - PHIÊN BẢN TIẾNG VIỆT
Sử dụng mô hình đã huấn luyện để dự đoán loại giấc mơ
"""

import pandas as pd
import numpy as np
import joblib
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.prompt import Prompt, FloatPrompt, IntPrompt

console = Console()

DREAM_LABELS = {0: 'Ác mộng', 1: 'Mơ đẹp', 2: 'Ngủ sâu'}
DREAM_DESCRIPTIONS = {
    0: '😱 Bạn có nguy cơ gặp ác mộng. Hãy giảm stress và cải thiện chất lượng ngủ!',
    1: '😊 Bạn sẽ có giấc ngủ ngon với những giấc mơ đẹp!',
    2: '😴 Bạn sẽ ngủ rất sâu và nghỉ ngơi hiệu quả!'
}

FEATURE_INFO = {
    'tuoi': {'name': 'Tuổi', 'min': 18, 'max': 80, 'unit': 'tuổi'},
    'gio_ngu': {'name': 'Giờ ngủ', 'min': 4, 'max': 12, 'unit': 'giờ'},
    'muc_stress': {'name': 'Mức stress', 'min': 0, 'max': 10, 'unit': 'điểm'},
    'caffeine': {'name': 'Lượng caffeine', 'min': 0, 'max': 5, 'unit': 'tách'},
    'phut_tap_luyen': {'name': 'Phút tập luyện', 'min': 0, 'max': 180, 'unit': 'phút'},
    'chat_luong_ngu': {'name': 'Chất lượng ngủ', 'min': 0, 'max': 10, 'unit': 'điểm'},
    'thoi_gian_man_hinh': {'name': 'Thời gian màn hình', 'min': 0, 'max': 12, 'unit': 'giờ'}
}

def load_model():
    """Tải mô hình đã huấn luyện"""
    try:
        model = joblib.load('mo_hinh_tot_nhat_vn.pkl')
        scaler = joblib.load('scaler_vn.pkl')
        console.print("[green]✓[/green] Đã tải mô hình thành công!")
        return model, scaler
    except FileNotFoundError:
        console.print("[red]✗[/red] Không tìm thấy file mô hình!")
        console.print("[yellow]💡 Hãy chạy 'python train_model_vn.py' trước![/yellow]")
        sys.exit(1)

def get_user_input_interactive():
    """Nhập dữ liệu từ người dùng (chế độ tương tác)"""
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]📝 NHẬP THÔNG TIN CỦA BẠN[/bold cyan]",
        border_style="cyan"
    ))
    
    data = {}
    
    for feature, info in FEATURE_INFO.items():
        while True:
            try:
                value = FloatPrompt.ask(
                    f"[cyan]{info['name']}[/cyan] ({info['min']}-{info['max']} {info['unit']})"
                )
                
                if info['min'] <= value <= info['max']:
                    data[feature] = value
                    break
                else:
                    console.print(f"[red]✗[/red] Giá trị phải nằm trong khoảng {info['min']}-{info['max']}")
            except:
                console.print("[red]✗[/red] Vui lòng nhập số hợp lệ!")
    
    return pd.DataFrame([data])

def get_sample_data(sample_type='random'):
    """Lấy dữ liệu mẫu để test"""
    
    samples = {
        'stress_cao': {
            'tuoi': 35,
            'gio_ngu': 5.5,
            'muc_stress': 9.0,
            'caffeine': 4,
            'phut_tap_luyen': 10,
            'chat_luong_ngu': 8.5,
            'thoi_gian_man_hinh': 6.0
        },
        'ngu_ngon': {
            'tuoi': 30,
            'gio_ngu': 8.0,
            'muc_stress': 2.0,
            'caffeine': 1,
            'phut_tap_luyen': 60,
            'chat_luong_ngu': 2.0,
            'thoi_gian_man_hinh': 2.0
        },
        'ngu_sau': {
            'tuoi': 28,
            'gio_ngu': 9.0,
            'muc_stress': 3.0,
            'caffeine': 0,
            'phut_tap_luyen': 90,
            'chat_luong_ngu': 1.5,
            'thoi_gian_man_hinh': 1.0
        },
        'random': None
    }
    
    if sample_type == 'random':
        data = {
            'tuoi': np.random.randint(20, 65),
            'gio_ngu': np.random.uniform(5, 9),
            'muc_stress': np.random.uniform(1, 8),
            'caffeine': np.random.randint(0, 4),
            'phut_tap_luyen': np.random.randint(10, 100),
            'chat_luong_ngu': np.random.uniform(2, 8),
            'thoi_gian_man_hinh': np.random.uniform(1, 6)
        }
    else:
        data = samples.get(sample_type, samples['random'])
    
    return pd.DataFrame([data])

def predict(model, scaler, data):
    """Thực hiện dự đoán"""
    
    # Chuẩn hóa dữ liệu
    data_scaled = scaler.transform(data)
    
    # Dự đoán
    prediction = model.predict(data_scaled)[0]
    
    # Dự đoán xác suất (nếu có)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(data_scaled)[0]
    else:
        probabilities = None
    
    return prediction, probabilities

def display_results(data, prediction, probabilities):
    """Hiển thị kết quả dự đoán"""
    
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]🎯 KẾT QUẢ DỰ ĐOÁN[/bold green]",
        border_style="green",
        box=box.DOUBLE
    ))
    
    # Hiển thị dữ liệu đầu vào
    input_table = Table(title="📊 Dữ Liệu Đầu Vào", box=box.ROUNDED)
    input_table.add_column("Đặc trưng", style="cyan")
    input_table.add_column("Giá trị", justify="right", style="yellow")
    
    for feature, value in data.iloc[0].items():
        feature_name = FEATURE_INFO.get(feature, {}).get('name', feature)
        unit = FEATURE_INFO.get(feature, {}).get('unit', '')
        input_table.add_row(feature_name, f"{value:.2f} {unit}")
    
    console.print(input_table)
    
    # Hiển thị kết quả
    console.print("\n")
    result_panel = Panel.fit(
        f"[bold cyan]Loại giấc mơ:[/bold cyan] [bold yellow]{DREAM_LABELS[prediction]}[/bold yellow]\n\n"
        f"{DREAM_DESCRIPTIONS[prediction]}",
        border_style="green",
        box=box.DOUBLE
    )
    console.print(result_panel)
    
    # Hiển thị xác suất
    if probabilities is not None:
        console.print("\n")
        prob_table = Table(title="📊 Xác Suất Dự Đoán", box=box.ROUNDED)
        prob_table.add_column("Loại giấc mơ", style="cyan")
        prob_table.add_column("Xác suất", justify="right", style="green")
        prob_table.add_column("Biểu đồ", style="yellow")
        
        for label_id, label_name in DREAM_LABELS.items():
            prob = probabilities[label_id]
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            
            style = "bold green" if label_id == prediction else "dim"
            prob_table.add_row(
                f"[{style}]{label_name}[/{style}]",
                f"[{style}]{prob*100:.2f}%[/{style}]",
                f"[{style}]{bar}[/{style}]"
            )
        
        console.print(prob_table)
    
    # Đưa ra lời khuyên
    console.print("\n")
    advice = get_advice(data.iloc[0], prediction)
    console.print(Panel(
        f"[bold cyan]💡 Lời Khuyên:[/bold cyan]\n\n{advice}",
        border_style="cyan",
        box=box.ROUNDED
    ))

def get_advice(data, prediction):
    """Đưa ra lời khuyên dựa trên dữ liệu và dự đoán"""
    
    advice_list = []
    
    # Kiểm tra stress
    if data['muc_stress'] > 7:
        advice_list.append("• Mức stress của bạn cao. Hãy thử thiền, yoga hoặc các kỹ thuật thư giãn.")
    
    # Kiểm tra giờ ngủ
    if data['gio_ngu'] < 6:
        advice_list.append("• Bạn ngủ quá ít. Nên ngủ 7-9 giờ mỗi đêm.")
    elif data['gio_ngu'] > 10:
        advice_list.append("• Bạn ngủ quá nhiều. Có thể ảnh hưởng đến chất lượng ngủ.")
    
    # Kiểm tra caffeine
    if data['caffeine'] > 2:
        advice_list.append("• Giảm lượng caffeine, đặc biệt sau 3 giờ chiều.")
    
    # Kiểm tra tập luyện
    if data['phut_tap_luyen'] < 30:
        advice_list.append("• Tăng thời gian tập luyện lên ít nhất 30 phút mỗi ngày.")
    
    # Kiểm tra thời gian màn hình
    if data['thoi_gian_man_hinh'] > 5:
        advice_list.append("• Giảm thời gian sử dụng màn hình trước khi ngủ ít nhất 1-2 giờ.")
    
    # Kiểm tra chất lượng ngủ
    if data['chat_luong_ngu'] > 7:
        advice_list.append("• Chất lượng ngủ kém. Hãy cải thiện môi trường ngủ (tối, yên tĩnh, mát mẻ).")
    
    if not advice_list:
        advice_list.append("• Thói quen ngủ của bạn khá tốt! Hãy duy trì!")
    
    return "\n".join(advice_list)

def main():
    """Hàm chính"""
    
    # Header
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]DỰ ĐOÁN GIẤC MƠ - TIẾNG VIỆT[/bold cyan]\n"
        "[dim]Dự đoán loại giấc mơ dựa trên thói quen sinh hoạt[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # Tải mô hình
    model, scaler = load_model()
    
    # Menu chọn chế độ
    console.print("\n[bold]Chọn chế độ:[/bold]")
    console.print("1. Nhập dữ liệu thủ công")
    console.print("2. Sử dụng dữ liệu mẫu (stress cao)")
    console.print("3. Sử dụng dữ liệu mẫu (ngủ ngon)")
    console.print("4. Sử dụng dữ liệu mẫu (ngủ sâu)")
    console.print("5. Sử dụng dữ liệu ngẫu nhiên")
    
    choice = IntPrompt.ask("\n[cyan]Lựa chọn của bạn[/cyan]", default=1)
    
    # Lấy dữ liệu
    if choice == 1:
        data = get_user_input_interactive()
    elif choice == 2:
        data = get_sample_data('stress_cao')
        console.print("[yellow]Sử dụng dữ liệu mẫu: Stress cao[/yellow]")
    elif choice == 3:
        data = get_sample_data('ngu_ngon')
        console.print("[yellow]Sử dụng dữ liệu mẫu: Ngủ ngon[/yellow]")
    elif choice == 4:
        data = get_sample_data('ngu_sau')
        console.print("[yellow]Sử dụng dữ liệu mẫu: Ngủ sâu[/yellow]")
    else:
        data = get_sample_data('random')
        console.print("[yellow]Sử dụng dữ liệu ngẫu nhiên[/yellow]")
    
    # Thực hiện dự đoán
    with console.status("[bold cyan]Đang dự đoán..."):
        prediction, probabilities = predict(model, scaler, data)
    
    # Hiển thị kết quả
    display_results(data, prediction, probabilities)
    
    # Hỏi tiếp tục
    console.print("\n")
    if Prompt.ask("[cyan]Bạn có muốn dự đoán tiếp?[/cyan] (y/n)", default="n").lower() == 'y':
        console.clear()
        main()
    else:
        console.print("\n[green]Cảm ơn bạn đã sử dụng! Chúc ngủ ngon! 😴[/green]\n")

if __name__ == "__main__":
    main()
