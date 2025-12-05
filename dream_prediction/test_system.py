"""
Script test toàn diện hệ thống dự đoán giấc mơ
Kiểm tra các edge cases và tính nhất quán
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# Khởi tạo console
console = Console()

def test_data_generation():
    """Test generate_data.py"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]TEST 1: Data Generation[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        from generate_data import generate_dream_data
        
        # Test với số lượng mẫu khác nhau
        with console.status("[cyan]Đang test data generation..."):
            for n in [10, 100, 1000]:
                df = generate_dream_data(n)
                assert len(df) == n, f"Expected {n} samples, got {len(df)}"
                assert 'dream_type' in df.columns, "Missing dream_type column"
                assert set(df['dream_type'].unique()).issubset({0, 1, 2}), "Invalid dream_type values"
                assert df['dream_type'].isnull().sum() == 0, "Null values in dream_type"
        
        console.print("[green]✓ Data generation: PASSED[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Data generation: FAILED - {e}[/red]")
        return False

def test_model_loading():
    """Test model và scaler loading"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]TEST 2: Model Loading[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        if not os.path.exists('best_dream_model.pkl'):
            console.print(Panel(
                "[bold yellow]⚠ Model file not found. Run train_model.py first.[/bold yellow]",
                border_style="yellow"
            ))
            return False
        
        if not os.path.exists('scaler.pkl'):
            console.print(Panel(
                "[bold yellow]⚠ Scaler file not found. Run train_model.py first.[/bold yellow]",
                border_style="yellow"
            ))
            return False
        
        with console.status("[cyan]Đang load model..."):
            model = joblib.load('best_dream_model.pkl')
            scaler = joblib.load('scaler.pkl')
        
        assert model is not None, "Model is None"
        assert scaler is not None, "Scaler is None"
        assert len(scaler.mean_) == 7, f"Expected 7 features, got {len(scaler.mean_)}"
        
        info_table = Table(box=box.SIMPLE, show_header=False)
        info_table.add_column("Metric", style="cyan")
        info_table.add_column("Value", style="green")
        info_table.add_row("Model type", type(model).__name__)
        info_table.add_row("Scaler type", type(scaler).__name__)
        info_table.add_row("Number of features", str(len(scaler.mean_)))
        console.print(info_table)
        
        console.print("[green]✓ Model loading: PASSED[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Model loading: FAILED - {e}[/red]")
        return False

def test_prediction_edge_cases():
    """Test prediction với edge cases"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]TEST 3: Prediction Edge Cases[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        with console.status("[cyan]Đang load model..."):
            model = joblib.load('best_dream_model.pkl')
            scaler = joblib.load('scaler.pkl')
        
        # Test cases
        test_cases = [
            ("Minimum values", {
                'age': 18, 'sleep_hours': 4.0, 'stress_level': 0.0,
                'caffeine_intake': 0, 'exercise_minutes': 0,
                'sleep_quality': 0.0, 'screen_time': 0.0
            }),
            ("Maximum values", {
                'age': 70, 'sleep_hours': 10.0, 'stress_level': 10.0,
                'caffeine_intake': 5, 'exercise_minutes': 120,
                'sleep_quality': 10.0, 'screen_time': 8.0
            }),
            ("Average values", {
                'age': 40, 'sleep_hours': 7.0, 'stress_level': 5.0,
                'caffeine_intake': 2, 'exercise_minutes': 30,
                'sleep_quality': 5.0, 'screen_time': 2.0
            }),
            ("High stress case", {
                'age': 30, 'sleep_hours': 5.0, 'stress_level': 9.0,
                'caffeine_intake': 4, 'exercise_minutes': 10,
                'sleep_quality': 3.0, 'screen_time': 6.0
            }),
            ("Low stress case", {
                'age': 35, 'sleep_hours': 8.5, 'stress_level': 1.0,
                'caffeine_intake': 0, 'exercise_minutes': 90,
                'sleep_quality': 9.0, 'screen_time': 0.5
            })
        ]
        
        results_table = Table(box=box.ROUNDED, show_header=True)
        results_table.add_column("Test Case", style="cyan")
        results_table.add_column("Prediction", justify="center", style="green")
        results_table.add_column("Status", justify="center", style="green")
        
        for desc, data in test_cases:
            df = pd.DataFrame([data])
            X_scaled = scaler.transform(df)
            prediction = model.predict(X_scaled)[0]
            
            assert prediction in [0, 1, 2], f"Invalid prediction: {prediction}"
            
            # Check probabilities if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)[0]
                assert abs(sum(proba) - 1.0) < 0.01, "Probabilities don't sum to 1"
                assert all(0 <= p <= 1 for p in proba), "Invalid probability values"
            
            dream_labels = {0: 'Ác mộng', 1: 'Mơ đẹp', 2: 'Ngủ sâu'}
            results_table.add_row(desc, dream_labels[prediction], "✓")
        
        console.print(results_table)
        console.print("[green]✓ Prediction edge cases: PASSED[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Prediction edge cases: FAILED - {e}[/red]")
        import traceback
        traceback.print_exc()
        return False

def test_data_consistency():
    """Test tính nhất quán của dữ liệu"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]TEST 4: Data Consistency[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        # Kiểm tra cả dữ liệu giả định và dữ liệu thật
        data_file = None
        if os.path.exists('dream_data_real.csv'):
            data_file = 'dream_data_real.csv'
            console.print("[dim]Sử dụng dữ liệu thật: dream_data_real.csv[/dim]")
        elif os.path.exists('dream_data.csv'):
            data_file = 'dream_data.csv'
            console.print("[dim]Sử dụng dữ liệu giả định: dream_data.csv[/dim]")
        else:
            console.print(Panel(
                "[bold yellow]⚠ Không tìm thấy dữ liệu (dream_data.csv hoặc dream_data_real.csv)[/bold yellow]",
                border_style="yellow"
            ))
            return False
        
        with console.status("[cyan]Đang kiểm tra dữ liệu..."):
            df = pd.read_csv(data_file)
            
            # Kiểm tra columns
            required_cols = ['age', 'sleep_hours', 'stress_level', 'caffeine_intake',
                            'exercise_minutes', 'sleep_quality', 'screen_time', 'dream_type']
            assert all(col in df.columns for col in required_cols), "Missing required columns"
            
            # Kiểm tra ranges
            assert df['age'].between(18, 70).all(), "Age out of range"
            assert df['sleep_hours'].between(4, 10).all(), "Sleep hours out of range"
            assert df['stress_level'].between(0, 10).all(), "Stress level out of range"
            assert df['caffeine_intake'].between(0, 5).all(), "Caffeine intake out of range"
            assert df['exercise_minutes'].between(0, 120).all(), "Exercise minutes out of range"
            assert df['sleep_quality'].between(0, 10).all(), "Sleep quality out of range"
            assert df['screen_time'].between(0, 8).all(), "Screen time out of range"
            assert df['dream_type'].isin([0, 1, 2]).all(), "Invalid dream_type"
            
            # Kiểm tra missing values
            assert df.isnull().sum().sum() == 0, "Found null values"
        
        info_table = Table(box=box.ROUNDED, show_header=False)
        info_table.add_column("Check", style="cyan")
        info_table.add_column("Result", style="green")
        info_table.add_row("Data shape", str(df.shape))
        info_table.add_row("Missing values", "0")
        info_table.add_row("All ranges valid", "✓")
        info_table.add_row("Dream type distribution", str(df['dream_type'].value_counts().to_dict()))
        console.print(info_table)
        
        console.print("[green]✓ Data consistency: PASSED[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Data consistency: FAILED - {e}[/red]")
        return False

def test_batch_prediction():
    """Test batch prediction"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]TEST 5: Batch Prediction[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        if not os.path.exists('test_data.csv'):
            console.print(Panel(
                "[bold yellow]⚠ test_data.csv not found[/bold yellow]",
                border_style="yellow"
            ))
            return False
        
        with console.status("[cyan]Đang test batch prediction..."):
            model = joblib.load('best_dream_model.pkl')
            scaler = joblib.load('scaler.pkl')
            
            df = pd.read_csv('test_data.csv')
            X = df.drop('dream_type', axis=1, errors='ignore')
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            
            assert len(predictions) == len(df), "Prediction count mismatch"
            assert all(p in [0, 1, 2] for p in predictions), "Invalid predictions"
        
        dream_labels = {0: 'Ác mộng', 1: 'Mơ đẹp', 2: 'Ngủ sâu'}
        results_table = Table(box=box.ROUNDED, show_header=True)
        results_table.add_column("Sample", style="cyan", justify="right")
        results_table.add_column("Prediction", style="green")
        
        for i, pred in enumerate(predictions):
            results_table.add_row(f"#{i+1}", dream_labels[pred])
        
        console.print(results_table)
        console.print(f"[green]✓ Processed {len(df)} samples[/green]")
        console.print("[green]✓ Batch prediction: PASSED[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Batch prediction: FAILED - {e}[/red]")
        return False

def test_file_outputs():
    """Test các file output"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]TEST 6: File Outputs[/bold cyan]",
        border_style="cyan"
    ))
    
    files_to_check = [
        ('dream_data.csv', 'Data file (synthetic)', False),
        ('dream_data_real.csv', 'Data file (real)', False),
        ('best_dream_model.pkl', 'Model file', True),
        ('scaler.pkl', 'Scaler file', True),
        ('model_comparison.png', 'Comparison chart', False),
        ('confusion_matrix.png', 'Confusion matrix', False)
    ]
    
    files_table = Table(box=box.ROUNDED, show_header=True)
    files_table.add_column("File", style="cyan")
    files_table.add_column("Description", style="white")
    files_table.add_column("Status", justify="center", style="green")
    files_table.add_column("Size", justify="right", style="blue")
    
    all_exist = True
    for filename, desc, required in files_to_check:
        exists = os.path.exists(filename)
        if exists:
            size = os.path.getsize(filename)
            files_table.add_row(filename, desc, "✓", f"{size:,} bytes")
        else:
            if filename.endswith('.png') or not required:
                # PNG files và data files là optional (có thể có 1 trong 2)
                if 'dream_data' in filename:
                    files_table.add_row(filename, desc, "⚠", "Optional")
                else:
                    files_table.add_row(filename, desc, "⚠", "Optional")
            else:
                files_table.add_row(filename, desc, "✗", "Missing")
                all_exist = False
    
    # Kiểm tra ít nhất 1 data file tồn tại
    has_data = os.path.exists('dream_data.csv') or os.path.exists('dream_data_real.csv')
    if not has_data:
        all_exist = False
    
    console.print(files_table)
    
    if all_exist:
        console.print("[green]✓ File outputs: PASSED[/green]")
    else:
        console.print("[yellow]⚠ File outputs: Some files missing[/yellow]")
    
    return all_exist

def main():
    """Chạy tất cả tests"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]HỆ THỐNG TEST TOÀN DIỆN - DREAM PREDICTION[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    results = []
    
    results.append(("Data Generation", test_data_generation()))
    results.append(("Data Consistency", test_data_consistency()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("Prediction Edge Cases", test_prediction_edge_cases()))
    results.append(("Batch Prediction", test_batch_prediction()))
    results.append(("File Outputs", test_file_outputs()))
    
    # Tổng hợp
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]📊 TỔNG HỢP KẾT QUẢ[/bold cyan]",
        border_style="cyan"
    ))
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    summary_table = Table(box=box.ROUNDED, show_header=True)
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Status", justify="center")
    
    for test_name, result in results:
        status = "[green]✓ PASSED[/green]" if result else "[red]✗ FAILED[/red]"
        summary_table.add_row(test_name, status)
    
    console.print(summary_table)
    
    percentage = passed * 100 // total if total > 0 else 0
    console.print(f"\n[bold]📈 Kết quả: {passed}/{total} tests passed ({percentage}%)[/bold]")
    
    if passed == total:
        console.print("\n")
        console.print(Panel.fit(
            "[bold green]🎉 TẤT CẢ TESTS ĐỀU PASSED![/bold green]",
            border_style="green",
            box=box.DOUBLE
        ))
        return 0
    else:
        console.print("\n")
        console.print(Panel(
            f"[bold yellow]⚠ {total - passed} test(s) failed[/bold yellow]",
            border_style="yellow"
        ))
        return 1

if __name__ == "__main__":
    sys.exit(main())

