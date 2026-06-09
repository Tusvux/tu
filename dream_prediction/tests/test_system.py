"""
Script test toàn diện hệ thống dự đoán giấc mơ
Kiểm tra các edge cases và tính nhất quán
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import DATA_DIR, MODELS_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Khởi tạo console
console = Console()

def test_data_generation():
    """Test generate_data_4_classes.py"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]TEST 1: Data Generation[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        from generate_data_4_classes import generate_dream_data_4_classes
        
        # Test với số lượng mẫu khác nhau
        with console.status("[cyan]Đang test data generation..."):
            for n in [10, 100, 1000]:
                df = generate_dream_data_4_classes(n)
                assert len(df) == n, f"Expected {n} samples, got {len(df)}"
                assert 'dream_type' in df.columns, "Missing dream_type column"
                assert set(df['dream_type'].unique()).issubset({0, 1, 2, 3}), "Invalid dream_type values"
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
        if not (MODELS_DIR / 'mo_hinh_tot_nhat_vn.pkl').exists():
            console.print(Panel(
                "[bold yellow]⚠ Model file not found. Run train_model_vn.py first.[/bold yellow]",
                border_style="yellow"
            ))
            return False
        
        if not (MODELS_DIR / 'scaler_vn.pkl').exists():
            console.print(Panel(
                "[bold yellow]⚠ Scaler file not found. Run train_model_vn.py first.[/bold yellow]",
                border_style="yellow"
            ))
            return False
        
        with console.status("[cyan]Đang load model..."):
            model = joblib.load(MODELS_DIR / 'mo_hinh_tot_nhat_vn.pkl')
            scaler = joblib.load(MODELS_DIR / 'scaler_vn.pkl')
        
        assert model is not None, "Model is None"
        assert scaler is not None, "Scaler is None"
        n_features = getattr(scaler, 'n_features_in_', None)
        if n_features is None:
            n_features = len(getattr(scaler, 'center_', getattr(scaler, 'mean_', [])))
        assert n_features == 7, f"Expected 7 features, got {n_features}"
        
        info_table = Table(box=box.SIMPLE, show_header=False)
        info_table.add_column("Metric", style="cyan")
        info_table.add_column("Value", style="green")
        info_table.add_row("Model type", type(model).__name__)
        info_table.add_row("Scaler type", type(scaler).__name__)
        info_table.add_row("Number of features", str(n_features))
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
            model = joblib.load(MODELS_DIR / 'mo_hinh_tot_nhat_vn.pkl')
            scaler = joblib.load(MODELS_DIR / 'scaler_vn.pkl')
        
        # Test cases
        test_cases = [
            ("Minimum values", {
                'tuoi': 18, 'gio_ngu': 4.0, 'muc_stress': 0.0,
                'caffeine': 0, 'phut_tap_luyen': 0,
                'chat_luong_ngu': 0.0, 'thoi_gian_man_hinh': 0.0
            }),
            ("Maximum values", {
                'tuoi': 70, 'gio_ngu': 10.0, 'muc_stress': 10.0,
                'caffeine': 5, 'phut_tap_luyen': 120,
                'chat_luong_ngu': 10.0, 'thoi_gian_man_hinh': 8.0
            }),
            ("Average values", {
                'tuoi': 40, 'gio_ngu': 7.0, 'muc_stress': 5.0,
                'caffeine': 2, 'phut_tap_luyen': 30,
                'chat_luong_ngu': 5.0, 'thoi_gian_man_hinh': 2.0
            }),
            ("High stress case", {
                'tuoi': 30, 'gio_ngu': 5.0, 'muc_stress': 9.0,
                'caffeine': 4, 'phut_tap_luyen': 10,
                'chat_luong_ngu': 3.0, 'thoi_gian_man_hinh': 6.0
            }),
            ("Low stress case", {
                'tuoi': 35, 'gio_ngu': 8.5, 'muc_stress': 1.0,
                'caffeine': 0, 'phut_tap_luyen': 90,
                'chat_luong_ngu': 9.0, 'thoi_gian_man_hinh': 0.5
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
            
            assert prediction in [0, 1, 2, 3], f"Invalid prediction: {prediction}"
            
            # Check probabilities if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)[0]
                assert abs(sum(proba) - 1.0) < 0.01, "Probabilities don't sum to 1"
                assert all(0 <= p <= 1 for p in proba), "Invalid probability values"
            
            dream_labels = {0: 'Ác mộng', 1: 'Mơ đẹp', 2: 'Ngủ sâu', 3: 'Không mơ'}
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
        if (DATA_DIR / 'dream_data_real.csv').exists():
            data_file = DATA_DIR / 'dream_data_real.csv'
            console.print("[dim]Sử dụng dữ liệu thật: dream_data_real.csv[/dim]")
        elif (DATA_DIR / 'dream_data.csv').exists():
            data_file = DATA_DIR / 'dream_data.csv'
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
            assert df['dream_type'].isin([0, 1, 2, 3]).all(), "Invalid dream_type"
            
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
        if not (DATA_DIR / 'test_data.csv').exists():
            console.print(Panel(
                "[bold yellow]⚠ test_data.csv not found[/bold yellow]",
                border_style="yellow"
            ))
            return False
        
        with console.status("[cyan]Đang test batch prediction..."):
            model = joblib.load(MODELS_DIR / 'mo_hinh_tot_nhat_vn.pkl')
            scaler = joblib.load(MODELS_DIR / 'scaler_vn.pkl')
            
            df = pd.read_csv(DATA_DIR / 'test_data.csv')
            X = df.drop('dream_type', axis=1, errors='ignore')
            X = X.rename(columns={
                'age': 'tuoi',
                'sleep_hours': 'gio_ngu',
                'stress_level': 'muc_stress',
                'caffeine_intake': 'caffeine',
                'exercise_minutes': 'phut_tap_luyen',
                'sleep_quality': 'chat_luong_ngu',
                'screen_time': 'thoi_gian_man_hinh'
            })
            if (MODELS_DIR / 'ten_dac_trung_vn.pkl').exists():
                X = X[joblib.load(MODELS_DIR / 'ten_dac_trung_vn.pkl')]
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            
            assert len(predictions) == len(df), "Prediction count mismatch"
            assert all(p in [0, 1, 2, 3] for p in predictions), "Invalid predictions"
        
        dream_labels = {0: 'Ác mộng', 1: 'Mơ đẹp', 2: 'Ngủ sâu', 3: 'Không mơ'}
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
        (DATA_DIR / 'dream_data.csv', 'Data file (synthetic)', False),
        (DATA_DIR / 'dream_data_real.csv', 'Data file (real)', False),
        (MODELS_DIR / 'mo_hinh_tot_nhat_vn.pkl', 'Model file', True),
        (MODELS_DIR / 'scaler_vn.pkl', 'Scaler file', True),
        (MODELS_DIR / 'ten_dac_trung_vn.pkl', 'Feature names file', True),
    ]
    
    files_table = Table(box=box.ROUNDED, show_header=True)
    files_table.add_column("File", style="cyan")
    files_table.add_column("Description", style="white")
    files_table.add_column("Status", justify="center", style="green")
    files_table.add_column("Size", justify="right", style="blue")
    
    all_exist = True
    for filename, desc, required in files_to_check:
        exists = filename.exists()
        if exists:
            size = filename.stat().st_size
            files_table.add_row(str(filename.relative_to(PROJECT_ROOT)), desc, "✓", f"{size:,} bytes")
        else:
            if filename.suffix == '.png' or not required:
                # PNG files và data files là optional (có thể có 1 trong 2)
                if 'dream_data' in filename.name:
                    files_table.add_row(str(filename.relative_to(PROJECT_ROOT)), desc, "⚠", "Optional")
                else:
                    files_table.add_row(str(filename.relative_to(PROJECT_ROOT)), desc, "⚠", "Optional")
            else:
                files_table.add_row(str(filename.relative_to(PROJECT_ROOT)), desc, "✗", "Missing")
                all_exist = False
    
    # Kiểm tra ít nhất 1 data file tồn tại
    has_data = (DATA_DIR / 'dream_data.csv').exists() or (DATA_DIR / 'dream_data_real.csv').exists()
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

