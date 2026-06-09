import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

from paths import DATA_DIR, MODELS_DIR, PROJECT_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent


def print_header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")


def run_script(script_name, description):
    script_path = SCRIPT_DIR / script_name
    print_header(description)

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
        )
        print(f"\nOK: {description} hoan thanh.")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"\nLOI: {description} that bai.")
        print(f"Exit code: {exc.returncode}")
        return False
    except FileNotFoundError:
        print(f"\nLOI: Khong tim thay file: {script_path}")
        return False


def check_data_exists():
    if (DATA_DIR / "dream_data_real.csv").exists():
        return DATA_DIR / "dream_data_real.csv", "DU LIEU THAT"
    if (DATA_DIR / "dream_data.csv").exists():
        return DATA_DIR / "dream_data.csv", "DU LIEU GIA DINH"
    return None, None


def run_demo_prediction():
    model = joblib.load(MODELS_DIR / "mo_hinh_tot_nhat_vn.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler_vn.pkl")

    test_data = pd.DataFrame(
        {
            "tuoi": [25, 30, 35, 28, 40],
            "gio_ngu": [7.5, 6.0, 8.5, 5.0, 8.0],
            "muc_stress": [6.0, 8.5, 3.0, 9.0, 2.0],
            "caffeine": [2, 3, 1, 4, 0],
            "phut_tap_luyen": [45, 20, 60, 10, 90],
            "chat_luong_ngu": [7.0, 5.0, 9.0, 3.0, 8.5],
            "thoi_gian_man_hinh": [2.0, 4.0, 1.0, 6.0, 0.5],
        }
    )

    predictions = model.predict(scaler.transform(test_data))
    labels = {
        0: "Ac mong",
        1: "Mo dep",
        2: "Ngu sau",
        3: "Khong mo",
    }

    print_header("BUOC 3: DU DOAN DEMO")
    for index, prediction in enumerate(predictions, start=1):
        print(f"Nguoi {index}: {labels[int(prediction)]}")


def print_outputs():
    files_to_check = [
        (DATA_DIR / "dream_data.csv", "Du lieu gia dinh"),
        (DATA_DIR / "dream_data_real.csv", "Du lieu that"),
        (MODELS_DIR / "mo_hinh_tot_nhat_vn.pkl", "Mo hinh tot nhat"),
        (MODELS_DIR / "scaler_vn.pkl", "Scaler"),
        (MODELS_DIR / "ten_dac_trung_vn.pkl", "Ten dac trung"),
    ]

    print("\nFILES DA TAO:")
    for path, description in files_to_check:
        if path.exists():
            print(f"  OK {path.relative_to(PROJECT_ROOT)} ({description}) - {path.stat().st_size:,} bytes")


def main():
    print_header("DREAM PREDICTION - MASTER PIPELINE")

    data_file, data_type = check_data_exists()
    if data_file:
        print(f"Da tim thay du lieu: {data_file.relative_to(PROJECT_ROOT)} ({data_type})")
        response = input("Ban co muon tao lai du lieu khong? (y/n): ").strip().lower()
        if response == "y":
            data_file = None

    if not data_file:
        print("\n1. Du lieu gia dinh nhanh")
        print("2. Du lieu that tu Kaggle (can setup rieng)")
        choice = input("Lua chon (1/2): ").strip()

        if choice == "2":
            print("\nHay chay: python src/load_real_data.py")
            print("Tam thoi tiep tuc voi du lieu gia dinh.")

        if not run_script("generate_data_4_classes.py", "BUOC 1: TAO DU LIEU"):
            sys.exit(1)

    if not run_script("train_model_vn.py", "BUOC 2: HUAN LUYEN MO HINH"):
        sys.exit(1)

    try:
        run_demo_prediction()
    except Exception as exc:
        print(f"Canh bao: demo prediction loi: {exc}")

    print_outputs()
    print("\nHoan thanh pipeline.")
    print("Chay du doan tuong tac: python src/predict_vn.py")


if __name__ == "__main__":
    main()
