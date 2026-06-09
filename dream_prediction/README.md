# Dream Prediction ML

Thu muc nay chua pipeline machine learning cho bai toan du doan giac mo.

## Cau truc

```text
src/              Ma nguon Python: train, predict, generate data, pipeline
tests/            Script kiem thu he thong
data/             CSV du lieu dau vao va du lieu da chuyen doi
models/           File model, scaler, feature names da train
reports/figures/  Bieu do va hinh anh bao cao
docs/             Tai lieu chi tiet cu
requirements.txt  Thu vien Python can cai
```

## Lenh hay dung

```powershell
cd F:\TGMT\tu\dream_prediction
pip install -r requirements.txt
python src\generate_data_4_classes.py
python src\train_model_vn.py
python src\predict_vn.py
python tests\test_system.py
```

Backend web doc model tu `models/`, nen sau khi train lai khong can copy file thu cong.
