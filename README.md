# Iris Classifier Api
یه API ساده با **FastAPI** که یه مدل طبقه‌بندی آموزش‌دیده روی دیتاست **Iris** رو سرو می‌کنه. مدل با `scikit-learn` آموزش دیده (رگرسیون لجستیک + StandardScaler) و با `joblib` ذخیره شده.

## پیش‌نیازها
- Python 3.10+
- pip

## نصب
```bash
pip install -r requirements.txt
```

## ساختار پروژه
```
project/
├── app/
│   ├── __init__.py
│   └── main.py            # اپ FastAPI
├── models/
│   └── model.joblib       # مدل + scaler ذخیره‌شده
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── pytest.ini
└── requirements.txt
```

## اجرا
```bash
uvicorn app.main:app --reload
```
سرور روی http://127.0.0.1:8000 بالا میاد. مستندات تعاملی: http://127.0.0.1:8000/docs

## Endpoints

### GET /health
```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```
### POST /predict
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
## تست
```bash
pytest -v
```

## تکنولوژی‌ها
FastAPI — وب فریم‌ورک

scikit-learn — آموزش مدل

joblib — ذخیره/بارگذاری مدل

pytest — تست