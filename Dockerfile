FROM python:3.11-slim

# کپی کردن ابزار uv از ایمیج رسمی‌شون
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# کپی فایل‌های وابستگی و نصب اون‌ها
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# کپی کردن کدها و مدل
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000

# اجرای اپ با uv
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]