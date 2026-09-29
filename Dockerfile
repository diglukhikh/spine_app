# --- Этап сборки ---
FROM python:3.10-slim AS builder

# ВАЖНО: git нужен здесь, потому что requirements.txt тянет пакет из GitHub
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 1. Создаем виртуальное окружение и активируем его для всех последующих шагов
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 2. ВАЖНО: Сначала устанавливаем CPU-версию PyTorch, чтобы зафиксировать её как базовую
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Принудительно удаляем triton, если он вдруг успел установиться
RUN pip uninstall -y triton || true

# 4. Устанавливаем именно CPU-версию TensorFlow (вместо обычной, которая лезет в GPU)
RUN pip install --no-cache-dir tensorflow-cpu

# 5. Устанавливаем ваши зависимости из файла. 
# (См. примечание ниже: удалите из самого requirements.txt слова 'tensorflow', 'torch', 'torchvision')
RUN pip install --no-cache-dir -r requirements.txt

# 6. Устанавливаем MobileSAM БЕЗ зависимостей (--no-deps), 
# чтобы он не попытался перезатереть наш CPU-версию PyTorch на CUDA-версию.
# Все его зависимости (timm, opencv и т.д.) у вас уже есть в requirements.txt.
RUN pip install --no-cache-dir --no-deps git+https://github.com/ChaoningZhang/MobileSAM.git

# --- Финальный этап ---
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY *.py .
COPY spine/ ./spine/
COPY class/ ./class/
COPY sustav/ ./sustav/
COPY jsonmy/ ./jsonmy/
COPY json_rep/ ./json_rep/
COPY examples/ ./examples/
COPY operation/ ./operation/
COPY output/ ./output/
COPY result/ ./result/
COPY sustavess/ ./sustavess/
COPY up_dicom/ ./up_dicom/
COPY upload_dicom/ ./upload_dicom/
COPY uploads/ ./uploads/
COPY vertebrae_debug/ ./vertebrae_debug/
COPY templates/ ./templates/
COPY static/ ./static/
COPY sam/ ./sam/

EXPOSE 5000
CMD ["gunicorn", "-w", "1", "-k", "gevent", "-b", "0.0.0.0:5000", "main_app:app"]
