# 輕量級 Python 3.10
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴 (for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先安裝 CPU 版的 Torch 和 Torchvision (這行最重要)
#  --no-cache-dir 減少暫存檔
#  --index-url 指定去 CPU 版本的倉庫下載
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# 再安裝其他的 requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有程式碼 (除了 .dockerignore 排除的)
COPY . .

# 設定環境變數 (讓 Python print 直接輸出到 Cloud Logging)
ENV PYTHONUNBUFFERED=1

# 開放 Port
EXPOSE 8080

# 啟動指令 (使用 main.py)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]