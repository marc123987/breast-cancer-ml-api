FROM python:3.13.7

WORKDIR /app

COPY modelo_cancer.pkl .
COPY api.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "api.py"]