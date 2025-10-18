FROM python:3.11-slim

# MLflow + przydatne dodatki (boto3/psycopg2 na przyszłość)
RUN pip install --no-cache-dir "mlflow[extras]==2.14.1" boto3 psycopg2-binary

# Użytkownik nie-root (bezpieczniej na wolumenach)
RUN useradd -m mlflow && mkdir -p /mlflow && chown -R mlflow:mlflow /mlflow
USER mlflow

EXPOSE 5000
WORKDIR /home/mlflow

# Serwer konfigurujemy w docker-compose (command)
ENTRYPOINT ["mlflow", "server"]
