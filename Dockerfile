FROM ghcr.io/mlflow/mlflow:latest

RUN apt-get -y update && \
    apt-get -y install python3-dev default-libmysqlclient-dev build-essential pkg-config && \
    pip install --upgrade pip && \
    pip install mysqlclient && \
    pip install psycopg2-binary \
    pip install boto3

CMD ["bash"]