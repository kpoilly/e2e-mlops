FROM python:3.13-slim-bullseye

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV MLFLOW_TRACKING_URI=http://localhost:5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]