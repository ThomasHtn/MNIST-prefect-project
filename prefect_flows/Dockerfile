FROM prefecthq/prefect:2.14.17

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["python", "training_flow.py"]