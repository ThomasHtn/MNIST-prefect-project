FROM prefecthq/prefect:3-latest

WORKDIR /app

COPY . /app

COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "training_flow.py"]