# AutoML Agent: Streamlit app + agents + MLflow in one image.
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime (on Mac this was libomp from Homebrew)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY .streamlit/ .streamlit/
COPY reports/ reports/

ENV PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
EXPOSE 8501 5000

# Default: the Streamlit app. (docker-compose also starts the MLflow dashboard.)
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", \
     "--server.port=8501", "--server.headless=true"]
