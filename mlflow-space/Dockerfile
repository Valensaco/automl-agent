# Read-only snapshot of the AutoML Agent's MLflow log, for Hugging Face Spaces.
FROM python:3.12-slim
RUN pip install --no-cache-dir "mlflow==3.17.2"
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user . .
USER user
ENV MLFLOW_DISABLE_AGENT_HINT=1
EXPOSE 7860
# Copy the snapshot to a temp folder at start, so any clicks on the public
# site never change the original (and every restart is a clean copy).
CMD cp -r /home/user/app /tmp/snap && cd /tmp/snap && \
    mlflow ui --backend-store-uri sqlite:////tmp/snap/mlflow.db \
              --host 0.0.0.0 --port 7860 --allowed-hosts "*"
