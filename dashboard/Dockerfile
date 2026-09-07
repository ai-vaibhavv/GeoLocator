# Hugging Face Spaces, Docker SDK. Dash is not one of the native Space SDKs,
# so the app is served by gunicorn on the port Spaces expects (7860).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Spaces runs the container as uid 1000; everything is installed for that user.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
# One worker: the model is loaded per process, and the free tier has 2 vCPU.
# The timeout covers the first request, which loads the checkpoint.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", \
     "--timeout", "120", "--preload", "app:server"]
