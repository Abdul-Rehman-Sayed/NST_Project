# Hugging Face Spaces (Docker SDK) image for the AdaIN style-transfer Flask app.
# The app code + model weights are pulled from the public GitHub repo at build
# time, so this Space only needs this Dockerfile + README.md.
FROM python:3.11-slim

# git is needed to clone the app repo during build.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run containers as a non-root user (uid 1000). Create it so the app
# can write its upload folder.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Pull the app (includes vgg_normalised.pth and models/decoder.pth).
RUN git clone --depth 1 https://github.com/Abdul-Rehman-Sayed/NST_Project.git .

RUN pip install --no-cache-dir --user -r requirements.txt

# 16 GB RAM on the free CPU Space -> run at full 512px for sharp faces.
# 2 vCPUs -> let torch use both.
ENV IMAGE_SIZE=512 \
    TORCH_NUM_THREADS=2

# HF Spaces routes traffic to port 7860.
EXPOSE 7860
CMD ["gunicorn", "app:app", "--workers", "1", "--timeout", "180", "--bind", "0.0.0.0:7860"]
