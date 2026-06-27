# StyleForge — Neural Style Transfer

A Flask web app for arbitrary neural style transfer that restyles a content photo in the look of any style image using the AdaIN (Adaptive Instance Normalization) method — in a single forward pass.

## Problem

Classical neural style transfer (Gatys et al.) produces beautiful results but requires hundreds of iterative optimization steps per image pair, making it too slow for interactive use. Per-style feed-forward networks are fast but lock you into one fixed style per trained model. StyleForge implements AdaIN-based arbitrary style transfer: a single trained decoder applies **any** style image to **any** content image in one forward pass, with a tunable alpha slider to control style strength — usable as a web app or a containerized HTTP service.

## Features

- **Arbitrary style transfer** — any content image + any style image, no per-style retraining
- **Single forward pass** — sub-second stylization on GPU, a few seconds on CPU
- **Alpha style-strength slider** (0.0 = original photo, 1.0 = full stylization)
- **Frozen VGG-19 encoder + custom-trained decoder** — only the decoder is trained; pretrained weights for both ship with the repo
- **Full training pipeline included** — multi-layer VGG style loss (content + per-channel mean/std style terms), Adam with LR decay, random-crop augmentation, checkpointing, and corrupt-image-skipping data loading
- **Production deployment-ready** — Gunicorn + Docker (Hugging Face Spaces, Render Blueprint, generic Procfile)
- **Safe upload handling** — extension allowlist (png/jpg/jpeg), `secure_filename`, 10 MB `MAX_CONTENT_LENGTH` cap with a 413 handler
- **Tunable inference** — env-driven `IMAGE_SIZE`, `TORCH_NUM_THREADS`, CUDA-fragmentation mitigation (`max_split_size_mb`)
- **CPU/GPU split** — separate PyTorch wheel installs for CPU-only deploy vs CUDA 12.1 GPU

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch 2.2.2 · torchvision 0.17.2 |
| Encoder | VGG-19 (frozen, `vgg_normalised.pth`, truncated at relu4_1) |
| Style Transfer | AdaIN (Adaptive Instance Normalization) |
| Decoder | Custom CNN (trained, `models/decoder.pth`) |
| Web Framework | Flask 3.1 · Flask-WTF · WTForms |
| WSGI Server | Gunicorn 23 |
| Frontend | Bootstrap 5.3 · Font Awesome 6 · Google Fonts (Space Grotesk, Inter) |
| Image I/O | Pillow 12 · torchvision.transforms |
| Config | python-dotenv |
| Containerization | Docker (`python:3.11-slim`) |
| Deployment Targets | Hugging Face Spaces · Render · generic Procfile |

## How It Works

1. **Upload** — User submits a content image and a style image through the Bootstrap UI and sets the alpha (style-strength) slider. Flask-WTF handles the multipart form (CSRF-protected) and saves files to `static/uploads`.
2. **Validation** — Extension allowlist (png/jpg/jpeg), `secure_filename`, and a 10 MB upload cap (413 handler on overflow).
3. **Preprocessing** — Both images are resized to `IMAGE_SIZE` (default 384, or 512 in the HF Docker image), converted to RGB, and turned into tensors (no normalization — the "normalised" VGG expects 0–1 input).
4. **Content feature extraction** — The content image is passed through the frozen VGG-19 encoder up to `relu4_1`, producing a content feature map.
5. **Style feature extraction** — The style image is passed through the same encoder, producing a style feature map.
6. **AdaIN** — Channel-wise mean and standard deviation of the content features are realigned to match the style:
   `AdaIN(c, s) = σ(s) · ((c − μ(c)) / σ(c)) + μ(s)`
7. **Alpha blending** — The AdaIN output is linearly interpolated with the original content features using the user's alpha value: `α·AdaIN + (1−α)·c`.
8. **Decoding** — The trained decoder (a structural mirror of the encoder with `ReflectionPad2d → Conv → ReLU` blocks and three `nn.Upsample(scale_factor=2)` stages) reconstructs the stylized image from the blended features in a single forward pass (`torch.no_grad`).
9. **Output** — The output tensor is clamped to [0, 1], converted to a PIL image, and saved with a UUID-based filename (prevents browser cache from serving stale results).
10. **Render** — The page re-renders with the stylized output and a download button. CUDA is auto-selected if available, otherwise CPU.

## Architecture

```
   ┌──────────────────────────────────────────────┐
   │  Flask UI  (Bootstrap 5 · Font Awesome)      │
   │  Content upload · Style upload · α slider    │
   └─────────────────────┬────────────────────────┘
                         │  multipart form (CSRF)
                         ▼
   ┌──────────────────────────────────────────────┐
   │  Flask app · Gunicorn                        │
   │  secure_filename · 10 MB cap                 │
   └────────────┬─────────────────────┬───────────┘
                │ content             │ style
                ▼                     ▼
       ┌──────────────────────────────────────┐
       │  VGG-19 Encoder  (frozen → relu4_1)  │
       │           vgg_normalised.pth         │
       └────────────┬────────────┬────────────┘
                fc  │            │  fs
                    ▼            ▼
                ┌───────────────────────┐
                │       AdaIN           │
                │  match μ, σ of fs     │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │  Alpha blending       │
                │  α·AdaIN + (1−α)·fc   │
                └──────────┬────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │  Decoder (trained)    │
                │  models/decoder.pth   │
                │  mirrors VGG-19       │
                └──────────┬────────────┘
                           ▼
                  ┌──────────────────┐
                  │ Stylized Output  │
                  │  (UUID filename) │
                  └────────┬─────────┘
                           ▼
              Browser preview + download

   Deployment: Docker → Hugging Face Spaces · Render · Procfile
```

## Setup

### Local

```bash
# Clone
git clone https://github.com/Abdul-Rehman-Sayed/styleforge.git
cd styleforge

# Virtual environment (Python 3.11)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies (CPU)
pip install -r requirements.txt

# Environment variables (.env)
# SECRET_KEY=replace_with_a_real_secret
# IMAGE_SIZE=384
# TORCH_NUM_THREADS=2

# Run (development)
python app.py
# → http://localhost:5000

# Run (production)
gunicorn -w 1 -b 0.0.0.0:8000 --timeout 120 app:app
```

### Docker

```bash
docker build -t styleforge .
docker run -p 7860:7860 -e SECRET_KEY=replace_here styleforge
```

> Pretrained weights `vgg_normalised.pth` and `models/decoder.pth` are committed to the repo, so the app is runnable as-is — no separate weight download required. Set a real `SECRET_KEY` in production (the default fallback is insecure).

### Training (optional)

```bash
python train.py \
  --content_dir /path/to/content_images \
  --style_dir   /path/to/style_images \
  --vgg         vgg_normalised.pth \
  --epochs 1 --batch_size 4 --lr 1e-4 \
  --style_weight 5 --content_weight 1.0
```

Defaults: content/style images resized to 512 then random-cropped to 256; Adam with `LambdaLR` decay (`lr_decay=5e-5`); checkpoints and preview grids saved per save interval under `experiment/<name>/`.

## Limitations & Future Work

- **Decoder-only training** — Only the decoder is trained; the encoder is a frozen pretrained VGG-19, so stylization quality is bounded by VGG features. Planned: experiment with joint encoder–decoder fine-tuning and modern backbones (ConvNeXt, EfficientNet) for better quality and smaller models.
- **No quantitative metrics** — Results are evaluated qualitatively only; no PSNR, SSIM, LPIPS, or reference-AdaIN benchmarks have been reported. Planned: add a formal evaluation suite tracking content/style loss and perceptual metrics.
- **Training dataset not bundled** — The large-scale content (MS-COCO) and style (e.g. WikiArt / "Painter by Numbers") datasets used during training are not redistributed in the repo, and `train.py` defaults point at hardcoded local paths. Planned: ship a small reproducibility subset and a dataset-download helper.
- **One-epoch defaults** — `train.py` defaults to `epochs=1`; the committed `decoder.pth` was trained separately. Reproducing quality from the in-repo defaults alone won't match the shipped weights. Planned: document and ship the real training recipe.
- **Single worker** — Production config runs a single Gunicorn worker with long timeouts (CPU inference is slow). Planned: queue + worker pool for concurrent requests.
