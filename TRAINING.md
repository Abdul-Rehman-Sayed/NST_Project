# Training a Better Decoder (GPU)

More epochs on the tiny sample set just memorizes those images. Real quality comes from
**more, more-diverse data** — the standard pairing is **MS-COCO** (content photos) +
**WikiArt** (style paintings). Do this on a machine with an NVIDIA GPU.

## 1. Set up on the GPU machine

Copy the project over (make sure `vgg_normalised.pth` is in the root), then:

```bash
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements-gpu.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

> If your driver is CUDA 11.8, change `cu121` -> `cu118` in `requirements-gpu.txt` first.

## 2. Get the datasets

```bash
# Content: MS-COCO (~19 GB, already a flat folder of photos)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip                # -> ./train2017/

# Style: WikiArt via Kaggle (needs ~/.kaggle/kaggle.json)
kaggle datasets download -d steubk/wikiart
unzip wikiart.zip -d wikiart_raw
```

> [!IMPORTANT]
> The dataset loader (`utils/utils.py`) is **NOT recursive** — it only reads images sitting
> _directly_ inside the folder you pass. COCO's `train2017/` is already flat. **WikiArt is
> nested in subfolders, so flatten it first** or training finds 0 images:
>
> ```bash
> # Linux
> mkdir style_flat && find wikiart_raw -type f \( -iname '*.jpg' -o -iname '*.png' \) -exec cp {} style_flat/ \;
> ```
>
> ```powershell
> # Windows
> New-Item -ItemType Directory style_flat; Get-ChildItem wikiart_raw -Recurse -Include *.jpg,*.png | Copy-Item -Destination style_flat
> ```
>
> Any folder of ~10k+ varied paintings works as the style set if you don't use Kaggle.

## 3. Train

```bash
python train.py \
  --content_dir train2017 \
  --style_dir style_flat \
  --vgg vgg_normalised.pth \
  --batch_size 8 \
  --lr 1e-4 \
  --style_weight 10 \
  --epochs 5 \
  --save_interval 1 \
  --experiment big1
```

- With COCO + WikiArt on a GPU, even **2-3 epochs** generalizes far better than the sample run.
  One COCO epoch is roughly 20-40 min on a mid-range GPU.
- `--style_weight 10` = stronger stylization (the paper's value). Lower if too stylized, raise if too weak.
- `--batch_size`: drop to 4 or 2 if you hit a CUDA out-of-memory error.
- Checkpoints + `output_N.png` preview grids land in `experiment/big1/`.
  **Judge quality by the preview images, not just the loss number.**

## 4. Use the best checkpoint in the app

```bash
cp experiment/big1/decoder_3.pth models/decoder.pth     # pick the best-looking one
```

The app auto-loads `models/decoder.pth`. Copy that ~13 MB file back to your deploy machine
(and commit it) to ship the improved model.

> For best-looking output at inference on a GPU, set `IMAGE_SIZE=512` in `.env`. The decoder
> trains at 256 but is fully convolutional, so it runs fine at higher resolution.
