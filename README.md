# Neural Style Transfer

A Flask web app that applies artistic style transfer to images using the AdaIN (Adaptive Instance Normalization) method.

## Setup

```bash
pip install -r requirements.txt
```

Download the pretrained VGG weights (`vgg_normalised.pth`) and place them in the project root.

## Training

```bash
python train.py --content_dir <path> --style_dir <path> --vgg <path_to_vgg_normalised.pth>
```

Trained decoder checkpoints are saved under `experiment/<name>/`.

## Running the App

Before running, set the decoder path in `app.py`:

```python
decoder.load_state_dict(torch.load("path/to/decoder_final.pth"))
```

Then start the server:

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

## Project Structure

```
NST_Project/
├── app.py              # Flask web app
├── train.py            # Training script
├── requirements.txt
├── utils/
│   ├── models.py       # VGGEncoder and Decoder definitions
│   └── utils.py        # AdaIN, dataset, transforms
├── templates/
│   └── index.html
├── content_data/       # Sample content images
└── style_data/         # Sample style images
```

## Notes

- `vgg_normalised.pth` is not included in this repo. Download it separately.
- Set the `SECRET_KEY` environment variable before deploying.
