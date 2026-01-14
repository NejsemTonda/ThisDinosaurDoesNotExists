from flask import Flask, Response, render_template_string
from PIL import Image
import io
import numpy as np
import vae

import os
import torch


app = Flask(__name__)

z_dim = 5
epoch = 2
class Arguments:
    seed = 122
    encoder_layers = [500, 500]
    decoder_layers = [500, 500]

    def __init__(self, z_dim):
        self.z_dim = z_dim

args = Arguments(z_dim)
model = vae.VAE(args)

model_dir = os.path.join("models", "vae", f"vae.py-bs=100,d=dataset,dl=[500, 500],el=[500, 500],zd={z_dim}")
model_path = os.path.join(model_dir, f"vae_model_{epoch}.pt")

vae.load_model(model, model_path)



def get_image_from_dataset():
    """
    For now, just retrun ramdon image until we implement the NN
    """
    import random
    import os
    DATASET_DIR = "dataset"
    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

    files = [
        f for f in os.listdir(DATASET_DIR)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
    ]
    
    if not files:
        raise FileNotFoundError("No images found in dataset folder")
    
    filename = random.choice(files)
    filepath = os.path.join(DATASET_DIR, filename)
    
    with open(filepath, "rb") as f:
        return f.read()

def get_image():
    with torch.no_grad():
        # z = torch.randn(1, args.z_dim)
        z = torch.rand(1, args.z_dim) * 20 - 10
        arr = model.decoder(z, training=False)[0].cpu().numpy()

    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@app.route("/")
def index():
    html = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>This Dinosaur Does Not Exists</title>
            <style>
                body {
                    margin: 0;
                    height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    background-color: #f5f5f5;
                }

                img {
                    max-width: 90%;
                    max-height: 90%;
                    object-fit: contain;
                }
            </style>
        </head>
        <body>
            <img src="/image" alt="Random Image" />
        </body>
    </html>
    """
    return render_template_string(html)


@app.route("/image")
def image():
    try:
        image_bytes = get_image()
        return Response(image_bytes, mimetype="image/png")
    except NotImplementedError:
        return Response(
            b"Image not implemented yet",
            mimetype="text/plain"
        )


if __name__ == "__main__":
    app.run(debug=True)

