#!/usr/bin/env python3
"""
ChatGPT: Write a python script that will download 16 images from from https://thispersondoesnotexist.com/ and will compile the into one image with 4x4 grid


Download 16 images from https://thispersondoesnotexist.com/ and compile into a 4x4 grid.

Dependencies:
  pip install requests pillow
"""

import io
import os
import time
import requests
from PIL import Image


URL = "https://thispersondoesnotexist.com/"
OUT_DIR = "tpdne_images"
GRID_OUT = "thisperson.jpg"
ROWS, COLS = 4, 4
TOTAL = ROWS * COLS

# A friendly User-Agent helps avoid some anti-bot blocks.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def fetch_image(session: requests.Session, idx: int, retries: int = 5, backoff: float = 0.8) -> Image.Image:
    """Fetch one image with retries and exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            r = session.get(URL, headers=HEADERS, timeout=20)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            return img
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"Failed to fetch image {idx} after {retries} attempts: {e}") from e
            time.sleep(backoff * (2 ** (attempt - 1)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    images = []
    with requests.Session() as session:
        for i in range(TOTAL):
            img = fetch_image(session, i)
            images.append(img)

            # Optional: save individual downloads
            img.save(os.path.join(OUT_DIR, f"face_{i:02d}.jpg"), quality=95)

            # Small delay can help reduce rate-limiting risk
            time.sleep(0.2)

    # Ensure all images are the same size (use the first as reference)
    w0, h0 = images[0].size
    images = [im.resize((w0, h0), Image.Resampling.LANCZOS) for im in images]

    # Create the grid canvas
    grid = Image.new("RGB", (COLS * w0, ROWS * h0), color=(0, 0, 0))

    # Paste images
    for idx, im in enumerate(images):
        r = idx // COLS
        c = idx % COLS
        grid.paste(im, (c * w0, r * h0))

    grid.save(GRID_OUT, quality=95)
    print(f"Saved grid to: {GRID_OUT}")

    [os.remove(f"{OUT_DIR}/{file}") for file in os.listdir(OUT_DIR)]
    os.removedirs(OUT_DIR)


if __name__ == "__main__":
    main()

