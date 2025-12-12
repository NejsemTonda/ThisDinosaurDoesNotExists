from flask import Flask, Response, render_template_string

app = Flask(__name__)

def get_image():
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

