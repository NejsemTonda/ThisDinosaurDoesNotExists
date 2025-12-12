import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

def squish_to_square(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.LANCZOS)


def process_single_image(args):
    input_path, output_path, size = args

    try:
        with Image.open(input_path) as img:
            img = squish_to_square(img, size)
            img.save(output_path)
        return f"Processed: {os.path.basename(input_path)}"
    except Exception as e:
        return f"Failed: {os.path.basename(input_path)} | Error: {e}"


def process_dataset_parallel(input_dir, output_dir, size, workers=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect tasks
    tasks = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            tasks.append((input_path, output_path, size))

    # Parallel execution
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_image, t) for t in tasks]
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    input_directory = "dataset"
    output_directory = "rescaled"
    target_size = 512         # Set your output square resolution
    num_workers = None        # None = use all CPU cores

    process_dataset_parallel(input_directory, output_directory, target_size, num_workers)
