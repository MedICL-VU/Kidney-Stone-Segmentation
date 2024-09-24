import os
from PIL import Image
from multiprocessing import Pool, cpu_count


def resize_image(image_path, max_size):
    with Image.open(image_path) as img:
        res = img.resize(max_size)
        res.save(image_path)


def process_image(args):
    image_path, max_size = args
    resize_image(image_path, max_size)


def resize_images_in_folder(folder_path, max_size):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png')):
                image_path = os.path.join(root, file)
                image_paths.append((image_path, max_size))

    with Pool(cpu_count()) as pool:
        pool.map(process_image, image_paths)


if __name__ == "__main__":
    folder_path = '../data/test_raindrop/images'  # Replace with the path to your folder
    max_size = (540, 540)  # Replace with your desired max size (width, height)
    resize_images_in_folder(folder_path, max_size)
