import os
from PIL import Image


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def check_image_dimensions(folder1, folder2):
    mismatched_images = []
    image_count = 0
    common_image_count = 0

    for subdir in os.listdir(folder1):
        subdir1_path = os.path.join(folder1, subdir)
        subdir2_path = os.path.join(folder2, subdir)

        if os.path.isdir(subdir1_path) and os.path.isdir(subdir2_path):
            folder1_images = {file.lower(): os.path.join(subdir1_path, file) for file in os.listdir(subdir1_path) if
                              file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))}
            folder2_images = {file.lower(): os.path.join(subdir2_path, file) for file in os.listdir(subdir2_path) if
                              file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))}

            common_images = set(folder1_images.keys()).intersection(set(folder2_images.keys()))
            image_count += len(folder1_images)
            common_image_count +=len(common_images)

            for image_name in common_images:
                size1 = get_image_size(folder1_images[image_name])
                size2 = get_image_size(folder2_images[image_name])
                if size1 != size2:
                    mismatched_images.append((os.path.join(subdir, image_name), size1, size2))

    print(f'Checked {common_image_count}/{image_count} images')
    if mismatched_images:
        print("Mismatched images:")
        for image_name, size1, size2 in mismatched_images:
            print(f"{image_name}: Folder1 size = {size1}, Folder2 size = {size2}")
    else:
        print("All matching images have the same dimensions.")


if __name__ == "__main__":
    folder1 = '../data/test_raindrop/images/'  # Replace with the path to the first folder
    folder2 = '../data/test_raindrop/masks/'  # Replace with the path to the second folder

    check_image_dimensions(folder1, folder2)
