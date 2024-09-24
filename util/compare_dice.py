import matplotlib.pyplot as plt
import shutil
import os

def read_scores(filename):
    names = []
    scores = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            names.append(parts[0])
            scores.append(float(parts[1]))
    return names, scores


def plot_scores(file1, file2, threshold):
    names1, scores1 = read_scores(file1)
    names2, scores2 = read_scores(file2)

    # if names1 != names2:
    #     raise ValueError("Names in both files do not match or are not in the same order.")

    plt.figure(figsize=(10, 6))
    plt.scatter(scores1, scores2)
    plt.title("Comparison of Dice Scores")
    plt.xlabel("Original")
    plt.ylabel("SDAN")

    max_score = max(max(scores1), max(scores2))
    plt.plot([0, max_score], [0, max_score], 'k--')
    plt.plot([0, max_score], [threshold, max_score+threshold], 'k--')

    # for i, name in enumerate(names1):
    #     plt.annotate(name, (scores1[i], scores2[i]))
    plt.grid(True)
    plt.show()
    # plt.savefig('Dice Comparison Raindrop.png')

    # Find names where the difference in scores exceeds the threshold
    significant_differences = []
    for name, score1, score2 in zip(names1, scores1, scores2):
        if score2 - score1 < threshold:
            significant_differences.append(name)

    os.makedirs(os.path.join(destination_folder, 'images'),exist_ok=True)
    os.makedirs(os.path.join(destination_folder, 'masks'), exist_ok=True)
    # # Copy files associated with significant names to a specified folder
    for name in significant_differences:
        # source_file = name + '.txt'  # Assuming the file names are based on the names in the scores files
        root, fname = os.path.split(name)
        root, video = os.path.split(root)
        root, type = os.path.split(root)
        source_image =  os.path.join('..', name)
        source_mask = os.path.join('..',root, 'masks', video, fname)
        destination_image = os.path.join(destination_folder, 'images', video + '_' +fname)
        destination_mask = os.path.join(destination_folder, 'masks', video + '_' + fname)

        shutil.copy(source_image, destination_image)
        shutil.copy(source_mask, destination_mask)
        # print(f"Copied {source_file} to {destination_file}")

    return significant_differences


threshold_value = -0.2  # Set the threshold for score differences
destination_folder = '../data/denoising_negative_raindrop_og'
significant_names = plot_scores('../checkpoints/unet_best_test/dice_scores.txt', '../checkpoints/unet_raindrop_test/dice_scores.txt', threshold_value)
print(len(significant_names))
print("Names with significant differences:", significant_names)

