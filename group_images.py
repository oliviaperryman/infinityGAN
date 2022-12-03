import glob
import json
import os
import pickle as pkl


def group_images(img_path,start,stop):
    """Group images by label.
    """
    # Get all the labels
    labels = sorted(glob.glob(f'{img_path}/labels/*.json'))
    # Create a dictionary to store the labels and the images that have them
    label_counts = {}
    # Loop through all the labels
    for label in labels[start:stop]:
        # Open the label file
        with open(label, "r") as f:
            # Load the label file as a dictionary
            label_dict = json.load(f)
        # Loop through all the labels in the label file
        for label in label_dict:
            # Get the description of the label
            label_desc = label['description']
            # If the label is not in the dictionary, add it
            if label_desc not in label_counts:
                label_counts[label_desc] = 0
            # Add the image to the label
            # label_dict[label_desc].append(label['image_path'])
            label_counts[label_desc] += 1
    print(dict(sorted(label_counts.items(), key=lambda item: item[1])))
    # Return the dictionary
    return label_dict

def get_avg_latent_for_label(img_path, chosen_label):
    label_paths = sorted(glob.glob(f'{img_path}/labels/*.json'))
    local_latents = []
    global_latents = []
    percent = 95
    for label_path in label_paths:
         with open(label_path, "r") as f:
            label_dict = json.load(f)
            all_labels = [label['description'] for label in label_dict]
            if "Landscape" not in all_labels and "Natural landscape" not in all_labels:
                continue
            for label in label_dict:
                label_desc = label['description']
                if label_desc == chosen_label:
                    if label["score"] > percent/100.0:
                        # Open latent representation for img
                        base_name = os.path.splitext(os.path.basename(label_path))[0]
                        latent_file = os.path.join(img_path, "latents", base_name + ".pkl")
                        latent = pkl.load(open(latent_file, "rb"))
                        img_number = int(base_name)
                        latent_index = img_number % 8
                        local_latents.append(latent.local_latent[latent_index])
                        global_latents.append(latent.global_latent[latent_index])
    print(len(local_latents), chosen_label, "found")
    avg_local_latent = sum(local_latents) / len(local_latents)
    avg_global_latent = sum(global_latents) / len(global_latents)
    
    subdir = os.path.join(img_path,"avg_latents")
    if (not os.path.exists(subdir)): os.makedirs(subdir)
    pkl.dump(avg_local_latent, open(f"{img_path}/avg_latents/avg_local_latent_{chosen_label}_{percent}.pkl", "wb"))
    pkl.dump(avg_global_latent, open(f"{img_path}/avg_latents/avg_global_latent_{chosen_label}_{percent}.pkl", "wb"))


if __name__ == '__main__':
    # group_images("logs/InfinityGAN-IOF/test/infinite_gen_197x197_dataset", 0,10)

    # get_avg_latent_for_label("logs/InfinityGAN-IOF/test/infinite_gen_197x197_dataset", "Tree")
    get_avg_latent_for_label("logs/InfinityGAN-IOF/test/infinite_gen_197x197_dataset", "Mountain")
    # get_avg_latent_for_label("logs/InfinityGAN-IOF/test/infinite_gen_197x197_dataset", "Lighthouse")
    # get_avg_latent_for_label("logs/InfinityGAN-IOF/test/infinite_gen_197x197_dataset", "Lake")
    # get_avg_latent_for_label("logs/InfinityGAN-IOF/test/infinite_gen_197x197_dataset", "Flower")
    # get_avg_latent_for_label("logs/InfinityGAN-IOF/test/infinite_gen_197x197_dataset", "Cloud")
    # get_avg_latent_for_label("logs/InfinityGAN-IOF/test/infinite_gen_197x197_dataset", "Landscape")
    # get_avg_latent_for_label("logs/InfinityGAN-IOF/test/infinite_gen_197x197_dataset", "Bridge")

