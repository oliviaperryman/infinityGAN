import glob
import json



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

if __name__ == '__main__':
    group_images("logs/InfinityGAN-IOF/test/infinite_gen_197x197_dataset", 0,900)
