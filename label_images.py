import glob
from google.cloud import vision
import io
import os
import json

def get_labels(client, image_path):
    with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
    image = vision.Image(content=content)
    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations
    # print("labels:", labels)
    return labels

def label_images(img_dir,start,stop):
    client = vision.ImageAnnotatorClient()

    images_list = sorted(glob.glob(f'{img_dir}/*.png'))

    subdir = os.path.join(img_dir,"labels")
    if (not os.path.exists(subdir)): os.makedirs(subdir)

    for image_path in images_list[start:stop]:
        labels = get_labels(client, image_path)
        label_file = os.path.join(subdir, os.path.splitext(os.path.basename(image_path))[0] + ".json")
        label_dicts = [] # Array that will contain all the EntityAnnotation dictionaries
        for label in labels:
            # Write each label (EntityAnnotation) into a dictionary
            dict = {'description': label.description, 'score': label.score, 'mid': label.mid, 'topicality': label.topicality}
            # Populate the array
            label_dicts.append(dict) 
        with open(label_file, "w") as f:
            json.dump(label_dicts, f)

if __name__ == '__main__':
    label_images("logs/InfinityGAN-IOF/test/infinite_gen_197x197_dataset", 0,900)
