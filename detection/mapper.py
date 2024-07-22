import os
import fiftyone as fo
import yaml
from skimage import io, transform
import matplotlib.pyplot as plt 
import numpy as np
import re


def normalize_brands(brand_list):
    normalized = []
    for brand in brand_list:
            # Umwandlung in Kleinbuchstaben
        brand = brand.lower()
            # Entfernen aller Wörter nach dem ersten
            #clean_brand = re.sub(r'\s.*', '', brand)
        normalized.append(brand)
    return normalized

def get_result_dict():
    CURRENT_DIR = os.getcwd()
    DATA_PATH = os.path.join(CURRENT_DIR, 'Dataset')
    yaml_file = os.path.join(DATA_PATH, "data.yaml")
    print(CURRENT_DIR) 

    ds = fo.load_dataset("sellpy-test")

    #limit = 10000
    # get all ids
    ds_temp = ds.match(fo.ViewField("Grounding_Dino") !=  None)
    print(ds_temp)
    ids = ds_temp.values("id")
    # check weather the sample contains a logo
    contains_logo = ["logo" in values for values in ds_temp.values("Grounding_Dino.detections.label")]
    # get ids of sample containing a logo 
    ids_contains_logo = np.array(ids)[contains_logo]
    # filter the data set to contain only samples with a logo found
    ds_with_logos = ds[ids_contains_logo]

    all_brands = ds_with_logos.count_values('brand.label')
    total_fiftyone_images = sum(all_brands.values())
    all_classes_in_fiftyone = list(all_brands.keys())
    with open(yaml_file, 'r') as file:
        yamlFile = yaml.safe_load(file)

    all_classes_in_yolo = (yamlFile['names'])


    all_classes_in_fiftyone_normalized = normalize_brands(all_classes_in_fiftyone)
    all_classes_in_yolo_normalized = normalize_brands(all_classes_in_yolo)
    matches = set(all_classes_in_yolo_normalized).intersection(set(all_classes_in_fiftyone_normalized))

    print(f'Total fiftyone Brands: {len(all_classes_in_fiftyone)}')
    print(f'Total fiftyone images: {total_fiftyone_images}')

    print(f'Total Logo3K Brands: {len(all_classes_in_yolo)}')

    # Check if there are any matches
    if matches:
        print("Total Matches found:", len(matches))
    else:
        print("No matches found.")
    model_fiftyone_labels_map = {all_classes_in_fiftyone[all_classes_in_fiftyone_normalized.index(b)]:all_classes_in_yolo_normalized.index(b) for b in matches}

    print('...............................................')
    idx = [all_classes_in_fiftyone_normalized.index(b) for b in matches]

    all_brands[all_classes_in_fiftyone[3268]]
    result_dict = {all_classes_in_fiftyone[key]:all_brands[all_classes_in_fiftyone[key]] for key in idx}
    print(result_dict)
    print('...............................................')
    print(f'Total number of images for matches: {sum(result_dict.values())}')
    print('...............................................')
    return result_dict