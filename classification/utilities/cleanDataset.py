import os
from PIL import Image

def delete_unopenable_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except (IOError, SyntaxError) as e:
                print(f"Deleting unopenable image: {file_path}")
                os.remove(file_path)

def rename_folders(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            new_name = item.capitalize()
            new_path = os.path.join(directory, new_name)
            if new_name != item:  # Rename only if the new name is different
                os.rename(item_path, new_path)
                print(f"Renamed '{item_path}' to '{new_path}'")

def process_dataset(directory):
    delete_unopenable_images(directory)
    rename_folders(directory)

# Specify the path to dataset directory
dataset_directory = 'Data:Logo2K'

# Call the main function to process the dataset
process_dataset(dataset_directory)
