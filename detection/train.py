import os
from ultralytics import YOLO
import torch

CURRENT_DIR = os.getcwd()
DATA_PATH = os.path.join(CURRENT_DIR, 'Dataset')

print(CURRENT_DIR)

num_gpus = torch.cuda.device_count()

print(f"Number of available GPUs: {num_gpus}")
# load a pretrained model 
model = YOLO('yolov8m.pt')

results = model.train(data=f'{DATA_PATH}/data.yaml', 
epochs=2, batch=-1, imgsz=640, cache=False, workers=8, project="exp", 
pretrained=True, optimizer="Adam", seed=42, lr0=0.001, lrf=0.0001, plots=True)