import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import pandas as pd
import seaborn as sn
import random
import os




def plot_results(history, target_dir, title):
    best_accuracy = history['best_accuracy']
    best_f1 = history['best_f1_score']
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot for Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['valid_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{title} - Loss')
    plt.savefig(f'{target_dir}/{title}_Loss.png')
    plt.show()

    # Plot for Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs, history['valid_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.axhline(y=best_accuracy, color='r', linestyle='--', label=f'Best Accuracy: {best_accuracy:.2f}%')
    plt.legend()
    plt.title(f'{title} - Accuracy')
    plt.savefig(f'{target_dir}/{title}_Accuracy.png')
    plt.show()

    # Plot for F1 Score
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_weighted_f1'], label='Training F1 Score')
    plt.plot(epochs, history['valid_weighted_f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.axhline(y=best_f1, color='r', linestyle='--', label=f'Best F1 Score: {best_f1:.2f}%')
    plt.legend()
    plt.title(f'{title} - F1 Score')
    plt.savefig(f'{target_dir}/{title}_F1_score.png')
    plt.show()


def plot_lr(learning_rates: list, target_dir:str):
    epochs = list(range(len(learning_rates)))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{target_dir}/Learning_Rate_Schedule.png')
    plt.show()

def plot_confusion_matrix(y_true: list, y_pred:list, class_names:list, target_dir:str):
        # Plot Confusion Matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index = range(len(class_names)), columns = range(len(class_names)))
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.title("Confusion Matrix for logos classification ")
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.savefig(f'{target_dir}/Confusion_Matrix.png')
    plt.show()
    print('-' * 80)
    

def get_classification_report(y_true: list, y_pred:list, target_dir:str):
    classificationReport = classification_report(y_true, y_pred)
    with open(f'{target_dir}/classificationReport.txt', 'w') as file:
        file.write(f'Classification report:\n {classificationReport}')
        file.close() 

def plot_img_before_after_transformation(train_dataset, transforms, class_names, target_dir):
    fig = plt.figure(figsize=(15, 9))
    rows, cols = 2, 4
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    for i in range(4):
        inx = random.randint(0, len(train_dataset))
        img, label = train_dataset[i]
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img) 
        ax.set_title(f"{class_names[label]} \nSize: {img.size}", color='white', backgroundcolor='green')
        ax.axis("off")

                # Apply transformation and plot transformed image
        transformed_image = transforms(img).permute(1, 2, 0)
        transformed_image = std * transformed_image + mean  # Denormalize
        transformed_image = np.clip(transformed_image.numpy(), 0, 1) 
        
        ax = fig.add_subplot(rows, cols, i + 5)
        ax.imshow(transformed_image)
        h, w, c = transformed_image.shape
        ax.set_title(f"{class_names[label]}\nSize: {(h, w)}", color='white', backgroundcolor='green')
        ax.axis("off")

        fig.text(0.5, 0.92, 'Original Images', ha='center', va='center', fontsize=16,  color='black')
        fig.text(0.5, 0.50, 'Transformed Images', ha='center', va='center', fontsize=16,  color='black')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    plt.savefig(os.path.join(target_dir, 'img_before_and_after_transformation.png'))
    plt.show()

def plot_decision_boundary(model, images, labels):
    """Plot decision boundaries using actual images from a dataset."""
    model.to("cpu")
    model.eval()

    # Assume images are already tensors of the correct shape [batch_size, channels, height, width]
    # Forward pass through the model
    with torch.no_grad():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        _, predictions = torch.max(probabilities, axis=1)

    # Reduce dimensions of images for plotting (using PCA to 2D)
    pca = PCA(n_components=2)
    images_flattened = images.view(images.shape[0], -1)  # Flatten images
    images_2d = pca.fit_transform(images_flattened.cpu().numpy())

    # Plot
    plt.figure(figsize=(10, 8))
    for i in range(len(torch.unique(labels))):
        idx = labels == i
        plt.scatter(images_2d[idx, 0], images_2d[idx, 1], label=f'Class {i}', alpha=0.5)
    plt.scatter(images_2d[predictions != labels, 0], images_2d[predictions != labels, 1], color='red', label='Misclassified', alpha=0.5)
    plt.legend()
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Decision Boundary Visualization')
    plt.show()


def plot_results_2(history, target_dir, title):
    best_accuracy = np.max(history['valid_acc'])
    best_f1 = np.max(history['valid_weighted_f1'])
    plt.figure(figsize=(18, 6))  # Adjusted for better spacing with three plots

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['valid_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['valid_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.axhline(y=best_accuracy, color='r', linestyle='--', label=f'Best Accuracy: {best_accuracy:.2f}')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['train_weighted_f1'], label='Training F1 Score')
    plt.plot(history['valid_weighted_f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.axhline(y=best_f1, color='r', linestyle='--', label=f'Best F1 Score: {best_f1:.2f}')
    plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to leave space for suptitle
    plt.savefig(f'{target_dir}/statistics-{title}.png')
    plt.show
    

def plot_results_inOne(history, target_dir, title):
    best_accuracy = np.max(history['valid_acc'])
    best_f1 = np.max(history['valid_weighted_f1'])
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(18, 6))  # Adjusted for better spacing with three plots

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['valid_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.yscale('log')  # Apply logarithmic scale to the y-axis
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs, history['valid_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.axhline(y=best_accuracy, color='r', linestyle='--', label=f'Best Accuracy: {best_accuracy:.2f}')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_weighted_f1'], label='Training F1 Score')
    plt.plot(epochs, history['valid_weighted_f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.axhline(y=best_f1, color='r', linestyle='--', label=f'Best F1 Score: {best_f1:.2f}')
    plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to leave space for suptitle
    plt.savefig(f'{target_dir}/statistics-{title}.png')
    plt.show()
