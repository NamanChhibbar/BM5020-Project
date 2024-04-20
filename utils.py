import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt, cm
from torch import nn
import torchvision.models as models
import torchvision.transforms as T

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# defining the model
class Resnest(nn.Module):
    def __init__(self, model_name='resnest101e'):
        super().__init__()
        self.model = models.resnet101(weights = models.ResNet101_Weights.IMAGENET1K_V2)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Identity(n_features)

    def forward(self, x):
        x = self.model(x)
        x = x.unsqueeze(2)
        x = x.unsqueeze(2)
        return {"out": x, "aux": x} 
    
# define the preprocessing function
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(image / 255)
    return image

def postprocess_output(output, num_classes):
    output = output.detach().numpy().squeeze()
    segmentation_map = np.argmax(output, axis=0)
    print(np.unique(segmentation_map))
    segmentation_map = cm.tab20(segmentation_map.astype(float) / num_classes)
    return segmentation_map

def visualize_results(image, masks, input_boxes=None):
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(image)
    ax[0].set_title('Input Image')
    
    ax[1].set_title('Segmented Image')
    ax[1].imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), ax[1], random_color=True)
    if input_boxes is not None:
        for box in input_boxes:
            show_box(+box.cpu().numpy(), plt.gca())
    plt.axis('off')
    plt.show()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def mIoU(predicted_mask, true_mask):
    intersection = np.logical_and(predicted_mask, true_mask)
    union = np.logical_or(predicted_mask, true_mask)
    return np.sum(intersection) / np.sum(union)

def F1_score(predicted_mask, true_mask):
    intersection = np.logical_and(predicted_mask, true_mask)
    precision = np.sum(intersection) / np.sum(predicted_mask)
    recall = np.sum(intersection) / np.sum(true_mask)
    return 2 * precision * recall / (precision + recall)

