import numpy as np, torch, cv2
from matplotlib import pyplot as plt, cm
from torch import nn
from torchvision import models
from torchvision import transforms as T

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
    output = output.cpu().numpy().squeeze()
    segmentation_map = np.argmax(output, axis=0)
    print(np.unique(segmentation_map))
    segmentation_map = cm.tab20(segmentation_map.astype(float) / num_classes)
    return segmentation_map

def overlay_segmentation_mask(image, segmentation_map):
    # if image and segmentation_map are tensors, convert them to numpy arrays
    if torch.is_tensor(image):
        image = image.cpu().numpy()
        
    if torch.is_tensor(segmentation_map):
        segmentation_map = segmentation_map.cpu().numpy()
    segmentation_map = segmentation_map[:, :, :3]
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.astype(np.float32)
    segmentation_map = segmentation_map.astype(np.float32)
    image = cv2.addWeighted(image, 0.05, segmentation_map, 0.95, 0)
    return image

def visualize_results(image, segmentation_map):
    _, axis = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axis:
        ax.axis("off")
    axis[0].imshow(image)
    axis[0].set_title('Input Image')
    axis[1].imshow(segmentation_map)
    axis[1].set_title('Segmentation Mask')
    axis[2].imshow(overlay_segmentation_mask(image, segmentation_map))
    axis[2].set_title('Overlay')
    plt.show()
