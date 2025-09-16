#Predicting single image
import torch
from torchvision import transforms
from PIL import Image
from model import build_model
import os
os.makedirs("models", exist_ok=True)

device = torch.device('cpu')

# Load the saved model
model = build_model(model_name='resnet50', num_classes=2, pretrained=False)
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the size expected by the model
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

def predict_image(image_path):
    """
    Predict the class of a single image.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    str: Predicted class label ('No Tumor' or 'Tumor').
    """
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return 'No Tumor' if predicted.item() == 0 else 'Tumor'

print(predict_image("brain_tumor_dataset/test_sample.jpg"))