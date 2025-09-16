from fastapi import FastAPI, UploadFile, File
from src.model import build_model
import torch
from torchvision import transforms      
from PIL import Image
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)


app = FastAPI()

# Load the pre-trained model
device = torch.device('cpu')
model = torch.jit.load("models/model_scripted.pt", map_location=device)
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert('RGB')  # Ensure image is in RGB format
    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return {'prediction': 'No Tumor' if predicted.item() == 0 else 'Tumor'}