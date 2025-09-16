import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


#using tranform to transform the image to tensor and normalize it
transform = transforms.Compose([
    transforms.Resize((128,128)), # Resize images to 128x128
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = datasets.ImageFolder('brain_tumor_dataset', transform=transform)

#Splitting dataset into traing, validation and test sets
train_size = int(0.8* len(dataset))
val_size =  len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_ds, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = 32)


# Load a pre-trained model and modify it for binary classification
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

# Move model to GPU if available
device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss funxtion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) # Move data to GPU if available

        optimizer.zero_grad() # Zero the gradients
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels)
        loss.backward() # Backpropagation
        optimizer.step() # Update weights

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1) #Get max value of each predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    
    acc = 100* correct/total
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%')


# Evaluate on validation set
model.eval()
correct, total = 0,0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")