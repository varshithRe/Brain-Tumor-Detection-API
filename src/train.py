import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import build_model

# Using transform to transform the image to tensor and normalize it
transform = transforms.Compose([
    transforms.Resize((224,224)), # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


dataset = datasets.ImageFolder('brain_tumor_dataset', transform=transform) # Load dataset

# Splitting dataset into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader =  DataLoader(train_ds, batch_size=32, shuffle = True)
test_loader = DataLoader(test_ds, batch_size=32)

# Build model using the build_model function from model.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(model_name='resnet50', num_classes=2, pretrained=True)
model = model.to(device) # Move model to GPU if available


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)# Learning rate scheduler

# Training loop
epochs = 20
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
        _, predicted = torch.max(outputs, 1) # Get max value of each predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print training statistics
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.4f}")

    scheduler.step()  # Update learning rate

# Save the trained model
torch.save(model.state_dict(), 'models/best_model.pth')

# Evaluate the model on the test set
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print test set statistics
print(f"Test Accuracy: {100*correct/total:.4f}")
