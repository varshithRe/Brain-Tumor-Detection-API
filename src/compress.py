import torch
from model import build_model

device = torch.device('cpu')

#load the saved model
model = build_model(model_name='resnet50', num_classes=2, pretrained=False)
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))

model.eval()

# TorchScript
example = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, example)
traced.save('models/model_scripted.pt')

# Quantized
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), 'models/model_quantized.pth')

# ONNX
torch.onnx.export(model, example, 'models/model.onnx', opset_version=11)

print("Models have been successfully saved in scripted, quantized, and ONNX formats.")