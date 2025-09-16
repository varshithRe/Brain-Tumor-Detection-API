# 🧠 Brain Tumor Detection API

This project implements a deep learning–based system for **brain tumor detection from MRI scans**. It includes training scripts, model evaluation, and an API for inference. The workflow covers model training, ONNX export, and quantization for deployment.

---

## 📊 Dataset

We used the **Brain Tumor MRI Dataset** available publicly on Kaggle:
🔗 [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

The dataset contains:

* **Yes** → MRI scans with tumors
* **No** → MRI scans without tumors

All images were resized to **224×224** before feeding into the model.

---

## 🏗️ Models and Training

### 1. **Baseline – ResNet18**

* Script: `train_od.py`
* Architecture: ResNet18
* Optimizer: Adam
* Learning rate: 0.001
* Epochs: 15
* Accuracy: **72.55%**

This gave us a baseline benchmark.

---

### 2. **Improved Model – ResNet50**

* Architecture: ResNet50 (deeper, more feature extraction capacity)
* Optimizer: Adam
* Learning rate: 0.0005
* Epochs: 20
* Accuracy: **86.43%**

By switching to ResNet50 and fine-tuning, we achieved a significant improvement in classification accuracy.

---

## 📈 Metrics

We evaluated using accuracy, precision, recall, and F1-score.

| Model    | Accuracy | Precision | Recall | F1-score |
| -------- | -------- | --------- | ------ | -------- |
| ResNet18 | 72.55%   | 71.4%     | 73.2%  | 72.2%    |
| ResNet50 | 86.43%   | 85.7%     | 87.1%  | 86.4%    |

Confusion matrices showed fewer false negatives with ResNet50, which is critical in medical diagnostics.

---

## 🔄 Model Conversion (ONNX & Quantization)

### 🔹 ONNX Export

We exported the PyTorch model to **ONNX** format for interoperability:

```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
```

ONNX allows running the model efficiently on different backends (CPU, GPU, mobile).

---

### 🔹 Quantization

We quantized the model to reduce its size and improve inference speed on CPU devices:

1. **Dynamic Quantization** – Applied to linear layers, reducing weights to int8.
2. **Resulting File** – `model_quantized.pth`
3. **Benefits**:

   * Model size reduced by \~75%
   * Inference time improved by \~2×
   * Accuracy drop was negligible (<1%)

This makes the model suitable for real-time applications and deployment in resource-limited environments.

---

## 🚀 Usage

### 1. Clone Repository

```bash
git clone https://github.com/your-username/brain_tumor.git
cd brain_tumor
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run API

```bash
uvicorn main:app --reload
```

Then open your browser at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📂 Project Structure

```
brain_tumor/
│── brain_tumor_dataset/   # dataset (ignored in Git)
│── myvenv/                # virtual environment (ignored)
│── train_od.py            # training script (ResNet18 baseline)
│── model.pth              # trained ResNet50 weights (not pushed to GitHub)
│── model.onnx             # exported ONNX model
│── model_quantized.pth    # quantized model (ignored in GitHub)
│── main.py                # FastAPI app for inference
│── requirements.txt       # dependencies
│── README.md              # this file
```

---

## ⚠️ Notes

* Large files (datasets, `.pth` models) are **not included in this repo**. Please download them from Kaggle or cloud storage.
* Only lightweight inference models (ONNX, quantized) should be deployed.

---

## 📌 Results

✅ **ResNet50 achieved 86.43% accuracy** on test data.
✅ ONNX and quantization made the model faster and smaller.
✅ API allows easy deployment for real-world usage.

---

## ✨ Future Work

* Experiment with **EfficientNet** for even better accuracy.
* Deploy to **Hugging Face Spaces** or **Docker container**.
* Add **Grad-CAM visualizations** for explainability.
