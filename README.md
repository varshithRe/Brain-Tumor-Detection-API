# Brain Tumor Detection API

This project is a deep learning-based API for brain tumor detection from MRI scans. I built it using PyTorch and FastAPI, and experimented with multiple architectures to improve performance and deployment readiness.

---

## Dataset

The dataset I used is the [**Brain Tumor MRI Dataset**](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) from Kaggle, which contains MRI images of brain scans labeled into tumor and non-tumor categories. 

---

## Training Results

I trained two models to compare performance:

* **ResNet18** → Accuracy: **72.55%**
* **ResNet50** → Accuracy: **88.43%** (best performance)

---

## Model Export and Optimization

To prepare the model for deployment, I exported it into multiple formats:

* **TorchScript (`model_scripted.pt`)** – for PyTorch serving.
* **ONNX (`model.onnx`)** – for interoperability with other frameworks and optimized inference.
* **Quantized Model (`model_quantized.pth`)** – attempted to reduce model size.

### Notes on Model Size

I tried quantizing the ResNet50 model to make it smaller and faster. Normally, quantization reduces model size by converting 32-bit weights to 8-bit. In my case, the model size did not shrink and even grew slightly (\~+1 MB). This happened because not all layers were quantized and extra metadata was added during saving.

---

## API Usage

The FastAPI backend allows MRI images to be uploaded and returns predictions on whether a brain tumor is present.

### Running the API

```bash
uvicorn app:app --reload
```

### Swagger UI

Once the server is running, open your browser at:

```
http://127.0.0.1:8000/docs
```

You can upload MRI images directly from the Swagger interface and view predictions.

### Example Request (cURL)

```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example_mri.jpg"
```

### Example Response

```json
{
  "prediction": "Tumor Detected"
}
```

---

## Future Work

* Explore **model pruning** and **distillation** to reduce size.
* Optimize the **ONNX export** with tools like `onnxruntime` or `TensorRT`.
* Improve accuracy further with more data and advanced architectures.


