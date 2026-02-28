# AI for Sudan – Smart Medical Assistant (PyTorch Powered)

A lightweight, explainable deep learning project for **pneumonia detection from chest X-rays**, designed for **low-resource healthcare environments**.

## Project Overview
This project provides a practical medical AI pipeline focused on accessibility, efficiency, and interpretability:
- Binary classification: **Pneumonia vs Normal**
- Two model options:
  - **Option A:** Custom lightweight CNN
  - **Option B:** Transfer learning with **MobileNetV2**
- Explainability using **Grad-CAM** (implemented from scratch)
- Deployment-friendly optimization via **dynamic quantization**

## Problem Statement
Many low-resource regions, including parts of Sudan, face challenges in timely radiology access. This project explores how AI can support frontline medical teams by offering rapid triage signals for pneumonia from chest X-rays. It is designed to run efficiently and provide visual reasoning support through heatmaps.

## Project Structure
```text
ai-for-sudan/
│
├── models/
│   └── model.py
├── train.py
├── evaluate.py
├── gradcam.py
├── inference.py
├── utils.py
├── requirements.txt
└── README.md
```

## Dataset Assumption
```text
dataset/
    train/
        pneumonia/
        normal/
    val/
        pneumonia/
        normal/
```

## Model Architecture
### Option A: Custom Lightweight CNN
- Multiple `Conv2d + BatchNorm + ReLU` blocks
- Max pooling for spatial reduction
- Dropout in feature extractor and classifier
- Adaptive average pooling for compact representation
- Fully connected head for binary classification
- Explicit weight initialization:
  - Kaiming for conv
  - Xavier for linear
  - BatchNorm scale/bias initialization

### Option B: MobileNetV2 Transfer Learning
- Uses torchvision MobileNetV2 backbone
- Replaces classifier head with:
  - Dropout
  - Linear layer to 2 classes
- Supports pretrained ImageNet weights

## Training Pipeline
- Modular train/validation epoch loop
- GPU support if available (`cuda` fallback to `cpu`)
- `tqdm` progress bars
- Per-epoch metrics logging:
  - Loss
  - Accuracy
  - Precision
  - Recall
  - F1-score
- `ReduceLROnPlateau` scheduler
- Early stopping on validation loss
- Best checkpoint saving

## Grad-CAM Explainability
`gradcam.py` implements Grad-CAM without external explainability libraries:
- Forward hook captures target layer activations
- Backward hook captures gradients
- Channel-wise gradient pooling creates attention weights
- Weighted activation map + ReLU generates heatmap
- Heatmap is resized and overlaid on original X-ray
- Saved to disk for inspection

## Quantization
The training script includes dynamic quantization for model compression:
- Applies `torch.quantization.quantize_dynamic` to linear layers
- Saves quantized weights separately
- Reports model size before and after quantization

## Installation
```bash
cd ai-for-sudan
pip install -r requirements.txt
```

## Training
### Custom CNN
```bash
python train.py \
  --data_dir dataset \
  --model_type custom_cnn \
  --epochs 20 \
  --batch_size 32 \
  --output_dir outputs
```

### MobileNetV2 Transfer Learning
```bash
python train.py \
  --data_dir dataset \
  --model_type mobilenet_v2 \
  --pretrained \
  --epochs 20 \
  --batch_size 32 \
  --output_dir outputs
```

## Evaluation
```bash
python evaluate.py \
  --data_dir dataset \
  --checkpoint outputs/checkpoints/best_model.pth \
  --model_type custom_cnn
```

## Inference + Grad-CAM
```bash
python inference.py \
  --image dataset/val/pneumonia/sample_xray.jpg \
  --checkpoint outputs/checkpoints/best_model.pth \
  --model_type custom_cnn \
  --output outputs/gradcam_inference.png
```

Inference output includes:
- Predicted class
- Confidence score
- Saved Grad-CAM overlay image

## Example Results
Add your run-specific outcomes here:
- Validation Accuracy: `XX.XX%`
- Validation Precision: `XX.XX%`
- Validation Recall: `XX.XX%`
- Validation F1-score: `XX.XX%`
- Quantization size reduction: `X.XX MB -> X.XX MB`

## Future Improvements
- Add k-fold cross-validation for robust estimates
- Extend to multi-class thoracic pathology detection
- Add confidence calibration for safer decision support
- Export optimized models to TorchScript/ONNX
- Integrate offline-first mobile inference app

## Ethical Considerations in Medical AI
- This system is a **decision-support tool**, not a clinical diagnosis replacement.
- Performance can vary across demographics, imaging devices, and hospitals.
- Dataset quality and representativeness strongly impact fairness and safety.
- Human clinician oversight is mandatory before medical action.
- Continuous auditing and revalidation are required in real deployments.

## Reflection
### Knowledge required
- PyTorch modeling fundamentals
- Transfer learning workflows
- Reliable training/validation engineering
- Metric design for medical binary classification
- Basic model deployment optimization concepts

### New concepts learned
- Building Grad-CAM from scratch with hooks and gradient-weighted activations
- Applying dynamic quantization for lightweight deployment
- Balancing explainability and efficiency for low-resource healthcare contexts
- Structuring hackathon-ready, production-style deep learning codebases
