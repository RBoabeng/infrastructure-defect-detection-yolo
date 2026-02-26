# Infrastructure Defect Detection using YOLOv8

This project implements an end-to-end Machine Learning Computer Vision pipeline to detect and classify structural defects in infrastructure (such as concrete cracks, spalling, and delamination) using deep learning.

## 📌 Project Overview
Monitoring infrastructure health is a highly manual and time-consuming process. This project leverages the YOLOv8 object detection model to automate the identification of structural anomalies. The model was trained on an academic computing cluster to rapidly process high-resolution imagery.

### Key Features:
* **Automated Data Pipeline:** Securely downloads and extracts dataset from Roboflow via API.
* **Programmatic Label Correction:** Automatically maps and fixes corrupted/default YOLO class labels.
* **High-Speed Inference:** Achieves ~80ms inference times per image on standard CPU hardware (12+ FPS).

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```
   git clone [https://github.com/yourusername/infrastructure-defect-detection-yolo.git](https://github.com/yourusername/infrastructure-defect-detection-yolo.git)
   cd infrastructure-defect-detection-yolo

   ```

2. **Set up the virtual environment:**

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Download the Data:**
Create a `.env` file in the root directory and add your Roboflow API key:
`ROBOFLOW_API_KEY=your_key_here`

Then run the download script:

```
python download_data.py
```

## 📊 Dataset & Discoveries
The model was trained on the "Infrastructure-Defects-Detection-4" dataset. During exploratory data analysis and validation, a severe class imbalance was identified.

The validation set was heavily skewed, containing exclusively `spalling` defects. Consequently, the model exhibits a bias toward predicting `spalling` in uncertain inference scenarios. This highlights the critical real-world MLOps necessity for strict data curation and balanced validation splits prior to production deployment.

## 🚀 Results
The trained custom weights (`yolov8_infrastructure_defects.pt`) are available in the `assets/` folder.

**Normalized Confusion Matrix:**

**Training Loss Results:**

## 💻 Running Inference
You can test the trained model on new images using the provided weights:

```
from ultralytics import YOLO

# Load the custom trained model
model = YOLO('assets/yolov8_infrastructure_defects.pt')

# Run inference
results = model('path/to/test/image.jpg')
results[0].show()
```