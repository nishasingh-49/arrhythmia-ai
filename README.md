# ECG Arrhythmia Detection & Deployment

An end-to-end deep learning system for detecting cardiac arrhythmias from ECG signals, built using a 1D Convolutional Neural Network and deployed as a FastAPI inference service for real-time prediction.

This project demonstrates the full ML lifecycle — from preprocessing and training to deployment and API-based inference.

---

## Project Highlights

- 1D CNN for ECG time-series classification  
- Class imbalance handling using class weights  
- Achieved ~98% accuracy and ~0.99 ROC-AUC  
- Consistent preprocessing using persisted scaler  
- Real-time inference via FastAPI REST API  
- Modular pipeline for reproducibility and scalability  

---

## System Architecture

ECG Dataset → Preprocessing → 1D CNN Training → Model + Scaler
↓
FastAPI Inference API → Real-time Prediction

---

## Model Performance

| Metric | Score |
|------|------|
| Accuracy | ~98% |
| ROC-AUC | ~0.99 |
| Abnormal Recall | ~95% |
| F1 Score | ~0.93 |

---

## Repository Structure

arrhythmia-ai/<br>
│<br>
├── app.py # FastAPI inference service<br>
├── train_arrhythmia.py # Training pipeline<br>
├── dataloader.py # Dataset loading<br>
├── label_processor.py # Label preprocessing<br>
├── beat_extractor.py # ECG beat segmentation<br>
├── test_api.py # API testing script<br>
├── requirements.txt<br>
├── arrhythmia_model.keras<br>
└── scaler.pkl

---

## Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/arrhythmia-ai.git
cd arrhythmia-ai
```
### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```
```bash
source venv/bin/activate  # Mac/Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## Sample API request

```bash
{
  "signal": [0.1, 0.2, 0.15, ... 187 values]
}
```
## Sample Response

```bash
{
  "prediction": "Normal",
  "confidence": 0.23
}

```

