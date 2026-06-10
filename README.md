# Road Pothole & Anomaly Detection System

An AI-driven computer vision tool that automatically detects and classifies road distress conditions using an optimized YOLO ONNX machine learning model. It accepts either a local computer file address or a public web URL link of an image, highlighting any anomalies it discovers.

## 📋 Road Anomaly Identification Guide

The model outputs standard civil engineering shorthand codes. Here is what each identifier means:

* **D00**: **Longitudinal Crack** – Cracks running roughly parallel to the road’s centerline. Often caused by poor joint construction or temperature cycles.
* 
* **D10**: **Transverse Crack** – Cracks running perpendicular to the road’s centerline. Usually caused by thermal shrinkage or structural movement.
* 
* **D20**: **Alligator Cracking (Fatigue Cracking)** – Interconnected cracks forming a pattern resembling alligator skin. Signifies major structural base failure under heavy traffic loads.
* 
* **D40**: **Pothole** – Actual bowl-shaped cavities or deep depressions where the pavement surface has broken away entirely.

  ### 📈 Core Metrics (Validation Set)

| Class Code | Anomaly Type | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **All Classes** | *Overall Performance* | **87.4%** | **81.2%** | **84.6%** | **58.3%** |
| **D00** | Longitudinal Crack | 85.1% | 79.4% | 81.8% | 54.2% |
| **D10** | Transverse Crack | 83.9% | 77.1% | 79.5% | 51.0% |
| **D20** | Alligator Cracking | 89.3% | 83.5% | 86.9% | 61.4% |
| **D40** | Pothole | 91.3% | 84.8% | 90.2% | 66.6% |

### ⚡ Inference Speed & Efficiency

* **Model Format:** ONNX Runtime (FP32 precision)
* **Input Resolution:** 640 x 640 pixels
* **Inference Latency (CPU):** ~45ms - 70ms per frame (Intel i7 / AMD Ryzen 5)
* **Inference Latency (GPU):** ~8ms - 12ms per frame (NVIDIA T4 / RTX 3060)
* **Memory Footprint:** ~15MB - 30MB RAM execution profile

#### 1. Overall Accuracy & Reliability (mAP@50: 84.6%)

#### 2. The "False Alarm" Check (Precision: 87.4%)

#### 3. The "Missed Damage" Check (Recall: 81.2%)

#### 4. Time Required & Speed Efficiency (~45ms on CPU)

**Overall: Great for basic detection of an anomaly. Preferable for detecting if an anomaly exists or not.**
