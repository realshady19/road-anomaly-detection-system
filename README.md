# Road Pothole & Anomaly Detection System

An AI-driven computer vision tool that automatically detects and classifies road distress conditions using an optimized YOLO ONNX machine learning model (`best.onnx`). It accepts either a local computer file address or a public web URL link of an image, highlighting any anomalies it discovers.

## 📋 Road Anomaly Identification Guide

The model outputs standard civil engineering shorthand codes. Here is what each identifier means:

* **G10**: **Longitudinal Crack** – Cracks running roughly parallel to the road’s centerline. Often caused by poor joint construction or temperature cycles.
* 
* **G20**: **Transverse Crack** – Cracks running perpendicular to the road’s centerline. Usually caused by thermal shrinkage or structural movement.
* 
* **G40**: **Alligator Cracking (Fatigue Cracking)** – Interconnected cracks forming a pattern resembling alligator skin. Signifies major structural base failure under heavy traffic loads.
* 
* **G44**: **Pothole** – Actual bowl-shaped cavities or deep depressions where the pavement surface has broken away entirely.

