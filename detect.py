import os
import urllib.request
import cv2
import numpy as np
from ultralytics import YOLO

# Define standard engineering shorthand codes
ANOMALY_MAPPING = {
    "g10": "Longitudinal Crack (Crack running parallel to the road centerline)",
    "g20": "Transverse Crack (Crack running perpendicular to the road centerline)",
    "g40": "Alligator Cracking (Advanced web-like structural fatigue cracks)",
    "g44": "Pothole (Deep structural cavity or bowl-shaped road depression)",
}

def load_image(path_or_url):
    """Loads an image from either a local absolute disk address or a web URL."""
    if path_or_url.startswith(('http://', 'https://')):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(path_or_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode the image file downloaded from the URL.")
            return img
        except Exception as e:
            raise Exception(f"Failed to download image from link: {e}")
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"No local file found at the address: {path_or_url}")
        img = cv2.imread(path_or_url)
        if img is None:
            raise ValueError("File found, but OpenCV could not read it as a valid image format.")
        return img

def run_pothole_detection(image_input, model_path="best.onnx"):
    # 1. Ensure model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing! Make sure '{model_path}' is in the same folder.")

    # 2. Load the YOLO ONNX model
    print("\nLoading model engine into memory...")
    model = YOLO(model_path, task='detect')

    # 3. Fetch image
    img = load_image(image_input)

    # 4. Perform Inference
    print("Analyzing image for road anomalies...")
    results = model(img, verbose=False)[0]

    print("\n======================================")
    print("         DETECTION RESULTS            ")
    print("======================================")

    # 5. Parse Detected Anomalies
    if len(results.boxes) == 0:
        print("No road anomalies detected. The pavement appears clear!")
    else:
        print(f"Detected {len(results.boxes)} distress marker(s):\n")
        
        for idx, box in enumerate(results.boxes):
            # Get the predicted class name/label from the model
            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower() # Usually g10, g20, g40, g44
            confidence = float(box.conf[0]) * 100

            # Map the shorthand code to human description
            description = ANOMALY_MAPPING.get(class_name, "Unknown road anomaly type")
            
            print(f"[{idx + 1}] Code Found : {class_name.upper()}")
            print(f"    Confidence : {confidence:.2f}%")
            print(f"    Definition : {description}")
            print("-" * 38)
            
        # Optional: Save a visual copy showing the boxes drawn over the image
        output_filename = "detected_anomaly.jpg"
        results.save(filename=output_filename)
        print(f"\nVisual results with drawn boxes saved as: '{output_filename}'")
        
    print("======================================\n")

if __name__ == "__main__":
    print("-------------------------------------------------")
    print("    Road Pothole & Distress Detector Console     ")
    print("-------------------------------------------------")
    user_input = input("Enter computer image address OR photo link URL: ").strip()
    
    # Strip terminal-injected quotes if user dragged and dropped local file
    user_input = user_input.strip("'\"")
    
    if not user_input:
        print("Input cannot be blank.")
    else:
        try:
            run_pothole_detection(user_input)
        except Exception as e:
            print(f"\n[ERROR]: {e}")