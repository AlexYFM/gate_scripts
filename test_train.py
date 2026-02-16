from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
# Test the formatting with a minimal run
results = model.train(
    data="dataset.yaml", 
    epochs=1,           # Just one pass
    imgsz=640, 
    batch=1,            # Small batch for speed
    plots=True          # Crucial: This generates 'train_batch0.jpg' showing your labels
)