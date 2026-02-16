from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)

# 1. Test the formatting with a minimal run
results = model.train(
    data="dataset.yaml", 
    epochs=1,           # Just one pass
    imgsz=640, 
    batch=1,            # Small batch for speed
    plots=True          # Crucial: This generates 'train_batch0.jpg' showing your labels
)

# 2. Production Training Run
# results = model.train(
#     data="dataset.yaml",
#     epochs=100,           # 100 is a good start; YOLO has early stopping by default
#     imgsz=640,            # Match the resolution used during generation
#     batch=16,             # Increase based on your GPU VRAM (16 or 32 is standard)
#     workers=8,            # Number of CPU threads for data loading
#     device=0,             # Use 0 for first GPU, or 'cpu' if no GPU available
#     project="gate_training",
#     name="v1_initial_run",
#     save=True,            # Save checkpoints
#     plots=True,           # Generates PR curves and val_batch examples
#     patience=20,          # Stop early if no improvement for 20 epochs
#     lr0=0.01,             # Initial learning rate
#     augment=True          # Use built-in augmentations (flips, color jitters)
# )
