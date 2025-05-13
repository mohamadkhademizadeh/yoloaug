from ultralytics import YOLO

# Initialize model (using yolov8x for best accuracy)
model = YOLO("yolov8x.pt")

# Train with morphology-preserving augmentations
results = model.train(
    data="/home/student/.cache/gstreamer-1.0/YOLOv8-Medical-Imaging/dataset.yaml",
    epochs=100,
    #patience=50,  # Longer patience before early stopping
    imgsz=640,
    batch=8,
    device='0',  # Use GPU if available
    
    # Supported augmentations
    augment=True,          # Enable basic augmentations

     # Data augmentation
    hsv_h=0.01,  # More aggressive color augmentation
    degrees=45,
    scale=0.5,
    fliplr=0.7,
    mosaic=1.0,
    mixup=0.3,
    

    # Parameters that affect confidence/sensitivity
    conf=0.01,        # Direct confidence threshold for validation
    iou=0.3,         # Lower IoU for more detections
    
    # Loss weights
    box=12.0,               # Emphasize bounding box accuracy
    cls=0.1,               # Slightly reduce class focus
    dfl=3.0,
    
    # Other parameters
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.95,
    name='cancer_detection_x'
)