import cv2
import os
import pandas as pd
from ultralytics import YOLO

# Configuration - UPDATE THESE PATHS
TEST_IMAGES_DIR = "C:/Users/LENOVO/Downloads/test"
LABELS_DIR = "C:/Users/LENOVO/Downloads/test"
MODEL_PATH = "C:/Users/LENOVO/Downloads/Programs/cancer_detection_x4/weights/best.pt"
OUTPUT_DIR = "C:/Users/LENOVO/Downloads/Documents"

def bbox_iou(box1, box2):
    """Calculate Intersection over Union for normalized boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    inter_area = max(0, min(x1+w1/2, x2+w2/2) - max(x1-w1/2, x2-w2/2)) * \
                 max(0, min(y1+h1/2, y2+h2/2) - max(y1-h1/2, y2-h2/2))
    union_area = w1*h1 + w2*h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def find_and_visualize_misses():
    # Setup output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/zoomed_misses", exist_ok=True)
    
    model = YOLO(MODEL_PATH)
    miss_data = []
    
    for img_file in os.listdir(TEST_IMAGES_DIR):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = f"{TEST_IMAGES_DIR}/{img_file}"
        label_path = f"{LABELS_DIR}/{os.path.splitext(img_file)[0]}.txt"
        
        if not os.path.exists(label_path):
            continue
            
        # Run detection
        results = model.predict(img_path, conf=0.001)[0]
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        has_miss = False
        
        # Load ground truth
        with open(label_path) as f:
            gt_boxes = [list(map(float, line.strip().split()[1:])) for line in f if len(line.strip().split()) == 5]
        
        # Check each ground truth
        for i, gt_box in enumerate(gt_boxes):
            matched = any(bbox_iou(gt_box, box.xywhn[0].cpu().numpy()) > 0.3 for box in results.boxes)
            
            if not matched:
                has_miss = True
                x, y, box_w, box_h = gt_box
                px_x, px_y = int(x*w), int(y*h)
                px_w, px_h = int(box_w*w), int(box_h*h)
                
                # Draw on image
                cv2.rectangle(img, 
                             (px_x-px_w//2, px_y-px_h//2),
                             (px_x+px_w//2, px_y+px_h//2),
                             (0, 255, 255), 2)  # Yellow
                
                # Save zoomed miss
                zoom_size = 256
                y1, y2 = max(0, px_y-zoom_size//2), min(h, px_y+zoom_size//2)
                x1, x2 = max(0, px_x-zoom_size//2), min(w, px_x+zoom_size//2)
                cv2.imwrite(f"{OUTPUT_DIR}/zoomed_misses/miss_{img_file}_{i}.jpg", img[y1:y2, x1:x2])
                
                miss_data.append({
                    'image': img_file,
                    'cell_id': i,
                    'x': x, 'y': y,
                    'width': box_w, 'height': box_h
                })
        
        # Save only if has misses
        if has_miss:
            cv2.imwrite(f"{OUTPUT_DIR}/misses_{img_file}", img)
    
    # Save report
    if miss_data:
        pd.DataFrame(miss_data).to_csv(f"{OUTPUT_DIR}/missed_detections.csv", index=False)
        print(f"Found {len(miss_data)} missed cells in {len(os.listdir(OUTPUT_DIR))-1} images")
    else:
        print("No missed detections found!")

if __name__ == "__main__":
    find_and_visualize_misses()