import os
import json
import cv2
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import base64
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
import logging

# Configure logging
logger = logging.getLogger('detection')

# Global variable to store the loaded model
loaded_model = None

def load_model():
    """Load the trained RCNN model"""
    global loaded_model
    if loaded_model is None:
        model_path = os.path.join(settings.MEDIA_ROOT, 'improved_rcnn_model.h5')
        print(f"Loading model from {model_path}")
        if os.path.exists(model_path):
            loaded_model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")
        else:
            print(f"Model not found at {model_path}")
    return loaded_model

def compute_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

def non_max_suppression(boxes, scores, iou_thresh=0.3, max_output_size=50):
    """Apply Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []

    boxes_tf = tf.convert_to_tensor(boxes, dtype=tf.float32)
    scores_tf = tf.convert_to_tensor(scores, dtype=tf.float32)

    selected = tf.image.non_max_suppression(
        boxes_tf, scores_tf,
        max_output_size=max_output_size,
        iou_threshold=iou_thresh
    )
    return selected.numpy()

def generate_region_proposals(image, max_proposals=1000):
    """Generate region proposals using Selective Search"""
    try:
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()

        rects = ss.process()
        proposals = []

        for x, y, w, h in rects[:max_proposals]:
            if w < 20 or h < 20:  # Filter very small regions
                continue

            proposal_box = [x, y, x + w, y + h]
            proposals.append((proposal_box, 0, 0))

        return proposals
    except Exception as e:
        print(f"Error in region proposal generation: {e}")
        return []

def predict_objects(model, image, confidence_threshold=0.3, nms_threshold=0.3):
    """Predict objects in an image using trained RCNN model"""
    
    # Categories from your training (you may need to adjust this based on your dataset)
    categories = {0: 'background', 1: 'object1', 2: 'object2', 3: 'object3'}  # Update with your actual classes
    
    # Generate proposals
    proposals = generate_region_proposals(image, max_proposals=1000)

    if not proposals:
        return [], [], []

    # Prepare ROIs
    rois = []
    proposal_boxes = []

    for prop_box, _, _ in proposals:
        x1, y1, x2, y2 = prop_box

        # Ensure valid coordinates
        x1 = max(0, min(x1, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        x2 = max(x1 + 1, min(x2, image.shape[1]))
        y2 = max(y1 + 1, min(y2, image.shape[0]))

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        roi_resized = cv2.resize(roi, (224, 224))
        roi_preprocessed = preprocess_input(roi_resized.astype(np.float32))

        rois.append(roi_preprocessed)
        proposal_boxes.append(prop_box)

    if not rois:
        return [], [], []

    rois = np.array(rois)

    # Make predictions
    cls_preds, bbox_preds = model.predict(rois, verbose=0)

    # Process predictions
    predicted_classes = np.argmax(cls_preds, axis=1)
    confidence_scores = np.max(cls_preds, axis=1)

    # Filter out background predictions and low confidence
    valid_indices = []
    for i in range(len(predicted_classes)):
        if predicted_classes[i] > 0 and confidence_scores[i] > confidence_threshold:
            valid_indices.append(i)

    if not valid_indices:
        return [], [], []

    # Apply NMS
    valid_boxes = np.array(proposal_boxes)[valid_indices]
    valid_scores = confidence_scores[valid_indices]
    valid_classes = predicted_classes[valid_indices]

    if len(valid_boxes) > 0:
        nms_indices = non_max_suppression(valid_boxes, valid_scores, nms_threshold)

        final_boxes = valid_boxes[nms_indices]
        final_scores = valid_scores[nms_indices]
        final_classes = valid_classes[nms_indices]

        return final_boxes.tolist(), final_classes.tolist(), final_scores.tolist()

    return [], [], []

def draw_boxes(image, boxes, labels=None, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on image"""
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if labels is not None:
            label_text = str(labels[i])
            cv2.putText(img, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 1, cv2.LINE_AA)
    return img

def index(request):
    """Render the main page"""
    return render(request, 'detection/index.html')

@csrf_exempt
def process_image(request):
    """Process uploaded image and return detection results"""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Load model if not already loaded
            model = load_model()
            if model is None:
                logger.error("Model not found at %s", os.path.join(settings.MEDIA_ROOT, 'improved_rcnn_model.h5'))
                return JsonResponse({'error': 'Model not found. Please ensure improved_rcnn_model.h5 is in the media folder.'}, status=500)
            
            # Get uploaded image
            uploaded_file = request.FILES['image']
            logger.info("Received image: %s", uploaded_file.name)
            
            # Save uploaded image temporarily
            file_path = default_storage.save(f'temp/{uploaded_file.name}', ContentFile(uploaded_file.read()))
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)
            logger.info("Saved image to: %s", full_path)
            
            # Read and process image
            image = cv2.imread(full_path)
            if image is None:
                logger.error("Failed to read image: %s", full_path)
                return JsonResponse({'error': 'Could not read image file'}, status=400)
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info("Image shape: %s", image.shape)
            
            # Make predictions
            pred_boxes, pred_classes, pred_scores = predict_objects(model, image_rgb, confidence_threshold=0.2)
            logger.info("Detected %d objects", len(pred_boxes))
            
            # Categories mapping (update with your actual classes)
            categories = {0: 'background', 1: 'object1', 2: 'object2', 3: 'object3'}
            
            # Create labels
            pred_labels = [f"{categories.get(cls, f'Class {cls}')} ({score:.2f})"
                          for cls, score in zip(pred_classes, pred_scores)]
            
            # Draw bounding boxes
            result_image = draw_boxes(image_rgb, pred_boxes, pred_labels, color=(0, 255, 0))
            
            # Convert result to base64
            result_pil = Image.fromarray(result_image)
            buffered = BytesIO()
            result_pil.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Clean up temporary file
            default_storage.delete(file_path)
            logger.info("Deleted temporary file: %s", file_path)
            
            # Prepare detection results
            detections = []
            for i, (box, cls, score) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
                detections.append({
                    'class': categories.get(cls, f'Class {cls}'),
                    'confidence': float(score),
                    'bbox': box
                })
            
            return JsonResponse({
                'success': True,
                'result_image': f'data:image/jpeg;base64,{img_str}',
                'detections': detections,
                'total_objects': len(pred_boxes)
            })
            
        except Exception as e:
            logger.exception("Error processing image: %s", str(e))
            return JsonResponse({'error': f'Processing error: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)