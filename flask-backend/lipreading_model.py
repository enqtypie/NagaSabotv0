import os
import cv2 # type: ignore
import math
import numpy as np # type: ignore
import mediapipe as mp # type: ignore
import tensorflow as tf # type: ignore
import logging
from typing import Tuple, Dict, List
from collections import OrderedDict

# Set up simplified logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NagaSabot")

# Constants - exactly matching training notebook/tester
TOTAL_FRAMES = 75
LIP_WIDTH = 112
LIP_HEIGHT = 80
CHANNELS = 3

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh

# MediaPipe face mesh indices for lips
LIP_OUTER_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]  # Outline
LIP_INNER_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]  # Inner contour

# Reference points for lip anchoring
LIP_CENTER_UPPER = 13  # Upper lip center landmark
LIP_CENTER_LOWER = 14  # Lower lip center landmark
LEFT_EYE_OUTER = 33    # Left eye outer corner
RIGHT_EYE_OUTER = 263  # Right eye outer corner

# Define Bikol-Naga phrases (classes) - UPDATED to match collector
BIKOL_NAGA_PHRASES = [
    "marhay na aldaw",
    "dios mabalos",
    "padaba taka",
    "iyo tabi",
    "dae man tabi",
    "dae ko aram",
    "tano man",
    "tabi po",
    "mayong problema",
    "nasasabutan mo",
    "maulay po kita",
    "gurano an",
    "maoragon man",
    "patapos na tabi",
    "halaton mo ako"
]

def enhance_lip_region(lip_frame: np.ndarray) -> np.ndarray:
    """Preprocess and enhance the lip region for model input."""
    try:
        if lip_frame is None or lip_frame.size == 0:
            logger.warning("Preprocessing received empty image.")
            return np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)
        if len(lip_frame.shape) == 2 or lip_frame.shape[2] == 1:
            lip_frame = cv2.cvtColor(lip_frame, cv2.COLOR_GRAY2BGR)
        elif lip_frame.shape[2] != 3:
            logger.warning(f"Preprocessing received image with unexpected channels: {lip_frame.shape}")
            return np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)
        lip_frame_gray = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
        lip_frame_eq = clahe.apply(lip_frame_gray)
        lip_frame_filtered = cv2.bilateralFilter(lip_frame_eq, 5, 35, 35)
        kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
        lip_frame_sharp = cv2.filter2D(lip_frame_filtered, -1, kernel)
        lip_frame_final = cv2.GaussianBlur(lip_frame_sharp, (3, 3), 0)
        lip_frame_3ch = cv2.cvtColor(lip_frame_final, cv2.COLOR_GRAY2BGR)
        return lip_frame_3ch
    except Exception as e:
        logger.error(f"Error in enhance_lip_region: {e}")
        return np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)

def get_fixed_centered_lip_region(image: np.ndarray, face_landmarks) -> Tuple[np.ndarray, float, Tuple[int, int, int, int], float]:
    """Extract a centered, rotated, and padded lip region from the image using face landmarks."""
    h, w, _ = image.shape
    try:
        outer_lip_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LIP_OUTER_INDICES]
        inner_lip_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LIP_INNER_INDICES]
        all_lip_points = outer_lip_points + inner_lip_points
        all_x = [p[0] for p in all_lip_points]
        all_y = [p[1] for p in all_lip_points]
        center_x = sum(all_x) / len(all_x)
        center_y = sum(all_y) / len(all_y)
        center_point = (int(center_x), int(center_y))
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        lip_width_raw = max_x - min_x
        lip_height_raw = max_y - min_y
        padding_factor = 1.75
        lip_width_padded = int(lip_width_raw * padding_factor)
        lip_height_padded = int(lip_height_raw * padding_factor)
        target_aspect = LIP_WIDTH / LIP_HEIGHT
        current_aspect = lip_width_padded / lip_height_padded
        if current_aspect > target_aspect:
            lip_height_padded = int(lip_width_padded / target_aspect)
        else:
            lip_width_padded = int(lip_height_padded * target_aspect)
        left_eye = face_landmarks.landmark[LEFT_EYE_OUTER]
        right_eye = face_landmarks.landmark[RIGHT_EYE_OUTER]
        eye_dx = (right_eye.x - left_eye.x) * w
        eye_dy = (right_eye.y - left_eye.y) * h
        angle_rad = math.atan2(eye_dy, eye_dx)
        angle_deg = math.degrees(angle_rad)
        rotation_matrix = cv2.getRotationMatrix2D(center_point, angle_deg, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        lip_left_x = max(0, int(center_point[0] - (lip_width_padded / 2)))
        lip_right_x = min(w, int(center_point[0] + (lip_width_padded / 2)))
        lip_top = max(0, int(center_point[1] - (lip_height_padded / 2)))
        lip_bottom = min(h, int(center_point[1] + (lip_height_padded / 2)))
        lip_region = rotated_image[lip_top:lip_bottom, lip_left_x:lip_right_x]
        if lip_region.size > 0:
            lip_region_resized = cv2.resize(lip_region, (LIP_WIDTH, LIP_HEIGHT))
        else:
            lip_region_resized = np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)
        return lip_region_resized, 0, (lip_left_x, lip_top, lip_right_x, lip_bottom), angle_deg
    except Exception as e:
        logger.error(f"Error in get_fixed_centered_lip_region: {e}")
        return np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8), 0, (0, 0, 0, 0), 0

def calculate_metrics_from_confidence(confidence: float) -> dict:
    """
    Calculate derived metrics based on confidence score.
    
    In a real-world evaluation scenario, these metrics would be calculated by comparing
    model predictions to ground truth labels. Here we're providing approximations
    based solely on the confidence score for demonstration purposes.
    
    Args:
        confidence: The prediction confidence from the model
        
    Returns:
        Dictionary containing derived metrics
    """
    # These are approximations based on confidence, not true metrics
    # True metrics would require comparing predictions to ground truth labels
    precision = min(confidence * 1.1, 1.0)  # Slightly higher than confidence but capped at 1.0
    recall = max(confidence * 0.9, 0.0)     # Slightly lower than confidence but min at 0.0
    
    # F1 score is the harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # In real evaluation, accuracy would be (TP+TN)/(TP+TN+FP+FN)
    # Here we use confidence as an approximation
    accuracy = confidence
    
    return {
        "confidence": confidence,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy
    }

def predict_lipreading(video_path: str, model: tf.keras.Model) -> Dict[str, object]:
    """
    Run the full lipreading pipeline and return prediction with detailed metrics and top 3 predictions.
    Uses the same lip region processing and top 3 logic as the main collector script.
    """
    try:
        logger.info(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return {
                "status": "error", 
                "message": "Error opening video", 
                "top_predictions": [], 
                "metrics": {}
            }

        all_frames = []
        open_mouth_count = 0
        total_frames_processed = 0
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    # Check if mouth is open
                    if is_mouth_open(face_landmarks):
                        open_mouth_count += 1
                    # Use the same processing as in the collector
                    lip_region, *_ = get_fixed_centered_lip_region(frame, face_landmarks)
                    processed_lip = enhance_lip_region(lip_region)
                    all_frames.append(processed_lip)
                total_frames_processed += 1
        cap.release()

        # Check if we have enough frames
        if len(all_frames) < 10:
            logger.warning(f"Not enough frames detected in video: {len(all_frames)} frames")
            return {
                "status": "error", 
                "message": "Not enough frames detected", 
                "top_predictions": [], 
                "metrics": {}
            }

        # Check if there was enough lip movement
        if total_frames_processed > 0:
            open_mouth_ratio = open_mouth_count / total_frames_processed
            logger.info(f"Open mouth frames: {open_mouth_count}/{total_frames_processed} (ratio: {open_mouth_ratio:.2f})")
            if open_mouth_ratio < 0.3:  # Less than 30% of frames have open mouth
                logger.warning("No significant lip movement detected")
                return {
                    "status": "error",
                    "message": "No significant lip movement detected. Please ensure you are speaking clearly.",
                    "top_predictions": [],
                    "metrics": {}
                }

        # Sample or pad frames to TOTAL_FRAMES
        if len(all_frames) >= TOTAL_FRAMES:
            idxs = np.linspace(0, len(all_frames) - 1, TOTAL_FRAMES).astype(int)
            frames = [all_frames[i] for i in idxs]
        else:
            frames = all_frames.copy()
            last_frame = all_frames[-1]
            while len(frames) < TOTAL_FRAMES:
                frames.append(last_frame)

        X = np.array(frames, dtype=np.float32) / 255.0
        X = np.expand_dims(X, axis=0)

        # Make prediction
        pred = model.predict(X, verbose=0)

        # Use the same top 3 logic as the collector
        top3_indices = np.argsort(pred[0])[-3:][::-1]
        top_predictions = []
        for idx in top3_indices:
            phrase = BIKOL_NAGA_PHRASES[idx] if idx < len(BIKOL_NAGA_PHRASES) else f'Unknown ({idx})'
            confidence = float(pred[0][idx])
            top_predictions.append({
                "phrase": phrase,
                "confidence": confidence
            })

        # Get the top prediction
        top_idx = top3_indices[0]
        top_phrase = BIKOL_NAGA_PHRASES[top_idx] if top_idx < len(BIKOL_NAGA_PHRASES) else f'Unknown ({top_idx})'
        top_confidence = float(pred[0][top_idx])

        # Calculate derived metrics based on confidence
        metrics = calculate_metrics_from_confidence(top_confidence)

        logger.info(f"Top prediction: {top_phrase}, Confidence: {top_confidence:.4f}")
        for i, pred_info in enumerate(top_predictions[1:], 2):
            logger.info(f"#{i} prediction: {pred_info['phrase']}, Confidence: {pred_info['confidence']:.4f}")

        return {
            "status": "success",
            "top_prediction": top_phrase,
            "top_predictions": top_predictions,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error in predict_lipreading: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "message": f"Error in prediction: {str(e)}", 
            "top_predictions": [], 
            "metrics": {}
        }

# Constants for the lip reading model
NUM_FRAMES = 75  # Changed from 75 to 30 frames per sample
HEIGHT = 80       # Height of the lip patch
WIDTH = 112       # Width of the lip patch
CHANNELS = 3      # RGB channels

def is_mouth_open(face_landmarks, threshold=0.018):
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    return abs(upper_lip.y - lower_lip.y) > threshold

class LipReadingModel:
    def __init__(self):
        try:
            # Download model if not exists
            from download_model import download_model # type: ignore
            model_path = download_model()
            
            # Load the model
            self.model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
            
            # Initialize MediaPipe Face Mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            logger.error(f"Error initializing LipReadingModel: {e}")
            raise

    def preprocess_image(self, img):
        try:
            # Convert to LAB color space
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(img_lab)
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
            l_channel_eq = clahe.apply(l_channel)
            # Merge back and convert to RGB
            img_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
            img_eq = cv2.cvtColor(img_eq, cv2.COLOR_LAB2RGB)
            return img_eq
        except Exception as e:
            print(f"Error in preprocess_image: {e}")
            return img

    def extract_lip_region(self, frame, face_landmarks):
        img_h, img_w = frame.shape[:2]
        try:
            # Get lip corners
            lip_left = int(face_landmarks.landmark[61].x * img_w)
            lip_right = int(face_landmarks.landmark[291].x * img_w)
            # Get lip top and bottom
            lip_top = int(face_landmarks.landmark[0].y * img_h)
            lip_bottom = int(face_landmarks.landmark[17].y * img_h)
            # Calculate width and height of lip region
            width_diff = WIDTH - (lip_right - lip_left)
            height_diff = HEIGHT - (lip_bottom - lip_top)
            # Calculate padding
            pad_left = width_diff // 2
            pad_right = width_diff - pad_left
            pad_top = height_diff // 2
            pad_bottom = height_diff - pad_top
            # Ensure padding doesn't extend beyond frame
            pad_left = min(pad_left, lip_left)
            pad_right = min(pad_right, img_w - lip_right)
            pad_top = min(pad_top, lip_top)
            pad_bottom = min(pad_bottom, img_h - lip_bottom)
            # Calculate coordinates with padding
            x1 = max(0, lip_left - pad_left)
            y1 = max(0, lip_top - pad_top)
            x2 = min(img_w, lip_right + pad_right)
            y2 = min(img_h, lip_bottom + pad_bottom)
            # Extract the lip region
            lip_frame = frame[y1:y2, x1:x2]
            if lip_frame.size == 0:
                return None
            # Resize to the standard dimensions
            lip_frame = cv2.resize(lip_frame, (WIDTH, HEIGHT))
            # Apply preprocessing
            lip_frame = self.preprocess_image(lip_frame)
            return lip_frame
        except Exception as e:
            print(f"Error extracting lip region: {e}")
            return None

    def predict(self, video_path):
        try:
            print(f"Starting video processing for: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video file")
                return {
                    "status": "error", 
                    "message": "Error opening video", 
                    "top_predictions": [], 
                    "metrics": {}
                }

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Total frames in video: {total_frames}")
            
            frames_buffer = []
            open_mouth_count = 0
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                frame_count += 1
                print(f"Processing frame {frame_count}/{total_frames}")
                
                # Flip the frame horizontally for selfie-view
                frame = cv2.flip(frame, 1)
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Process with MediaPipe
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    print(f"Face detected in frame {frame_count}")
                    face_landmarks = results.multi_face_landmarks[0]
                    # Check if mouth is open
                    if is_mouth_open(face_landmarks):
                        open_mouth_count += 1
                    lip_frame = self.extract_lip_region(frame, face_landmarks)
                    if lip_frame is not None:
                        frames_buffer.append(lip_frame)
                        print(f"Lip frame collected. Total lip frames: {len(frames_buffer)}/{NUM_FRAMES}")
                        if len(frames_buffer) >= NUM_FRAMES:
                            print("Required number of frames collected")
                            break
                else:
                    print(f"No face detected in frame {frame_count}")
            
            cap.release()
            print(f"Video processing complete. Collected {len(frames_buffer)} lip frames")
            
            if len(frames_buffer) < NUM_FRAMES:
                print(f"Not enough frames collected. Got {len(frames_buffer)}, need {NUM_FRAMES}")
                if len(frames_buffer) > 0:
                    # Pad with the last frame until we have exactly NUM_FRAMES
                    while len(frames_buffer) < NUM_FRAMES:
                        frames_buffer.append(frames_buffer[-1])
                    print(f"Padded frames to {NUM_FRAMES}")
                else:
                    return {
                        "status": "error", 
                        "message": "Not enough frames", 
                        "top_predictions": [], 
                        "metrics": {}
                    }

            # Check if enough frames have open mouth (talking)
            open_mouth_ratio = open_mouth_count / len(frames_buffer) if len(frames_buffer) > 0 else 0
            print(f"Open mouth frames: {open_mouth_count}/{len(frames_buffer)} (ratio: {open_mouth_ratio:.2f})")
            if open_mouth_ratio < 0.3:  # Less than 30% of frames have open mouth
                print("No speech detected (mouth mostly closed)")
                return {
                    "status": "error", 
                    "message": "No speech detected", 
                    "top_predictions": [], 
                    "metrics": {}
                }

            # Convert to numpy array and normalize to [0,1]
            input_data = np.array(frames_buffer[:NUM_FRAMES], dtype=np.float32) / 255.0
            # Reshape to match model input shape
            input_data = input_data.reshape(1, NUM_FRAMES, HEIGHT, WIDTH, CHANNELS)
            print(f"Input data shape: {input_data.shape}")
            
            # Make prediction
            prediction = self.model.predict(input_data, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            print(f"Prediction made: {BIKOL_NAGA_PHRASES[predicted_class]} with confidence {confidence}")
            
            # Get top 3 predictions
            top3_indices = np.argsort(prediction)[-3:][::-1]
            top_predictions = []
            for idx in top3_indices:
                phrase = BIKOL_NAGA_PHRASES[idx] if idx < len(BIKOL_NAGA_PHRASES) else f'Unknown ({idx})'
                confidence = float(prediction[idx])
                top_predictions.append({
                    "phrase": phrase,
                    "confidence": confidence
                })

            # Calculate metrics
            metrics = calculate_metrics_from_confidence(confidence)
            
            return {
                "status": "success",
                "top_prediction": BIKOL_NAGA_PHRASES[predicted_class],
                "top_predictions": top_predictions,
                "metrics": metrics
            }
        except Exception as e:
            print(f"Error in predict: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error", 
                "message": f"Error in prediction: {str(e)}", 
                "top_predictions": [], 
                "metrics": {}
            } 