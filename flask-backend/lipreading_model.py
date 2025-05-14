import os
import cv2 # type: ignore
import math
import numpy as np # type: ignore
import mediapipe as mp # type: ignore
import tensorflow as tf # type: ignore
import logging
import gc
from typing import Tuple, Dict, List, Optional, Union # type: ignore
from collections import OrderedDict, deque # type: ignore
import uuid
from datetime import datetime

# Initialize TensorFlow memory settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Memory growth needs to be set before GPUs have been initialized
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Optionally limit memory usage
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
        )
    except RuntimeError as e:
        print(f"Memory growth setting error: {e}")

# Set up simplified logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NagaSabot")

# Constants - exactly matching training notebook
TOTAL_FRAMES = 75
LIP_WIDTH = 112
LIP_HEIGHT = 80
CHANNELS = 3

# Head tilt thresholds
HEAD_TILT_WARNING_THRESHOLD = 15.0  # Degrees
HEAD_TILT_ERROR_THRESHOLD = 30.0  # Degrees

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# MediaPipe face mesh indices for lips
LIP_OUTER_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]  # Outline
LIP_INNER_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]  # Inner contour

# Reference points for lip anchoring
LIP_CENTER_UPPER = 13  # Upper lip center landmark
LIP_CENTER_LOWER = 14  # Lower lip center landmark
LIP_LEFT_CORNER = 78   # Left corner of mouth
LIP_RIGHT_CORNER = 308 # Right corner of mouth

# Face reference points for head tilt detection
LEFT_EYE_OUTER = 33    # Left eye outer corner
RIGHT_EYE_OUTER = 263  # Right eye outer corner

# Define Bikol-Naga phrases (classes)
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

# NEW: Lip distance threshold for mouth closed detection
LIP_DISTANCE_THRESHOLD = 15.0

# NEW: Word to Phrase mapping for individual word recognition
WORD_TO_PHRASE_MAP = {
    "marhay": "marhay na aldaw",
    "aldaw": "marhay na aldaw",
    "dios": "dios mabalos",
    "mabalos": "dios mabalos",
    "padaba": "padaba taka",
    "taka": "padaba taka",
    "iyo": "iyo tabi",
    "dae": ["dae man tabi", "dae ko aram"],
    "tabi": ["iyo tabi", "dae man tabi", "patapos na tabi"],
    "man": ["dae man tabi", "tano man", "maoragon man"],
    "aram": "dae ko aram",
    "ko": "dae ko aram",
    "tano": "tano man",
    "po": ["tabi po", "maulay po kita"],
    "mayong": "mayong problema",
    "problema": "mayong problema",
    "nasasabutan": "nasasabutan mo",
    "mo": ["nasasabutan mo", "halaton mo ako"],
    "maulay": "maulay po kita",
    "kita": "maulay po kita",
    "gurano": "gurano an",
    "an": "gurano an",
    "maoragon": "maoragon man",
    "patapos": "patapos na tabi",
    "na": ["marhay na aldaw", "patapos na tabi"],
    "halaton": "halaton mo ako",
    "ako": "halaton mo ako"
}

def is_mouth_open(face_landmarks, threshold=0.018):
    """Check if mouth is open based on face landmarks."""
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    return abs(upper_lip.y - lower_lip.y) > threshold

def enhance_lip_region(lip_frame, transformed_outer=None, transformed_inner=None):
    """
    Apply preprocessing to standardize the lip region:
    - Convert to monochrome
    - Enhance contrast
    - Apply filters for better feature visibility
    - Add lip outline visualization using transformed landmarks
    """
    try:
        # Ensure input is valid
        if lip_frame is None or lip_frame.size == 0:
            logger.warning("Preprocessing received empty image.")
            return np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)  # Return black image

        # Ensure the image is BGR
        if len(lip_frame.shape) == 2 or lip_frame.shape[2] == 1:  # Handle grayscale if it somehow occurs
            lip_frame = cv2.cvtColor(lip_frame, cv2.COLOR_GRAY2BGR)
        elif lip_frame.shape[2] != 3:
            logger.warning(f"Preprocessing received image with unexpected channels: {lip_frame.shape}")
            return np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)  # Return black image
            
        # Convert to grayscale (monochrome)
        lip_frame_gray = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
        lip_frame_eq = clahe.apply(lip_frame_gray)
        
        # Apply bilateral filter to preserve edges while reducing noise
        lip_frame_filtered = cv2.bilateralFilter(lip_frame_eq, 5, 35, 35)
        
        # Apply sharpening kernel for better lip contour definition
        kernel = np.array([[-1, -1, -1],
                            [-1,  9, -1],
                            [-1, -1, -1]])
        lip_frame_sharp = cv2.filter2D(lip_frame_filtered, -1, kernel)
        
        # Apply slight Gaussian blur to reduce noise from sharpening
        lip_frame_final = cv2.GaussianBlur(lip_frame_sharp, (3, 3), 0)
        
        # Convert back to 3-channel (though all channels will be the same)
        lip_frame_3ch = cv2.cvtColor(lip_frame_final, cv2.COLOR_GRAY2BGR)
        
        # Add lip outline visualization using transformed landmarks
        if transformed_outer is not None and transformed_inner is not None and len(transformed_outer) > 0 and len(transformed_inner) > 0:
            lip_frame_with_outline = lip_frame_3ch.copy()
            outer_lip_points = np.array(transformed_outer, dtype=np.int32)
            inner_lip_points = np.array(transformed_inner, dtype=np.int32)
            cv2.polylines(lip_frame_with_outline, [outer_lip_points], True, (255, 255, 255), 2)  # White color, 2px thickness
            cv2.polylines(lip_frame_with_outline, [inner_lip_points], True, (255, 255, 255), 2)  # White color, 2px thickness
            alpha = 0.7
            lip_frame_final = cv2.addWeighted(lip_frame_3ch, 1-alpha, lip_frame_with_outline, alpha, 0)
        else:
            lip_frame_final = lip_frame_3ch
        return lip_frame_final
    except cv2.error as e:
        logger.error(f"OpenCV Error during preprocessing: {e}")
        logger.error(f"Image shape: {lip_frame.shape if lip_frame is not None else 'None'}, dtype: {lip_frame.dtype if lip_frame is not None else 'None'}")
        return np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)  # Fallback
    except Exception as e:
        logger.error(f"General Error in preprocessing: {e}")
        return np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)  # Fallback

def get_fixed_centered_lip_region(image, face_landmarks):
    """
    Extract lip region with lip contour always centered regardless of movement:
    - Uses the actual user's lip contour as the template
    - Ensures lip contour is always in the center of the frame
    - Only captures opening/closing motion without zooming
    - Returns transformed lip landmarks for the cropped region
    """
    h, w, _ = image.shape
    try:
        # Extract all lip landmarks for the contour
        outer_lip_points = []
        inner_lip_points = []
        all_lip_points = []
        for i in LIP_OUTER_INDICES:
            pt = face_landmarks.landmark[i]
            x, y = int(pt.x * w), int(pt.y * h)
            outer_lip_points.append((x, y))
            all_lip_points.append((x, y))
        for i in LIP_INNER_INDICES:
            pt = face_landmarks.landmark[i]
            x, y = int(pt.x * w), int(pt.y * h)
            inner_lip_points.append((x, y))
            all_lip_points.append((x, y))
        all_x = [p[0] for p in all_lip_points]
        all_y = [p[1] for p in all_lip_points]
        center_x = sum(all_x) / len(all_x)
        center_y = sum(all_y) / len(all_y)
        center_point = (int(center_x), int(center_y))
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
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
        lip_left_x = int(center_point[0] - (lip_width_padded / 2))
        lip_right_x = int(center_point[0] + (lip_width_padded / 2))
        lip_top = int(center_point[1] - (lip_height_padded / 2))
        lip_bottom = int(center_point[1] + (lip_height_padded / 2))
        lip_left_x = max(0, lip_left_x)
        lip_right_x = min(w, lip_right_x)
        lip_top = max(0, lip_top)
        lip_bottom = min(h, lip_bottom)
        upper_lip_y = face_landmarks.landmark[LIP_CENTER_UPPER].y * h
        lower_lip_y = face_landmarks.landmark[LIP_CENTER_LOWER].y * h
        lip_distance = abs(lower_lip_y - upper_lip_y)
        lip_region = rotated_image[lip_top:lip_bottom, lip_left_x:lip_right_x]
        # Transform landmarks to cropped region and resize
        def transform_points(points):
            transformed = []
            for (x, y) in points:
                # Shift to cropped region
                x_cropped = x - lip_left_x
                y_cropped = y - lip_top
                # Scale to output size
                x_resized = int(x_cropped * (LIP_WIDTH / (lip_right_x - lip_left_x)))
                y_resized = int(y_cropped * (LIP_HEIGHT / (lip_bottom - lip_top)))
                transformed.append((x_resized, y_resized))
            return transformed
        transformed_outer = transform_points(outer_lip_points)
        transformed_inner = transform_points(inner_lip_points)
        if lip_region.size > 0:
            lip_region_resized = cv2.resize(lip_region, (LIP_WIDTH, LIP_HEIGHT))
        else:
            lip_region_resized = np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)
        lip_distance_color = (0, 0, 255) if lip_distance < LIP_DISTANCE_THRESHOLD else (255, 255, 255)
        cv2.putText(lip_region_resized, f"Lip Distance: {lip_distance:.2f}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, lip_distance_color, 2)
        tilt_color = (0, 255, 0) if abs(angle_deg) <= HEAD_TILT_WARNING_THRESHOLD else (0, 0, 255)
        cv2.putText(lip_region_resized, f"Head Angle: {angle_deg:.1f}° {'(TILT WARNING!)' if abs(angle_deg) > HEAD_TILT_WARNING_THRESHOLD else ''}", 
                (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tilt_color, 2)
        # Return transformed landmarks for drawing
        return lip_region_resized, lip_distance, (lip_left_x, lip_top, lip_right_x, lip_bottom), angle_deg, transformed_outer, transformed_inner
    except Exception as e:
        logger.error(f"Error in get_fixed_centered_lip_region: {e}")
        return np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8), 0, (0, 0, 0, 0), 0, [], []

# Global model variable to avoid reloading
_global_model = None

def get_model():
    global _global_model
    if _global_model is None:
        try:
            model_path = "nagsabot_full_model_complete15.keras"
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at: {model_path}")
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            _global_model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    return _global_model

def save_lip_frames_as_video(frames: List[np.ndarray], output_dir: str = 'uploads') -> str:
    """
    Save preprocessed lip frames as a video file.
    
    Args:
        frames: List of preprocessed lip frame images
        output_dir: Directory to save the video
        
    Returns:
        Path to the saved video file
    """
    if not frames:
        raise ValueError("No frames provided to save")
    
    # Create unique filename
    filename = f"preprocessed_lips_{uuid.uuid4()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = os.path.join(output_dir, filename)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dimensions from first frame
    height, width = frames[0].shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    try:
        # Write each frame
        for frame in frames:
            out.write(frame)
        
        logger.info(f"Saved preprocessed lip video to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving preprocessed video: {e}")
        raise
    finally:
        out.release()

def is_mouth_closed(lip_distances, threshold=LIP_DISTANCE_THRESHOLD):
    """
    Determine if mouth is closed based on recent lip distances
    
    Args:
        lip_distances: list of recent lip distance measurements
        threshold: distance below which lips are considered closed
        
    Returns:
        bool: True if mouth is determined to be closed
    """
    if not lip_distances:
        return False
    
    # Check if all recent measurements are below threshold
    all_below_threshold = all(dist < threshold for dist in lip_distances)
    
    # Check for minimal movement (low variance)
    if len(lip_distances) >= 3:
        variance = np.var(lip_distances)
        low_movement = variance < 5.0  # Adjust this threshold as needed
    else:
        low_movement = True
    
    return all_below_threshold and low_movement

def find_matching_words(predicted_phrase, confidence):
    """
    Check if the predicted phrase matches any individual words
    and return both the word and its corresponding phrase
    
    Args:
        predicted_phrase: The phrase predicted by the model
        confidence: The confidence score of the prediction
        
    Returns:
        list: Contains tuples of (word, corresponding_phrase, confidence)
    """
    results = []
    
    # If confidence is high, just return the full phrase
    if confidence > 0.6:
        return [(predicted_phrase, predicted_phrase, confidence)]
    
    # Check if any individual words from our mapping are in the predicted phrase
    for word, phrases in WORD_TO_PHRASE_MAP.items():
        # Check if the word matches parts of the predicted phrase
        if word in predicted_phrase.lower().split():
            # If the phrases value is a list, add all corresponding phrases
            if isinstance(phrases, list):
                for phrase in phrases:
                    # Adjust confidence based on word length (longer words are more reliable)
                    word_confidence = min(0.9, confidence + (len(word) / 20))
                    results.append((word, phrase, word_confidence))
            else:
                # Single phrase mapping
                word_confidence = min(0.9, confidence + (len(word) / 20))
                results.append((word, phrases, word_confidence))
    
    # If no matches were found, return the original phrase
    if not results:
        return [(predicted_phrase, predicted_phrase, confidence)]
    
    # Sort by confidence (highest first)
    results.sort(key=lambda x: x[2], reverse=True)
    return results

def predict_lipreading(video_path: str, model=None) -> Dict[str, object]:
    """
    Process video and predict lip reading result.
    Now includes saving of preprocessed lip frames and word-to-phrase mapping.
    """
    if not os.path.exists(video_path):
        return {'status': 'error', 'message': 'Video file not found'}
    
    try:
        start_time = datetime.now()
        
        # Extract lip frames with metrics
        result = _extract_lip_frames_efficiently(video_path)
        if result is None:
            return {
                'status': 'error',
                'message': 'Failed to extract lip frames from video'
            }
        
        frames_array, open_mouth_ratio = result
        
        # Check if mouth remained closed during the video
        # Get lip distances from the frames processing stage
        if open_mouth_ratio < 0.1:  # If less than 10% of frames show open mouth
            return {
                'status': 'success',
                'top_prediction': 'Mouth Closed',
                'top_predictions': [
                    {'phrase': 'Mouth Closed', 'confidence': 1.0},
                    {'phrase': '', 'confidence': 0.0},
                    {'phrase': '', 'confidence': 0.0}
                ],
                'metrics': {
                    'open_mouth_ratio': float(open_mouth_ratio),
                    'frames_processed': int(frames_array.shape[0]),
                    'confidence_score': 1.0,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            }
        
        # Save preprocessed frames as video
        try:
            preprocessed_video_path = save_lip_frames_as_video(
                [frames_array[i] for i in range(frames_array.shape[0])],
                'uploads'
            )
        except Exception as e:
            logger.error(f"Failed to save preprocessed video: {e}")
            preprocessed_video_path = None
        
        # Get or load model
        if model is None:
            model = get_model()
            if model is None:
                return {
                    'status': 'error',
                    'message': 'Failed to load model'
                }
        
        # Prepare input
        processed_frames = frames_array.astype('float32') / 255.0
        processed_frames = np.expand_dims(processed_frames, axis=0)
        
        # Make prediction
        predictions = model.predict(processed_frames)
        predictions = predictions[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        top_predictions = []
        
        # Process each prediction with word matching
        for idx in top_indices:
            phrase = BIKOL_NAGA_PHRASES[idx]
            confidence = float(predictions[idx])
            
            # Find matching words for each prediction
            word_matches = find_matching_words(phrase, confidence)
            if word_matches:
                word, matched_phrase, adjusted_confidence = word_matches[0]
                top_predictions.append({
                    'phrase': phrase,
                    'confidence': float(confidence),  # Ensure confidence is a Python float
                    'matched_word': word if word != phrase else None,
                    'matched_phrase': matched_phrase if matched_phrase != phrase else None,
                    'adjusted_confidence': float(adjusted_confidence)  # Ensure adjusted_confidence is a Python float
                })
            else:
                top_predictions.append({
                    'phrase': phrase,
                    'confidence': float(confidence)  # Ensure confidence is a Python float
                })
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'status': 'success',
            'top_prediction': top_predictions[0]['phrase'],
            'top_predictions': top_predictions,
            'metrics': {
                'open_mouth_ratio': float(open_mouth_ratio),
                'frames_processed': int(frames_array.shape[0]),
                'confidence_score': float(np.max(predictions)),
                'processing_time': processing_time
            }
        }
        
        # Add preprocessed video path if available
        if preprocessed_video_path:
            result['preprocessed_video_path'] = preprocessed_video_path
        
        return result
        
    except Exception as e:
        logger.error(f"Error in predict_lipreading: {str(e)}")
        return {
            'status': 'error',
            'message': f'Error processing video: {str(e)}'
        }
    finally:
        # Force garbage collection
        gc.collect()

def _extract_lip_frames_efficiently(video_path: str) -> Optional[Tuple[np.ndarray, float]]:
    """Extract lip frames from video efficiently using MediaPipe."""
    frames = []
    open_mouth_frames = 0
    total_frames = 0
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video file")
        return None
        
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while len(frames) < TOTAL_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            frame.flags.writeable = True
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Extract lip region
                lip_frame, lip_distance, _, head_angle, transformed_outer, transformed_inner = get_fixed_centered_lip_region(frame, face_landmarks)
                
                if lip_frame is not None and lip_frame.size > 0:
                    # Process the lip frame with face landmarks
                    processed_lip = enhance_lip_region(lip_frame, transformed_outer, transformed_inner)
                    frames.append(processed_lip)
                    
                    # Check if mouth is open
                    if lip_distance > LIP_DISTANCE_THRESHOLD:
                        open_mouth_frames += 1
                        
    cap.release()
    
    if not frames:
        logger.warning("No frames were extracted from the video")
        return None
        
    # Calculate ratio of frames with open mouth
    open_mouth_ratio = open_mouth_frames / total_frames if total_frames > 0 else 0
    
    # Pad or truncate frames to match TOTAL_FRAMES
    if len(frames) < TOTAL_FRAMES:
        # Pad with last frame
        last_frame = frames[-1] if frames else np.zeros((LIP_HEIGHT, LIP_WIDTH, 3), dtype=np.uint8)
        while len(frames) < TOTAL_FRAMES:
            frames.append(last_frame.copy())
    elif len(frames) > TOTAL_FRAMES:
        # Keep only the first TOTAL_FRAMES frames
        frames = frames[:TOTAL_FRAMES]
    
    return np.array(frames), open_mouth_ratio

class LipReadingModel:
    def __init__(self):
        try:
            # Set memory-efficient TensorFlow session configuration
            self._setup_tf_config()
            
            # Use the shared model
            self.model = get_model()
            
            # Initialize MediaPipe Face Mesh with reduced resources
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

    def _setup_tf_config(self):
        """Configure TensorFlow for memory efficiency."""
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # Don't pre-allocate all GPU memory
        config.gpu_options.per_process_gpu_memory_fraction = 0.6  # Limit to 60% of GPU memory
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    def _load_model_efficiently(self, model_path: str) -> tf.keras.Model:
        """Load model with memory-efficient settings."""
        try:
            # Load model with optimized settings
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Use float16 for reduced memory usage (if supported by hardware)
            if tf.config.list_physical_devices('GPU'):
                try:
                    from tensorflow.keras.mixed_precision import experimental as mixed_precision # type: ignore
                    policy = mixed_precision.Policy('mixed_float16')
                    mixed_precision.set_global_policy(policy)
                    logger.info("Using mixed precision for model inference")
                except:
                    logger.info("Mixed precision not available, using default precision")
            
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, video_path: str) -> Dict[str, object]:
        """Process video and make prediction using memory-efficient approach."""
        try:
            logger.info(f"Starting video processing for: {video_path}")
            
            # Use the optimized standalone function for prediction
            return predict_lipreading(video_path, self.model)
            
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error", 
                "message": f"Error in prediction: {str(e)}", 
                "top_predictions": [], 
                "metrics": {}
            }

    def _extract_lip_frames_stream(self, video_path: str) -> Optional[Tuple[np.ndarray, float]]:
        """Extract lip frames from video in a streaming manner to reduce memory usage."""
        # This method is kept for backward compatibility
        # But functionality is moved to _extract_lip_frames_efficiently 
        return _extract_lip_frames_efficiently(video_path)

    def _predict_efficiently(self, frames_array: np.ndarray) -> Dict[str, object]:
        """Method kept for backward compatibility."""
        # Implementation moved to predict_lipreading function
        # Reshape data for prediction
        with tf.device('/CPU:0'):
            input_data = frames_array.reshape(1, TOTAL_FRAMES, LIP_HEIGHT, LIP_WIDTH, CHANNELS)
            prediction = self.model.predict(input_data, verbose=0)[0]
        
        # Get top prediction indices
        top3_indices = np.argsort(prediction)[-3:][::-1]
        predicted_class = top3_indices[0]
        confidence = float(prediction[predicted_class])
        
        # Format predictions
        top_predictions = []
        for idx in top3_indices:
            phrase = BIKOL_NAGA_PHRASES[idx] if idx < len(BIKOL_NAGA_PHRASES) else f'Unknown ({idx})'
            pred_confidence = float(prediction[idx])
            top_predictions.append({
                "phrase": phrase,
                "confidence": pred_confidence
            })
        
        # Return only actual metrics
        metrics = {
            "confidence": confidence
        }
        
        return {
            "status": "success",
            "top_prediction": BIKOL_NAGA_PHRASES[predicted_class],
            "top_predictions": top_predictions,
            "metrics": metrics
        }