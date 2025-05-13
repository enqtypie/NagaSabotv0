from flask import Flask, request, jsonify # type: ignore
from flask_cors import CORS # type: ignore
from werkzeug.utils import secure_filename # type: ignore       
import os
from datetime import datetime
import uuid
from lipreading_model import predict_lipreading
import tensorflow as tf # type: ignore
import logging
import gc
import gdown # type: ignore
import traceback
import requests # type: ignore
import numpy as np # type: ignore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NagaSabot")

# Configure TensorFlow to be memory-efficient
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth - prevents TF from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU memory growth enabled for {len(gpus)} GPUs")
    except Exception as e:
        logger.error(f"Failed to configure GPU memory growth: {e}")

# Use mixed precision if possible
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    logger.info("Mixed precision policy set to mixed_float16")
except Exception as e:
    logger.error(f"Failed to set mixed precision policy: {e}")

app = Flask(__name__)

# Configure CORS with specific settings
CORS(app, 
     resources={r"/*": {
         "origins": [
             "http://localhost:4200",  # Angular dev server
             "https://nagasabotv0-quvx.onrender.com",  # Your deployed frontend
             "https://*.onrender.com"  # Any Render subdomain
         ],
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "expose_headers": ["Content-Range", "X-Content-Range"],
         "supports_credentials": True,
         "max_age": 600  # Cache preflight requests for 10 minutes
     }},
     supports_credentials=True
)

# Add CORS headers to all responses
ALLOWED_ORIGINS = [
    "http://localhost:4200",
    "https://nagasabotv0-quvx.onrender.com"
]

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')
    if origin in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Vary'] = 'Origin'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max-limit

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Model file path - Updated to new model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'nagsabot_full_model_complete15.keras')

# Download and load the model at startup
try:
    # Check if model exists, if not download it
    if not os.path.exists(MODEL_PATH):
        logger.info("Model not found, downloading from Google Drive...")
        model_url = "https://drive.google.com/uc?id=13cTvPLbw2ldDC8qohurcGup2YDS8qOo_"
        gdown.download(model_url, MODEL_PATH, quiet=False)
        logger.info("Model downloaded successfully")
    else:
        logger.info("Model already exists, skipping download")
    logger.info(f"Loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # Don't compile to save memory
    model_loaded = True
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Supabase configuration
SUPABASE_URL = "https://dhblfdgqtvsyyfjifhgn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRoYmxmZGdxdHZzeXlmamlmaGduIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NjU0MTQwNSwiZXhwIjoyMDYyMTE3NDA1fQ.wK5aygRzve2SufjpKTEFi8qtjdcPKZvTibayQ4cQzoU"
VIDEOS_BUCKET = "videos"  # For original videos
PREPROCESSED_BUCKET = "preprocessed-lips"  # For preprocessed lip videos

def upload_to_supabase(file_path, dest_filename, bucket_name):
    """Upload file to specified Supabase bucket"""
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket_name}/{dest_filename}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/octet-stream"
    }
    with open(file_path, "rb") as f:
        response = requests.post(url, headers=headers, data=f)
    if response.status_code in (200, 201):
        return f"{SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{dest_filename}"
    else:
        raise Exception(f"Supabase upload failed: {response.text}")

def process_video(filepath):
    """Common video processing logic for both upload and predict endpoints"""
    if model is None:
        raise Exception('Model not loaded')
    
    logger.info(f"Processing video: {filepath}")
    result = predict_lipreading(filepath, model)
    
    if result.get('status') == 'error':
        raise Exception(result.get('message', 'Unknown error in video processing'))
    
    # Extract and format metrics
    metrics = result.get('metrics', {})
    processed_result = {
        'status': 'success',
        'top_prediction': result.get('top_prediction', 'No phrase detected'),
        'top_predictions': result.get('top_predictions', []),
        'metrics': {
            'open_mouth_ratio': metrics.get('open_mouth_ratio', 0.0),
            'frames_processed': metrics.get('frames_processed', 0),
            'confidence_score': metrics.get('confidence_score', 0.0),
            'processing_time': metrics.get('processing_time', 0.0)
        }
    }
    
    # If preprocessed video path is provided in the result, upload it
    if 'preprocessed_video_path' in result:
        try:
            preprocessed_filename = f"preprocessed_{os.path.basename(filepath)}"
            preprocessed_url = upload_to_supabase(
                result['preprocessed_video_path'], 
                preprocessed_filename,
                PREPROCESSED_BUCKET
            )
            processed_result['preprocessed_video_url'] = preprocessed_url
            logger.info(f"Uploaded preprocessed video to Supabase: {preprocessed_url}")
            
            # Clean up preprocessed video file
            try:
                os.remove(result['preprocessed_video_path'])
                logger.info(f"Deleted preprocessed video file: {result['preprocessed_video_path']}")
            except Exception as e:
                logger.warning(f"Failed to delete preprocessed video file: {result['preprocessed_video_path']}, error: {e}")
        except Exception as e:
            logger.error(f"Failed to upload preprocessed video: {str(e)}")
            # Don't fail the whole process if preprocessed upload fails
            processed_result['preprocessed_video_url'] = None
    
    return processed_result

@app.route('/predict', methods=['POST'])
def predict():
    """New endpoint for direct prediction without storage"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided', 'status': 'error'}), 400
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'error'}), 400
    
    if not allowed_file(video_file.filename):
        return jsonify({'error': 'File type not allowed', 'status': 'error'}), 400
    
    try:
        # Save temporarily
        unique_filename = f"temp_{uuid.uuid4()}_{secure_filename(video_file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        video_file.save(filepath)
        
        # Process video
        result = process_video(filepath)
        
        # Cleanup
        try:
            os.remove(filepath)
            logger.info(f"Deleted temporary file: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {filepath}, error: {e}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e), 'status': 'error'}), 500
    finally:
        del video_file
        gc.collect()

@app.route('/upload', methods=['POST','GET'])
def upload_video():
    """Updated upload endpoint with storage"""
    if request.method == 'GET':
        return jsonify({'message': 'Upload endpoint is live. Use POST with a video file.'}), 200
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided', 'status': 'error'}), 400
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'error'}), 400
    
    if not allowed_file(video_file.filename):
        return jsonify({'error': 'File type not allowed', 'status': 'error'}), 400
    
    try:
        # Generate unique filename and save
        original_filename = secure_filename(video_file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        video_file.save(filepath)
        
        # Upload original video to videos bucket
        video_url = upload_to_supabase(filepath, unique_filename, VIDEOS_BUCKET)
        logger.info(f"Uploaded original video to Supabase: {video_url}")
        
        # Process video
        result = process_video(filepath)
        result['video_url'] = video_url  # URL of the original video
        
        # Cleanup
        try:
            os.remove(filepath)
            logger.info(f"Deleted uploaded file: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to delete uploaded file: {filepath}, error: {e}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e), 'status': 'error'}), 500
    finally:
        del video_file
        gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
