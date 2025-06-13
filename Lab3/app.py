from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from PIL import Image
import io
import numpy as np
import base64
from tensorflow.keras.models import load_model

import time
import logging
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/fastapi.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Instrument FastAPI với Prometheus
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Custom Prometheus metrics cho model monitoring
model_inference_total = Counter(
    'model_inference_total', 
    'Tổng số lần model inference được thực hiện',
    ['model_name', 'status']
)

model_inference_duration = Histogram(
    'model_inference_duration_seconds', 
    'Thời gian inference của model (seconds)',
    ['model_name', 'processing_type'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

model_confidence_score = Gauge(
    'model_confidence_score', 
    'Điểm confidence của model prediction',
    ['model_name', 'predicted_class']
)

cpu_inference_time = Histogram(
    'cpu_inference_time_seconds',
    'CPU processing time cho model inference',
    ['model_name'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

gpu_inference_time = Histogram(
    'gpu_inference_time_seconds',
    'GPU processing time cho model inference', 
    ['model_name'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

api_errors_total = Counter(
    'api_errors_total',
    'Tổng số lỗi API',
    ['error_type', 'endpoint']
)

# Load model
try:
    model = load_model("models/model.keras")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    api_errors_total.labels(error_type='model_load_error', endpoint='/').inc()
    raise

label_map = {0:"cat", 1:"dog", 2:"panda", 3:"snake"}
templates = Jinja2Templates(directory="templates")

def preprocess_image(image: Image.Image):
    """Preprocess image cho model input"""
    try:
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        api_errors_total.labels(error_type='preprocessing_error', endpoint='/predict').inc()
        raise

def predict_with_metrics(input_data, model_name="animal_classifier"):
    """Model prediction với comprehensive metrics tracking"""
    
    # Track total inference attempts
    model_inference_total.labels(model_name=model_name, status='started').inc()
    
    # Measure total inference time
    with model_inference_duration.labels(model_name=model_name, processing_type='total').time():
        try:
            # Measure CPU processing time
            cpu_start = time.perf_counter()
            
            # Simulate CPU preprocessing time
            time.sleep(0.01)  # Simulate preprocessing overhead
            
            # Actual model prediction
            prediction_start = time.perf_counter()
            preds = model.predict(input_data, verbose=0)
            prediction_end = time.perf_counter()
            
            cpu_end = time.perf_counter()
            
            # Record CPU time (total processing time)
            cpu_time = cpu_end - cpu_start
            cpu_inference_time.labels(model_name=model_name).observe(cpu_time)
            
            # Record GPU time (actual prediction time - giả sử model chạy trên GPU)
            gpu_time = prediction_end - prediction_start
            gpu_inference_time.labels(model_name=model_name).observe(gpu_time)
            
            # Calculate confidence score
            confidence = float(np.max(preds))
            predicted_class = int(np.argmax(preds))
            predicted_label = label_map.get(predicted_class, "unknown")
            
            # Update confidence gauge
            model_confidence_score.labels(
                model_name=model_name, 
                predicted_class=predicted_label
            ).set(confidence)
            
            # Track successful inference
            model_inference_total.labels(model_name=model_name, status='success').inc()
            
            logger.info(f"Inference completed - Class: {predicted_label}, Confidence: {confidence:.3f}, CPU time: {cpu_time:.3f}s, GPU time: {gpu_time:.3f}s")
            
            return preds, confidence, predicted_class
            
        except Exception as e:
            # Track failed inference
            model_inference_total.labels(model_name=model_name, status='failed').inc()
            api_errors_total.labels(error_type='inference_error', endpoint='/predict').inc()
            logger.error(f"Model inference error: {e}")
            raise

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    """Main page endpoint"""
    try:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": None, 
            "img_data": None
        })
    except Exception as e:
        api_errors_total.labels(error_type='template_error', endpoint='/').inc()
        logger.error(f"Template rendering error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """Image prediction endpoint với comprehensive monitoring"""
    
    start_time = time.perf_counter()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            api_errors_total.labels(error_type='invalid_file_type', endpoint='/predict').inc()
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        
        if len(contents) == 0:
            api_errors_total.labels(error_type='empty_file', endpoint='/predict').inc()
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Load image
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            api_errors_total.labels(error_type='image_load_error', endpoint='/predict').inc()
            logger.error(f"Image loading error: {e}")
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Preprocess image
        input_data = preprocess_image(image)
        
        # Run prediction with metrics
        preds, confidence, pred_class = predict_with_metrics(input_data)
        pred_label = label_map.get(pred_class, "Unknown")
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Create result text with confidence
        result_text = f"Nhãn dự đoán: {pred_label} (class {pred_class}) - Confidence: {confidence:.3f}"
        
        # Log successful prediction
        total_time = time.perf_counter() - start_time
        logger.info(f"Prediction successful - Total time: {total_time:.3f}s")
        
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": result_text, 
            "img_data": img_str
        })
        
    except HTTPException:
        raise
    except Exception as e:
        api_errors_total.labels(error_type='unexpected_error', endpoint='/predict').inc()
        logger.error(f"Unexpected error in prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint cho monitoring"""
    try:
        # Simple health check - có thể mở rộng thêm
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "model_loaded": model is not None
        }
    except Exception as e:
        api_errors_total.labels(error_type='health_check_error', endpoint='/health').inc()
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/metrics")
async def get_metrics():
    """Expose Prometheus metrics endpoint"""
    try:
        return generate_latest()
    except Exception as e:
        logger.error(f"Metrics generation error: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")

if __name__ == "__main__":
    logger.info("Starting FastAPI application with Prometheus monitoring")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "formatter": "default",
                    "class": "logging.FileHandler",
                    "filename": "/app/logs/fastapi.log",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default", "file"],
            },
        }
    )
