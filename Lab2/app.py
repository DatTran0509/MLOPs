from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from PIL import Image
import io
import numpy as np
import base64
from tensorflow.keras.models import load_model

app = FastAPI()

model = load_model("./models/model.keras")
label_map = {0:"cat", 1:"dog", 2:"panda", 3:"snake"}

templates = Jinja2Templates(directory="templates")

def preprocess_image(image: Image.Image):
    image = image.resize((224,224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "img_data": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_data = preprocess_image(image)
    preds = model.predict(input_data)
    pred_class = int(np.argmax(preds, axis=1)[0])
    pred_label = label_map.get(pred_class, "Unknown")

    # Chuyển ảnh sang base64 để nhúng trong html
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    result_text = f"Nhãn dự đoán: {pred_label} (class {pred_class})"

    return templates.TemplateResponse("index.html", 
                                      {"request": request, 
                                       "result": result_text, 
                                       "img_data": img_str})
