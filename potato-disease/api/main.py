from fastapi import FastAPI,File,UploadFile
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.layers import TFSMLayer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
path = r"C:\Users\Admin\OneDrive\Documents\potato_disease_classification\potato-disease\Models\1"
MODEL = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(256, 256, 3)),
    tf.keras.layers.TFSMLayer(path, call_endpoint='serving_default')
])
CLASS_NAMES = ["Early Blight","Late Blight","Healthy"]

@app.get("/ping")

async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))
    return np.array(image)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,axis=0)
    predictions = MODEL.predict(img_batch)
    output = list(predictions.values())[0]
    predicted_class = CLASS_NAMES[np.argmax(output[0])]
    confidence = float(np.max(output[0]))
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)