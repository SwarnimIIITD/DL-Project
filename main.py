from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

app = FastAPI()

# CORS settings to allow frontend access
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://your-frontend-url.com"  # Add your frontend URL here if deploying
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model directly within the FastAPI app
MODEL_PATH = "./1.keras" # Update with your actual model path
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]  # Update with your actual class names

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    """Reads and converts image to numpy array."""
    image = np.array(Image.open(BytesIO(data)).resize((256, 256)))  # Resize if necessary for your model
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint to receive an image and return prediction."""
    # Convert the image file to a numpy array
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # Add batch dimension

    # Get model predictions
    predictions = model.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Return prediction results
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
