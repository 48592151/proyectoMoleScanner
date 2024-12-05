import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

#evitar logs innecesarios de tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suprime los logs de nivel INFO

app = FastAPI ()

model_path = "moleScanner.h5"

# Cargar el modelo
try:
    model = load_model(model_path)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    raise

# Función para predecir usando el modelo
def predict_image(img):
    img = np.expand_dims(img, axis=0)  # Agregar dimensión para batch
    img = img / 255.0  # Normalizar la imagen
    prediction = model.predict(img)
    prob_benigno = prediction[0][0]
    prob_maligno = prediction[0][1]
    return prob_benigno, prob_maligno

# Endpoint para predicción
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Validar el tipo de archivo
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen JPEG o PNG.")

    # Leer el contenido del archivo
    contents = await file.read()

    # Convertir el archivo a una imagen
    try:
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen. Verifica el archivo.")
        
        # Redimensionar la imagen al tamaño esperado por el modelo
        img = cv2.resize(img, (224, 224))  # Cambia (224, 224) si tu modelo espera otro tamaño
        img = img.astype("float32")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {e}")

    # Realizar predicción
    prob_benigno, prob_maligno = predict_image(img)

    # Retornar resultados
    return {
        "Probabilidad de que sea benigno": f"{prob_benigno * 100:.2f}%",
        "Probabilidad de que sea maligno": f"{prob_maligno * 100:.2f}%"
    }
# probar en render
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)