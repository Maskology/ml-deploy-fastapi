from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.ml.script import predict
from app.ml.utils import img_array_from_url, img_array_from_upload_file

input_shape = (128, 128, 3)
model_path = './model/model.h5'

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


@app.get('/')
async def root():
    return {"message": "Welcome to Balinese mask classifier API!"}


@app.post('/predict/img_link')
async def predict_img_link(img_link: str):
    img_array = await img_array_from_url(img_link, input_shape)
    return predict(model_path=model_path, img_array=img_array)


@app.post('/predict/img_upload')
async def predict_img_upload(file: UploadFile):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not an image.")
    img_array = await img_array_from_upload_file(file, input_shape)
    return predict(model_path=model_path, img_array=img_array)
