import io

import numpy as np
from PIL import Image
from fastapi import File

from keras.utils import get_file, load_img, img_to_array

DEFAULT_INPUT_SHAPE = (128, 128, 3)


async def img_array_from_url(url: str, input_shape=DEFAULT_INPUT_SHAPE):
    img_path = get_file(origin=url)
    img = load_img(img_path, target_size=input_shape[:-1])
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


async def img_array_from_upload_file(file: File, input_shape=DEFAULT_INPUT_SHAPE):
    img = Image.open(io.BytesIO(await file.read()))
    img = img.resize(input_shape[:-1])
    img_array = np.array(img).reshape(input_shape)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array
