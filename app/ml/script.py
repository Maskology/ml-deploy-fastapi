from operator import itemgetter

from keras import Model
from keras.models import load_model

labels = sorted([
    'keras',
    'sidakarya',
    'wijil',
    'penasar',
    'tua',
    'dalem',
    'bujuh'
])


def predict(model_path, img_array):
    model: Model = load_model(model_path)
    prediction = model.predict(img_array)
    result = [(label, pred*100) for label, pred in zip(labels, prediction[0])]
    return dict(sorted(result, key=itemgetter(1), reverse=True))
