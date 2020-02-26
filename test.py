from keras.engine.saving import model_from_json
from matplotlib import image
import numpy as np

from app import SKIN_CLASSES

f = open('E:\\STUDY\PYTHON\\New folder (2)\\akiec2.jpg', 'r')
print("naerm "+ f.name)
path = 'static/data/' + f.name
f.save(path)
j_file = open('modelnew.json', 'r')
loaded_json_model = j_file.read()
j_file.close()
model = model_from_json(loaded_json_model)
model.load_weights('modelnew.h5')
img1 = image.load_img(f, target_size=(224, 224))
img1 = np.array(img1)
img1 = img1.reshape((1, 224, 224, 3))
img1 = img1 / 255
prediction = model.predict(img1)
pred = np.argmax(prediction)
disease = SKIN_CLASSES[pred]
accuracy = prediction[0][pred]
