import numpy as np
import cv2
import os
import imutils
from keras.models import load_model,model_from_json
from keras.preprocessing.image import img_to_array

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Model is Loaded")

            

                               # Test For A Image                  
                    
                    
image = cv2.imread('examples/water-sceneries-hd.jpg')
orig = image.copy()

image = cv2.resize(image, (64, 64))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

(not_gun, gun) = loaded_model.predict(image)[0]

label = "Gun" if gun > not_gun else "Not Gun"
proba = gun if gun > not_gun else not_gun
label = "{}: {:.2f}%".format(label, proba * 100)

output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

cv2.imshow("Result", output)
cv2.waitKey(0)