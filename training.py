import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from cnn import CNN
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import imutils
from keras.models import load_model,model_from_json

NO_OF_EPOCHS = 25
BATCH_SIZE = 32
INIT_LR = 1e-3  #Initial Learning Rate

data = []
labels = []

imagePaths1 = sorted(list(paths.list_images("images/not_gun")))
random.seed(56)
random.shuffle(imagePaths1)
imagePaths1 = imagePaths1[0:3794]

imagePaths2 = sorted(list(paths.list_images("images/gun")))
random.seed(56)
random.shuffle(imagePaths2)

imagePaths = imagePaths1 + imagePaths2
random.seed(56)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	image = img_to_array(image)
	data.append(image)

	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "images/gun" else 0
	labels.append(label)


data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=25)


trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)


aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


print("Model is Compiling")
model = CNN.ntwrk(width=64, height=64, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / NO_OF_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("Training the Network")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BATCH_SIZE,
	epochs=NO_OF_EPOCHS, verbose=1)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Model is Saved")



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Model is Loaded")

            

                               # Test For A Image                     
                    
                    
image = cv2.imread('examples/162522888.jpg')
orig = image.copy()

image = cv2.resize(image, (64, 64))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

(not_gun, gun) = loaded_model.predict(image)[0]

label = "Gun" if gun > 0.70 else "Not Gun"
proba = gun if gun > not_gun else not_gun
label = "{}: {:.2f}%".format(label, proba * 100)

output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

cv2.imshow("Result", output)
cv2.waitKey(0)