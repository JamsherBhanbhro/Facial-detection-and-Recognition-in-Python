import os
import cv2
import numpy as np
from PIL import Image
import pickle


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(BASE_DIR,"images")

current_id=0
label_ids={}
x_train = []
y_labels = []
for root, dir, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path=os.path.join(root, file)
            label=os.path.basename(os.path.dirname(path).replace(" ", "-").lower())
            #print(label, path)
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id +=1
            id_=label_ids[label]
            #print(label_ids)
            pil_image=Image.open(path).convert("L")
            Image_array=np.array(pil_image, "uint8")
            #print(Image_array)
            faces = face_cascade.detectMultiScale(Image_array, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi = Image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
#print(y_labels)
#print(x_train)
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")