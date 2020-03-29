import cv2,time
import pickle
import numpy as np
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

smile_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {"person-name":1}
with open("labels.pickle", 'rb') as f:
    oglabels=pickle.load(f)
    labels={v:k for  k,v in oglabels.items()}
# for live streming use following code
video = cv2.VideoCapture(0)
#video = cv2.imread('jam.jpg')
#from a video
#video=cv2.VideoCapture('class.mp4')
#video=cv2.VideoCapture('class.mp4')
while True:
    check, frame=video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,h,w) in faces:
        #print(x,y,w,h)
        gray_roi=gray[y:y+h, x:x+w]

        color_roi=frame[y:y+h, x:x+w]
        img_color="jam.png"
        img_item="jamsher.png"
        cv2.imwrite(img_item,color_roi)
        id_, conf=recognizer.predict(gray_roi)
        if conf>=45 and conf<=85:
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            color= (255,255,255)
            stroke=2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        cv2.imwrite(img_item, gray_roi)
        cv2.imwrite(img_color, color_roi)
        Color = (0,255,0)
        stroke=2
        x_end = x+w
        y_end = y+h
        cv2.rectangle(frame, (x,y),(x_end, y_end), Color, stroke)

        eyes = eye_cascade.detectMultiScale(gray_roi)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(color_roi,(ex, ey), (ex+ew, ey+eh),(0,255,0),2)
        subitems = smile_cascade.detectMultiScale(gray_roi)
        for (ex, ey, ew, eh) in subitems:
            cv2.rectangle(color_roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow("image",frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()


