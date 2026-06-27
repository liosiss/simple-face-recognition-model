#PROBLEM: the pretrained model isn't as accurate as we want it to be,
#because in the picture that we use it detects two faces in one head and when we change the 3rd value to 6 (in the function detectMultiScale),
#not only it doen't fix the problem(make it more accurate) but also it doesn't detect another face.

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread('people.jpg')
img = cv2.resize(img, (1000, 800))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
cnt = 0
for (x, y, w, h) in faces:
    cnt = cnt + 1
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)

print("the number of people in the image is: "+str(cnt))

cv2.imshow("IMAGE:", img)
cv2.waitKey(0)
cv2.destroyAllWindows()




