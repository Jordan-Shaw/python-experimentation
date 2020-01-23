#used for turning img into an array of smaller things to iterate over
import numpy as np

#openCV stuff
import cv2

#the graphing stuff which will display img
import matplotlib.pyplot as plt

#used for getting current directory, should make path work on other machines
import os



# assigns image to variable img
img = cv2.imread('barack.jpg')

img_copy = img.copy()

#converts img to grayscale
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

#function that should convert back into colour when invoked, isn't currently working
def convertToRGB(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#assigns the classifier (which is the instructions of what to look for in img) to a variable
haar_cascade_face = cv2.CascadeClassifier(
    os.getcwd() + '/data/haarcascade_frontalface_default.xml')

#invokes cascade on img
faces_rects = haar_cascade_face.detectMultiScale(img_gray, scaleFactor=1.7, minNeighbors=5)

#prints number of faces found to console
print('Faces found: ', len(faces_rects))

#draws rectangle around face
for (x,y,w,h) in faces_rects:
  cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

#should convert img back to color and assign it to recoloured
#recoloured = convertToRGB(img_gray)

#plt.imshow(convertToRGB(img_gray))

#displays img
plt.imshow(convertToRGB(img_copy))
plt.show()
