print('Lesson 11: AI model. Threat image')

import cv2
from PIL import Image


image_cat_path = 'Cat.png'
image_glasses_path = 'minecraft glasses.png'
image_cat = cv2.imread(image_cat_path)

cat_face_handler = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')

cat_face_coordinates = cat_face_handler.detectMultiScale(image_cat)

for (x, y, w, h) in cat_face_coordinates:
    cv2.rectangle(image_cat, (x, y), (x + w, y + h), (255, 0, 0), 3)

cv2.imshow('Shocked Cat', image_cat)

cat = Image.open(image_cat_path)
glasses = Image.open(image_glasses_path)

cat = cat.convert('RGB')
glasses = glasses.convert('RGB')

(x, y, w, h) = cat_face_coordinates[0]
glasses = glasses.resize((w, int(h/3)))

cat.paste(glasses, (x, int(y + h/4)))
cat.save('cat_with_glasses.png')

cat_with_glasses = cv2.imread('cat_with_glasses.png')
cv2.imshow('Shocked Cat with glasses', cat_with_glasses)

cv2.waitKey()

