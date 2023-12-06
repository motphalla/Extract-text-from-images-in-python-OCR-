import pytesseract
import PIL.Image
import cv2

#language
import os
from PIL import Image
os.environ['TESSDATA_PREFIX'] = 'C:\\Program Files\\Tesseract-OCR\\tessdata'

#インストールしたTesseract-OCRのパスを環境変数「PATH」へ追記する。
#OS自体に設定してあれば以下の2行は不要
path='C:\\Program Files\\Tesseract-OCR'
os.environ['PATH'] = os.environ['PATH'] + path

#required files: jpn.traineddata and jpn_vert.traineddata
#https://github.com/tesseract-ocr/tessdata_best

img_url = "japanese-text.jpg"
myconfig = r"--psm 3 --oem 3"

text = pytesseract.image_to_string(PIL.Image.open(img_url), lang="jpn", config=myconfig)

print(text)

"""
Page segmentation mode: 
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR.
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
			bypassing hacks that are Tesseract-specific.
"""

"""
OCR Engine Mode:
0 = Original Tesseract only.
1 = Neural nets LSTM only.
2 = Tesseract + LSTM.
3 = Default, based on what is available.
"""


#showing in image box
myconfig2 = r"--psm 11 --oem 3"
img = cv2.imread(img_url)
height, width, _ = img.shape

boxes = pytesseract.image_to_boxes(img, config=myconfig2)
for box in boxes.splitlines():
    box = box.split(" ")
    img = cv2.rectangle(img, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)

#2023/11/20
""" 
still testing on Japanese OCR

"""
#2023/11/21
#Japanese OCR is done, refer to line 5-16