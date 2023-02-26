import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

def show(imgCrop, fixedImg):
    #Now we are showing the cropped image
    cv2.imshow("ImageCrop",imgCrop)
    cv2.imshow("ImageWhite", fixedImg)

# 0 is id number for main camera
cap = cv2.VideoCapture(0) 
#can only track one hand
detector=HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["Bye","Good", "Morning","Nice", "Thankyou", "ToSee", "You"]

# folder = "Data/1"
# counter = 0
offset = 30
imgSize = 400


while True:
    
    success, img = cap.read() 
    #detects hands
    hands,img = detector.findHands(img)
    #print(hands)
    if hands:
        
        if len(hands) == 1:
            #0 because we are ony taking one hadn which is the first hand in the array
            hand = hands[0]
            x,y,w,h= hand['bbox']
            #dimensions we want to crop it at
        
            #fixed width adjustment
            #8 bit defines the color value on RGB
        
        
            fixedImg = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y-offset :y+ h +offset, x-offset: x + w+offset]
        
            #put image crop matrix into image white matrix
            imgCropShape = imgCrop.shape

            aspectRatio = h/w
            
            #proportionally size image, and center on white screen
            if aspectRatio > 1:
                #height adjustment and centering
                constant = imgSize/h
                wCal = math.ceil(constant*w)
                imgResize = cv2.resize(imgCrop,(wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal)/2)
                fixedImg[:, wGap: wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(fixedImg)
                # print(prediction, index)
            
            #So far, we have a fixed, but we need to stretch the width and center the image to make it more appealing
            #calculating the width gap between white image and the new width and dividing by 2
                
            else:
                #width adjustment
                constant = imgSize/w
                hCal = math.ceil(constant*h)
                imgResize = cv2.resize(imgCrop,(imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal)/2)
                fixedImg[hGap: hCal + hGap, :] = imgResize         
                prediction, index = classifier.getPrediction(fixedImg)
                
                
                
        
        
            show(imgCrop, fixedImg)
            
        if len(hands) == 2:
            #0 because we are ony taking one hadn which is the first hand in the array
            hand1 = hands[0]
            x1,y1,w1,h1= hand1['bbox']
            hand2 = hands[1]
            x2,y2,w2,h2= hand2['bbox']
            #dimensions we want to crop it at
        
            #fixed width adjustment
            #8 bit defines the color value on RGB
        
        
            fixedImg = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            
            h = h1 if (max(y1,y2) == y1) else h2 
            w = w1 if (max(x1,x2) == x1) else w2 
            imgCrop = img[min(y1,y2)-offset : max(y1,y2) + (h)+ offset, min(x1,x2)-offset: max(x1,x2) + (w) + offset]
        
            #put image crop matrix into image white matrix
            imgCropShape = imgCrop.shape

            aspectRatio = h/w
            
            #proportionally size image, and center on white screen
            if aspectRatio > 1:
                #height adjustment and centering
                constant = imgSize/h
                wCal = math.ceil(constant*w)
                imgResize = cv2.resize(imgCrop,(wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal)/2)
                fixedImg[:, wGap: wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(fixedImg)
                # print(prediction, index)
            
            #So far, we have a fixed, but we need to stretch the width and center the image to make it more appealing
            #calculating the width gap between white image and the new width and dividing by 2
                
            else:
                #width adjustment
                constant = imgSize/w
                hCal = math.ceil(constant*h)
                imgResize = cv2.resize(imgCrop,(imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal)/2)
                fixedImg[hGap: hCal + hGap, :] = imgResize         
                prediction, index = classifier.getPrediction(fixedImg)
                
                
        
        
            show(imgCrop, fixedImg)
        
    cv2.imshow("image",img)
    cv2.waitKey(1)
    #1 mili second delay
    

