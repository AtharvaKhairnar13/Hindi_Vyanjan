import cv2
import mediapipe as mp
import time 
import os
import HandTrackingModule as htm
import numpy as np
import tensorflow as tf


hindi_character = ['ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'क', 'न', 'प', 'फ', 'ब', 'भ', 'म',\
                    'य', 'र', 'ल', 'व', 'ख', 'श', 'ष', 'स', 'ह', 'ॠ', 'त्र', 'ज्ञ', 'ग', 'घ', 'ङ', 'च', 'छ',\
                    'ज', 'झ', '0', '१', '२', '३', '४', '५', '६', '७', '८', '९']

def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model


def get_n_predictions(pred_prob,n=5):

    pred_prob = np.squeeze(pred_prob)
    
    top_n_max_idx = np.argsort(pred_prob)[::-1][:n] # Get index of top n predictions
    top_n_max_val = list(pred_prob[top_n_max_idx])  # Get actual top n predictions
    
    # Get the coresponding hindi character for top n predictions
    top_n_class_name=[]
    for i in top_n_max_idx:
        top_n_class_name.append(hindi_character[i])
    
    return top_n_class_name,top_n_max_val

def clipDrawing():
    roi = imgCanvas[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, (32, 32))  # Assuming your model expects 28x28 input size
    roi_normalized = roi_resized / 255.0  # Normalize to [0, 1]
    roi_normalized = np.expand_dims(roi_normalized, axis=-1)  # Add channel dimension
    roi_normalized = np.expand_dims(roi_normalized, axis=0) 
    pred_prob = model.predict(roi_normalized)
    class_name , confidense = get_n_predictions(pred_prob)
    return class_name[0]

model = load_model()
cap= cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector=htm.handDetector(detectionCon=0.85)
xp=0
yp=0

print("Hi")
rect_start = (480, 260)
rect_end = (880, 560)
imgCanvas=np.zeros((720,1280,3),np.uint8)
while True:
    success, img=cap.read()
    img=cv2.flip(img,1)
    cv2.rectangle(img, rect_start, rect_end, (0,0, 255), 3)

    img=detector.findHands(img)
    lmlist=detector.findPositions(img,draw=False)

    if len(lmlist)!=0:
        #print(lmlist)
        x1,y1=lmlist[8][1:]


        fingers=detector.fingersUp()
        #print(fingers)

        if all(finger==0 for finger in fingers):
            imgCanvas=np.zeros((720,1280,3),np.uint8)
            xp,yp=0,0
            '''predicted_class=clipDrawing()
            cv2.putText(img,f"{predicted_class}",(600,600),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),5,cv2.LINE_AA)'''
            print("Erase the selection")
        
        elif len(fingers) > 1 and fingers[1] == 1 and all(element == 0 for i, element in enumerate(fingers) if i!=1) :
            print("Drawing")
            if xp==0 and yp==0:
                xp,yp=x1,y1

            if rect_start[0] +20 < x1 < rect_end[0]-20 and rect_start[1]+20 < y1 < rect_end[1]-20:
                cv2.line(img, (xp, yp), (x1, y1), (0, 255, 0), thickness=15)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 255, 0), thickness=15)
                xp, yp = x1, y1

        elif len(fingers) > 1 and fingers[1] == 1 and fingers[2]==1 and all(element == 0 for i, element in enumerate(fingers) if (i!=1 and i!=2)):
            cv2.circle(img,(x1,y1),15,(0,255,0),cv2.FILLED)
            xp,yp=0,0
        else:
            print("Not Drawing")
            xp,yp=0,0




    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv= cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgCanvas)
    
    cv2.imshow("Image",img)
    
    if cv2.waitKey(1)==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
