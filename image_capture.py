import numpy as np
import operator
import cv2
import sys, os
from keras.models import load_model
from keras.models import model_from_json
import json
from PIL import Image
import pygame

pygame.init()
screen = pygame.display.set_mode((900,900),pygame.RESIZABLE)

CLIP_X1 = 160
CLIP_Y1 = 140
CLIP_X2 = 400
CLIP_Y2 = 360


'''with open('model.json','r') as f:
    model_json = json.load(f)
loaded_model = model_from_json(json.dumps(model_json))
loaded_model.load_weights('model.h5')
'''
json_file = open('C:/Users/shiv taneja/capstone project/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:/Users/shiv taneja/capstone project/model.h5")
#loaded_model = load_model('fi')

cap = cv2.VideoCapture(0)

while True:
    _, FrameImage = cap.read()
    FrameImage = cv2.flip(FrameImage, 1)
    cv2.imshow("", FrameImage)
    cv2.rectangle(FrameImage, (CLIP_X1, CLIP_Y1), (CLIP_X2, CLIP_Y2), (0,255,0) ,1)

    ROI = FrameImage[CLIP_Y1:CLIP_Y2, CLIP_X1:CLIP_X2]
    ROI = cv2.resize(ROI, (200, 200)) 
    #ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    #ROI= cv2.add(ROI,np.array([40.0]))
    _, output = cv2.threshold(ROI, 200, 255, cv2.THRESH_BINARY) # adjust brightness
    
    '''
    SHOWROI = cv2.resize(ROI, (256, 256)) 
    _, output2 = cv2.threshold(SHOWROI, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", output2)
    '''
    result = loaded_model.predict(np.reshape(ROI, [-1, 200, 200, 3]))
    predict =   { 'downdog':    result[0][0],
                  'goddess':    result[0][1],    
                  'plank':    result[0][2],
                  'tree':    result[0][3],
                  'warrior2':    result[0][4],
                  }
    
    predict = sorted(predict.items(), key=operator.itemgetter(1), reverse=True)
    
    if(predict[0][1] >= 1.0):
        predict_img  = pygame.image.load('C:/Users/shiv taneja/capstone project' + '/dataset/' + predict[0][0] + '.jpg')
    else:
        predict_img  = pygame.image.load('C:/Users/shiv taneja/capstone project' + '/dataset/nosign.png')
    predict_img = pygame.transform.scale(predict_img, (900, 900))
    screen.blit(predict_img, (0,0))
    pygame.display.flip()
    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == ord('q'): # esc key
        break
            
pygame.quit()
cap.release()
cv2.destroyAllWindows()
