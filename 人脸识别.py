import os
import cv2 as cv
import numpy as np


#加载数据文件
recognizer =  cv.face.LBPHFaceRecognizer_create()
recognizer.read('D:\\PycharmProjects\\pythonProject1\\trainer\\trainer.yml')


def face_detect_demo(img,minW,minH,str_array):

    #将图片转为灰色
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #加载特征数据
    face_detector = cv.CascadeClassifier('D:\\opencv\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(
        gray,
        minNeighbors=3,
        minSize=(int(minW), int(minH)))

    #识别
    if len(faces)>0:
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        id,confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if confidence < 80:
            confidence = "{0}%".format(round(100 - confidence))
        else:
            confidence = "{0}%".format(round(100 - confidence))

        font = cv.FONT_HERSHEY_SIMPLEX
        #cv.putText(img, str('chenchunyu'), (x + 5, y - 5), font, 1, (0, 0, 255), 1)
        cv.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)
        str_array.append('标签id：'+str(id)+'置信评分：'+str(confidence))
        #print('标签id:',id,'置信评分',confidence)
        cv.imshow('result', img)
    else:
        cv.imshow('result', img)



if __name__ == '__main__':
    cap = cv.VideoCapture('D:\\PycharmProjects\\pythonProject1\\video1.mp4')
    minW = 0.11 * cap.get(3)
    minH = 0.11 * cap.get(4)
    str_array = []
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        face_detect_demo(frame, minW, minH, str_array)
        if ord('q') == cv.waitKey(30):
            break

    cv.destroyAllWindows()
    print(str_array)

