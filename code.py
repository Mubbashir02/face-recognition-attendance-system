import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd


path = 'images'
images = []  # to store read images
personNames = []  # to store person name
myList = os.listdir(path)
print(myList)

for cu_img in myList:  # to read image and store image + info in list
    current_Img = cv2.imread(f'{path}/{cu_img}')   # to read image by path + name
    images.append(current_Img)  # to add read images to list
    personNames.append(os.path.splitext(cu_img)[0])   # to add person name to list by split img name
print(personNames)

#    0    1
# '1ahad.jpeg'

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        dateList = []
        time_now = datetime.now()
        dStr = time_now.strftime('%d/%m/%Y')
        tStr = time_now.strftime('%H:%M:%S')
        att = False
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            dateList.append(entry[-1])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},\t{tStr},\t{dStr}')







encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

# HOG Algorithm (Histogram of Oriented Gradients)

# laptop camera 0 & for external camera 1

window = tk.Tk()
window.attributes('-fullscreen',True)
window.resizable(True,False)
frame1 = tk.Frame(window,bg="#800000")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, 0.45)  # match store image to frame image
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)   # calculate distance
        print(faceDis)
        matchIndex = np.argmin(faceDis)  # store minimum face distance

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # make face frame
            cv2.rectangle(frame, (x1, y2 ), (x2, y2+30), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 + 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)  # print personName on frame
            attendance(name)
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # make face frame
            cv2.rectangle(frame, (x1, y2), (x2, y2 + 30), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, 'Unknown', (x1 + 6, y2 + 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:    # press enter to exit
        break

cap.release()
cv2.destroyAllWindows()




window.mainloop()