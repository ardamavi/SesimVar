import cv2
import os
import time

goz_bulma = False
bekleme = 5
yakınlık = 150

face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def seslendir():
    os.system("say Merhabalar! Sizi görebiliyorum. Sesim Var! projesi konuşma engellilere ses sağlamak için, Arda Mavi ve Zümra Uğur tarafından yapılmıştır.")
    return

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    yüz_yakınlık = 0
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        if w > yüz_yakınlık:
            yüz_yakınlık = w
        if goz_bulma:
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    cv2.imshow('Arda Mavi',img)
    if cv2.waitKey(1) == 27: # Decimal 27 = Esc
        break
    if yüz_yakınlık > yakınlık:
        seslendir()
        time.sleep(5)

cap.release()
cv2.destroyAllWindows()
