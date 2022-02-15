import cv2
import numpy as np
import dlib
from math import hypot

inreg = cv2.VideoCapture(0)

#INTERFATA

#cvSet(img, CV_BGR(0,0,0))
interfata = np.zeros((800, 1200, 3), np.uint8)
elemente_interfata = {0: "DA", 1: "NU", 2: "NU Stiu", 3: "FOAME", 4: "SETE", 5: "OBOSEALA"}
culoare1=(0, 255, 0) #apasat
culoare2=(0, 0, 255) #liber

#plasamentul butoanelor
def buton(nr_buton, text, buton_apasat):
    if nr_buton == 0:
        x = 0
        y = 0
    if nr_buton == 1:
        x = 400
        y = 0
    if nr_buton == 2:
        x = 800
        y = 0
    if nr_buton == 3:
        x = 0
        y = 200
    if nr_buton == 4:
        x = 0
        y = 400
    if nr_buton == 5:
        x = 0
        y = 600

    inaltime = 200
    if nr_buton <=2:
        latime = 400
    else:
        latime = 1200
    th = 3

    if buton_apasat is True:
        cv2.rectangle(interfata, (x + th, y + th), (x + latime - th, y + inaltime - th), culoare1, -1)
    else:
        cv2.rectangle(interfata, (x + th, y + th), (x + latime - th, y + inaltime -th), culoare2, th)
    # TEXT
    font_litera = cv2.FONT_HERSHEY_PLAIN
    dimensiune = 7
    font_th = 4
    text_size = cv2.getTextSize(text, font_litera, dimensiune, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((latime - width_text) / 2) + x
    text_y = int((inaltime + height_text) / 2) + y
    cv2.putText(interfata, text, (text_x, text_y), font_litera, dimensiune, (255, 0, 0, 0), font_th)

#DETECTOARE-dlib
detector_fata = dlib.get_frontal_face_detector()
detector_ochi = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#FUNCTII CLIPIRE PRIVIRE
def mijloc(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def getClipire(puncte_ochi , structura_fata):
    extremitate_stanga = (structura_fata.part(puncte_ochi[0]).x, structura_fata.part(puncte_ochi[0]).y)
    center_top = mijloc(structura_fata.part(puncte_ochi[1]), structura_fata.part(puncte_ochi[2]))
    extremitate_dreapta = (structura_fata.part(puncte_ochi[3]).x, structura_fata.part(puncte_ochi[3]).y)
    center_bottom = mijloc(structura_fata.part(puncte_ochi[4]), structura_fata.part(puncte_ochi[5]))
    #raportul dintre linia verticala si cea orizontala
    raport = hypot((extremitate_stanga[0] - extremitate_dreapta[0]), (extremitate_stanga[1] - extremitate_stanga[0])) /hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    return raport

def getPrivire(puncte_ochi, structura_fata):

    zona_ochi = np.array([(structura_fata.part(puncte_ochi[0]).x, structura_fata.part(puncte_ochi[0]).y),
                          (structura_fata.part(puncte_ochi[1]).x, structura_fata.part(puncte_ochi[1]).y),
                          (structura_fata.part(puncte_ochi[2]).x, structura_fata.part(puncte_ochi[2]).y),
                          (structura_fata.part(puncte_ochi[3]).x, structura_fata.part(puncte_ochi[3]).y),
                          (structura_fata.part(puncte_ochi[4]).x, structura_fata.part(puncte_ochi[4]).y),
                          (structura_fata.part(puncte_ochi[5]).x, structura_fata.part(puncte_ochi[5]).y)], np.int32)
    #generare MASK

    inaltime, latime, _ = frame.shape
    mask = np.zeros((inaltime, latime), np.uint8)
    cv2.polylines(frame, [zona_ochi], True, 255, 2)
    cv2.fillPoly(mask, [zona_ochi], 255)

    min_x = np.min(zona_ochi[:, 0])
    max_x = np.max(zona_ochi[:, 0])
    min_y = np.min(zona_ochi[:, 1])
    max_y = np.max(zona_ochi[:, 1])

    ochi= cv2.bitwise_and(gray, gray, mask=mask)
    gray_eye= ochi[min_y: max_y, min_x: max_x]

	#generare THRESHOLD

    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    inaltime, latime = threshold_eye.shape
    threshold_stanga = threshold_eye[0: inaltime, 0: int(latime / 2)]
    zona_alba_stanga = cv2.countNonZero(threshold_stanga)
    threshold_dreapta = threshold_eye[0: inaltime, int(latime / 2): latime]
    zona_alba_dreapta = cv2.countNonZero(threshold_dreapta)

    if zona_alba_stanga == 0:
        privire = 1
    elif zona_alba_dreapta == 0:
        privire = 5
    else:
        privire = zona_alba_stanga / zona_alba_dreapta
    return privire

#VARIABILE
frames = 0
frames_clipire = 0
nr_buton = 0
text=""

board = np.zeros((500, 500), np.uint8)
board[:] = 255

while True:
    _, frame = inreg.read()

    interfata[:] = (0, 0, 0)
    buton_selectat = elemente_interfata[nr_buton]
    frames += 1
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    elemente = detector_fata(gray)
    for element in elemente:
        #print(element)
        x2, y2  = element.left(), element.top()
        x1, y1 = element.right(), element.bottom()
        cv2.rectangle(frame, (x2,y2), (x1,y1), (0, 255, 0), 2)
        landmarks = detector_ochi(gray, element)
##CLIPIRE

        ochi_stang = getClipire([36, 37, 38, 39, 40, 41], landmarks)
        ochi_drept = getClipire([42, 43, 44, 45, 46, 47], landmarks)
        clipire = (ochi_stang + ochi_drept)/2

        if clipire > 10:
            cv2.putText(frame, "Clipesti", (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), thickness=3)
            frames_clipire += 1
            frames -= 1
            if frames_clipire == 5:
                text += buton_selectat
        else:
            frames_clipire = 0
        landmarks = detector_ochi(gray, element)
##PRIVIRE
    privire_ochi_stang = getPrivire([36, 37, 38, 39, 40, 41], landmarks)
    privire_ochi_drept = getPrivire([42, 43, 44, 45, 46, 47], landmarks)
    privire = (privire_ochi_stang + privire_ochi_drept) / 2

    if privire <= 0.87:
        cv2.putText(frame, "STANGA", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        new_frame[:] = (0, 0, 255)
    elif 0.87 < privire < 1.26:
        cv2.putText(frame, "CENTRU", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    else:
        new_frame[:] = (255, 0, 0)
        cv2.putText(frame, "DREAPTA", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
##BUTOANE
    if frames == 15:
        nr_buton += 1
        frames = 0
    if nr_buton == 6:
        nr_buton = 0

    for i in range(6):
        if i == nr_buton:
            selectare = True
        else:
            selectare = False
        buton(i, elemente_interfata[i], selectare)
#afisare selectie
    cv2.putText(board, text, (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, 0, 3)
#FERESTRE
    cv2.imshow("Frame", frame)
    # cv2.imshow("New frame", new_frame)
    cv2.imshow("Interfata utilizator", interfata)
    cv2.imshow("TextPad", board)
    key = cv2.waitKey(1)
    if key == 27:
        break


inreg.release()
cv2.destroyAllWindows()