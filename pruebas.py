# importar bibliotecas
import cv2 as cv
import numpy as np

# VENTANAS
window_name = 'threshold'
color_white = (255,255,255)

# KERNEL
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

# TRACKBARS
from trackBar import create_trackbar, on_trackbar, get_trackbar_value

trackbar_window_name = 'Trackbars'
cv.namedWindow(trackbar_window_name)

max_value_thresh = 255
thresh_name = 'Threshold'
create_trackbar(thresh_name, trackbar_window_name, max_value_thresh)

max_value_morph = 15
opening_name = 'Opening'
closing_name = 'Closing'
dilate_name = 'Dilate'
erode_name = 'Erode'
create_trackbar(opening_name, trackbar_window_name, max_value_morph)
create_trackbar(closing_name, trackbar_window_name, max_value_morph)
create_trackbar(erode_name, trackbar_window_name, max_value_morph)
create_trackbar(dilate_name, trackbar_window_name, max_value_morph)


while True:

    # IMAGEN
    img = cv.imread('proyectos\proyecto3\levadura.png')
    img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # segmentación con threshold
    thresh_value = get_trackbar_value(thresh_name, trackbar_window_name)
    opening_value = get_trackbar_value(opening_name, trackbar_window_name)
    closing_value = get_trackbar_value(closing_name, trackbar_window_name)


    _, thresh = cv.threshold(img2, thresh_value, max_value_thresh, cv.THRESH_BINARY)
    openinig = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=opening_value)
    closing = cv.morphologyEx(openinig, cv.MORPH_OPEN, kernel, iterations=closing_value)

    cv.imshow('threshold', thresh)
    # cv.imshow('opening', openinig)
    # cv.imshow('closing', closing) # NO SE MODIFICA CON LA TRACKBAR

    # semillas o marcadores
    erode_value = get_trackbar_value(erode_name, trackbar_window_name)
    dilate_value = get_trackbar_value(dilate_name, trackbar_window_name)

    foreground = cv.erode(closing, kernel, iterations=erode_value)
    background = cv.dilate(closing, kernel, iterations=dilate_value)
    unknown = cv.subtract(background, foreground)

    # individualización con componentes conectados
    number, markers = cv.connectedComponents(foreground)
    markers = markers + 1 # para ponerle al fondo una semilla
    markers[unknown == 255] = 0 # para marcar lo desconocido como desconocido (para watershed la etiqueta 0 es para lo que no tiene semilla)

    map = np.uint8(255 * markers / number)
    color_map = cv.applyColorMap(map, cv.COLORMAP_JET)
    cv.imshow('color_map', color_map)

    # segmentación con watershed
    markers = cv.watershed(img, markers)
    img[markers == -1] = [0,0,255] # se toma el área del borde como -1

    numberSTR = str(number-1)
    number_cell = 'cantidad celulas:'+ numberSTR
    cv.putText(img, number_cell, (10, 800), cv.FONT_HERSHEY_SIMPLEX , 1, color=color_white)
    cv.imshow('watershed', img)

    # Clasificación por áreas
    if cv.waitKey(1) == ord("c"): # es necesario esperar a la actualización d elas trackbars
        
        # contornos
        contours, hierachy = cv.findContours(image=foreground, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)

        # áreas
        areas = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            areas.append(area)
        
        # clasificación
        clasification_value = 960
        small = []
        big = []
        i = 0
        while i < len(areas):
            if areas[i] < clasification_value:
                small.append(contours[i])
            else:
                big.append(contours[i])
            i = i+1

        color_green = color_green = (0, 255, 0)
        color_blue = (255, 0, 0)
        cv.drawContours(img, small, -1, color_green, 3)
        cv.putText(img, 'celulas pequenas', (10, 40), cv.FONT_HERSHEY_SIMPLEX , 1, color=color_green)
        cv.drawContours(img, big, -1, color_blue, 3)
        cv.putText(img, 'celulas grandes', (10, 70), cv.FONT_HERSHEY_SIMPLEX , 1, color=color_blue)
        cv.imshow('Clasificacion por area', img)


    tecla = cv.waitKey(30)  # espera 30 ms para que se presione una tecla . El mínimo es 1 ms.  tecla == 0 si no se pulsó ninguna.
    if tecla == 27:	# tecla ESC para salir
        break