import torch
import cv2
import numpy as np



# leer el modelo 


model = torch.hub.load('ultralytics/yolov5','custom',
                        path ='/home/gbl-net/Documentos/Yolo_2/model/best_2.pt')

# videocapture 

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    # realizamos la detencion 

    detect = model (frame)

    # Mostrar FPS
    cv2.imshow('Detencion de objetos', np.squeeze(detect.render()))
     #leer el teclado 
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()    
cv2.destroyAllWindows()



