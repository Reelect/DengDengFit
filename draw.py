import cv2
import numpy as np

drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

# mouse callback function
def draw(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), True)
        else:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)


img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('Window')
cv2.setMouseCallback('Window',draw)
while(1):
       cv2.imshow('Window',img)
       k=cv2.waitKey(1)&0xFF
       if k==27:
           break
cv2.destroyAllWindows()