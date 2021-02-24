from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import sys
import tkinter
import tkinter.font


# we are using argparse for taking the address of image as argument

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    k = maxWidth * 297 / 210
    dst = np.array([
        [tl[0] + 0, tl[1] + 0],
        [tl[0] + maxWidth - 1, tl[1] + 0],
        [tl[0] + maxWidth - 1, tl[1] + k - 1],
        [tl[0] + 0, tl[1] + k - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

    return warped


'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())'''

# now we will loade the image from th provided address and
# clone it in a variable called orig

image = cv2.imread("C:/Users/fxuvh/Desktop/opencv/notebooks/image/A5.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# now we will convert the image in gray scale for reducing dimensions.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# Step one comes here with showing the result detecting edges.

print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# now we wil find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    print(len(approx))
    if len(approx) == 4:
        screenCnt = approx
        print(screenCnt)
        break

# showing the the outline of the paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(screenCnt.reshape(4, 2))
# now we will use four point transform to obtain a top-down of the image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
cv2.waitKey(0)

# A4 위치에 29cm 표시
warped = imutils.resize(warped, height=500)
gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break
screenCnt = screenCnt.reshape(4, 2)
a4 = np.sqrt(((screenCnt[0][0] - screenCnt[1][0]) ** 2) + ((screenCnt[0][1] - screenCnt[1][1]) ** 2)) * (
            650 / warped.shape[0])

cv2.drawContours(warped, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

contours, hierachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(warped, [contours[0]], 0, (0, 0, 255), 2)

gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# 시-토마스의 코너 검출 메서드
corners = cv2.goodFeaturesToTrack(gray, 100, 0.05, 15, blockSize=5, useHarrisDetector=True, k=0.03)
# 실수 좌표를 정수 좌표로 변환
corners = np.int32(corners)

# 좌표에 동그라미 표시
for corner in corners:
    x, y = corner[0]
    cv2.circle(warped, (x, y), 1, (255, 0, 0), 8, cv2.LINE_AA)

i = 0
x1 = 0
y1 = 0
# 마우스 콜백 함수: 연속적인 원을 그리기 위한 콜백 함수

cv2.namedWindow('drawing event')
warped = imutils.resize(warped, height=650)
cv2.imshow('drawing event', warped)
font = cv2.FONT_HERSHEY_DUPLEX


def DrawConnectedCircle(event, x, y, flags, param):
    global drawing
    global i
    global x1, y1
    # 마우스 왼쪽 버튼이 눌리면 드로윙을 시작함
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(warped, (x, y), 2, (0, 0, 255), -1)
        if i == 0:
            x1 = x
            y1 = y
            i += 1
        else:
            cv2.line(warped, (x1, y1), (x, y), (0, 0, 255), 2)
            k = round(((np.sqrt((x1 - x) ** 2 + (y1 - y) ** 2) / a4) * 29), 2)
            print(k, "cm")
            centi = str(k)
            cv2.putText(warped, centi + " cm", ((int)((x1 + x) / 2), (int)((y1 + y) / 2)), font, 0.7, (255, 128, 114),
                        2, 1)

            i = 0
        cv2.imshow('drawing event', warped)


    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.destroyAllWindows()


cv2.setMouseCallback('drawing event', DrawConnectedCircle)

out = 0
i = 0

k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()