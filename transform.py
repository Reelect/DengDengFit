# following lines show how to import different packages
import numpy as np
import cv2


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
    axis = 0
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    if maxWidth > maxHeight :
        axis = 1
        k = maxHeight*297/210
        dst = np.array([
            [tl[0] + 0, tl[1] + 0],
            [tl[0] + k - 1, tl[1] + 0],
            [tl[0] + k - 1, tl[1] + maxHeight - 1],
            [tl[0] + 0, tl[1] + maxHeight - 1]], dtype="float32")
    else:
        k = maxWidth*297/210
        dst = np.array([
            [tl[0] + 0, tl[1] + 0],
            [tl[0] + maxWidth - 1, tl[1] + 0],
            [tl[0] + maxWidth - 1, tl[1] + k - 1],
            [tl[0] + 0, tl[1] + k - 1]], dtype="float32")


    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

    return warped, axis