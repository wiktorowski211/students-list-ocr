import cv2
import numpy as np
from imutils.perspective import order_points


def deshadow(image):
    rgb_planes = cv2.split(image)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result = cv2.merge(result_planes)
    return result


def relight_clahe(bgr, clipLimit=2.0, tileGridSize=(8, 8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_dims = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    lab_dims[0] = clahe.apply(lab_dims[0])
    lab = cv2.merge(lab_dims)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image and it's perspective transform
    return warped, M


# load the image and return only the piece of paper
# returns new_image, transform tuple
def into_page(image, clipMax=31):
    clipLimit = 1

    while clipLimit <= clipMax:
        clipLimit = clipLimit + 0.5
        # CLAHE uses LAB
        light = relight_clahe(image, clipLimit=clipLimit)

        gray = cv2.cvtColor(light, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1]

        # get rid of the checkerboard and/or text to detect white piece of paper
        kernel = np.ones((8, 8), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)

        # Find contours and sort for largest contour
        cnts = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        displayCnt = None

        for c in cnts:
            # Perform contour approximation
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                rect = cv2.minAreaRect(approx)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                rect_area = cv2.contourArea(box)
                polygon_area = cv2.contourArea(approx)
                if rect_area > 1 and (polygon_area / rect_area) > 0.9:
                    displayCnt = approx
                    break

        if displayCnt is None:
            continue
            # if the captured "page" is actually just some small element
        elif cv2.contourArea(displayCnt) < image.shape[0] * image.shape[1] * 0.5:
            continue
        else:
            return four_point_transform(image, displayCnt.reshape(4, 2))
    return image, None


def resize(image, size, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if h > w:
        # print("there")
        r = size / float(h)
        # print(f'r: {r}')
        dim = (int(w * r), size)
        # print(f'dim: {dim}')
        # print(f'{h}x{w}')
        resized = cv2.resize(image, dim, interpolation=inter)

        (h, w) = resized.shape[:2]

        padding = int((h - w) / 2)
        res = cv2.copyMakeBorder(resized, 0, 0, padding, padding, cv2.BORDER_CONSTANT)
    else:
        # print("here")
        r = size / float(w)
        dim = (size, int(h * r))
        resized = cv2.resize(image, dim, interpolation=inter)

        (h, w) = resized.shape[:2]

        padding = int((w - h) / 2)
        res = cv2.copyMakeBorder(resized, padding, padding, 0, 0, cv2.BORDER_CONSTANT)

    res = cv2.resize(res, (size, size), interpolation=inter)

    return res
