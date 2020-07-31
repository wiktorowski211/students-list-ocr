import cv2
import numpy as np
from imutils.perspective import order_points


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
                if (cv2.contourArea(approx) / rect_area) > 0.9:
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

    # uses a PAGE image


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
    result_norm = cv2.merge(result_norm_planes)
    return result


# uses GRAY or BINARY image
def clean_border(img):
    height, width = img.shape
    return cv2.rectangle(img, (0, 0), (width - 1, height - 1), 255, 10)


# uses GRAY image
def has_grid(img):
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    len_cnts = len(cnts)
    inv_thresh = cv2.bitwise_not(thresh)
    inv_cnts = cv2.findContours(inv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    inv_len_cnts = len(inv_cnts)
    return len_cnts > 1000 or inv_len_cnts > 1000


THRESHOLD = 38


def preprocess(image):
    image = cv2.medianBlur(image, 3)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return 255 - image


def get_block_index(image_shape, yx, block_size):
    y = np.arange(max(0, yx[0] - block_size), min(image_shape[0], yx[0] + block_size))
    x = np.arange(max(0, yx[1] - block_size), min(image_shape[1], yx[1] + block_size))
    return np.meshgrid(y, x)


def adaptive_median_threshold(img_in):
    med = np.median(img_in)
    img_out = np.zeros_like(img_in)
    img_out[img_in - med < THRESHOLD] = 255
    return img_out


def block_image_process(image, block_size):
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[block_idx] = adaptive_median_threshold(image[block_idx])

    return out_image


# uses gray image
def clean_grid(image, BLOCK_SIZE=50):
    image_in = preprocess(image)
    image_out = block_image_process(image_in, BLOCK_SIZE)

    return image_out


# uses BINARY cleaned image
def slice_image_lines(img, minimum=20, padding=10, draw=False):
    ## (2) threshold
    th, threshed = cv2.threshold(img, 127, 255, cv2.cv2.THRESH_OTSU)

    ## (5) find and draw the upper and lower boundary of each lines
    hist = np.mean(threshed, axis=1)

    th = 4
    H, W = img.shape[:2]
    lowers = [y for y in range(H - 1) if hist[y] <= th and hist[y + 1] > th]
    uppers = [y for y in range(H - 1) if hist[y] > th and hist[y + 1] <= th]

    lines = []

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for lower, upper in zip(lowers, uppers):
        if upper - lower > minimum:
            if lower < padding:
                lower = padding

            if H - upper < padding:
                upper = W - padding
            line = (lower - padding, upper + padding)
            # draw
            if draw:
                color = colors.pop()
                cv2.line(img, (100, line[0]), (100, line[1]), color, 5)
                colors.insert(0, color)
            lines.append(line)

    return lines, img


def mark_words(page):
    no_shadow = deshadow(page)
    gray = cv2.cvtColor(no_shadow, cv2.COLOR_BGR2GRAY)

    if has_grid(gray):
        minigrid = clean_grid(gray, BLOCK_SIZE=50)
        minigrid = cv2.medianBlur(minigrid, 3)
    else:
        minigrid = gray

    minigrid = clean_border(minigrid)
    binary = cv2.threshold(minigrid, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    contours = page.copy()
    cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # format: (top_left, bottom_right)
    text_bits = []
    for c in cnts:
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(c)

        height, width, channels = page.shape
        # filter too big rectangles
        if rect_w * rect_h > height * width / 10:
            continue
        if rect_w * rect_h >= 20 and rect_h <= 120:
            # text_bits.append(((rect_x, rect_y), (rect_x+rect_w+20, rect_y+rect_h)))
            text_bits.insert(0, ((rect_x, rect_y), (rect_x + rect_w + 20, rect_y + rect_h)))
            # enhance binary image
            cv2.rectangle(binary, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), 255, -2)
        else:
            # cover small grains
            cv2.rectangle(binary, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), 0, -2)

    # inv_binary = cv2.bitwise_not(binary)
    lines = slice_image_lines(binary)[0]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for ((top_left_x, top_left_y)), ((bottom_right_x, bottom_right_y)) in text_bits:
        color = None
        for n, (line_lower, line_upper) in enumerate(lines):
            box_middle = (top_left_y + bottom_right_y + 10) / 2
            if line_lower <= box_middle <= line_upper:
                # color = n+1
                color = colors[n % 3]
                break
        if color is not None:
            cv2.rectangle(contours, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, -1)
        else:
            cv2.rectangle(contours, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 255, 0), -1)
    return contours


def revert_transform(image, page, transform=None):
    if transform is None:
        return page
    max_height, max_width, _ch = image.shape
    retval, inv_transform = cv2.invert(transform)
    return cv2.warpPerspective(page, inv_transform, (max_width, max_height), borderValue=cv2.BORDER_TRANSPARENT)


def call(image):
    page, transf = into_page(image)

    marked_lines = mark_words(page)

    page_on_image = revert_transform(image, marked_lines, transf)

    return page_on_image
