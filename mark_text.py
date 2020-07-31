import cv2
import numpy as np

from preprocess import into_page, deshadow

THRESHOLD = 38


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
    lowers = [y for y in range(H - 1) if hist[y] <= th < hist[y + 1]]
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
