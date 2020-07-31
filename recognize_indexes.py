import cv2
import numpy as np

import classifier
import preprocess


def remove_noise(img):
    kernel_size = 25
    repair_kernel_size = 6

    img = preprocess.deshadow(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (255, 255, 255), 2)

    # Remove vertical
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (255, 255, 255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, repair_kernel_size))
    result = 255 - cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (repair_kernel_size, 1))
    result = 255 - cv2.morphologyEx(255 - result, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return result


def slice_image_lines(img, minimum=20, padding=10):
    th, threshed = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)

    ## (5) find and draw the upper and lower boundary of each lines
    hist = np.mean(threshed, axis=1)

    th = 4
    height, _width = img.shape[:2]
    lowers = [y for y in range(height - 1) if hist[y] <= th and hist[y + 1] > th]
    uppers = [y for y in range(height - 1) if hist[y] > th and hist[y + 1] <= th]

    if len(uppers) and len(lowers):
        if uppers[0] < lowers[0]:
            upper = uppers.pop(0)

    lines = []

    for lower, upper in zip(lowers, uppers):
        if upper - lower > minimum:
            if lower < padding:
                lower = padding
            if height - upper < padding:
                upper = height - padding
            line = img[lower - padding:upper + padding, ]
            lines.append(line)

    return lines


def slice_words(img, minimum=5, padding=10, block_size=30):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _ret, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hist = np.mean(threshed, axis=0)

    histogram = []

    block = []

    for index, number in enumerate(hist):
        block.append(number)

        if index % block_size == block_size - 1:
            mean_of_block = np.mean(block)

            for i in block:
                histogram.append(mean_of_block)
            block = []

    th = 0
    _height, width = img.shape[:2]
    lowers = [x for x in range(width - block_size) if histogram[x] <= th and histogram[x + 1] > th]
    uppers = [x for x in range(width - block_size) if histogram[x] > th and histogram[x + 1] <= th]

    words = []

    for lower, upper in zip(lowers, uppers):
        if upper - lower > minimum:
            if lower < padding:
                lower = padding

            if width - upper < padding:
                upper = width - padding

            word = img[:, lower - padding:upper + padding]
            words.append(word)

    return words


def slice_digits(img, minimum=5, padding=10):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _ret, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hist = np.mean(threshed, axis=0)

    th = 0
    _height, width = img.shape[:2]
    lowers = [x for x in range(width - 1) if hist[x] <= th and hist[x + 1] > th]
    uppers = [x for x in range(width - 1) if hist[x] > th and hist[x + 1] <= th]

    digits = []

    for lower, upper in zip(lowers, uppers):
        if upper - lower > minimum:
            if lower < padding:
                lower = padding

            if width - upper < padding:
                upper = width - padding

            digit = img[:, lower - padding:upper + padding]
            digits.append(digit)

    return digits


def call(image):
    model = classifier.load_model()

    page, _ = preprocess.into_page(image)

    cleaned = remove_noise(page)

    lines = slice_image_lines(cleaned)

    indexes = []

    for line in lines:
        words = slice_words(line, block_size=20)

        word_digits = []
        digits_size = 0

        for word in reversed(words):
            digits = slice_digits(word)
            digits_size += len(digits)
            word_digits.extend(reversed(digits))
            # min 3 "digits"
            if digits_size >3:
                break
        word_digits = reversed(word_digits)

        predicted_digits = []

        for digit in word_digits:
            (h, w) = digit.shape[:2]
            if h > 300 or w > 300:
                print("Invalid digit")
                continue
            resized_digit = preprocess.resize(digit, 28)

            predicted_digit, _probab = classifier.predict(model, resized_digit)

            predicted_digits.append(predicted_digit)

        index = ''.join([str(x) for x in predicted_digits])
        indexes.append(index)

    return indexes
