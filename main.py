import os
import re
import shutil
import sys

import cv2

import mark_text
import recognize_indexes


def sorted_files(path):
    x = os.listdir(path)
    x.sort(key=lambda f: int(re.sub('\D', '', f)))
    return x


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def remove_dir_if_exists(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)


def extract_file_number(file_name):
    res = int(re.sub('\D', '', file_name))
    return res


def process_images(input_data_dir, output_data_dir, n_files):
    remove_dir_if_exists(output_data_dir)
    create_dir_if_not_exists(output_data_dir)

    for number, file_name in enumerate(sorted_files(input_data_dir)):
        if number >= n_files:
            break

        file_number = extract_file_number(file_name)

        input_file_path = '{}/{}'.format(input_data_dir, file_name)
        output_indexes_file_path = '{}/{}-indeksy.txt'.format(output_data_dir, file_number)
        output_words_file_path = '{}/{}-wyrazy.png'.format(output_data_dir, file_number)

        image = cv2.imread(input_file_path)

        marked_text = mark_text.call(image)
        cv2.imwrite(output_words_file_path, marked_text)

        recognized_indexes = recognize_indexes.call(image)
        with open(output_indexes_file_path, 'w') as f:
            f.write('\n'.join(recognized_indexes))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("To few args")
        exit(1)

    input_data_dir = sys.argv[1]
    output_data_dir = sys.argv[3]
    n_files = int((sys.argv[2]))

    process_images(input_data_dir, output_data_dir, n_files)
