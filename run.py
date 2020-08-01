#!/usr/bin/env python

import os
import shutil
import sys

import cv2

import mark_text
import recognize_indexes


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def remove_dir_if_exists(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)


def process_images(input_data_dir, output_data_dir, n_files):
    remove_dir_if_exists(output_data_dir)
    create_dir_if_not_exists(output_data_dir)

    for file_number in range(0, n_files):
        input_file_path = f'{input_data_dir}/{file_number}.png'
        output_indexes_file_path = f'{output_data_dir}/{file_number}-indeksy.txt'
        output_words_file_path = f'{output_data_dir}/{file_number}-wyrazy.png'

        image = cv2.imread(input_file_path)

        print(f"{file_number}. Mark text")
        marked_text = mark_text.call(image)
        cv2.imwrite(output_words_file_path, marked_text)
        print(f"{file_number}. Recognize index")
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
