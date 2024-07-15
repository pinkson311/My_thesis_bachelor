'''
Purpose: Generates csv file of annotations from .txts
'''
import pandas as pd
import os
from tqdm import tqdm
import argparse
import math
import cv2



# parse arguments
INPUTDIR = 'original_data/train_annots/'
FILENAME = 'annotations.csv'
folder_img = os.path.join('/home/minh/Desktop/DATN_Son/My_thesis_bachelor/original_data/train_images')

df = pd.DataFrame(columns=['filename', 'class', 'x_min', 'y_min', 'x_max', 'y_max', 'distance'])
                          

def assign_values(filename, idx, list_to_assign):
    df.at[idx, 'filename'] = os.path.join('original_data/train_images', 
                                          filename.replace('.txt', '.png'))
    df.at[idx, 'class'] = list_to_assign[0]

    # xmin = float(list_to_assign[4])
    # ymin = float(list_to_assign[5])
    # xmax = float(list_to_assign[6])
    # ymax = float(list_to_assign[7])

    # # bbox coordinates yolo format
    # x_center = (xmin + xmax) / 2 / img_width
    # y_center = (ymin + ymax) / 2 / img_height
    # width = (xmax - xmin) / img_width
    # height = (ymax - ymin) / img_height

    df.at[idx, 'x_min'] = list_to_assign[4]
    df.at[idx, 'y_min'] = list_to_assign[5]
    df.at[idx, 'x_max'] = list_to_assign[6]
    df.at[idx, 'y_max'] = list_to_assign[7]

    df.at[idx, 'distance'] = math.sqrt(pow(float(list_to_assign[11]), 2) +
                                       pow(float(list_to_assign[12]), 2) +
                                       pow(float(list_to_assign[13]), 2))

    

all_files = sorted(os.listdir(INPUTDIR))
pbar = tqdm(total=len(all_files), position=1)

count = 0
for idx, f in enumerate(all_files):
    pbar.update(1)

    # img_path = os.path.join(folder_img, f.replace('.txt', '.png'))
    # img = cv2.imread(img_path)
    # height, width = img.shape[:2]

    file_object = open(INPUTDIR + f, 'r')
    file_content = [x.strip() for x in file_object.readlines()]

    for line in file_content:
        elements = line.split()
        if elements[0] == 'DontCare':
            continue

        assign_values(f, count, elements)
        count += 1

df.to_csv(FILENAME, index=False)
