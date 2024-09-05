from glob import glob
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='none')
parser.add_argument('--data_set', type=int, default=1)
args = parser.parse_args()


def update(file, old_str, new_str):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str,new_str)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)

# your_dataset_path = args.data_root
# all_txt_file = glob(os.path.join('AFEW_*.txt'))
# for txt_file in all_txt_file:
#     update(txt_file, "./AFEW", your_dataset_path)

your_dataset_path = args.data_root
all_txt_file = glob(os.path.join('DFEW_set_{data_set}*.txt'.format(data_set=str(args.data_set))))
for txt_file in all_txt_file:
    update(txt_file, "/home/user/datasets/DFEW_Face/", your_dataset_path)

# your_dataset_path = ".../FERV39K/"
# all_txt_file = glob(os.path.join('FERV39K_*.txt'))
# for txt_file in all_txt_file:
#     update(txt_file, "/home/user/datasets/FERV39K/", your_dataset_path)
#

# python script.py --data_root D:\Dataset\DFEW\data_affine\single_label\data\