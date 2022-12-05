import os
import json
from natsort import natsorted
from sklearn.model_selection import train_test_split


def json_save(data_list, dtype):
    save_data_type = dtype + '_list.json'
    with open(save_data_type, 'w') as jf:
        json.dump(data_list, jf)


def data_split():
    img_root_path = "./images"
    img_list = natsorted(os.listdir(img_root_path))

    train_list, test_list = train_test_split(img_list, test_size=0.2, shuffle=True, random_state=123)
    val_list, test_list = train_test_split(test_list, test_size=0.5, shuffle=True, random_state=123)

    json_save(train_list, 'train')
    json_save(test_list, 'test')
    json_save(val_list, 'val')


if __name__ == "__main__":
    data_split()