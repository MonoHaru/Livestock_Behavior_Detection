import json
import os
import re
import pickle

base_path = r'E:\Kim\ccc\yolov5-master\data\asd'
types = None

def data(classes, file_name, file_list):
    global types
    file_name = file_name[0:-5]
    txt_path = r'E:\Kim\ccc\yolov5-master\data\realdata'
    if types == 'train':
        with open(txt_path+'/train/'+file_name+".txt", 'w') as file:
            for i in range(len(file_list)):
                file.write(file_list[i])
    elif types == 'valid':
        with open(txt_path+'/valid/'+file_name+".txt", 'w') as file:
            for i in range(len(file_list)):
                file.write(file_list[i])

def bbox2yolov5(annotations, class_name, images):
    xywh_list = []

    full_x, full_y = images['width'], images['height']

    # 클래스(classes) 설정
    if "cow" in class_name:
        classes = 0
    else:
        classes = 1

    anno = annotations

    # 바운딩 박스 값 -> yolov5 레이블 형태로 변환
    for i in range(len(annotations)):
        x, y, w, h = anno[i]['bbox']
        x_ = (x + w) / 2 / full_x
        y_ = (y + h) / 2 / full_y
        w_ = (w - x) / full_x
        h_ = (h - y) / full_y

        xywh_list.extend([str(classes), ' ', str(x_), ' ', str(y_), ' ', str(w_), ' ', str(h_), '\n'])
    data(classes,class_name, xywh_list)


if __name__ == "__main__":

    train_Or_valid = os.listdir(base_path)
    for i in range(len(train_Or_valid)):
        path = os.path.join(base_path, train_Or_valid[i])
        if train_Or_valid[i] == 'train':
            types = 'train'
        elif train_Or_valid[i] == 'valid':
            types = 'valid'
        # train이나 valid의 리스트
        path_list = os.listdir(path)
        for list in path_list:
            with open(os.path.join(path, list), "r") as js:
                json_data = json.load(js)
                bbox2yolov5(json_data['label_info']['annotations'], list, json_data['label_info']['image'])