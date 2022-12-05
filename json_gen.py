import os
import json
import numpy as np
from tqdm import tqdm

import argparse


def _bbox_to_center_and_scale(bbox):
    x, y, w, h = bbox
    pixel_std = 200
    
    center = np.zeros(2, dtype=np.float32)
    center[0] = x + w / 2.0
    center[1] = y + h / 2.0
    
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    
    return center, scale


def cvt_kps(kps):
    joints = []
    
    for i in range(0, len(kps), 3):
        joints.append([kps[i], kps[i+1]])
        
    return joints


def gen_json(img_filenames, dtype, all_joints, centers, scales):

    assert len(img_filenames) == len(all_joints) == len(centers) == len(scales)

    dict_list = []
    for img_filename, joints, center, scale in tqdm(zip(img_filenames, all_joints, centers, scales), total=len(img_filenames)):
        joints_vis = [1 for i in range(14)]
        dict = {
            "center": center,
            "scale": float(scale),
            "image": img_filename,
            "joints_vis": joints_vis,
            "joints": joints
        }
        dict_list.append(dict)
    
    save_path = "./gt_json"
    os.makedirs(save_path, exist_ok=True)
    
    save_json_file = os.path.join(save_path, dtype+".json")
    with open(save_json_file, 'w') as save_jf:
        json.dump(dict_list, save_jf)


def make_data(dtype):
    root_path = "./labels"
    annot_json_file = "./" + dtype + "_list.json"
    
    with open(annot_json_file, 'r') as jf:
        img_list = json.load(jf)
    
    json_list = [os.path.splitext(file)[0] + '.json' for file in img_list]
    
    centers = []
    scales = []
    img_filenames = []
    all_joints = []
    
    for i, json_file in enumerate(json_list):
        json_path = os.path.join(root_path, json_file)
        
        with open(json_path, 'r') as jf:
            json_data = json.load(jf)
    
        bbox = json_data['label_info']['annotations'][0]['bbox']
        keypoints = json_data['label_info']['annotations'][0]['keypoints']
        if len(keypoints) // 3 != 14: continue
        img_file = json_data['label_info']['image']['file_name']

        joint = cvt_kps(keypoints)
        
        center, scale = _bbox_to_center_and_scale(bbox)
        scale = scale[0] if scale[0] > scale[1] else scale[1]
        center = [round(center[0]), round(center[1])]
        
        img_filenames.append(img_file)
        all_joints.append(joint)
        centers.append(center)
        scales.append(scale)
        
    gen_json(img_filenames, dtype, all_joints, centers, scales)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make data for the keypoints model.")
    parser.add_argument('--dtype', type=str, default='train', help='Select data type; train, test, val (default: train)')

    args = parser.parse_args()

    make_data(args.dtype)
