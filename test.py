import torch
from PIL import Image
import cv2

model = torch.hub.load('E:\Kim\ccc\yolov5-master', 'custom', path='best.pt', source='local')

# image1 = Image.open('data/images/valid/livestock_cow_bbox_006111.jpg')
# result1 = model(image1)

# image2 = cv2.imread('data/images/valid/livestock_cow_bbox_006111.jpg')[..., ::-1]
# result2 = model(image2)

# result1.print()
# print(result1.xyxy[0])
# print(result1.pandas().xyxy[0])


base_path = r'E:\Kim\ccc\yolov5-master\data\images\valid'
import os
data_list = os.listdir(base_path)
import pandas as pd

def transform2pandas(result, image_):
    df = None
    img_txt = image_
    result_ = result
    for i in range(len(result_)):
        new_df = pd.DataFrame([{'ImageID':img_txt,
                               'LabelName':result_['name'][i],
                               'Conf':result_['confidence'][i],
                               'XMin':result_['xmin'][i],
                               'XMax':result_['xmax'][i],
                               'YMin':result_['ymin'][i],
                               'YMax':result_['ymax'][i]}])
        df = pd.concat([df, new_df])
    return df

final_df = None

for image_txt in data_list:
    image = Image.open(base_path+'/'+image_txt)
    result = model(image, augment=True)
    df = transform2pandas(result.pandas().xyxy[0], image_txt)
    final_df = pd.concat([final_df, df])

final_df.to_csv('new_output.csv', index=False)