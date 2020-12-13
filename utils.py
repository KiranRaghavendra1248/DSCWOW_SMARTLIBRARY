import os
import torch
from torchvision import transforms
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import time
from datetime import date
import pickle
from PIL import Image


def save_data(frames, boxes, model):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    for f in range(len(frames)):
        img = np.asarray(frames[f])
        box = boxes[f]
        if len(box.shape) == 3:
            # Go into loop only when there is atleast 1 face in image
            # Loop for num of boxes in each image
            for b in range(box.shape[1]):
                start = (np.clip(int(box[0][b][0]) - 15, 0, 480), np.clip(int(box[0][b][1]) - 50, 0, 640))
                end = (np.clip(int(box[0][b][2]) + 15, 0, 480), np.clip(int(box[0][b][3]) + 20, 0, 640))
                crop_pic = img[start[1]:end[1], start[0]:end[0]]
            img_crop = Image.fromarray(crop_pic)
            img_crop = transform(img_crop)
            img_crop = torch.unsqueeze(img_crop, 0)
            save_tensor = model(img_crop)
            return save_tensor

def detect_imgs(img,transform,threshold,model,faces,usn_nums):
    global current_person
    person_ = None
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img = model(img)
    minimum = torch.tensor(99)
    for face_, usn in zip(faces, usn_nums):
        temp = torch.min(torch.norm((face_- img), dim=1))
        if temp < minimum and temp < threshold:
            minimum = temp
            person_ = usn
    return person_, minimum.item()


def show_images(frames, boxes, color,transform,threshold,model,faces,usn_nums,usn_number):
    temp = None
    for f in range(len(frames)):
        img = np.asarray(frames[f])
        box = boxes[f]
        if len(box.shape) == 3:
            # Go into loop only when there is atleast 1 face in image
            # Loop for num of boxes in each image
            for b in range(box.shape[1]):
                start = (np.clip(int(box[0][b][0]) - 15, 0, 600), np.clip(int(box[0][b][1]) - 20, 0, 600))
                end = (np.clip(int(box[0][b][2]) + 15, 0, 600), np.clip(int(box[0][b][3]) + 15, 0, 600))
                img = cv2.rectangle(img, start, end, color, 2)
                crop_pic = img[start[1]:end[1], start[0]:end[0]]
                crop_pic = Image.fromarray(crop_pic)
                person, diff,  = detect_imgs(crop_pic,transform,threshold,model,faces,usn_nums)
                if person is not None:
                    cv2.putText(img, person + ': ' + '{:.2f}'.format(diff), (start[0], start[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    temp = 1
                else:
                    cv2.putText(img, 'Unknown' + ': ' + '{0}'.format(diff), (start[0], start[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    temp = 0
        if temp == 1 and usn_number==person:
            return img,1
        else:
            return img,0