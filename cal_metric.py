from __future__ import annotations
import os
import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from monai.transforms import LoadImage
import argparse


def filter_label(x:torch.Tensor, cls_idx:List[int]):
    ret = torch.zeros_like(x).to(bool)
    if isinstance(cls_idx, list):
        for idx in cls_idx:
            ret[x == idx] = 1
    else: 
        ret[x == cls_idx] = 1
    return ret


def eval_cls(pred:torch.Tensor, label:torch.Tensor, cls_idx:List[int],filter=True):
    if filter:
        pred_filtered = filter_label(pred, cls_idx)
        label_filtered = filter_label(label, cls_idx)
    else:
        pred_filtered = pred, cls_idx
        label_filtered = label, cls_idx
    # if torch.sum(label_filtered) == 0 and torch.sum(pred_filtered)==0: 
    #     return 1.
    # elif torch.sum(label_filtered) == 0 and torch.sum(pred_filtered)>0:
    #     return -1
    # intersection = pred_filtered & label_filtered
    # return 2 * torch.sum(intersection) / (torch.sum(pred_filtered) + torch.sum(label_filtered))
    if torch.sum(label_filtered) == 0: return 1.
    intersection = pred_filtered & label_filtered
    return 2 * torch.sum(intersection) / (torch.sum(pred_filtered) + torch.sum(label_filtered))

def eval_cls_multi_class(pred:torch.Tensor, label:torch.Tensor, idx):
    pred_filtered = pred[:,idx,::]
    label_filtered = label[:,idx,::]
    if torch.sum(label_filtered) == 0: return 1.
    intersection = pred_filtered & label_filtered
    return 2 * torch.sum(intersection) / (torch.sum(pred_filtered) + torch.sum(label_filtered))



def eval_res(label_dict:Dict[str, List | int], labels_folder:str, preds_folder:str):
    res_dict = {}
    valid_dice = []
    fake_sample = 0
    for label_file_name in tqdm(os.listdir(labels_folder)):
        label_file = os.path.join(labels_folder, label_file_name)
        pred_file = os.path.join(preds_folder, label_file_name)
        label = LoadImage()(label_file)
        pred = LoadImage()(pred_file)
        gt_num = np.count_nonzero(label == 3)
        # print(label_file_name, label.shape, torch.sum((label != 0) & (label != 1) & (label != 2)))
        for label_name in label_dict:
            if label_name not in res_dict: res_dict[label_name] = 0.
            dice = eval_cls(pred, label, label_dict[label_name])
            # if gt_num>0 and label_name=="cls3":
            if label_name=="cls3":
                print(label_name, dice)
                if gt_num>0:
                    valid_dice.append(dice)
            res_dict[label_name] += dice
    
    avg_dict = {label_name: res_dict[label_name] / len(os.listdir(labels_folder)) for label_name in res_dict}
    print("avg", avg_dict)
    print("cls avg", np.mean([avg_dict[k] for k in avg_dict]))
    print("valid dice: ",f"{np.mean(valid_dice):.4f}","fake_sample:",fake_sample)
    
@torch.no_grad()   
def eval_res_online(model,loader,label_dict):
    model.eval()
    print("cur dataset length: ",len(loader))
    res_dict = {}
    valid_dice = []
    for (x,y) in tqdm(loader,desc="validation"):
        x, y = x.cuda(), y.cuda()
        y_pred = model.forward(x)
        y_pred_binary = (y_pred > 0.5).float()
        # 将二值化的结果转换为类别
        # y_pred_converted = torch.zeros_like(y_pred_binary)
        # y_pred_converted[y_pred_binary[:, 0, :, :] == 1] = 1
        # y_pred_converted[y_pred_binary[:, 1, :, :] == 1] = 2
        # y_pred_converted[y_pred_binary[:, 2, :, :] == 1] = 3
        y_pred_converted = y_pred_binary.cpu().numpy().astype(np.uint8)
        
        label = y.cpu().numpy().astype(np.uint8)
        pred = y_pred_converted
        gt_num = np.sum(label[:,-1,::])
        # print(label_file_name, label.shape, torch.sum((label != 0) & (label != 1) & (label != 2)))
        for idx,label_name in enumerate(label_dict):
            if label_name not in res_dict: res_dict[label_name] = 0.
            dice = eval_cls_multi_class(torch.tensor(pred), torch.tensor(label), idx)
            if gt_num>0 and label_name=="cls3":
                # print(label_name, dice)
                valid_dice.append(dice)
            res_dict[label_name] += dice
    
    avg_dict = {label_name: res_dict[label_name] / len(loader) for label_name in res_dict}
    if "cls3" in avg_dict:
        avg_dict['cls3'] = np.mean(valid_dice)
    print("avg", avg_dict)
    print("cls avg", np.mean([avg_dict[k] for k in avg_dict]))
    print("valid dice: ",np.mean(valid_dice))
    return np.mean([avg_dict[k] for k in avg_dict])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_folder", default="",type=str)
    parser.add_argument("--label_folder", default="",type=str)
    args = parser.parse_args()
    eval_res(
        label_dict={
            "cls1": 1,
            "cls2": [2,3],
            "cls3": 3
        },
        labels_folder=args.label_folder,
        preds_folder=args.pred_folder
    )
    
        
