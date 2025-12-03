import os
import re
import json
import argparse
from collections import defaultdict
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU, prepare_texts_my
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
import cv2
import os.path as osp
import copy
import functools
import gc
import logging
import pickle
from collections.abc import Mapping
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union,Dict,Iterable
import numpy as np
import os
import json
import torch
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data.dataloader import default_collate
from PIL import Image
import random
from torchvision.transforms.functional import InterpolationMode
import copy
from PIL import Image

from minigpt4.datasets.datasets.ego_irgbench_dataset import EgoIRGBenchDataset


def list_of_str(arg):
    return list(map(str, arg.split(',')))

def eval_parser():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--name", type=str, default='A2', help="evaluation name")
    parser.add_argument("--ckpt", type=str, help="path to configuration file.")
    parser.add_argument("--eval_opt", type=str, default='all', help="path to configuration file.")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="max number of generated tokens")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    return parser

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='caption', help="dataset to evaluate")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")
args = parser.parse_args()

cfg = Config(args)

model, vis_processor,text_processor = init_model(args)
model.eval()
CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""

model.eval()
save_path = cfg.run_cfg.save_path


batch_size = 10
max_new_tokens = 50
dt = EgoIRGBenchDataset(vis_processor=vis_processor, text_processor=text_processor, 
                        vis_root="/mnt/nvme1/suyuejiao/Final_rgbd_dataset/ANNEXE/RGB/",
                         ann_paths="/mnt/nvme1/suyuejiao/Final_rgbd_dataset/ANNEXE/dataset_split/new_new_test/",
                         depth_folder="/mnt/nvme1/suyuejiao/Final_rgbd_dataset/ANNEXE/depth/",
                         mask_folder="/mnt/nvme1/suyuejiao/Final_rgbd_dataset/ANNEXE/mask/")
val_loader = DataLoader(dt, batch_size=batch_size, shuffle=True)

output_save_path = '/mnt/nvme1/suyuejiao/Final_rgbd_dataset/ANNEXE/MiniGPT-4/results/newnew/'
save_mask_path = output_save_path + 'mask/'
save_depth_path = output_save_path + 'depth/'
output_json = output_save_path + 'output.json'

os.makedirs(save_mask_path,exist_ok = True)
os.makedirs(save_depth_path,exist_ok = True)


results = dict()
dict_list = []
m_query = []
c_query = []
m_caption = []
c_caption = []
mae_depth = []
cls_acc = []
ciou = []
i = 0

import nltk
from nltk.translate import meteor_score
nltk.download('punkt')
nltk.download('wordnet')  
nltk.download('omw-1.4')
nltk.download('punkt_tab')
def calculate_meteor_for_lists(A, B):
    meteor_scores = []
    min_length = min(len(A), len(B))
    for i in range(min_length):
        if A[i]:
            tokenized_hypothesis = nltk.word_tokenize(B[i]) 
            tokenized_reference = nltk.word_tokenize(A[i])   
            
            score = meteor_score.meteor_score([tokenized_reference], tokenized_hypothesis)
            meteor_scores.append(score)
        else:
            meteor_scores.append(0)
    final_meteor = sum(meteor_scores)/len(meteor_scores)
    return final_meteor

from pycocoevalcap.cider.cider import Cider
def calculate_cider_for_lists(pred, gt):
    ref = {}
    ref_gt = {}
    for index, item in enumerate(pred, start=1):
        ref[str(index)] = [item] 
    for index, item in enumerate(gt, start=1):
        ref_gt[str(index)] = [item] 
    for key in ref.keys():
        if ref[key]=='':
            ref[key] = 'none'
        if ref_gt[key]=='':
            ref_gt[key] = 'none'
        if ref[key] == 'none' and ref_gt[key]=='':
            score = 10.0
        else:
            cider_scorer = Cider()  
            score, _ = cider_scorer.compute_score(ref, ref_gt)
    return score

def calculate_mae(depth_map1, depth_map2):
    if len(depth_map1.shape) == 2 and len(depth_map2.shape)==3:
        depth_map1 = depth_map1.unsqueeze()
    if len(depth_map1.shape) == 3 and len(depth_map2.shape)==2:
        depth_map2 = depth_map2.unsqueeze()
    if depth_map1.shape != depth_map2.shape:
        print(depth_map1.shape, depth_map2.shape)
        raise ValueError("different shape")
    mae = torch.mean(torch.abs(depth_map1 - depth_map2))
    return mae.item()

def calculate_iou(mask, ground_truth, num_classes=4):
  
    if mask.shape != ground_truth.shape:
        raise ValueError("shape")
    iou = []
    for cls in range(num_classes):
        if torch.sum(ground_truth==cls).item()==0:
            pass
        else:
            intersection = torch.sum((mask == cls) & (ground_truth == cls)).item()
            union = torch.sum((mask == cls) | (ground_truth == cls)).item()
            iou_cls = intersection / (union + 1e-6) 
        iou.append(iou_cls)
    iou_final = sum(iou)/len(iou)
    return iou_final

with open(output_json, 'w') as json_file:
    json_file.write('[\n')

print(len(val_loader))
for batch_data in val_loader:
    img_paths = batch_data['image_path']
    images = batch_data['image']
    ori_shapes = batch_data['ori_shape']
    caption_answers = batch_data['answer']
    instruction_inputs = batch_data['instruction_input']
    with_masks = batch_data['with_mask']
    vqa_querys = batch_data['vqa_query']
    vqa_answers = batch_data['vqa_answer']
    depth = batch_data['depth']
    mask = batch_data['mask']

    # print(img_path, image, ori_shape, caption_answer, instruction_input, with_mask, vqa_query, vqa_answer)
    vqa_querys = prepare_texts_my(vqa_querys, conv_temp)  # warp the texts with conversation template
    instruction_inputs = prepare_texts_my(instruction_inputs, conv_temp) 
    # print(vqa_querys, instruction_inputs)

    answers = model.generate(images, instruction_inputs,vqa_querys, img_paths, max_new_tokens=max_new_tokens, do_sample=False)
    # print(answers)

    query_pred_answer = answers['answer_vqa']
    caption_pred_answer = answers['answer_caption']
    query_gt = vqa_answers
    caption_gt = caption_answers
    # print('-',query_pred_answer, query_gt)
    # print("--",caption_pred_answer, caption_gt)
    meteor_vqa = calculate_meteor_for_lists(query_pred_answer, query_gt)
    meteor_cap = calculate_meteor_for_lists(caption_pred_answer, caption_gt)
    print(meteor_vqa, meteor_cap)
    m_query.append(meteor_vqa)
    m_caption.append(meteor_cap)

    cider_vqa = calculate_cider_for_lists(query_pred_answer, query_gt)
    cider_caption = calculate_cider_for_lists(caption_pred_answer, caption_gt)
    print(cider_vqa, cider_caption)
    c_query.append(cider_vqa)
    c_caption.append(cider_caption)

    # ====depth ==========
    depth_norm = answers['depth_norm']
    # print(depth_norm.shape, depth.shape)
    depth = depth.squeeze().to(depth_norm.device)
    mae = calculate_mae(depth_norm, depth)
    print(mae)
    mae_depth.append(mae)

    #===with mask cls===
    mask_index_pred = answers['with_mask']
    # print(mask_index_pred)
    # print(with_masks)
    with_masks = with_masks.to(mask_index_pred.device)

    mask_index_pred = [1 if x > 0.5 else 0 for x in mask_index_pred]
    correct_predictions = sum(1 for a, b in zip(with_masks, mask_index_pred) if a == b)
    acc = correct_predictions / len(mask_index_pred)
    print(acc)
    cls_acc.append(acc)
    
    # ===mask====
    mask_pred = answers['mask']
    # print("------------88",mask_pred.shape)
    mask_pred = torch.argmax(mask_pred, dim=1)
    print("*****(((())))",np.unique(np.array(mask_pred.cpu())))
    mask = mask.to(mask_pred.device)
    iou = calculate_iou(mask_pred, mask)
    print(iou)
    ciou.append(iou)

  
    for query_pred_ans, caption_pred_ans, query_g, caption_g,  vqa_query, instruction_input, img_path in zip(
        query_pred_answer, caption_pred_answer, query_gt, caption_gt, vqa_querys, instruction_inputs, img_paths):
        results = {}
        query_pred_ans = query_pred_ans.replace("<unk>","").replace("  "," ").strip()
        caption_pred_ans = caption_pred_ans.replace("<unk>","").replace("  "," ").strip()
        results['vqa_ans_pred'] = query_pred_ans
        results['cap_ans_pred'] = caption_pred_ans
        results['vqa_que'] = vqa_query
        results['cap_que'] = instruction_input
        results['vqa_ans_gt'] = query_g
        results['cap_ans_gt'] = caption_g
        results['img_path'] = img_path

        with open(output_json,'a') as json_file:
            json.dump(results, json_file)  
            json_file.write(',\n') 


METEOR_query_final = sum(m_query) / len(m_query)
CIDEr_query_final = sum(c_query) / len(c_query)
METEOR_caption_final = sum(m_caption) / len(m_caption)
CIDEr_caption_final = sum(c_caption)/len(c_caption)
mae_final = sum(mae_depth) / len(mae_depth)
ciou_final = sum(ciou) / len(ciou)

with open(output_json,'a') as json_file:       
    json_file.write(']\n') 
    json_file.write(str(METEOR_query_final)) 
    json_file.write('\n') 
    json_file.write(str(CIDEr_query_final)) 
    json_file.write('\n') 
    json_file.write(str(METEOR_caption_final)) 
    json_file.write('\n') 
    json_file.write(str(CIDEr_caption_final)) 
    json_file.write('\n') 
    json_file.write(str(mae_final)) 
    json_file.write('\n') 
    json_file.write(str(ciou_final)) 
    json_file.write('\n') 