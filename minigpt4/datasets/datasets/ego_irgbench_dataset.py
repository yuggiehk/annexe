# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from torch.utils.data import Dataset
import numpy as np
import os
import json
import torch
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from PIL import Image
import random
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
import cv2
import random

class EgoIRGBenchDataset(Dataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None, depth_folder=None, mask_folder=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        self.qa_folder = ann_paths
        self.rgb_folder = vis_root
        self.depth_folder = "/mnt/nvme1/suyuejiao/Final_rgbd_dataset/ANNEXE/depth/"
        self.mask_folder = "/mnt/nvme1/suyuejiao/Final_rgbd_dataset/ANNEXE/mask/"

        data_list = []
        for json_filename in os.listdir(self.qa_folder):
            data_info = dict()
            image_filaneme = json_filename.split('.')[0]
        
            data_info['img_path'] = os.path.join(self.rgb_folder, image_filaneme+'.jpg')  
            data_info['depth_path'] = os.path.join(self.depth_folder, image_filaneme+'.png')
            data_info['msk_path'] = os.path.join(self.mask_folder, image_filaneme+'.png')
            assert osp.isfile(data_info['img_path'])
            assert osp.isfile(data_info['depth_path'])
            assert osp.isfile(data_info['msk_path'])

            with open(os.path.join(self.qa_folder, json_filename), 'r') as json_file:
                json_data = json.load(json_file)
                for i in range(len(json_data)):
                    json_data_i = json_data[i]
                    len_question = len(json_data_i['questions'])
                    random_number = random.randint(1, len_question)
                    data_info['masks_idx'] = json_data_i['masks_idx']
                    data_info['with_mask'] = json_data_i['with_mask']
                    data_info['caption'] = json_data_i['caption']
                    data_info['answer'] = json_data_i['answer']
                    data_info['tokens'] = json_data_i['tokens']
                    data_info['query'] = json_data_i['questions'][random_number-1]['query']
                    data_info['is_sentence'] = json_data_i['questions'][random_number-1]['is_sentence']
                    data_info['query_id'] = json_data_i['questions'][random_number-1]['id']
                    data_list.append(data_info)

        self.annotation = data_list
    
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.instruction_pool = [
            'Briefly describe the interaction between person and objects in this image.',
            'Provide a concise depiction of the interaction between hands and objects in this image.',
            'Present a short description of the interaction between hands and objects in this image.',
            'Summarize the interaction between hands and objects in this image in a few words.',
            'A short image caption of the interaction between hands and objects',
            'A short image description of the interaction between hands and objects',
            'A photo of the interaction between hands and objects',
            'An image that shows the interaction between hands and objects',
            'Write a short description for the image about the interaction between hands and objects. ',
            'Write a description for the interaction between hands and objects in this photo.',
            'Provide a description of what is presented of the interaction between hands and objects in the photo.',
            'Briefly describe the interaction between hands and objects of the image.',
            'Can you briefly explain what you see about the interaction between hands and objects in the image?'
        ]

        self._add_instance_ids()

        self.rgb_transform = None

        self.depth_transform = transforms.Compose(
            [
                transforms.Resize((448,448),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor()
            ]
        )
    
    def load_depth(self, item):
        """ get depth map from item['depth_path']=image path
        """
        depth_path = item['depth_path']
        depth_np = Image.open(depth_path)
        return depth_np
    
    def load_mask(self, item):
        """get label from json['masks']['points']
        Returns:
            dict, optional: The mapping from old classes in cls.METAINFO to
                new classes in self._metainfo
        """
        mask_path = item['msk_path']
        # print(item['masks_idx'])
        mask = Image.open(mask_path) # this is correct
        mask_np = np.array(mask).squeeze()
        mask_size = mask_np.shape #torch.Size([1080,1920])
        # print(np.unique(mask_np))

        new_mask = np.zeros(mask_size)
        id = 1
        for idx in item['masks_idx']:
            if idx != 0:
                position = (mask_np==idx)
                new_mask[position] = id
            id = id + 1
        new_mask = torch.from_numpy(new_mask)
        # print(new_mask.shape) # torch.Size([1080, 1920])
        return new_mask

    def __getitem__(self, index):

        ann = self.annotation[index]

        img_file = ann["img_path"]
        image = Image.open(img_file).convert("RGB")
        ori_shape = (1080, 1920)
        if self.rgb_transform:
            image = self.rgb_transform(image)
        if self.vis_processor:
            image = self.vis_processor(image)

        caption = ann['caption']
        if self.text_processor:
            caption = self.text_processor(ann["caption"])

        instruction = random.choice(self.instruction_pool)
        instruction = "<Img><ImageHere></Img> [caption] {} ".format(instruction)

        query = ann['query']
        query = "<Img><ImageHere></Img> [vqa] Based on the image, describe the things need to be segmented accorting to this prompt: {}".format(query)

        tokens = ann['tokens']
        final_token = tokens[0]
        for i in range(1,4):
            if tokens[i] != '':
                final_token = final_token + ' and ' + tokens[i]

        depth_img = self.load_depth(ann)
        if self.depth_transform:
            depth_img = self.depth_transform(depth_img)
        depth_img =(depth_img-depth_img.min())/(depth_img.max()-depth_img.min()) 

        mask = self.load_mask(item=ann)    

        return {
            "image_path":img_file, # path
            "image": image, # tensor, shape [b,3,448,448], after resize and norm
            "ori_shape": ori_shape, # ori image shape, [1080, 1920]
            "answer": caption,
            "instruction_input": instruction,
            "with_mask": ann['with_mask'],
            "vqa_query": query,
            "vqa_answer": final_token,
            "depth": depth_img, # tensor, shape [b,1,448,448], after resize and norm
            "mask": mask # tensor, shape [b,1080,1920]
        }
    
    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = idx
            if 'query_id' in ann.keys():
                del ann['query_id']
    

# if __name__=='__main__':
#     dt = EgoInterDataset(vis_root="/mnt/nvme1/suyuejiao/Final_rgbd_dataset/RGB/",
#                          ann_paths="/mnt/nvme1/suyuejiao/Final_rgbd_dataset/small_EGOINTER/train/",
#                          depth_folder="/mnt/nvme1/suyuejiao/Final_rgbd_dataset/depth/",
#                          mask_folder="/mnt/nvme1/suyuejiao/Final_rgbd_dataset/mask/")
#     train_loader = DataLoader(dt, batch_size=4, shuffle=True)

#     print(len(train_loader))
#     for item in train_loader:
#         # print(item.keys())   
#         # print(item['image'].shape) # torch.Size([4, 3, 448, 448])
#         # print(item['depth'].shape) # torch.Size([4, 1, 448, 448])
#         # print(item['mask'].shape)  # torch.Size([4, 448, 448])
#         print(item['ori_shape'])
        
        # bsz, _, h, w  = item['image'].shape
        # for i in range(bsz):
            # image_path = item['image_path'][i]
            # print(image_path)
            # img = cv2.imread(image_path)
            # cv2.imwrite('/mnt/nvme1/suyuejiao/Final_rgbd_dataset/Depth-Anything-V2/pretrained/sample1.jpg', img)
            
            # image = item['image'][i]
            # image_pil = transforms.ToPILImage()(image)
            # image_pil.save('/mnt/nvme1/suyuejiao/Final_rgbd_dataset/Depth-Anything-V2/pretrained/sample.jpg')

            # depth = item['depth'][i]
            # depth = depth*255
            # depth_pil = transforms.ToPILImage()(depth)
            # depth_pil.save('/mnt/nvme1/suyuejiao/Final_rgbd_dataset/Depth-Anything-V2/pretrained/sample2.jpg')

        #     print('==',item['vqa_query'][i])