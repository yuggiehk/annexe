import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.base_model import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList
from minigpt4.conversation.conversation import StoppingCriteriaSub
import math
import torch.nn.functional as F
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
from PIL import Image,ImageDraw, ImageFont
import torch.nn.init as init
import os

class DepthDecoder(nn.Sequential):
    '''Fusion module

    From Adabins
    
    '''
    def __init__(self, skip_input, output_features):
        super(DepthDecoder, self).__init__()
        self.convA = nn.ConvTranspose2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.convB = nn.ConvTranspose2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)                
        )

    def forward(self, x):
        up_x = F.interpolate(x, size=(112,112), mode='bilinear', align_corners=True)
        x = self.convB(self.convA(torch.cat([up_x], dim=1)))
        x = self.decoder(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)  
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x)) 
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x

from .depth_anything_v2.dpt import DepthAnythingV2

def load_depth_model():
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

    encoder = 'vitb' # or 'vits', 'vitb', 'vitg'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f"/home/suyuejiao/MiniGPT-4/minigpt4/models/checkponits/depth_anything_v2_vitb.pth", weights_only=True,map_location='cpu'))

    for name, param in model.named_parameters():
        # print(name)
        param.requires_grad = False
        if 'depth_head' in name:
            if 'output_conv' in name:
                param.requires_grad = True
            if '4' in name:
                param.requires_grad = True

    return model

class MaskDecoder(nn.Module):
    def __init__(self):
        super(MaskDecoder, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 32x32 -> 64x64
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 64x64 -> 128x128
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)    # 128x128 -> 256x256
        self.deconv4 = nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1)     # 256x256 -> 448x448
        self._initialize_weights()

    def forward(self, x):
        x = self.deconv1(x)  # 32x32 -> 64x64
        x = self.deconv2(x)  # 64x64 -> 128x128
        x = self.deconv3(x)  # 128x128 -> 256x256
        x = self.deconv4(x)  # 256x256 -> 448x448
        x = nn.functional.interpolate(x, size=(1080, 1920), mode='bilinear', align_corners=False)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)  
                init.zeros_(m.bias)

class ANNEXEBase(BaseModel):

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        llama_model="",
        max_txt_len=32,
        max_context_len=3800,
        prompt_template="",
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        lora_r=0,  # lora_r means lora is not used
        lora_target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
    ):
        super().__init__()
        
        self.output_visulization_folder = '/mnt/nvme1/suyuejiao/Final_rgbd_dataset/ANNEXE/MiniGPT-4/results/new/'
        os.makedirs(self.output_visulization_folder+'depth/',exist_ok=True)
        os.makedirs(self.output_visulization_folder+'mask/',exist_ok=True)

        self.llama_model, self.llama_tokenizer = self.init_llm(
            llama_model_path=llama_model,
            low_resource=low_resource,
            low_res_device=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, freeze_vit
        )

        self.max_txt_len = max_txt_len
        self.max_context_len = max_context_len
        self.end_sym = end_sym

        self.prompt_template = prompt_template
        self.prompt_list = []

        self.depth_decoder = DepthDecoder(256,128)
        self.with_mask_classfier = Classifier()

        # self.sam = build_sam_vit_b(checkpoint="/mnt/nvme1/suyuejiao/Final_rgbd_dataset/MiniGPT-4/minigpt4/models/checkponits/sam_vit_b_01ec64.pth")
        # for name, param in self.sam.named_parameters():
        #     if 'image_encoder' in name:  
        #         param.requires_grad = False
        #     if 'prompt_encoder' in name:
        #         param.requires_grad = False

        self.depth_estimation_model = load_depth_model()

        self.mask_decoder = MaskDecoder()
        self.linear_layer = nn.Linear(768, 256)

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def get_context_emb(self, prompt, img_list):
        device = img_list[0].device
        prompt_segs = prompt.split('<ImageHere>')

        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i==0).to(device).input_ids # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def prompt_wrap(self, img_embeds, atts_img, prompts, lengths=None):
        if prompts is None or len(prompts) == 0:
            # prompts is not provided, just return the original image embedding
            return img_embeds, atts_img
        elif img_embeds is None:
            # prompt is provided but there is no image embedding. return the prompt embedding in right padding
            self.llama_tokenizer.padding_side = "right"
            prompt_tokens = self.llama_tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            ).to(self.device)
            prompt_embeds = self.embed_tokens(prompt_tokens.input_ids)
            atts_prompt = prompt_tokens.attention_mask
            return prompt_embeds, atts_prompt
        else:
            # return the multi-modal embedding in right padding
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)

            for idx, (each_img_embed, each_prompt) in enumerate(zip(img_embeds, prompts)):
                pn = each_img_embed.shape[-2]
                if lengths is not None:
                    each_img_embed = each_img_embed.reshape(-1, each_img_embed.shape[-1])
                    each_img_embed = each_img_embed[:lengths[idx] * pn]
                p_segs = each_prompt.split('<ImageHere>')
                interleave_emb = []
                for idx, seg in enumerate(p_segs[:-1]):
                    p_tokens = self.llama_tokenizer(
                        seg, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                    p_embed = self.embed_tokens(p_tokens.input_ids)
                    interleave_emb.append(torch.cat([p_embed, each_img_embed[None][:, idx * pn:(idx + 1) * pn]], dim=1))
                wrapped_emb = torch.cat(interleave_emb, dim=1)
                p_tokens = self.llama_tokenizer(
                    p_segs[-1], return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_embed = self.embed_tokens(p_tokens.input_ids)
                wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=1)
                emb_lists.append(wrapped_emb)

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))

            max_length = max(emb_lens) if max(emb_lens) < self.max_context_len else self.max_context_len
            wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch.int, device=img_embeds.device)
            
            for i, emb in enumerate(emb_lists):
                length = emb_lens[i] if emb_lens[i] < self.max_context_len else self.max_context_len
                wrapped_embs[i, :length] = emb[:, :length]
                wrapped_atts[i, :length] = 1
            return wrapped_embs, wrapped_atts

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        """
        Concatenate the batched input embedding and batched output embedding together.
        Both the input and the output embedding should be right padded.
        """
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def tokenize_conversation(self, conv_q, conv_a):
        """concatenate conversation and make sure the model is only trained to regress the answer"""

        to_regress_token_ids_list = []
        targets_list = []

        batch_size = len(conv_q)
        for batch_idx in range(batch_size):
            questions, answers = conv_q[batch_idx], conv_a[batch_idx]
            questions = [self.llama_tokenizer(self.llama_tokenizer.bos_token + q,
                                              return_tensors="pt",
                                              add_special_tokens=False).to(self.device) for q in questions[1:]]  # the first question is handled in the prompt wrap function, skip it
            answers = [self.llama_tokenizer(a + self.end_sym,
                                            return_tensors="pt",
                                            add_special_tokens=False).to(self.device) for a in answers]
            cur_id = []
            cur_target = []
            for i in range(len(questions)):
                cur_id.append(answers[i].input_ids)
                cur_target.append(answers[i].input_ids)
                cur_id.append(questions[i].input_ids)
                cur_target.append(torch.ones_like(questions[i].input_ids) * -100)

            cur_id.append(answers[-1].input_ids)
            cur_target.append(answers[-1].input_ids)

            cur_id = torch.cat(cur_id, dim=1)
            cur_target = torch.cat(cur_target, dim=1)
            to_regress_token_ids_list.append(cur_id)
            targets_list.append(cur_target)

        max_len = min(max([target.shape[1] for target in targets_list]), self.max_txt_len)
        to_regress_token_ids = torch.ones([batch_size, max_len],
                                          dtype=cur_id.dtype, device=self.device) * self.llama_tokenizer.pad_token_id
        targets = torch.ones([batch_size, max_len],
                                          dtype=cur_id.dtype, device=self.device) * -100
        for batch_idx in range(batch_size):
            cur_len = to_regress_token_ids_list[batch_idx].shape[1]
            to_regress_token_ids[batch_idx, :cur_len] = to_regress_token_ids_list[batch_idx][0, :max_len]
            targets[batch_idx, :cur_len] = targets_list[batch_idx][0, :max_len]

        to_regress_token_attn = (to_regress_token_ids != self.llama_tokenizer.pad_token_id).to(torch.int)

        return to_regress_token_ids, to_regress_token_attn, targets
    
    def preparing_embedding_caption(self, samples):
        ### prepare input tokens
        if 'image' in samples:
            img_embeds, img_atts = self.encode_img(samples["image"])
        else:
            img_embeds = img_atts = None
        sole_img_embeds = img_embeds

        if "instruction_input" in samples:
            instruction = samples["instruction_input"]
          
            if hasattr(self, 'chat_template') and self.chat_template:
                instruction = [self.prompt_template.format(instruct) for instruct in instruction]

            if 'length' in samples:
                # the input is a image train (like videos)
                bsz, pn, hs = img_embeds.shape
                img_embeds = img_embeds.reshape(len(samples['image']), -1, pn, hs)
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction, samples['length'])
            else:
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction)

            ### prepare target tokens
            self.llama_tokenizer.padding_side = "right"
            text = [t + self.end_sym for t in samples["answer"]]

            regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(self.device)

            regress_token_ids = regress_tokens.input_ids
            regress_atts = regress_tokens.attention_mask
            part_targets = regress_token_ids.masked_fill(
                regress_token_ids == self.llama_tokenizer.pad_token_id, -100
            )

        regress_embeds = self.embed_tokens(regress_token_ids)

        return sole_img_embeds, img_atts, cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets
    
    def preparing_embedding_vqa(self, samples, img_embeds, img_atts):

        if 'vqa_query' in samples:
            instruction = samples["vqa_query"]

            if hasattr(self, 'chat_template') and self.chat_template:
                instruction = [self.prompt_template.format(instruct) for instruct in instruction]

            if 'length' in samples:
                # the input is a image train (like videos)
                bsz, pn, hs = img_embeds.shape
                img_embeds = img_embeds.reshape(len(samples['image']), -1, pn, hs)
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction, samples['length'])
            else:
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction)

            ### prepare target tokens
            self.llama_tokenizer.padding_side = "right"
            text = [t + self.end_sym for t in samples["vqa_answer"]]

            regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(self.device)

            regress_token_ids = regress_tokens.input_ids
            regress_atts = regress_tokens.attention_mask
            part_targets = regress_token_ids.masked_fill(
                regress_token_ids == self.llama_tokenizer.pad_token_id, -100
            )

        regress_embeds = self.embed_tokens(regress_token_ids)

        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets

    def forward(self, samples, reduction='mean'):
        loss = {}
        # --------------analyzing ----------------
        # prepare the embedding to condition and the embedding to regress
        sole_img_embeds, img_atts, cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets = \
            self.preparing_embedding_caption(samples)
        
        # concat the embedding to condition and the embedding to regress
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)

        # get bos token embedding
        bos = torch.ones_like(part_targets[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = cond_atts[:, :1]

        # add bos token at the begining
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

        # ensemble the final targets
        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)
        for i, target in enumerate(part_targets):
            targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  # plus 1 for bos

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                reduction=reduction
            )
    
        loss_caption = outputs.loss
        loss['caption_loss'] = loss_caption
        
        # =================answering=================
        # prepare the embedding to condition and the embedding to regress
        cond_embeds_vqa, cond_atts_vqa, regress_embeds_vqa, regress_atts_vqa, part_targets_vqa = \
            self.preparing_embedding_vqa(samples, sole_img_embeds, img_atts)

        # concat the embedding to condition and the embedding to regress
        inputs_embeds_vqa, attention_mask_vqa, input_lens_vqa = \
            self.concat_emb_input_output(cond_embeds_vqa, cond_atts_vqa, regress_embeds_vqa, regress_atts_vqa)

        # get bos token embedding
        bos_vqa = torch.ones_like(part_targets_vqa[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds_vqa = self.embed_tokens(bos_vqa)
        bos_atts_vqa = cond_atts_vqa[:, :1]

        # add bos token at the begining
        inputs_embeds_vqa = torch.cat([bos_embeds_vqa, inputs_embeds_vqa], dim=1)
        attention_mask_vqa= torch.cat([bos_atts_vqa, attention_mask_vqa], dim=1)

        # ensemble the final targets
        targets_vqa = torch.ones([inputs_embeds_vqa.shape[0], inputs_embeds_vqa.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)

        for i, target in enumerate(part_targets_vqa):
            targets_vqa[i, input_lens_vqa[i]+1:input_lens_vqa[i]+len(target)+1] = target  # plus 1 for bos


        with self.maybe_autocast():
            outputs_vqa = self.llama_model(
                inputs_embeds=inputs_embeds_vqa,
                attention_mask=attention_mask_vqa,
                return_dict=True,
                labels=targets_vqa,
                reduction=reduction
            )
  
        loss_vqa = outputs_vqa.loss
        loss['vqa_loss'] = loss_vqa

        # ===============Depth===========
        depth, image_features = self.depth_estimation_model(samples['image']) # this image_features is a tuple, which is [B, 768] for the last layer

        depth_norm = (depth-depth.min())/(depth.max()-depth.min()) # normalize the depth prediction
        depth_gt = samples['depth']
        huber_loss = nn.L1Loss()
        depth_loss = huber_loss(depth_norm, depth_gt)
        loss['depth_loss'] = depth_loss*0.5
        depth = depth_norm*255
        img_path = samples['image_path']
        for i in range(4):
            depth_i = depth[i]
            depth_i = depth_i.cpu().detach().numpy().astype(np.uint8)
            depth_i = Image.fromarray(depth_i.squeeze())
            filename = img_path[i].split('/')[-1].split('.')[0]
            depth_i.save(self.output_visulization_folder+'depth/'+str(filename)+'.png')

        #============with_mask classification==========
        max_ = 400

        bsz, c, hw = inputs_embeds_vqa.shape
        # print(inputs_embeds_vqa.shape) #[b,x,4096]
        h = w = int(math.sqrt(hw))
        reshaped_tensor = inputs_embeds_vqa.view(bsz, c, h,-1)
        random_indices = torch.randperm(c)[:256]  
        select_feature = reshaped_tensor[:, random_indices, :, :]

        with_mask_pred = self.with_mask_classfier(select_feature)
        with_mask_pred_r = torch.sigmoid(with_mask_pred)

        with_mask_gt = samples['with_mask'].unsqueeze(1).float()

        loss_function = nn.BCEWithLogitsLoss()
        loss_cls = loss_function(with_mask_pred, with_mask_gt)
        loss['cls_loss'] = loss_cls

        # # ===================mask=============
        select_feature_from_cls = select_feature # [B, 256, 64, 64]
        b,c, h,w = select_feature_from_cls.shape
        select_feature_from_cls = select_feature_from_cls.view(b,c,-1)
        max_pool = nn.MaxPool1d(kernel_size=4, stride=4)
        select_feature_from_cls = max_pool(select_feature_from_cls)  


        feature_from_depth = image_features[-1][-2]
        b, hw, c = feature_from_depth.shape
        feature_from_depth = self.linear_layer(feature_from_depth)
        feature_from_depth = feature_from_depth.view(b, -1, hw)


        input_feature_mask = feature_from_depth + select_feature_from_cls
        b, c, hw = input_feature_mask.shape
        h = w = int(math.sqrt(hw))
        input_feature_mask = input_feature_mask.view(b,c,h,w)
        output_mask = self.mask_decoder(input_feature_mask)

        output_mask = output_mask * with_mask_pred_r.unsqueeze(1).unsqueeze(2)
        
        mask_gt = samples['mask'].to(output_mask.device)


        # vis mask pred
        txt = samples['vqa_answer']
        for i in range(b):
            new_output_mask = torch.argmax(output_mask,dim=1)
            msk = new_output_mask[i].cpu().detach()
            msk = msk.squeeze().numpy().astype(np.uint8)
            msk = msk / msk.max()*255
            msk = np.uint8(msk)
  
            msk = Image.fromarray(msk.squeeze())
            
            filename = img_path[i].split('/')[-1].split('.')[0]
            msk.save(self.output_visulization_folder+'mask/'+str(filename)+'.png')
       
        criterion = nn.CrossEntropyLoss()
        loss_mask = criterion(output_mask, mask_gt.long())
        loss['mask_loss'] = loss_mask*3
        print(loss_mask)


        sum_loss = 0
        for key in loss.keys():
            sum_loss = sum_loss + loss[key]
        sum_loss = sum_loss / 5
        return {"loss": sum_loss,"loss_cap":loss_caption, "loss_vqa":loss_vqa,
        "loss_dep":depth_loss, "loss_msk":loss_mask, "loss_cls":loss_cls}

    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds
    
    def generate_caption(
        self,
        images,
        texts,
        num_beams=1,
        max_new_tokens=20,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=1,
        do_sample=False,
        stop_words_ids=[2],
    ):
        '''
            function for generate test use
        '''

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])

        img_embeds, atts_img = self.encode_img(images.to(self.device))
        image_lists = [[image_emb[None]] for image_emb in img_embeds]

        batch_embs = [self.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]

        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device

        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1

        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                # stopping_criteria=stopping_criteria,
            )

        # with self.maybe_autocast():
        #     outputs = self.llama_model.generate(
        #         inputs_embeds=embs,
        #         attention_mask=attn_mask,
        #         max_new_tokens=max_new_tokens,
        #         num_beams=num_beams,
        #         do_sample=do_sample,
        #         # stopping_criteria=stopping_criteria,
        #     )
        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('</s>')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)

        return answers, img_embeds, embs

    @torch.no_grad()
    def generate(
        self,
        images,
        texts_caption,
        texts_vqa,
        image_path,
        num_beams=1,
        max_new_tokens=20,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=1,
        do_sample=False,
        stop_words_ids=[2],
    ):
        '''
            function for generate test use
        '''

        answers = {}

        vqa_ans, img_embeds, cond_embeds_vqa = self.generate_caption(images,
        texts_vqa,
        num_beams,
        max_new_tokens,
        min_length,
        top_p,
        repetition_penalty,
        length_penalty,
        temperature,
        do_sample,
        stop_words_ids)
        answers['answer_vqa'] = vqa_ans

        caption_ans,_,_ = self.generate_caption(images,
        texts_caption,
        num_beams,
        max_new_tokens,
        min_length,
        top_p,
        repetition_penalty,
        length_penalty,
        temperature,
        do_sample,
        stop_words_ids)
        answers['answer_caption'] = caption_ans


        #  ======== depth ============
        bsz, c, hw = cond_embeds_vqa.shape
        images = images.to(self.device)
        depth, image_features = self.depth_estimation_model(images) # this image_features is a tuple, which is [B, 768] for the last layer
        # print("===depth pred shape ===",depth.shape) #torch.Size([4, 448, 448])
        depth_norm = (depth-depth.min())/(depth.max()-depth.min()) # normalize the depth prediction
        # depth visualization 
        depth = depth_norm*255
        for i in range(bsz):
            depth_i = depth[i]
            # print(depth_i.shape)
            depth_i = depth_i.cpu().detach().numpy().astype(np.uint8)
            depth_i = Image.fromarray(depth_i.squeeze())
            filename = image_path[i].split('/')[-1].split('.')[0]
            depth_i.save(self.output_visulization_folder+'depth/'+str(filename)+'.png')
        answers['depth_norm'] = depth_norm

        
        #============with_mask classification==========
        max_ = 400
       
        bsz, c, hw = cond_embeds_vqa.shape
        h = w = int(math.sqrt(hw))
        reshaped_tensor = cond_embeds_vqa.view(bsz, c, h,-1)
        random_indices = torch.randperm(c)[:256]  # 随机选择 256 个索引
        select_feature = reshaped_tensor[:, random_indices, :, :]
        print(select_feature.shape) 
        with_mask_pred = self.with_mask_classfier(select_feature.float())
        with_mask_pred = torch.sigmoid(with_mask_pred)
        answers['with_mask'] = with_mask_pred

        #============mask==========
    
        select_feature_from_cls = select_feature # [B, 256, 64, 64]
        b,c, h,w = select_feature_from_cls.shape
        select_feature_from_cls = select_feature_from_cls.view(b,c,-1)
        max_pool = nn.MaxPool1d(kernel_size=4, stride=4)
        select_feature_from_cls = max_pool(select_feature_from_cls)  
        # print(select_feature_from_cls.shape)

        feature_from_depth = image_features[-1][-2]
        b, hw, c = feature_from_depth.shape
        feature_from_depth = self.linear_layer(feature_from_depth)
        feature_from_depth = feature_from_depth.view(b, -1, hw)
        # print(feature_from_depth.shape) #[B, 256, 1024]

        input_feature_mask = feature_from_depth + select_feature_from_cls
        b, c, hw = input_feature_mask.shape
        h = w = int(math.sqrt(hw))
        input_feature_mask = input_feature_mask.view(b,c,h,w)
        output_mask = self.mask_decoder(input_feature_mask)
        output_mask = output_mask * with_mask_pred.unsqueeze(1).unsqueeze(2)
        print("output mask shape**************",output_mask.shape) #torch.Size([4, 4, 1080, 1920])
        answers['mask'] = output_mask

        return answers

    @torch.no_grad()
    def multi_select(self, images, texts, answers, num_cand=None):
        all_losses = []
        for answer in answers:
            choice_samples = {
                'image': images,
                'instruction_input': texts,
                'answer': answer
            }
            loss = self.forward(choice_samples, reduction='none')['loss'].reshape(-1, 1)
            all_losses.append(loss)
            torch.cuda.empty_cache()
        all_losses = torch.cat(all_losses, dim=-1)
        if num_cand is not None:
            for i in range(all_losses.shape[0]):
                all_losses[i, num_cand[i]:] = 9999
        output_class_ranks = torch.argsort(all_losses, dim=-1)
        return output_class_ranks.tolist()
