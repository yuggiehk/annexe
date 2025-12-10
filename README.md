<p align="center">
    <h2 align="center">ANNEXE: Unified Analyzing, Answering, and Pixel Grounding for Egocentric Interaction</h2>
    <p align="center">
        <a href="https://github.com/yuggiehk/annexe">Yuejiao Su</a> · 
        <a href="https://scholar.google.com/citations?user=MAG909MAAAAJ&hl=en">Yi Wang</a> ·  
        <a>Qiongyang Hu</a> ·  
        <a href="https://scholar.google.com/citations?user=37S_Zz4AAAAJ&hl=zh-CN">Chuang Yang</a> ·  
        <a href="https://scholar.google.com/citations?user=MYREIH0AAAAJ&hl=en">Lap-Pui Chau</a>
        <br>
        <a href="https://www.polyu.edu.hk/">The Hong Kong Polytechnic University
        <br>
        <br>
        <a href="https://arxiv.org/abs/2504.01472">
            <img src='https://github.com/yuggiehk/annexe/raw/refs/heads/main/imgs/arxiv.svg' alt='Paper PDF'>
        </a>
        <a href='https://yuggiehk.github.io/annexe/'>
            <img src='https://github.com/yuggiehk/annexe/raw/refs/heads/main/imgs/project.svg' alt='Project Page'>
        </a>
        <a href='https://huggingface.co/datasets/yuki3585/Ego-IEGBench'>
            <img src='https://github.com/yuggiehk/annexe/raw/refs/heads/main/imgs/model.svg'>
        </a>
        <br>
    </p>
</p>


-----

This repo is the official pytorch implementation of ANNEXE: Unified Analyzing, Answering, and Pixel Grounding for Egocentric Interaction (CVPR 2025).


## **Abstract**
Egocentric interaction perception is one of the essential branches in investigating human-environment interaction, which lays the basis for developing next-generation intelligent systems. However, existing egocentric interaction understanding methods cannot yield coherent textual and pixel-level responses simultaneously according to user queries, which lacks flexibility for varying downstream application requirements. To comprehend egocentric interactions exhaustively, this paper presents a novel task named Egocentric Interaction Reasoning and pixel Grounding (Ego-IRG). Taking an egocentric image with the query as input, Ego-IRG is the first task that aims to resolve the interactions through three crucial steps: analyzing, answering, and pixel grounding, which results in fluent textual and fine-grained pixel-level responses. Another challenge is that existing datasets cannot meet the conditions for the Ego-IRG task. To address this limitation, this paper creates the Ego-IRGBench dataset based on extensive manual efforts, which includes over 20k egocentric images with 1.6 million queries and corresponding multimodal responses about interactions. Moreover, we design a unified ANNEXE model to generate text- and pixel-level outputs utilizing multimodal large language models, which enables a comprehensive interpretation of egocentric interactions. The experiments on the Ego-IRGBench exhibit the effectiveness of our ANNEXE model compared with other works.

## 📑 **News**
- [Finished.] Release the code.
- [Finished] Release the dataset.
- [Finished] The [Project Page](https://yuggiehk.github.io/annexe/) is released.
- [Finished] The paper is accepted by [CVPR 2025](https://cvpr.thecvf.com/).

## **Training**
### **Dataset Preparation**
The Ego-IRGBench dataset has been uploaded to [Hugginface](https://huggingface.co/datasets/yuki3585/Ego-IEGBench). Please download and unzip it in your data_root.
The structure of the data folder should be organized as follows,
```
- Data_root
	|- RGB
	|- depth
	|- mask
	|- dataset_split
        |- train
            |- ZY20210800001_H1_C11_N07_S185_s02_T2_00104.json
            |- ZY20210800001_H1_C11_N07_S185_s02_T2_00212.json
            |- ...
        |- test
            |- ZY20210800001_H1_C11_N07_S185_s02_T2_00069.json
            |- ...
        |- val
            |- ZY20210800001_H1_C11_N07_S185_s02_T2_00079.json
            |- ...
```

Replace the configurations in minigpt4/configs/datasets/egoirgbench/config.yaml with your saved path for dataset. Also, replace the depth and mask path in minigpt4/datasets/datasets/ego_irgbench_dataset.py with your own path.


### **Setup**
Create the environment by:

```
conda env create -f environment.yml
conda activate annexe
```
Download this repo:
```
git clone https://github.com/yuggiehk/annexe.git
cd annexe
```

### Download the model

Download the LLAMA model [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) and [Mini-GPT model](https://drive.google.com/file/d/1Vi_E7ZtZXRAQcyz4f8E6LtLh2UXABCmu/view), replace the corresponding path in train_configs/annexe.yaml (line 8 and line 9). The directory path of output should also be set.

### Running 
```
torchrun train.py --cfg-path train_configs/annexe.yaml
```

## Validation

Environment setup by:
```
export PYTHONPATH=$PYTHONPATH:/path/to/directory/of/ANNEXE
```

Run the testing by:
```
torchrun --master-port ${port} --nproc_per_node ${node} eval_scripts/eval_egoirgbench.py  --cfg-path eval_configs/annexe_val.yaml --dataset ego_irgbench

```
## **Acknowledgements**
This code is developed based on [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [MiniGPT-4V2](https://github.com/Vision-CAIR/MiniGPT-4), and [Depth-Anything](https://github.com/LiheYoung/Depth-Anything). We appreciate their contributions for their open-sourced research.

The research work was conducted in the JC STEM Lab of Machine Learning and Computer Vision funded by The Hong Kong Jockey Club Charities Trust.


## 🔗 BibTeX
If you find [ANNEXE](https://yuggiehk.github.io/annexe/) useful for your research and applications, please cite ANNEXE using this BibTeX:

```BibTeX
@INPROCEEDINGS{su2025annexe,
  author={Su, Yuejiao and Wang, Yi and Hu, Qiongyang and Yang, Chuang and Chau, Lap-Pui},
  booktitle={2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={ANNEXE: Unified Analyzing, Answering, and Pixel Grounding for Egocentric Interaction}, 
  year={2025},
  volume={},
  number={},
  pages={9027-9038},
  keywords={Computer vision;Image resolution;Grounding;Large language models;Computational modeling;Cognition;Pattern recognition;Intelligent systems;Next generation networking},
  doi={10.1109/CVPR52734.2025.00844}}
```

