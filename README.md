# TlMamba: Enhancing Ancient Tai Palm-Leaf Manuscript Character Recognition through Hybrid Vision Mamba Framework

## 1. Overview
Recognizing handwritten characters in ancient Tai palm-leaf manuscripts poses significant challenges due to high inter-character similarity, visually similar glyphs, and long-tail category distributions. This paper introduces TlMamba, a hybrid Vision Mamba-based framework, to address these issues, The model incorporates a Structure-Aware Patch Embedding (SPE) module to enhance fine-grained local feature modeling and a Structure-Semantics Dual-Branch (SSD) module to jointly model local stroke structures and contextual semantic relationships. A Two-Stage Knowledge Distillation (TSKD) framework transfers knowledge from a large-scale pre-trained Transformer to the Mamba model, improving rare character recognition. Experiments on the newly constructed HLDLC1.0 dataset demonstrate TlMamba's superior performance with 96.17% accuracy, outperforming other models and showing strong generalization on Oracle-MNlST, Devanagari, and Tamil datasets.
<img width="578" height="515" alt="image" src="https://github.com/user-attachments/assets/be07a1e7-4b57-4369-b4f6-ab73c2db6b9d" />

## 2. Environment and Dependencies
- pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 
- pip install packaging
- pip install timm==0.4.12  
- pip install pytest chardet yacs termcolor  
- pip install submitit tensorboardX 
- pip install triton==2.0.0  
- pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
- pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
- pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs

## 3.Dataset Description

The data are organized into training, validation, and test sets to ensure fair evaluation and reproducibility.

### Directory Structure

```text
data/
├─ train/
├─ val/
└─ test/


## 4.Training and Evaluation
### Training
python TSKD_train.py 
### Evaluation
python test.py 

## Citation

If you use this code or data in your research, please cite the following paper:

**"[TlMamba: Enhancing Ancient Tai Palm-Leaf Manuscript Character Recognition through Hybrid Vision Mamba Framework]"**, *The Visual Computer*

```bibtex
@article{,
  title   = {TlMamba: Enhancing Ancient Tai Palm-Leaf Manuscript Character Recognition through Hybrid Vision Mamba Framework},
  author  = {JingYing Zhao},
  journal = {The Visual Computer},
  year    = {2026}
}

