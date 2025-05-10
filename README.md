# STDA: Spatio-Temporal Deviation Alignment Learning for Cross-city Fine-grained Urban Flow Inference

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

This is the official PyTorch implementation of the paper:  
**"STDA: Spatio-Temporal Deviation Alignment Learning for Cross-city Fine-grained Urban Flow Inference"**

## 📌 Overview
STDA is a meta-transfer-learning framework for cross-city fine-grained urban flow inference, enabling knowledge transfer from data-rich source cities to target cities with limited data.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.7+
- PyTorch 1.8+
- Install dependencies: `pip install -r requirements.txt`

### 2. Data Preparation
```bash
cd datasets
unzip XiAn.zip    
unzip ChengDu.zip 
unzip Beijing.zip 


### 3. Meta-Training

```bash
python meta_train.py --model=STDA --rec_cities=['XiAn','ChengDu'] --tar_city=['BeiJing']


### 4. Few-Shot Adaptation

```bash
python few_sample_adaptation.py --model=STDA --rec_cities=['XiAn','ChengDu'] --tar_city=['BeiJing']



📜 Citation
 
