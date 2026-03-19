# HW1 - Multi-Granularity Visual Recognition

## Introduction

This repository contains my implementation for Homework 1: Image Classification in *Selected Topics in Visual Recognition using Deep Learning*. The task is a 100-class image classification problem with class imbalance and fine-grained characteristics. According to the assignment requirements, the model backbone must come from the ResNet family, pretrained weights are allowed, no external data can be used, and the total model size must be less than 100M parameters.

My final method is a multi-granularity visual recognition framework based on a pretrained ResNeXt-101-32x8d backbone. The model further integrates SE-based channel attention, PMG-style multi-branch feature learning, GeM pooling, jigsaw-based multi-granularity training, a finetuning stage for classifier refinement, and test-time augmentation with multi-branch prediction fusion during inference.

The goal of this project is to improve discriminative feature learning for fine-grained multi-class recognition and to build a complete training-to-inference pipeline.

Total Params: 99.02M

## Environment Setup

### Recommended Environment
- Python 3.9 or higher

### Main Dependencies
- PyTorch
- torchvision
- numpy
- pandas
- tqdm
- Pillow

You can create a virtual environment first, then install the required packages. If you are using pip, install PyTorch and torchvision according to your CUDA version, then install the remaining Python packages.

### Expected Folder Structure
```text
.
├── model.py
├── train.py
├── finetune.py
├── infer.py
└── cv_hw1_data
    └── data
        ├── train
        ├── val
        └── test
```

## Usage

### 1. Train the Main Model
```
python train.py
```

Run `train.py` to start the main training stage.

This stage includes:
- imbalance-aware sampling
- strong data augmentation
- jigsaw-based multi-granularity supervision
- PMG-style branch learning
- EMA-based checkpoint selection

The best checkpoint from this stage will be saved as `best_model.pth`.

### 2. Finetune the Model
```
python finetune.py
```

Run `finetune.py` after the main training stage.

This stage starts from `best_model.pth` and uses a milder setting than the main training stage. Jigsaw-based learning, Mixup, and CutMix are removed during finetuning so that the model can further refine the decision boundary.

The best checkpoint from this stage will be saved as `best_finetune.pth`.

### 3. Run Inference
```
python infer.py
```
Run `infer.py` to generate predictions on the test set.

The inference script:
- loads `best_finetune.pth`
- applies test-time augmentation
- aggregates the softmax outputs of all four prediction heads
- writes the final predictions to `prediction.csv`

## Performance Snapshot

The model was evaluated under three settings:

- **Baseline without jigsaw-based training**  
  Best validation accuracy = **94.33%**

- **With jigsaw-based training**  
  Best validation accuracy = **95.00%**

- **With finetuning**  
  Best validation accuracy = **95.67%**

These results show that jigsaw-based multi-granularity training improves the baseline model, and the finetuning stage further refines the learned representation to achieve the best final validation performance.

### Key Observations
- The jigsaw-based model achieves higher validation accuracy than the baseline.
- The best checkpoint appears before the final epoch, so checkpoint selection is important.
- Finetuning provides an additional gain after the main training stage.

### Final Inference Strategy
The final prediction is obtained by:
- applying test-time augmentation
- summing the softmax outputs from branch head 1, branch head 2, branch head 3, and the fusion head

## References

Main references used in this project include:
- Du et al., *Fine-grained Visual Classification via Progressive Multi-granularity Training of Jigsaw Patches*, ECCV 2020.
- Xie et al., *Aggregated Residual Transformations for Deep Neural Networks*, CVPR 2017.
- Facebook Research ResNeXt repository.
