# KTN: Knowledge Transfer Network for Multi-person Densepose Estimation

# Introduction
This is the implementation of KTN: Knowledge Transfer Network for Multi-person Densepose Estimation.In this work, we address the multi-person densepose estimation problem, which aims at learning dense correspondences between 2D pixels of human body and 3D surface. It still poses several challenges due to real-world scenes with scale variations and noisy labels with insufficient annotations. In particular, we address two main problems: 1) how to design a simple yet effective pipeline for densepose estimation; and 2) how to equip this pipeline with the ability of handing the issue of noise labels(i.e., limited annotation and class-unbalance). To solve these problems, we develop a novel densepose estimation framework based on the two-stage pipeline, called \textit{Knowledge Transfer Network} (KTN). Unlike existing works which directly propagate the pyramidal base feature among regions, we enhance the representation power of base features through a {\normalsize\bf{multi-instance decoder(MID)}}, which preserves more details of foreground instances while suppresses the activations of backgrounds. Then, we introduce a {\normalsize\bf{knowledge transfer machine(KTM)}}, which estimates densepose through the external commonsense knowledge. 
We discover that aside from the knowledge transfer machine, current densepose estimation systems either from RCNN based methods or fully-convolutional frameworks can be improved in terms of the accuracy of human densepose estimation.
Solid experiments on densepose estimation benchmarks demonstrate the superiority and generalizability of our approach.
![](https://github.com/cfm-wxh/TSN/blob/master/visualization/KTN.png)
# Main Results on Densepose-COCO validation set
![](https://github.com/cfm-wxh/TSN/blob/master/visualization/main_results.jpg)
# Environment
The code is developed based on the ！[Detectron2]{https://github.com/facebookresearch/detectron2} platform. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA RTX GPU cards. Other platforms or GPU cards are not fully tested.
# Installation
Please follow the ![installation instruction]{https://github.com/facebookresearch/detectron2} to build environment.
