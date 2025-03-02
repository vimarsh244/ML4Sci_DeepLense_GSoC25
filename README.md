# DeepLense Projects for GSOC25 - ML4Sci

This repository contains assignments for the DeepLense project, part of the GSoC'25 for the ML4Sci organization. 

Assignment questions from [here](https://docs.google.com/document/d/1a-5JiHph3K59gV3-kEZWzKYTFMvDeYiJvoE0U2I4x0w/edit?usp=sharing)


Just attempted most of these questions as midsems were coming up and wanted to do something. These problems were pretty cool and doable for the most part, hence attempted as many as I could. So far haven't made the focus on getting the best performance (accuracy, and all) just that the model is learning and that the task is being solved to an extent.

## Assignments Overview

- **Q5_PINN**  
    The goal was to implement a physics-informed neural network for gravitational lensing. 
    To do this task:
    - Built resnet backbone for feature extraction
    - add physics based loss components
    - this comes from gravitational lensing equation:
    - ... to add here the eqn
    - train with that

    There are still some issues as the val loss after a point goes haywire and havent been able to figure out why that is happening

- **Q6_FoundationModel**  
    - Using a Masked Autoencoder (MAE) for pretraining on “no_sub” (no substructure) samples followed by:
    - Fine-tuning on multi-class classification (no_sub, cdm, axion) using a the earlier trained encoder.
    - Super-resolution fine-tuning on low-resolution (LR) to high-resolution (HR) image tasks.
    
- **Q4_Diffusion**  
    Contains resources and examples related to diffusion models for image generation and reconstruction.
    UNDER PROGRESS

- **Q1_Classification**

- **Q2_IdentifyingLenses**
    Binary classification but with weighted data



All the details are in each's respective READMEs.


##### Datasets:
Download these from the google drive folder from the assignment page:
```
dataset/*
Dataset 3B/*
SuperRes_Dataset/*
Diffusion_Samples/*
lens-finding-test/*
```