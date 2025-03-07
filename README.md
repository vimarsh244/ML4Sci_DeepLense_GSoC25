# Assignments for DeepLense Projects for GSoC'25 - ML4Sci

This repository contains assignment solutions for the DeepLense project, part of the GSoC'25 for the ML4Sci organization. 

Assignment questions from [here](https://docs.google.com/document/d/1a-5JiHph3K59gV3-kEZWzKYTFMvDeYiJvoE0U2I4x0w/edit?usp=sharing) or [pdf](/GSoC25_DeepLense_Tests.pdf)


Just attempted most of these questions as midsems were coming up and wanted to do something interesting. These problems were pretty cool and doable for the most part, hence attempted as many as I could. So far haven't made the focus on getting the best performance (accuracy, and all) just that the model is learning and that the task is being solved to an extent. a lot of code is from my older implementations for image classificationm etc. 

### assignments overview

doccumentation in progress, but below are the questions attempted and their readme's with details.

- **Q1_Classification**

    [readme](Q1_Classification/README.md)



- **Q2_IdentifyingLenses**

    [readme](Q2_IdentifyingLenses/README.md)


    Binary classification but with weighted data


- **Q3_SuperResolution**

    [readme](Q3_SuperResolution/README.md)

- **Q4_Diffusion**  
    [readme](Q4_Diffusion/README.md)


    Contains resources and examples related to diffusion models for image generation and reconstruction.
    UNDER PROGRESS

- **Q5_PINN**  
    [readme](Q5_PINN/README.md)


    The goal is to implement a physics-informed neural network for gravitational lensing. 
    To do this task:
    - Built resnet backbone for feature extraction
    - added physics based loss components
    - this comes from gravitational lensing equation:
    - ... to add here the eqn
    - train with that

    There are still some issues as the val loss after a point goes haywire and havent been able to figure out why that is happening

- **Q6_FoundationModel**  
    [readme](Q6_FoundationModel/README.md)


    - Using a Masked Autoencoder (MAE) for pretraining on “no_sub” (no substructure) samples followed by:
    - Fine-tuning on multi-class classification (no_sub, cdm, axion) using a the earlier trained encoder.
    - Super-resolution fine-tuning on low-resolution (LR) to high-resolution (HR) image tasks.
    





All the details are in each's respective READMEs. They are more like notes rn, .. like logs


### datasets:
Download these from the google drive folder from the assignment page:
```
dataset/*
Dataset 3B/*
SuperRes_Dataset/*
Diffusion_Samples/*
lens-finding-test/*
```