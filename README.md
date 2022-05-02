# Training Efficient Wav2Vec2.0 Automatic Speech Recognition Engine Transformer Model from Scratch with Different Heads

## Abstract

Wav2Vec2.0 is a state-of-the-art model when it comes to Automatic Speech Recognition (ASR) due to its unique style of self-supervised training. The objective is to both pre-train and fine-tune the Wav2Vec2.0 model on unlabeled and labeled data to efficiently build an ASR engine from scratch. In doing so, opens endless possibilities to training a unique and individual speech recognition engine that allows for flexibility to a user’s specific use case. Along the way, this project aims to apply every piece of a machine learning pipeline developed from scratch using well known software skills that adds to the flexibility in creating ASR engines. Furthermore, Wav2Vec2.0 is said to have the elasticity to be applied to other audio and speech datasets, in which we set out to give a finite answer to this hypothesis. 

## Research Goal

The goal of this research project is to create a machine learning pipeline using custom-built methods, to allow for pre-training the Wav2Vec2.0 model from scratch, to then fine-tuning the model for speech recognition, to finally adding a classification head to the model to implement a speech classification model.

In doing so, opens up research to allow for the ability to pre-train a model which is typically difficult and time-consuming. Furthermore, we set out to compete with other professional machine learning engineers to achieve competitive results and create our own version of the Wav2Vec2.0 model. 

## Results for Fine-Tuning and Speech Classification

Below include the results on fine-tuning the Wav2Vec2.0 model on the TI-MIT dataset, along with the results from speech classification on the RAVDESS emotional classification dataset. There are 3 models that were created for comparision that you will see below:

1) Original Wav2Vec2.0 Model: ~95 million trainable parameters
2) Medium Wav2Vec2.0 Model: ~61 million trainable parameters
3) Small Wav2Vec2.0 Model: ~36 million trainable parameters

The idea behind creating different versions of the Wav2Vec2.0 model is to see if we can create a smaller and more condensed model, with the hopes to decrease training time yet still achieve accurate results on all the datasets. 

Firstly, are the results from fine-tuning all the models on the TI-MIT speech recognition dataset:

|  | WER | Loss | Total Time |
| --- | :---: | :---: | :---: |
| **Original Wav2Vec2.0 Model** | **16.298%** | **0.1254** | **17:14:36** |
| Medium Wav2Vec2.0 Model | 31.578% | 69.999 | 13:48:26 |
| Small Wav2Vec2.0 Model | 53.434% | 103.774 | 12:40:58 |

Secondly, we can look at the results from applying a classification head to the end of each of the above models to correctly classify the emotion of different speakers on the RAVDESS dataset:

|  | Accuracy | Loss | Total Time |
| --- | :---: | :---: | :---: |
| **Original Wav2Vec2.0 Model** | **89.121%** | **0.205** | **31:21:56** |
| Medium Wav2Vec2.0 Model | 88.897% | 0.223 | 26:20:33 |
| Small Wav2Vec2.0 Model | 85.776% | 0.309 | 23:47:32 |

Looking at the results above, we can see that the original Wav2Vec2.0 model was by far the best model on the TI-MIT dataset with a WER of 16.298%, and the medium-sized model not performing as well as we thought with a WER of 31.578%, and the small-sized model not even coming close with a WER of 53.434%. Even though we were able to decrease training time, the accuracy did not correlate with the training time. 

The results of the speech classification dataset on RAVDESS were much more promising, with the original model still having the highest accuracy at 89.121%, but with the medium-sized model less than a quarter of a percent within the original model at 88.897%. The small-sized model also produced good results at an accuracy of 85.776%. The training times prove that we were able to decrease the overall size of the model and hold high-levels of accuracy on speech classification. 

## Project Repository

This repository is split up into 4 different sections: 

1) **Code:** This directory includes all of the code that makes up the entire research project, all with instructions on how to recreate the results above with some flexibility in creating your own models. To see how the details of the code directory and how to run each piece, click [HERE](https://github.com/tristinjohnson/PreTraining-ASR-From-Scratch/tree/main/Code).


2) **Final Research Report:** This directory includes the research report containing and in-depth analysis of the entire project. From every detail about Wav2Vec2.0, to the overall project approach/objective, to the results, feel free to read the report [HERE](https://github.com/tristinjohnson/PreTraining-ASR-From-Scratch/tree/main/Final%20Research%20Report).


3) **Final Research Presentation:** This directory contains the presentation that was used to conclude the Master's of Science in Data Science degree. This presentation provides a birds-eye-view of the entire project that highlights the important details throughout the research project, which can be seen [HERE](https://github.com/tristinjohnson/PreTraining-ASR-From-Scratch/tree/main/Final%20Research%20Presentation).


4) **Research Proposal:** This directory contains the initial project proposal that was submitted for acceptance by the department of Data Science, and was a qualified proposal that led to the creation of this research project and repository. To view the proposal, click [HERE](https://github.com/tristinjohnson/PreTraining-ASR-From-Scratch/tree/main/Project%20Proposal).

Feel free to explore any of the sections above, as there is a lot of information that can be learned from this project. Furthermore, other researcher's out there have the ability to re-create this project and even add their own twist to try and compete with the results above. 
