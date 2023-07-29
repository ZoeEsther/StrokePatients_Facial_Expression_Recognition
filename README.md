# StrokePatients_Facial_Expression_Recognition

This project is a facial expression recognition algorithm for stroke patients, which is designed with ViT as the baseline.
Please refer to the paper at https://doi.org/10.3390/brainsci12121626

[Note] This code is a simple implementation version of the algorithm described in this paper.

![image](https://github.com/ZoeEsther/StrokePatients_Facial_Expression_Recognition/assets/119051069/5a66620a-905b-4c4c-ad54-10c31596cd58)

## Overall framework of algorithm 
  In order to occupy fewer computing resources to identify eight facial expressions of stroke patients accurately, we propose a lightweight FER model, named the Facial Expression Recognition with Patch-Convolutional Vision Transformer (FER-PCVT). The FER-PCVT designed with ViT as the baseline mainly consists of three modules: the Convolutional Patch Embedding (CPE), the Pyramid Transformer (PTF), and the Valence-Arousal-Like Classifier (V-ALC). The first two modules combine to form the backbone network, Patch-Convolutional Vision Transformer (PCVT). The V-ALC is an expression classifier designed based on the Valence-Arousal (V-A) emotion theory.
![image](https://github.com/ZoeEsther/StrokePatients_Facial_Expression_Recognition/assets/119051069/84258fbb-2bc2-40fa-9968-8122de775f82)

## Detailed heat maps captured by algorithmic features
![image](https://github.com/ZoeEsther/StrokePatients_Facial_Expression_Recognition/assets/119051069/ba06f08e-64e1-4542-b50b-99a1a07d097d)
