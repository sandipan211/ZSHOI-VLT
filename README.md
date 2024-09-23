[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-zero-shot-human-object-interaction/zero-shot-human-object-interaction-detection)](https://paperswithcode.com/sota/zero-shot-human-object-interaction-detection?p=boosting-zero-shot-human-object-interaction)

# Boosting Zero-shot Human-Object Interaction Detection with Vision-Language Transfer (ICASSP 2024)

Official code for the ICASSP 2024 paper that implements a one-stage DETR-based network for zero-shot HOI detection boosted with vision-language transfer.

##  👓 At a glance
This repository contains the official PyTorch implementation of our [ICASSP 2024](https://2024.ieeeicassp.org) paper : [Boosting Zero-Shot Human-Object Interaction Detection with Vision-Language Transfer](https://ieeexplore.ieee.org/document/10445910), a work done by Sandipan Sarma, Pradnesh Kalkar, and Arijit Sur at [Indian Institute of Technology Guwahati](https://www.iitg.ac.in/cse/). 

- Human-Object Interaction (HOI) detection is a crucial task that involves localizing interactive human-object pairs and identifying the actions being performed. In this work, our primary focus is improving HOI detection in images, particularly in zero-shot scenarios.
- The query vectors in our DETR-based framework are vital in projecting an idea about “what” visual information about the human-object pairs to look for, with each vector element suggesting “where” to look for these pairs within the image. Since the final task is to detect human-object pairs, unified query vectors for human-object pairs are important.
- Despite the unavailability of certain actions and objects (such as in UA and UO settings), our method is better at detecting unseen interactions in such challenging settings.
<p align="center">
  <img width="584" alt="zshoid" src="https://github.com/user-attachments/assets/3115c1c5-632e-4fe2-acc0-29833b87b088">
    <br>
    <em>The framework</em>
</p>



## 💪 Pre-trained models
- Download the [params](https://mega.nz/folder/bFUGHSiZ#i-ECSp_MtYbEfO5seXvkIA) folder and put it outside all folders for DETR-based pretrained models.
- Outside all folders, make a new folder called ```ckpt``` and download the pretrained model of CLIP for [CLIP50x16](https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt) inside it.

## 📝 Generating semantics
Generate the object, action, and interaction CLIP semantics for offline use by running:
```bash
cd models
python generate_clip_semantics.py
```

## :bullettrain_side: Training and evaluation
Run the scripts from the scripts folder which contain ```DETR_CLIP``` in the file name. For example, to train the model for UA setting, run the command:
```bash
cd scripts
sh train_DETR_CLIP_UA.sh
```

## 🏆Zero-shot Results on HICO-DET
![image](https://github.com/user-attachments/assets/34923e96-7f0f-491b-ad3a-08afade66bc7)

## :gift: Citation
If you use our work for your research, kindly star :star: our repository and consider citing our work using the following BibTex:
```
@INPROCEEDINGS{10445910,
  author={Sarma, Sandipan and Kalkar, Pradnesh and Sur, Arijit},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Boosting Zero-Shot Human-Object Interaction Detection with Vision-Language Transfer}, 
  year={2024},
  volume={},
  number={},
  pages={6355-6359},
  keywords={Visualization;Semantics;Detectors;Transformers;Feature extraction;Task analysis;Speech processing;Human-object interaction;transformer;CLIP;zero-shot learning},
  doi={10.1109/ICASSP48485.2024.10445910}}
```
## 🙏Acknowledgments
This work partially borrows codes from [CDN](https://github.com/YueLiao/CDN) and [ConsNet](https://github.com/yeliudev/ConsNet)
