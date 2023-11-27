<div align="center">

# <br> Scene Text Recognition Models Explainability Using Local Features (STRExp) </br>

[![arXiv preprint](http://img.shields.io/badge/arXiv-2310.09549-b31b1b)](https://arxiv.org/abs/2310.09549)
[![In Proc. ICIP 2023](http://img.shields.io/badge/ICIP-2023-10222406)](https://ieeexplore.ieee.org/abstract/document/10222406)

[**Mark Vincent Ty**](https://github.com/markytools) and [**Rowel Atienza**](https://github.com/roatienza)

Electrical and Electronics Engineering Institute </br>
University of the Philippines, Diliman

</div>

 **We release a framework that merges Explainable AI (XAI) into Scene Text Recognition (STR). This tool builds on the captum library and applies explainability to existing Scene Text Recognition models by leveraging their local explanations.**

## Download
* Pretrained STR models. After unzipping, simply put the "pretrained/" folder into the cloned strexp directory.
```
    https://zenodo.org/record/7476285/files/pretrained.zip
```
* STR LMDB Real Test Datasets and their segmentations. After unzipping, simply put the "datasets/" folder into the cloned strexp directory.
```
    https://zenodo.org/record/7478796/files/datasets.zip
```

## STR Model Evaluations
* Before running anything, you need to edit the file ```settings.py```. Set the STR model (vitstr, parseq, srn, abinet, trba, matrn), the segmentation directory, and the STR real test dataset name (IIIT5k_3000, SVT, IC03_860, IC03_867, IC13_857, IC13_1015, IC15_1811, IC15_2077, SVTP, CUTE80).
* To run/evaluate (vitstr, srn, abinet, trba, matrn), pip install timm==0.4.5
* To run/evaluate (parseq), pip install timm==0.6.7


* Run STRExp on VITSTR: 
```
CUDA_VISIBLE_DEVICES=0 python captum_improve_vitstr.py --eval_data datasets/data_lmdb_release/evaluation \
--benchmark_all_eval --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --sensitive \
--data_filtering_off  --imgH 224 --imgW 224 --TransformerModel=vitstr_base_patch16_224 \
--saved_model pretrained/vitstr_base_patch16_224_aug.pth --batch_size=1 --workers=0 --scorer mean --blackbg
```

* Run STRExp on PARSeq:
```
CUDA_VISIBLE_DEVICES=0 python captum_improve_parseq.py --eval_data datasets/data_lmdb_release/evaluation \
--benchmark_all_eval --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --sensitive \
--data_filtering_off  --imgH 32 --imgW 128 --TransformerModel=vitstr_base_patch16_224 --batch_size=1 --workers=0 --scorer mean --blackbg --rgb
```

* Run STRExp on TRBA:
```
CUDA_VISIBLE_DEVICES=0 python captum_improve_trba.py --eval_data datasets/data_lmdb_release/evaluation --benchmark_all_eval \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --batch_size 1 --workers=0 --data_filtering_off \
--saved_model pretrained/trba.pth --confidence_mode 0 --scorer mean --blackbg --imgH 32 --imgW 100
```

* Run STRExp on SRN:
```
CUDA_VISIBLE_DEVICES=0 python captum_improve_srn.py --eval_data datasets/data_lmdb_release/evaluation \
--saved_model pretrained/srn.pth --batch_size=1 --workers=0 --imgH 32 --imgW 100 --scorer mean
```

* Run STRExp on ABINET (change also the dataset.test.roots dataset name in ```configs\train_abinet.yaml``` to settings.py TARGET_DATASET):
```
CUDA_VISIBLE_DEVICES=0 python captum_improve_abinet.py --config=configs/train_abinet.yaml --phase test --image_only --scorer mean --blackbg \
--checkpoint pretrained/abinet.pth --imgH 32 --imgW 128 --rgb
```

* Run STRExp on MATRN (change also the dataset.test.roots dataset name in ```configs\train_matrn.yaml``` to settings.py TARGET_DATASET):
```
CUDA_VISIBLE_DEVICES=0 python captum_improve_matrn.py --imgH 32 --imgW 128 --checkpoint=pretrained/matrn.pth --scorer mean --rgb
```

## Acquiring the Selectivity AUC

* After running the experiments above, you will have an output pickle file in the current directory. The name of this pickle file can be found in the variable "outputSelectivityPkl", which is written just below the "acquireSingleCharAttrAve()" function. For example in captum_improve_vitstr.py, you can see the variable "outputSelectivityPkl" in line 194.
* In order to acquire the selectivity AUC, you need to replace the pickle file in captum_test.py to the output pickle filename. After this, uncomment the line 670 in captum_improve_vitstr.py (comment out line 671 too) and run the same code above to produce the metrics of STRExp evaluated on VITSTR.
* Repeat this step for all other captum_improve pyfiles.


## Experiments
![alt text](https://github.com/markytools/strexp/blob/master/data/VITSTR_PARSeq_TRBA_SRN_ABINET_MATRN.png?raw=true)</br>
From top to bottom: VITSTR(1st & 2nd figure), PARSeq(3rd & 4th figure), TRBA(5th & 6th figure), SRN(7th & 8th figure), ABINET(9th & 10th figure) and MATRN(11th & 12th figure) quantitative results.
</br>
</br>
![alt text](https://github.com/markytools/strexp/blob/master/data/parseq_srn_trba.png?raw=true)</br>
From top to bottom: PARSeq(1st & 2nd figure), SRN(3rd & 4th figure), and TRBA(5th & 6th figure) qualitative results.
</br>
</br>

## Reference 

Full thesis manuscript: https://drive.google.com/file/d/1KBFXfjZL6Gf4HYU5cw5nWylU93gCPLlv/view?usp=sharing
</br>
If you find the codes useful, please cite this paper
```
@inproceedings{ty2023scene,
  title={Scene Text Recognition Models Explainability Using Local Features},
  author={Ty, Mark Vincent and Atienza, Rowel},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  pages={645--649},
  year={2023},
  organization={IEEE}
}
```
