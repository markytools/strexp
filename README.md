# Scene Text Recognition Explainability (STRExp)

 **We release a framework that attemps to merge Explainable AI (XAI) into Scene Text Recognition (STR). Light and portable version of Network Dissection in pyTorch at [NetDissect-Lite](https://github.com/CSAILVision/NetDissect-Lite). It is much faster      than this first version and the code structure is cleaned up, without any complex shell commands. It takes about 30 min for a resnet18 model and 2 hours for a          densenet161. If you have questions, please open issues at NetDissect-Lite**
 
## Introduction
This repository contains the demo code for the [CVPR'17 paper](http://netdissect.csail.mit.edu/final-network-dissection.pdf) Network Dissection: Quantifying Interpretability of Deep Visual Representations. You can use this code with naive [Caffe](https://github.com/BVLC/caffe), with matcaffe and pycaffe compiled. We also provide a [PyTorch wrapper](script/rundissect_pytorch.sh) to apply NetDissect to probe networks in PyTorch format. There are dissection results for several networks at the [project page](http://netdissect.csail.mit.edu/).

This code includes

* Code to run network dissection on an arbitrary deep convolutional
    neural network provided as a Caffe deploy.prototxt and .caffemodel.
    The script `rundissect.sh` runs all the needed phases.

* Code to create the merged Broden dataset from constituent datasets
    ADE, PASCAL, PASCAL Parts, PASCAL Context, OpenSurfaces, and DTD.
    The script `makebroden.sh` runs all the needed steps.


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
If you find the codes useful, please cite this paper
```
@inproceedings{netdissect2017,
  title={Network Dissection: Quantifying Interpretability of Deep Visual Representations},
  author={Bau, David and Zhou, Bolei and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
  booktitle={Computer Vision and Pattern Recognition},
  year={2017}
}
```
