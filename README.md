
# HIBMatch
PyTorch implementation of "[HIBMatch: Hypergraph Information Bottleneck for Semi-supervised Alzheimer’s Progression](https://ieeexplore.ieee.org/document/11259074)" (IEEE J-BHI 2025).

## Install environment

```
conda create -n HIBMatch python=3.7.13
conda activate HIBMatch
pip install -r requirements.txt
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch-lts -c nvidia
```

## Run HIBMatch

Please use the script `submit_train_HGIB_gpu` to submit a Condor job to train the HIBMatch model. Alternatively, run the following command
```python
python train_ssl_cross_val.py --gpu_ids 0, --lr 0.001 --name HGIB/CV1 --netD fc --focal --model HGIB_semi_unlabeled_consistency_ema --label_time m24 --onefc --control MCI --K_neigs 20 --continue_train --load_weight non_image/complete-modality-info10 --niter 0 --niter_decay 2000  --beta 10 --split 10 --num_graph_update 100 --save_latest_freq 100 --save_epoch_freq 400 --weight_u 0.1 --use_strong_aug --train_encoders --fold 4
```

### Data preparation
To train and test a model on ADNI, please specify `root_path` in `utils/NiftiDataset_cls_densenet_non_imaging_cross_val.py`, `utils/NiftiDataset_cls_densenet_non_imaging_test.py`, and `utils/NiftiDataset_cls_semisup_non_imaging.py` (using your ADNI dataset path). Since we do not have the right to redistribute the ADNI data, please download the dataset following the [official website of ADNI](https://adni.loni.usc.edu/). For the details on data pre-processing, please refer to Section IV.A-C in [our paper](https://ieeexplore.ieee.org/document/11259074).

The data should be organized as follows:

```
|-- data
    |-- ADNI2
        |-- Month0 (data folder of baseline visit)
                |-- MRI
                    |-- xxx (folder of the subjects IDs containing MRI images)
                |-- PET
                    |-- xxx (folder of the subjects IDs containing PET images)
        |-- Year2 (data folder of follow-up visit at month 24, this folder is optional as we do not use images at follow-up visits)
                |-- MRI
                    |-- xxx (folder of the subjects IDs containing MRI images)
                |-- PET
                    |-- xxx (folder of the subjects IDs containing PET images)
    |-- train_info10.csv (file of non-image features for ADNI)
    |-- test_info10.csv (file of non-image features for ADNI)
    |-- train_split10.pkl (file storing image paths)
    |-- test_split10.pkl (file storing image paths)
```

Please note that the images in `train_split10.pkl` and `test_split10.pkl` (please see [this link](https://github.com/Zhongying-Deng/SAM-Brain3D_HyDA/tree/main/data) to download these two files and other necessary files) will be mixed together and split into 5 folds for cross validation.




## Citations
```
@ARTICLE{11259074,
  author={Deng, Zhongying and Wang, Shujun and Aviles-Rivero, Angelica I and Kourtzi, Zoe and Schönlieb, Carola-Bibiane},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={HIBMatch: Hypergraph Information Bottleneck for Semi-Supervised Alzheimer's Progression}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Feature extraction;Magnetic resonance imaging;Alzheimer's disease;Data mining;Data models;Perturbation methods;Diffusion tensor imaging;Vectors;Three-dimensional displays;Prognostics and health management;Alzheimer's disease;progression prediction;hypergraph information bottleneck;multimodal data;consistency regularisation},
  doi={10.1109/JBHI.2025.3634534}}
```
