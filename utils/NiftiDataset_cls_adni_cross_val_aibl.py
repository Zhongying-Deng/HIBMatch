import SimpleITK as sitk
import os
import itertools
import numpy as np
import random
import glob
import scipy.ndimage.interpolation as interpolation
import math
import torch
import torch.utils.data
import pandas as pd
import pickle
import pdb
from torch.utils.data import Dataset

from monai.transforms import LoadImage, apply_transform

class NifitDataSetADNICrossValAIBL(torch.utils.data.Dataset):
    def __init__(self, data_path,
                 transforms=None,
                 train=False,
                 test=False, 
                 phase='train',
                 unlabeled_only = False,
                 use_strong_aug=False,
                 transforms_strong=None,
                 label_time='m24',
                 control = None,
                 root_path = "/rds/project/rds-LlrDsbHU5UM/sw991/Project_AD/data/ADNI-rawdata/",
                 split=3, fold=1, label_ratio=None, binary_class=True, remove_mmse=False, n_splits=5):

        # Init membership variables
        self.use_strong_aug = use_strong_aug
        self.transforms_strong = transforms_strong
        self.binary_class = binary_class
        self.data_path = data_path
        self.root = root_path
        df = pd.read_csv(os.path.join(root_path, 'scripts/ADNIMERGE.csv'), low_memory=False)
        # self.df_adni_mem_ef = pd.read_csv(os.path.join(root_path, 'scripts/adni_mem.csv'), low_memory=False)
        path_img = os.path.join(root_path, 'ADNI2')
        
        self.read_files(df, label_time)

        self.MRIList = []
        self.PETList = []
        self.DemoList = []
        self.LabelList = []
        self.NonImageList = []
        self.subject_ids = []
        self.save_nonimg_info = []
        self.phase = phase
        self.label_ratio = label_ratio # control the number of subjects in each class
        self.length = 0
        self.unlabeled_only = unlabeled_only
        if not remove_mmse:
            self.non_image_features = ['AGE', 'PTGENDER', 'MMSE']
            self.non_image_features_range = {'AGE':[55.0, 95.0], 'PTGENDER':['Male', 'Female'], 'MMSE':[0.0,30.0]}
        else:
            print('Remove MMSE in Non-Imaging data!')
            # self.non_image_features = ['AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'ADNI_MEM', 'ADNI_EF',]
            # self.non_image_features_range = {'AGE':[55.0, 91.4,], 'PTGENDER':['Male', 'Female'], 'PTEDUCAT':[6.0, 20.0],
            #                                 'APOE4':[0.0,2.0], 'ADNI_MEM':[-2.863, 3.055], 'ADNI_EF':[-2.855,2.865]}

            # to match the non-imaging information of AIBL
            self.non_image_features = ['AGE', 'PTGENDER']
            self.non_image_features_range = {'AGE':[55.0, 95.0], 'PTGENDER':['Male', 'Female']}

        self.gender_map = {'Male': 0, 'Female': 1}
        self.read_feasible_image(path_img, labeltime=label_time, control=control)
        if not self.unlabeled_only:
            seed = 0
            random.seed(seed)
            paired = list(zip(self.LabelList, self.MRIList, self.PETList, self.NonImageList, self.subject_ids))
            random.shuffle(paired)
            self.LabelList, self.MRIList, self.PETList, self.NonImageList, self.subject_ids = zip(*paired)
            self.LabelList, self.MRIList, self.PETList, self.NonImageList, self.subject_ids = \
                list(self.LabelList), list(self.MRIList), list(self.PETList), list(self.NonImageList), list(self.subject_ids)

            self.subjects_num_train = len(self.LabelList)
            fold_subjects_num = self.subjects_num_train//n_splits
            
            self.cross_val_label_list = {}
            self.cross_val_MRI_list = {}
            self.cross_val_PET_list = {}
            self.cross_val_nonimg_list = {}
            for idx in range(n_splits):
                if idx != n_splits - 1:
                    self.cross_val_label_list[idx] = self.LabelList[idx*fold_subjects_num:(idx+1)*fold_subjects_num]
                    self.cross_val_MRI_list[idx] = self.MRIList[idx*fold_subjects_num:(idx+1)*fold_subjects_num]
                    self.cross_val_PET_list[idx] = self.PETList[idx*fold_subjects_num:(idx+1)*fold_subjects_num]
                    self.cross_val_nonimg_list[idx] = self.NonImageList[idx*fold_subjects_num:(idx+1)*fold_subjects_num]
                else:
                    self.cross_val_label_list[idx] = self.LabelList[idx*fold_subjects_num:]
                    self.cross_val_MRI_list[idx] = self.MRIList[idx*fold_subjects_num:]
                    self.cross_val_PET_list[idx] = self.PETList[idx*fold_subjects_num:]
                    self.cross_val_nonimg_list[idx] = self.NonImageList[idx*fold_subjects_num:]
            
            if self.phase == 'test' or self.phase == 'val':
                self.LabelList = self.cross_val_label_list[fold]
                self.PETList = self.cross_val_PET_list[fold]
                self.MRIList = self.cross_val_MRI_list[fold]
                self.NonImageList = self.cross_val_nonimg_list[fold]
            else:
                del self.cross_val_label_list[fold]
                del self.cross_val_PET_list[fold]
                del self.cross_val_MRI_list[fold]
                del self.cross_val_nonimg_list[fold]
                
                self.LabelList = list(self.cross_val_label_list.values())
                self.LabelList = list(itertools.chain.from_iterable(self.LabelList))
                self.PETList = list(self.cross_val_PET_list.values())
                self.PETList = list(itertools.chain.from_iterable(self.PETList))
                self.MRIList = list(self.cross_val_MRI_list.values())
                self.MRIList = list(itertools.chain.from_iterable(self.MRIList))
                self.NonImageList = list(self.cross_val_nonimg_list.values())
                self.NonImageList = list(itertools.chain.from_iterable(self.NonImageList))
            
                if self.label_ratio is not None:
                    self.LabelList_new = []
                    self.PETList_new = []
                    self.MRIList_new = []
                    self.NonImageList_new = []
                    for i in range(max(self.LabelList)+1):
                        num = math.ceil(self.label_ratio * self.LabelList.count(i))
                        idx_all = np.where(np.array(self.LabelList)==i)
                        idx_chosen = random.sample(list(idx_all[0]), num)
                        self.LabelList_new.extend([self.LabelList[idx] for idx in idx_chosen])
                        self.PETList_new.extend([self.PETList[idx] for idx in idx_chosen])
                        self.MRIList_new.extend([self.MRIList[idx] for idx in idx_chosen])
                        self.NonImageList_new.extend([self.NonImageList[idx] for idx in idx_chosen])
                    
                    self.LabelList = self.LabelList_new
                    self.PETList = self.PETList_new
                    self.MRIList = self.MRIList_new
                    self.NonImageList = self.NonImageList_new
                
        self.transforms = transforms

        self.train = train
        self.test = test
        self.loader = LoadImage(None, True, np.float32)
        self.bit = sitk.sitkFloat32
    
    def read_files(self, df, label_time):
        # filter out NAN based on label at month 24, after filtering, the rows go from 3217 to 2635 rows for training set
        # and 804 to 648 rows for testing set
        df = df[df['DX'].notna()]  # df['DX'].notna() returns True/False for each row
        # another round of filtering, this time we get 1460 rows for training set, and the same 648 rows for testing
        if label_time == 'bl':
            df = df[df['DX_bl'].notna()]
            self.df = df
            self.df = self.df[self.df['VISCODE'] == label_time].reset_index(drop=True)
        else:
            df = df[df['DX'].notna()]
            self.df = df
            self.df = self.df[(self.df['VISCODE']==label_time)|(self.df['VISCODE']=='bl')].reset_index(drop=True)
        if label_time == 'bl':
            self.label_map = {'CN': 0, 'EMCI': 1, 'LMCI': 2, 'AD': 3}
        else:
            self.label_map = {'CN':0, 'MCI':1, 'Dementia':2}
            if self.binary_class:
                self.label_map = {'MCI':0, 'Dementia':1}


    def read_feasible_image(self, path_img, img_time='bl', labeltime='m24', control=None):
        '''
        Store the imaging data path into a list, and non-imaging data features into a list. When call forward, we can just get the image directly according to these lists.
        '''
        self.df = self.df.fillna(-1)
        self.df = self.df.groupby(['PTID']) # group the subjects according to their ID: PTID
        
        time_map = {'bl':'Month0', 'm03':'Month3', 'm06':'Month6','m12':'Year1','m24':'Year2'}
        label_map = self.label_map
        # filter the patient id with both MRI and PET data
        count = 0
        cnt_group = 0
        for name, group in self.df:
            # total group number for training is 892, for testing is 218
            cnt_group += 1
            # print(group)
            # group may contain two or even more rows, but with the same subject ID
            group = group.reset_index(drop=True) # reset the index of group to 0,1,...
            subject = group.iloc[0].loc['PTID'] # get the subject ID
            
            if labeltime == 'bl':
                label = group.loc[group['VISCODE'] == labeltime].iloc[0].loc['DX_bl']
            else:
                if group.loc[group['VISCODE'] == labeltime].empty:
                    count += 1
                    continue
                else:
                    # img_time is always baseline ('bl')
                    folder_name = time_map[img_time]
                    MRI_path = os.path.join(path_img, folder_name, 'MRI', subject)
                    PET_path = os.path.join(path_img, folder_name, 'PET', subject)
                    if os.path.exists(MRI_path): # and os.path.exists(PET_path):
                        full_path_mri = glob.glob(MRI_path+'/SS_r*.nii')
                        # full_path_pet = glob.glob(PET_path+'/sSS_rr*.nii')
                        if len(full_path_mri) == 0: # or len(full_path_pet) == 0:
                            continue                            
                    else:
                        continue
                    label = group.loc[group['VISCODE'] == labeltime].iloc[0].loc['DX']
                    label_bl = group.loc[group['VISCODE'] == labeltime].iloc[0].loc['DX_bl']
            if not self.unlabeled_only:
                if label not in label_map:
                    continue
           
            # the input arg control is 'MCI' in the readme.md
            if control is not None:
                # 'MC' not in label_bl, only 'MC' in the label of baseline visit, this case is added to training set
                if control[0:2] not in label_bl:
                    continue

            # store the information
            if not self.unlabeled_only:
                self.LabelList.append(label_map[label])
            else:
                self.LabelList.append(-1)
            self.MRIList.append(full_path_mri[0])
            # Note that PET list is the copy of MRI list as we do not use it (to match the modality info of AIBL)
            self.PETList.append(full_path_mri[0])
            self.subject_ids.append(subject)
            
            if group.loc[group['VISCODE'] == img_time].empty:
                self.NonImageList.append(self.read_non_image(group.loc[group['VISCODE'] == labeltime].iloc[0]))
                nonimg_info = group.loc[group['VISCODE'] == labeltime].iloc[0]
                nonimg_info.loc['DX'] = group.loc[group['VISCODE'] == labeltime]['DX']
                self.save_nonimg_info.append(nonimg_info)
            else:
                self.NonImageList.append(self.read_non_image(group.loc[group['VISCODE'] == img_time].iloc[0]))
                nonimg_info = group.loc[group['VISCODE'] == img_time].iloc[0]
                nonimg_info.loc['DX'] = group.loc[group['VISCODE'] == labeltime]['DX'].iloc[0]
                self.save_nonimg_info.append(nonimg_info)
            # if len(self.subject_ids) >= 40:
            #     break

        print('Length of label list: {}, non-image list: {}, MRI/PET list: {}'.format(
            len(self.LabelList), len(self.NonImageList), len(self.MRIList)))
        if not self.unlabeled_only:
            print('Image number for class 0-2: {}, {}, {}'.format(
                self.LabelList.count(0), self.LabelList.count(1), self.LabelList.count(2)))
            print('total group number {}, empty number {}'.format(cnt_group, count))
            df = pd.DataFrame(self.subject_ids)
            save_subject_file = 'labeled_{}_subject_ids.csv'.format(self.phase)
            if not os.path.exists(save_subject_file):
                df.to_csv(save_subject_file, index=False)
            save_subject_pkl = 'labeled_{}_subject_ids.pkl'.format(self.phase)
            if not os.path.exists(save_subject_pkl):
                with open(save_subject_pkl, 'wb') as f:
                    pickle.dump(self.subject_ids, f)
            df_nonimg = pd.DataFrame(self.save_nonimg_info)
            save_subject_nonimg = 'labeled_{}_nonimg_info.csv'.format(self.phase)
            if not os.path.exists(save_subject_nonimg):
                df_nonimg.to_csv(save_subject_nonimg, index=False)
        else:
            print('-------- read unlabeled data successfully ---------')

    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        image = reader.Execute()
        return image

    def read_non_image(self, group):
        features = []
        for item in self.non_image_features:
            if item == "PTGENDER":
                features.append(self.gender_map[group.loc[item]])
            else:
                features.append(self.normalize_non_image(group.loc[item], item))
                # features.append(group.loc[item])
        return features

    def normalize_non_image(self, num, item):
        mmin = self.non_image_features_range[item][0]
        mmax = self.non_image_features_range[item][1]
        return ((num-mmin)/(mmax-mmin+1e-5))*2.0-1.0

    def __getitem__(self, index):
        label = self.LabelList[index]
        # read image and label
        MRI_path = os.path.join(self.root, self.MRIList[index])
        # PET_path = os.path.join(self.root, self.PETList[index])
        image_MRI = self.loader(MRI_path)
        # image_PET = self.loader(PET_path)
        non_image = np.array(self.NonImageList[index])

        MRI_np = image_MRI
        # PET_np = image_PET
        if self.transforms is not None:
            MRI_np = apply_transform(self.transforms, MRI_np, map_items=False)
            # PET_np = apply_transform(self.transforms, PET_np, map_items=False)
        assert not np.any(np.isnan(non_image))
        if self.use_strong_aug:
            MRI_str_aug = image_MRI #fda.mix_amplitude(image_MRI, image_PET)
            #MRI_str_aug = self.rand_gamma(image_MRI)
            #axis = random.randint(0,2)
            #rand_flip = tio.RandomFlip(axis, 0.5)
            #MRI_str_aug = rand_flip(MRI_str_aug)
            if self.transforms_strong is not None:
                MRI_str_aug = apply_transform(self.transforms_strong, MRI_str_aug, map_items=False)
            else:
                MRI_str_aug = apply_transform(self.transforms, MRI_str_aug, map_items=False)
            return MRI_np, label, torch.FloatTensor(non_image[:]), MRI_str_aug
        return MRI_np, label, torch.FloatTensor(non_image[:])

    def __len__(self):
        return len(self.LabelList)


if __name__ == '__main__':
    from monai.data import DataLoader
    from monai.transforms import (
        EnsureChannelFirst,
        Compose,
        ScaleIntensity,
        CenterSpatialCrop,
        RandAdjustContrast,
        RandFlip,
        RandRotate
    )
    data_path = ''
    fold = 1
    trainTransforms = Compose([RandFlip(prob=0.5), ScaleIntensity(), EnsureChannelFirst(), CenterSpatialCrop(96)])
    train_set = NifitDataSetADNICrossValAIBL(data_path, transforms=trainTransforms, train=True, phase='train', label_time='m36', control='MCI', split=10, fold=fold, label_ratio=0.8)
    print('length labeled train list:', len(train_set))
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)  # Here are then fed to the network with a defined batch size
    train_set_u = NifitDataSetADNICrossValAIBL(data_path, transforms=trainTransforms, train=True, phase='train', label_time='m36', control='MCI', split=10, fold=fold, unlabeled_only=True)
    train_loader_u = DataLoader(train_set_u, batch_size=2, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_set = NifitDataSetADNICrossValAIBL(data_path, transforms=trainTransforms, train=True, phase='test', label_time='m36', control='MCI', split=10, fold=fold)
    print('length labeled train list:', len(test_set))
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    for mri, label, non_img in train_loader_u:
        print(mri.shape)
        break
    # bl as image time and m36 as the label time, 119 subject from MCI to MCI (112)/AD (7) with MRI and non-imaging data. No one goes from MCI to CN