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

class NifitDataSetTest(torch.utils.data.Dataset):
    def __init__(self, data_path,
                 transforms=None,
                 control = None,
                 root_path = '/rds/project/rds-LlrDsbHU5UM/sw991/Project_AD/data/ADNI-rawdata',
                 binary_class=False, remove_mmse=False):

        # Init membership variables
        self.binary_class = binary_class
        self.data_path = data_path
        self.phase = 'inference'
        # change it bafore running
        df = pd.read_csv(os.path.join(root_path, 'scripts/ADNIMERGE.csv'), low_memory=False)
        self.df_adni_mem_ef = pd.read_csv(os.path.join(root_path, 'scripts/adni_mem.csv'), low_memory=False)
        path_img = os.path.join(root_path, 'ADNI2')
        save_subject_pkl = 'labeled_train_subject_ids.pkl'
        with open(save_subject_pkl, 'rb') as f:
            subject_ids_train = pickle.load(f)
        save_subject_pkl = 'labeled_test_subject_ids.pkl'
        with open(save_subject_pkl, 'rb') as f:
            subject_ids_val = pickle.load(f)
        self.subject_ids_train_val = subject_ids_train + subject_ids_val
        
        self.baseline_time = ['m12', 'm24']
        self.label_time = ['m36', 'm48']
        self.read_files(df)

        self.MRIList = []
        self.PETList = []
        self.LabelList = []
        self.NonImageList = []
        self.subject_ids = []
        self.save_nonimg_info = []
        self.length = 0
        if not remove_mmse:
            self.non_image_features = ['AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'MMSE', 'ADNI_MEM', 'ADNI_EF',]
            self.non_image_features_range = {'AGE':[55.0, 91.4,], 'PTGENDER':['Male', 'Female'], 'PTEDUCAT':[6.0, 20.0],
                                            'APOE4':[0.0,2.0], 'MMSE':[9.0,30.0], 'ADNI_MEM':[-2.863, 3.055], 'ADNI_EF':[-2.855,2.865]}
        else:
            print('Test set operation: Remove MMSE in Non-Imaging data!')
            self.non_image_features = ['AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4', 'ADNI_MEM', 'ADNI_EF',]
            self.non_image_features_range = {'AGE':[55.0, 91.4,], 'PTGENDER':['Male', 'Female'], 'PTEDUCAT':[6.0, 20.0],
                                            'APOE4':[0.0,2.0], 'ADNI_MEM':[-2.863, 3.055], 'ADNI_EF':[-2.855,2.865]}

        self.gender_map = {'Male': 0, 'Female': 1}
        self.read_feasible_image(path_img, control=control)
        self.subjects_num = len(self.LabelList)

        self.transforms = transforms

        self.loader = LoadImage(None, True, np.float32)
        self.bit = sitk.sitkFloat32
    
    def read_files(self, df):
        # filter out NAN based on label at month 24, after filtering, the rows go from 3217 to 2635 rows for training set
        # and 804 to 648 rows for testing set
        df = df[df['DX'].notna()]  # df['DX'].notna() returns True/False for each row
        # another round of filtering, this time we get 1460 rows for training set, and the same 648 rows for testing
        self.df = df
        self.df = self.df[(self.df['VISCODE']==self.label_time[0])|(self.df['VISCODE']==self.label_time[1])|
                          (self.df['VISCODE']==self.baseline_time[0])|(self.df['VISCODE']==self.baseline_time[1])
                          ].reset_index(drop=True)
        self.df_adni_mem_ef = self.df_adni_mem_ef.fillna(-1)
        # self.df_adni_mem_ef = self.df_adni_mem_ef.groupby(['RID'])
        self.label_map = {'CN':0, 'MCI':1, 'Dementia':2}
        if self.binary_class:
            self.label_map = {'MCI':0, 'Dementia':1}


    def read_feasible_image(self, path_img, control):
        '''
        Store the imaging data path into a list, and non-imaging data features into a list. When call forward, we can just get the image directly according to these lists.
        '''
        self.df = self.df.fillna(-1)
        self.df = self.df.groupby(['PTID']) # group the subjects according to their ID: PTID
        
        time_map = {'bl':'Month0', 'm03':'Month3', 'm06':'Month6','m12':'Year1','m24':'Year2'}
        baseline_time = self.baseline_time #['m12', 'm24']
        label_time = self.label_time
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
            rid = group.iloc[0].loc['RID']
            for img_time, labeltime in zip(baseline_time, label_time):   
                if subject in self.subject_ids_train_val or subject in self.subject_ids:
                    # this subject has been included in train or val set or already in test/inference set
                    # skip it for testing/inference phase
                    continue
                img_time_info = group.loc[group['VISCODE']==img_time]
                label_time_info = group.loc[group['VISCODE']==labeltime] 
                if img_time_info.empty or label_time_info.empty:
                    # 319 empty for training, 79 empty for testing
                    count += 1
                    continue
                else:
                    label = label_time_info.iloc[0].loc['DX']
                    label_pseudo_bl = img_time_info.iloc[0].loc['DX']
                    # the input arg control is 'MCI' in the readme.md
                    # 'MC' not in label_bl, only 'MC' in the label of baseline visit, this case is added to training set
                    if control[0:2] not in label_pseudo_bl or label not in label_map:
                        continue
                    
                    folder_name = time_map[img_time]
                    MRI_path = os.path.join(path_img, folder_name, 'MRI', subject)
                    PET_path = os.path.join(path_img, folder_name, 'PET', subject)
                    if os.path.exists(MRI_path) and os.path.exists(PET_path):
                        full_path_mri = glob.glob(MRI_path+'/SS_r*.nii')
                        full_path_pet = glob.glob(PET_path+'/sSS_rr*.nii')
                        if len(full_path_mri) == 0 or len(full_path_pet) == 0:
                            continue                            
                    else:
                        continue

                # store the information
                self.LabelList.append(label_map[label])
                self.MRIList.append(full_path_mri)
                self.PETList.append(full_path_pet)
                self.subject_ids.append(subject)
                self.NonImageList.append(self.read_non_image(img_time_info.iloc[0], rid, img_time, labeltime))
                nonimg_info = img_time_info.iloc[0]
                nonimg_info.loc['DX'] = label_time_info['DX'].iloc[0]
                self.save_nonimg_info.append(nonimg_info)
                # if len(self.subject_ids) >= 16:
                #     break

        print('Length of label list: {}, non-image list: {}, MRI/PET list: {}'.format(
            len(self.LabelList), len(self.NonImageList), len(self.MRIList)))
        print('Image number for class 0-2: {}, {}, {}'.format(
            self.LabelList.count(0), self.LabelList.count(1), self.LabelList.count(2)))
        # for training set, these two prints 200 200 200; and 44 118 38 respectively
        # for testing set, 48 48 48; 14 28 6
        # the following one print: total group number 892, empty number 319
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

    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        image = reader.Execute()
        return image

    def read_non_image(self, group, rid, img_time, labeltime):
        features = []
        for item in self.non_image_features:
            if item == "PTGENDER":
                features.append(self.gender_map[group.loc[item]])
            elif item == 'ADNI_MEM' or item == 'ADNI_EF':
                rid_feats = self.df_adni_mem_ef[self.df_adni_mem_ef['RID']==rid]
                if rid_feats[rid_feats['VISCODE']==img_time].empty:
                    feat = rid_feats[rid_feats['VISCODE']==labeltime].iloc[0].loc[item]
                else:
                    feat = rid_feats[rid_feats['VISCODE']==img_time].iloc[0].loc[item]
                features.append(self.normalize_non_image(feat, item))
            else:
                features.append(self.normalize_non_image(group.loc[item], item))
        return features

    def normalize_non_image(self, num, item):
        mmin = self.non_image_features_range[item][0]
        mmax = self.non_image_features_range[item][1]
        return ((num-mmin)/(mmax-mmin+1e-5))*2.0-1.0

    def __getitem__(self, index):
        label = self.LabelList[index]
        # read image and label
        MRI_path = self.MRIList[index]
        PET_path = self.PETList[index]
        image_MRI = self.loader(MRI_path)
        image_PET = self.loader(PET_path)
        non_image = np.array(self.NonImageList[index])

        MRI_np = image_MRI
        PET_np = image_PET
        if self.transforms is not None:
            MRI_np = apply_transform(self.transforms, MRI_np, map_items=False)
            PET_np = apply_transform(self.transforms, PET_np, map_items=False)
        assert not np.any(np.isnan(non_image))
        return MRI_np, PET_np, label, torch.FloatTensor(non_image[:])

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
    trainTransforms = Compose([RandFlip(prob=0.5), ScaleIntensity(), EnsureChannelFirst(), CenterSpatialCrop(96)])
    test_set = NifitDataSetTest(data_path, transforms=trainTransforms, control='MCI')
    print('length labeled train list:', len(test_set))
    train_loader = DataLoader(test_set, batch_size=2, shuffle=True, num_workers=len([0]), pin_memory=True, drop_last=True)  # Here are then fed to the network with a defined batch size
    