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

class NifitAIBLCrossVal(torch.utils.data.Dataset):
    def __init__(self, data_path,
                 transforms=None,
                 phase='train',
                 img_time = 'bl',
                 label_time='m36',
                 control = None,
                 root_path = "/home/zd294/rds/rds-foudation-model-Bt2PicqoQF4/data/AIBL/",
                 fold=1, label_ratio=None, binary_class=True, remove_mmse=False):

        # Init membership variables
        self.binary_class = binary_class
        self.data_path = data_path
        self.root = root_path
        self.label_time_to_visit = {'bl': 0, 'm18': 18, 'm36': 36, 'm54': 54, 'm72': 72}
        non_img_file_path = os.path.join(self.root, 'AIBL_TOTAL_DATA.csv')
        self.img_path = os.path.join(self.root, 'processed')
        df = pd.read_csv(non_img_file_path, low_memory=False)
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
        if not remove_mmse:
            # Note that DX may be -4 or 7, Age can be 0.
            # "Using artificial intelligence to learn optimal regimen plan for Alzheimer’s disease" (2023)
            # from Journal of the American Medical Informatics Association use 'LIMMTOTAL', 'LDELTOTAL', 'CDGLOBAL' as features
            # self.non_image_features = ['AGE', 'PTGENDER', 'MMSE', 'LIMMTOTAL', 'LDELTOTAL', 'CDGLOBAL']
            # self.non_image_features_range = {'AGE':[55.0, 95.0], 'PTGENDER':['Male', 'Female'], 'LIMMTOTAL':[0.0, 24.0],
            #                                 'LDELTOTAL':[0.0,23.0], 'MMSE':[0.0,30.0], 'CDGLOBAL':[0, 3.0]}

            # to match the non-imaging information with ADNI
            self.non_image_features = ['AGE', 'PTGENDER', 'MMSE']
            self.non_image_features_range = {'AGE':[55.0, 95.0], 'PTGENDER':['Male', 'Female'], 'MMSE':[0.0,30.0]}
        else:
            print('Remove MMSE in Non-Imaging data!')
            # ADNI dataset also has 'LDELTOTAL'
            self.non_image_features = ['AGE', 'PTGENDER', 'LIMMTOTAL', 'LDELTOTAL', 'CDGLOBAL']
            self.non_image_features_range = {'AGE':[55.0, 95.0], 'PTGENDER':['Male', 'Female'], 'LIMMTOTAL':[0.0, 24.0],
                                            'LDELTOTAL':[0.0, 23.0], 'MMSE':[0.0,30.0], 'CDGLOBAL':[0, 3.0]}
        self.gender_map = {1: 0, 2: 1} # 1 is male in AIBL while 2 is female
        self.read_feasible_image(img_time=img_time, labeltime=label_time, control=control)
        self.subjects_num_train = len(self.LabelList)
        if self.phase != 'test_only':
            # random shuffle labels
            seed = 0
            random.seed(seed)
            paired = list(zip(self.LabelList, self.PETList, self.MRIList, self.NonImageList, self.subject_ids))
            random.shuffle(paired)
            self.LabelList, self.PETList, self.MRIList, self.NonImageList, self.subject_ids = zip(*paired)
            self.LabelList, self.PETList, self.MRIList, self.NonImageList, self.subject_ids = \
                list(self.LabelList), list(self.PETList), list(self.MRIList), list(self.NonImageList), list(self.subject_ids)

            fold_subjects_num = self.subjects_num_train//5
            self.cross_val_label_list = {0: self.LabelList[:(fold_subjects_num)], 
                                        1: self.LabelList[(fold_subjects_num):(2*fold_subjects_num)],
                                        2: self.LabelList[(2*fold_subjects_num):(3*fold_subjects_num)],
                                        3: self.LabelList[(3*fold_subjects_num):(4*fold_subjects_num)],
                                        4: self.LabelList[(4*fold_subjects_num):],
                                        }
            self.cross_val_PET_list = {0: self.PETList[:(fold_subjects_num)], 
                                    1: self.PETList[(fold_subjects_num):(2*fold_subjects_num)], 
                                    2: self.PETList[(2*fold_subjects_num):(3*fold_subjects_num)],
                                    3: self.PETList[(3*fold_subjects_num):(4*fold_subjects_num)],
                                    4: self.PETList[(4*fold_subjects_num):]
                                    }
            self.cross_val_MRI_list = {0: self.MRIList[:(fold_subjects_num)], 
                                    1: self.MRIList[(fold_subjects_num):(2*fold_subjects_num)],
                                    2: self.MRIList[(2*fold_subjects_num):(3*fold_subjects_num)],
                                    3: self.MRIList[(3*fold_subjects_num):(4*fold_subjects_num)],
                                    4: self.MRIList[(4*fold_subjects_num):],
                                        }
            self.cross_val_nonimg_list = {0: self.NonImageList[:(fold_subjects_num)], 
                                        1: self.NonImageList[(fold_subjects_num):(2*fold_subjects_num)],
                                        2: self.NonImageList[(2*fold_subjects_num):(3*fold_subjects_num)],
                                        3: self.NonImageList[(3*fold_subjects_num):(4*fold_subjects_num)],
                                        4: self.NonImageList[(4*fold_subjects_num):]
                                        }
            
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
        else:
            print('Use AIBL only as testing set.')
                
        self.transforms = transforms

        self.loader = LoadImage(None, True, np.float32)
        self.bit = sitk.sitkFloat32
    
    def read_files(self, df, label_time):
        # filter out NAN based on label at month 24, after filtering, the rows go from 3217 to 2635 rows for training set
        # and 804 to 648 rows for testing set
        df = df[df['DX'].notna()]  # df['DX'].notna() returns True/False for each row
        # another round of filtering, this time we get 1460 rows for training set, and the same 648 rows for testing
        if label_time == 'bl':
            df = df[df['VISM_IN'].notna()]
            self.df = df
            self.df = self.df[self.df['VISM_IN'] == self.label_time_to_visit[label_time]].reset_index(drop=True)
        else:
            self.df = df
            self.df = self.df[(self.df['VISM_IN']== self.label_time_to_visit[label_time])|
                              (self.df['VISM_IN']== self.label_time_to_visit['bl'])].reset_index(drop=True)
        self.label_map = {'CN': 0, 'MCI': 1, 'Dementia': 2}
        if self.binary_class:
            self.label_map = {'MCI': 1, 'Dementia': 2}

    def read_feasible_image(self, img_time='bl', labeltime='m36', control=None):
        '''
        Store the imaging data path into a list, and non-imaging data features into a list. When call forward, we can just get the image directly according to these lists.
        '''
        self.df = self.df.fillna(-4)
        self.df = self.df.groupby(['PTID']) # group the subjects according to their ID: PTID
        
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
                label = group.loc[group['VISM_IN'] == self.label_time_to_visit[labeltime]].iloc[0].loc['DX']
            else:
                if group.loc[group['VISM_IN'] == self.label_time_to_visit[labeltime]].empty or group.loc[group['VISM_IN'] == self.label_time_to_visit[img_time]].empty:
                    # 319 empty for training, 79 empty for testing
                    count += 1
                    continue
                else:
                    # img_time is always baseline ('bl')
                    subject_folder = os.path.join(self.img_path, str(self.label_time_to_visit[img_time]), str(subject))
                    PET_file = os.path.join(subject_folder, 'PET_TregPET_SS_MN152.nii')
                    MRI_file = os.path.join(subject_folder, 'T1_TregT1_SS_restore_MN152.nii')
                    if os.path.exists(MRI_file): #and os.path.exists(PET_file):
                        label = group.loc[group['VISM_IN'] == self.label_time_to_visit[labeltime]].iloc[0].loc['DX']
                        label_bl = group.loc[group['VISM_IN'] == self.label_time_to_visit[img_time]].iloc[0].loc['DX']
                    else:
                        continue
            
            if label not in label_map.values():
                continue # this if statement filters out nothing
            # For training set, after the above filtering
            # Length of label list: 297, non-image list: 297, MRI/PET list: 297
            # Image number for class 0-2: 126, 122, 49
            # For testing set, length of label list: 82, non-image list: 82, MRI/PET list: 82
            # Image number for class 0-2: 38, 30, 14

            # the input arg control is 'MCI' in the readme.md
            if control is not None:
                if self.label_map[control] != label_bl:
                    continue

            # store the information
            if self.binary_class:
                self.LabelList.append(label-1)
            else:
                self.LabelList.append(label)
            self.MRIList.append(MRI_file)
            self.PETList.append(PET_file)
            self.subject_ids.append(subject)
            
            if group.loc[group['VISM_IN'] == self.label_time_to_visit[img_time]].empty:
                self.NonImageList.append(
                    self.read_non_image(group.loc[group['VISM_IN'] == self.label_time_to_visit[labeltime]].iloc[0])
                    )
                nonimg_info = group.loc[group['VISM_IN'] == self.label_time_to_visit[labeltime]].iloc[0]
                nonimg_info.loc['DX'] = group.loc[group['VISM_IN'] == self.label_time_to_visit[labeltime]]['DX']
                self.save_nonimg_info.append(nonimg_info)
            else:
                self.NonImageList.append(
                    self.read_non_image(group.loc[group['VISM_IN'] == self.label_time_to_visit[img_time]].iloc[0])
                    )
                nonimg_info = group.loc[group['VISM_IN'] == self.label_time_to_visit[img_time]].iloc[0]
                nonimg_info.loc['DX'] = group.loc[group['VISM_IN'] == self.label_time_to_visit[labeltime]]['DX'].iloc[0]
                self.save_nonimg_info.append(nonimg_info)
            # if len(self.subject_ids) >= 40:
            #     break

        print('Length of label list: {}, non-image list: {}, MRI/PET list: {}'.format(
            len(self.LabelList), len(self.NonImageList), len(self.MRIList)))
        if self.binary_class:
            print('Image number for class 0-1: {}, {}, max label: {}'.format(
                self.LabelList.count(0), self.LabelList.count(1), max(self.LabelList)))
        else:
            print('Image number for class 0-2: {}, {}, {}'.format(
                self.LabelList.count(0), self.LabelList.count(1), self.LabelList.count(2)))
        # for training set, these two prints 200 200 200; and 44 118 38 respectively
        # for testing set, 48 48 48; 14 28 6
        # the following one print: total group number 892, empty number 319
        print('total group number {}, without label number {}'.format(cnt_group, count))
        df = pd.DataFrame(self.subject_ids)
        save_subject_file = 'AIBL_labeled_{}_subject_ids.csv'.format(self.phase)
        if not os.path.exists(save_subject_file):
            df.to_csv(save_subject_file, index=False)
        save_subject_pkl = 'AIBL_labeled_{}_subject_ids.pkl'.format(self.phase)
        if not os.path.exists(save_subject_pkl):
            with open(save_subject_pkl, 'wb') as f:
                pickle.dump(self.subject_ids, f)
        df_nonimg = pd.DataFrame(self.save_nonimg_info)
        save_subject_nonimg = 'AIBL_labeled_{}_nonimg_info.csv'.format(self.phase)
        if not os.path.exists(save_subject_nonimg):
            df_nonimg.to_csv(save_subject_nonimg, index=False)

    def read_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        image = reader.Execute()
        return image

    def read_non_image(self, group):
        features = []
        for item in self.non_image_features:
            if item == "PTGENDER":
                # features.append(float(group.loc[item]))
                features.append(self.gender_map[group.loc[item]])
            else:
                features.append(self.normalize_non_image(float(group.loc[item]), item))
                # features.append(float(group.loc[item]))
        return features

    def normalize_non_image(self, num, item):
        mmin = self.non_image_features_range[item][0]
        mmax = self.non_image_features_range[item][1]
        return ((num-mmin)/(mmax-mmin+1e-5))*2.0-1.0

    def __getitem__(self, index):
        label = self.LabelList[index]
        # read image and label
        MRI_path = self.MRIList[index]
        # PET_path = self.PETList[index]
        image_MRI = self.loader(MRI_path)
        # Note that PET images will not be used in our case
        # image_PET = self.loader(PET_path)
        non_image = np.array(self.NonImageList[index])

        MRI_np = image_MRI
        # PET_np = image_PET
        if self.transforms is not None:
            MRI_np = apply_transform(self.transforms, MRI_np, map_items=False)
            # PET_np = apply_transform(self.transforms, PET_np, map_items=False)
        assert not np.any(np.isnan(non_image))
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
    train_set = NifitAIBLCrossVal(data_path, transforms=trainTransforms, phase='train', img_time='bl', label_time='m36', control='MCI', fold=fold, binary_class=True)
    print('length labeled train list:', len(train_set))
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=len([0]), pin_memory=True, drop_last=True)  # Here are then fed to the network with a defined batch size
    val_set = NifitAIBLCrossVal(data_path, transforms=trainTransforms, phase='val', img_time='bl', label_time='m36', control='MCI', fold=fold, binary_class=True)
    print('length labeled val list:', len(val_set))
    test_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=len([0]), pin_memory=True, drop_last=True)
    # bl as image time and m18 as the label time, 271 subjects from MCI to MCI (261)/AD (10) with MRI and non-imaging data. No one goes from MCI to CN
    #                                             42 subjects with all three modalities and all of them are MCI -> MCI
    # bl as image time and m36 as the label time, 119 subject from MCI to MCI (112)/AD (7) with MRI and non-imaging data. No one goes from MCI to CN
    # bl as image time and m54 as the label time, 98 subject from MCI to MCI (93)/AD (5) with MRI and non-imaging data. No one goes from MCI to CN