import torch
from torch.utils.data import Dataset, DataLoader
from numpy.random import choice as npc
import time
import pandas as pd
import sys

import random, os
import torch
import numpy as np
seed = 32
random.seed(seed)
os.environ['PYTHONHASHSEED'] =str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic =True


class Train_Siamese(Dataset):

    def __init__(self, dataPath, testCouple):
        super(Train_Siamese, self).__init__()
        self.top_samples = 128
        self.datas = self.loadToMem(dataPath, testCouple)
    def loadToMem(self, dataPath, testCouple):

        datas = []

        df = pd.read_csv(dataPath)

        # couple_list = [102,104,105,117,126,133,135,155,177,183,188,708,709,710,712,713,714,715,204,716,206,207,717,718,719,720,721,722,723,724,725,726,727,729,730,731,732,733,735,736,737,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,755,756,757,758,759,760,762,763,765,767,768,769,771,772,773,774,775,776,777,779,781,783,784,786,787,789,790,791,793,794,796,799,800,907,908,913,916,924,930,945,953,966]

        # for couple in couple_list:
        #     testCouple = couple

        train_df = df.loc[df['ID'] != testCouple]
        train_df_stressed = train_df.loc[train_df['Stress'] != 0].drop(columns=['Stress', 'Gender', 'ID', 'Race', 'Report']).to_numpy()
        train_df_unstressed = train_df.loc[train_df['Stress'] == 0].drop(columns=['Stress', 'Gender', 'ID', 'Race', 'Report']).to_numpy()

        test_df = df.loc[df['ID'] == testCouple]
        select_count = int(len(test_df) * 0.4)
        test_df = test_df.iloc[:select_count , :]
        test_df_stressed = test_df.loc[test_df['Stress'] != 0].drop(columns=['Stress', 'Gender', 'ID', 'Race', 'Report']).to_numpy()
        test_df_unstressed = test_df.loc[test_df['Stress'] == 0].drop(columns=['Stress', 'Gender', 'ID', 'Race', 'Report']).to_numpy()

        # print(len(test_df_stressed))
        # print(len(test_df_unstressed))
        self.stress_ratio = float(len(test_df_stressed)) / len(test_df)

        all_unstressed_distance = []
        for i in range(len(test_df_unstressed)):
            unstressed_distance_tmp = [np.linalg.norm(test_df_unstressed[i]-tmp) for tmp in train_df_unstressed]
            all_unstressed_distance.append(unstressed_distance_tmp)
        all_unstressed_distance = np.transpose(np.array(all_unstressed_distance))
        unstressed_distance = [min(tmp) for tmp in all_unstressed_distance]

        all_stressed_distance = []
        for i in range(len(test_df_stressed)):
            stressed_distance_tmp = [np.linalg.norm(test_df_stressed[i]-tmp) for tmp in train_df_stressed]
            all_stressed_distance.append(stressed_distance_tmp)
        all_stressed_distance = np.transpose(np.array(all_stressed_distance))
        stressed_distance = [min(tmp) for tmp in all_stressed_distance]
  
        train_dict_stressed = dict(zip(stressed_distance, train_df_stressed))
        train_dict_unstressed = dict(zip(unstressed_distance, train_df_unstressed))


        top_stressed_keys = list(sorted(train_dict_stressed.keys()))[:self.top_samples]
        top_unstressed_keys = list(sorted(train_dict_unstressed.keys()))[:self.top_samples]

        top_stressed_samples = [train_dict_stressed[tmp] for tmp in top_stressed_keys]
        top_unstressed_samples = [train_dict_unstressed[tmp] for tmp in top_unstressed_keys]


        # couple_data_0_center = np.mean(top_unstressed_samples, axis=0)
        # couple_data_1_center = np.mean(top_stressed_samples, axis=0)

        # dist = np.linalg.norm(couple_data_0_center-couple_data_1_center)

        # print(top_stressed_keys)
        # print(top_unstressed_keys)


        for i in range(self.top_samples):
            datas.append([test_df_stressed[np.random.choice(test_df_stressed.shape[0], size=1), :][0], top_stressed_samples[i], torch.from_numpy(np.array([1.0], dtype=np.float32))])
            datas.append([test_df_stressed[np.random.choice(test_df_stressed.shape[0], size=1), :][0], top_unstressed_samples[i], torch.from_numpy(np.array([0.0], dtype=np.float32))])
            datas.append([test_df_unstressed[np.random.choice(test_df_unstressed.shape[0], size=1), :][0], top_unstressed_samples[i], torch.from_numpy(np.array([1.0], dtype=np.float32))])
            datas.append([test_df_unstressed[np.random.choice(test_df_unstressed.shape[0], size=1), :][0], top_stressed_samples[i], torch.from_numpy(np.array([0.0], dtype=np.float32))])

        # self.couple_data_0_center = np.mean(top_unstressed_samples+[test_df_unstressed], axis=0)
        # self.couple_data_1_center = np.mean(top_stressed_samples+[test_df_stressed], axis=0)


        self.train_X = top_stressed_samples + list(test_df_stressed) + top_unstressed_samples + list(test_df_unstressed)
        # print(len(self.KNN_X))
        self.train_y = [1] * (len(top_stressed_samples)+len(list(test_df_stressed))) + [0] * (len(top_unstressed_samples)+len(list(test_df_unstressed)))
        self.train_stress_count = len(top_stressed_samples)+len(list(test_df_stressed))
        self.train_unstress_count = len(top_unstressed_samples)+len(list(test_df_unstressed))
        # print(np.linalg.norm(self.couple_data_0_center-self.couple_data_1_center))
        # print(len(self.train_X))
        # print(len(list(test_df_unstressed)))
        # exit()
        random.shuffle(datas)
        return datas

    def __len__(self):
        # print(30*num_sample)
        # return  self.epoch
        return 4*self.top_samples

    def __getitem__(self, index):
  
        return self.datas[index]


# trainSet = Train_Siamese('../leave_one_couple_out_analysis/preprocessed_data_valid_couples_v1.csv', 102)








