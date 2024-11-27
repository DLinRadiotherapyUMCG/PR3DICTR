# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:53:31 2024

@author: HoekL02
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os

import cv2
import matplotlib.pyplot as plt
""""
Data handling plan:

//zkh/appdata/RTDicom/Projectline_HNC_modelling/Users/Luuk vd Hoek/Projects collection/Project1/Plan data handling project 1.docx

"""



#%% Load all the endpoint data and patient information
def PatRated_survey_data(path=r'C:\Users\Luuk\Desktop\project1\data\tox info/', tox = ''):
    """
    This function is highly data-specific 
    
    All weeks of survey collection for this dataset
    weeks = [0,1,2,3,4,5,6,7,12,24,48,72,96,120,144,168,192,216,240]
    Weeks_select by default excludes week 120, 168 and 216 (all <1% response rate)
    
    """
    
    if tox == 'xerostomia':
        path = path + 'Data_voor_Luuk_xerostomia.xlsx'
    elif tox == 'taste loss':
        path = path + 'Data_voor_Luuk_taste_loss.xlsx'
    elif tox == 'dysphagia':
        path = path + 'Data_voor_Luuk_Dysphagia.xlsx'
    else:
        print('No such tox, selected default xerostomia')
        path = path + 'Data_voor_Luuk_xerostomia.xlsx'
     
    df = pd.read_excel(path)    
    df = df.to_numpy()
    
    umcg_n = df[:,0] 
    df = df[:,1:]
    
    df[np.where(df == 'Een beetje')] = 1
    df[np.where(df == 'Heel erg')] = 3
    df[np.where(df == 'Helemaal niet')] = 0
    df[np.where(df == 'Nogal')] = 2    
    
    df = df.astype(float)

    return np.round(df/2.1), df, umcg_n

def physRated_survey_data(path = r'C:\Users\Luuk\Desktop\project1\data\tox info\Data_voor_Luuk_Dysphagia.xlsx'):
    
    df = pd.read_excel(path)
   
    df = df.to_numpy()
    
    umcg_n = df[:,0]
    df = df[:,1:]
    
    df[np.where(df == 'Grade 0-1 (regular diet)')] = 0
    df[np.where(df == 'Grade 2 (soft food)')] = 2
    df[np.where(df == 'Grade 3 (liquids only )')] = 3
    df[np.where(df == 'Grade 4-5 (tube feeding dependent)')] = 4
    
    df[np.where(df == 'wel slikklachten, maar kan wel alles eten')] = 1
    df[np.where(df == 'geen slikklachten/normaal eetpatroon')] = 0
    df[np.where(df == 'alleen gepureerd/zacht voedsel')] = 2
    df[np.where(df == 'alleen vloeibaar voedsel, geen sondevoeding')] = 3
    df[np.where(df == 'neussonde of PEG, geen enkele orale intake meer mogelijk')] = 5
    df[np.where(df == 'neussonde of PEG, maar nog wel orale intake mogelijk')] = 4
    
    df = df.astype(float)
    df_bin = df.copy()
    df_bin[np.where(df_bin < 2)] = 0
    df_bin[np.where(df_bin > 1)] = 1
    
    return df_bin, df, umcg_n

def full_tox_data(tox = '',binary = False):
    if tox == 'Dysphagia' or tox == 'DS':
        if binary:
            df,_,umcg_n = physRated_survey_data()
        else:
            _,df,umcg_n = physRated_survey_data()
    else:
        if binary:
            df,_,umcg_n = PatRated_survey_data(tox=tox)
        else:
            _,df,umcg_n = PatRated_survey_data(tox=tox)        
    
    return df,umcg_n

def imputed_full_tox_data(tox = '',binary = False, min_avail_pdat = 8):
    if tox == 'Dysphagia' or tox == 'DS':
        _,df,umcg_n = physRated_survey_data()
    else:
        _,df,umcg_n = PatRated_survey_data(tox=tox)       
    
    critera_indx = np.where(np.sum(~np.isnan(df),axis = 1) > min_avail_pdat)[0]

    df = df.astype(float)
    LR_imputer = IterativeImputer(max_iter = 300, random_state= 0)
    LR_imputer.fit(df)
    IterativeImputer(random_state=0)
    df = LR_imputer.transform(df)
    
    if binary:
        if tox == 'Dysphagia' or tox == 'DS':
            df = np.round(df/3.1)
        else:
            df = np.round(df/2.1)
    
    return df,umcg_n, critera_indx

def get_umcg_n(path = r'C:\Users\Luuk\Desktop\project1\data\patients'):
    
    umcg_n = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    
    return np.array(umcg_n)

def get_endpoint_data(X, time_point_idx = 1, imputed = True, binary = False, tox = ''):
    """
    Helper function for Data_loader (and can be used to check patient response vector)
    
    Input:
        X: Vector containing patient ID's
        tox: string indicating desired tox [Xerostomia, Dysphagia, Taste loss, Sticky saliva]
    
    Output:
        X: Vector containing patient ID's 
        y: vector of vectors containing all end points for the requested ID's
        
    """
    
    if imputed:
        df,umcg_n,critera_indx = imputed_full_tox_data(tox = tox, binary = binary)
    else:
        df,umcg_n = full_tox_data(tox = tox, binary = binary)
        
    umcg_n_avail = get_umcg_n()
    
    if imputed == True:
        umcg_n_rm = umcg_n[critera_indx]
        umcg_n_avail = umcg_n_rm[np.in1d(umcg_n_rm.astype(float),umcg_n_avail.astype(float))]
        umcg_n_avail_set = umcg_n_avail[np.in1d(umcg_n_avail.astype(float),X.astype(float))]
        X = X[np.in1d(X.astype(float),umcg_n_avail_set.astype(float))]
        
        indx = []
        rm = []
        for i in range(len(X)):
            if len(np.where(umcg_n.astype(float) == X[i].astype(float))[0]) > 0:
                indx.append(np.where(umcg_n.astype(float) == X[i].astype(float))[0][0])
            else:
                rm.append(i)
        X = np.delete(X,rm)
        end_points = df[indx,time_point_idx]        
    
    if imputed == False:     
        indx = []
        rm = []
        for i in range(len(X)):
            if len(np.where(umcg_n.astype(float) == X[i].astype(float))[0]) > 0:
                indx.append(np.where(umcg_n.astype(float) == X[i].astype(float))[0][0])
            else:
                rm.append(i)
        X = np.delete(X,rm)
        end_points = df[indx,time_point_idx]
        
        X = X[~np.isnan(end_points)]
        end_points = end_points[~np.isnan(end_points)]
        
    if binary:
        end_points = np.clip(end_points,0,1)
    else:
        end_points = np.round(end_points)
        if tox  == 'Dysphagia' or tox == 'DS':
            end_points = end_points.clip(0,5)
        else:
            end_points = end_points.clip(0,3)
    return X, end_points

#%% Loading grouped patients

def get_Loctum2_IDs(loctum = "Larynx"):
    """
    Types as in Loctum2 = 'Categorie: overig'(6), 'Hypofarynx'(100), 'Larynx'(504), 'Nasopharynx'(49), 'Neus(bij)holte'(9), 'Oral Cavity'(66), 'Oropharynx'(433), 'nan'(1)
    
    Input: loctum type
    Output: patient IDs
    
    """
    
    df = pd.read_excel(r'C:\Users\Luuk\Desktop\project1\data\patients_all_features.xlsx')
    np_df = np.array(df)
    
    IDs = np_df[:,0][np.where(np_df[:,5] == loctum)]
    
    return IDs.astype(int)

def get_TN_IDS(T = 'None', N = 'None'):
    """
    Types as in Loctum2 = 'Categorie: overig'(6), 'Hypofarynx'(100), 'Larynx'(504), 'Nasopharynx'(49), 'Neus(bij)holte'(9), 'Oral Cavity'(66), 'Oropharynx'(433), 'nan'(1)
    
    Input: loctum type
    Output: patient IDs
    
    """
    
    df = pd.read_excel(r'C:\Users\Luuk\Desktop\project1\data\patients_all_features.xlsx')
    np_df = np.array(df)
    
    if T == 'None' and N != 'None':
        IDs = np_df[:,0][np.where(np_df[:,8] == N)]
        
    elif N == 'None' and T != 'None':
        IDs = np_df[:,0][np.where(np_df[:,7] == T)]
        
    elif N == T:
        print('Select T and or N stage')
    
    else:
        IDs = np_df[:,0][np.where(np_df[:,8] == N and np_df[:7] == T)]
    
    return IDs.astype(int)


#%% Splitter
def Data_split(fd, split = [.7,.15,.15],seed = 8):
    """
    Splits complete dataset [rows,columns] (rows = patients, columns = variables)
    into three subsets: train, validation and test
    
    """    
    np.random.seed(seed)
    indx = np.arange(len(fd))
    np.random.shuffle(indx)
    
    train = fd[indx[:int(split[0]*len(indx))]]
    validation = fd[indx[int(split[0]*len(indx)):int(split[0]*len(indx)) +int(split[1]*len(indx))]]
    test = fd[indx[int(split[0]*len(indx)) +int(split[1]*len(indx)):]]
    
    return train, validation, test

#%% Artificial end-point equality

def label_equalizer(X,y, method = "up"):
    
    X_c = X.copy()
    y_c = y.copy()
    
    instance_labels =[]
    C_labels = len(np.unique(y))

    for i in range(C_labels):
        instance_labels.append(np.sum(y == np.unique(y)[i]))
        
    freq_max = np.max(instance_labels)    
    freq_min = np.min(instance_labels)
    
    if method == 'up':
        for i in range(len(instance_labels)):
            for ii in range(freq_max-instance_labels[i]):
                r_indx = np.where(y_c == np.unique(y_c)[i])[0][np.random.randint(0,instance_labels[i]-1)]
                X = np.append(X,X_c[r_indx])
                y = np.append(y,y_c[r_indx])
        return X,y
    
    
    if method == 'down':
        X_D = []
        y_D = []
        for i in range(len(instance_labels)):
            r_indx = np.where(y_c == np.unique(y_c)[i])[0]
            np.random.shuffle(r_indx)
            X_D.extend(X[r_indx[:freq_min]])
            y_D.extend(y[r_indx[:freq_min]])
        return np.array(X_D),np.array(y_D)
    
    if method == "None":
        return X,y
    



#%% Dataset
class Data_loader(Dataset):
    """
    Data loader to convert information from preprocessed image data folder into useable tensors.
    
    Preprocessed folder structure:
        -main directory (one folder named XXX)
            -patient ID (N folders named xxx)
                - ct.npy (array, size: 1x100x100x100, value range: [-1000,1000])
                - rtdose.npy (array, size: 1x100x100x100, value range: [0,8000])
                - segmentation_map.npy (array, size: 1x100x100x100, value range: [0,1])
    
    Input:
        X: vector of patient ID's
        y: vector of vectors of patients end-points
        bs: batch size
    
    Output: 
        img: processed and stacked ct and rtdose torch Tensor shape: bsx2x96x96x96
    
    """
    
    
    def __init__(self, X, y, 
                 imgpath= r'C:\Users\Luuk\Desktop\project1\data\patients/',
                 transform = None):
        super(Data_loader, self).__init__()  
        
        self.X = X
        self.y = y
        
        self.imgpath = imgpath

        self.transform = transform

    def __len__(self):
      return len(self.X)
  

    def __getitem__(self, idx):
            patient_folder_path = self.imgpath + self.X[idx]
            patient_CT_path = patient_folder_path + '/ct.npy'
            patient_dose_path = patient_folder_path + '/rtdose.npy'
            patient_seg_path = patient_folder_path + '/segmentation_map.npy'
                
            ct = np.load(patient_CT_path)
            dose = np.load(patient_dose_path)
            seg = np.load(patient_seg_path)
            
            image = np.vstack((ct,dose,seg))
            
            label = self.y[idx]
            
            image = torch.Tensor(image)
            label = torch.Tensor(np.array(label))
            
            if self.transform:
                image = self.transform(image) 

            
            return [image, label]
        
class Data_loader_features(Dataset):
    """
    Data loader to convert information from preprocessed image data folder into useable tensors.
    
    Preprocessed folder structure:
        -main directory (one folder named XXX)
            -patient ID (N folders named xxx)
                - ct.npy (array, size: 1x100x100x100, value range: [-1000,1000])
                - rtdose.npy (array, size: 1x100x100x100, value range: [0,8000])
                - segmentation_map.npy (array, size: 1x100x100x100, value range: [0,1])
    
    Input:
        X: vector of patient ID's
        y: vector of vectors of patients end-points
        bs: batch size
    
    Output: 
        img: processed and stacked ct and rtdose torch Tensor shape: bsx2x96x96x96
    
    """
    
    
    def __init__(self, X, y,
                 features_list = ['Xer_w1_0','Xer_w1_1','Xer_W1_2_3','sex','age'],
                 features_path = r'C:\Users\Luuk\Desktop\project1\data\clin_features.xlsx',
                 imgpath= r'C:\Users\Luuk\Desktop\project1\data\patients/',
                 transform = None):
        super(Data_loader_features, self).__init__()  
        
        
        self.features_list = features_list
        self.df = pd.read_excel(features_path)
        
        self.X = X
        self.y = y
        
        self.imgpath = imgpath

        self.transform = transform

    def __len__(self):
      return len(self.X)
  

    def __getitem__(self, idx):
            patient_folder_path = self.imgpath + self.X[idx]
            patient_CT_path = patient_folder_path + '/ct.npy'
            patient_dose_path = patient_folder_path + '/rtdose.npy'
            patient_seg_path = patient_folder_path + '/segmentation_map.npy'
                
            ct = np.load(patient_CT_path)
            dose = np.load(patient_dose_path)
            seg = np.load(patient_seg_path)
            
            image = np.vstack((ct,dose,seg))
            
            label = self.y[idx]
            
            image = torch.Tensor(image)
            label = torch.Tensor(np.array(label))
            
            if self.transform:
                image = self.transform(image) 

            clin_features = [self.df[self.df["PatientID"] == self.df.iloc[idx]['PatientID']][feature].values[0] for feature in self.features_list]
            
            clin_features = np.array(clin_features).astype(np.float32)
            
            return [image, clin_features, label]
        
#%% Augmentations
"""
All input for augmentations are expected to be 3*x100x100x100 

* 3 if segmentations are included

"""

class segmentation_adjustment():
    def __init__(self,set_labels = [0,1,1,1,1,0,0,0,0,.1,0,.1,.1,0,0,0,0]):
        self.set_labels = set_labels
    def __call__(self,img):
        for i in range(len(self.set_labels)):
            img[2][img[2,:,:,:] ==i] = self.set_labels[i]
        return img


    
class windowing(object):
    def __init__(self, w_min_CT=-200, w_max_CT=400,i_min_CT=0,i_max_CT=1,
                     w_min_DS=0, w_max_DS=8000,i_min_DS=0,i_max_DS=1,):
        self.w_min_CT = w_min_CT
        self.w_max_CT = w_max_CT
        self.i_min_CT = i_min_CT
        self.i_max_CT = i_max_CT
        
        self.w_min_DS = w_min_DS
        self.w_max_DS = w_max_DS
        self.i_min_DS = i_min_DS
        self.i_max_DS = i_max_DS
               
    def __call__(self, img):
        img[0,:,:,:] = (img[0,:,:,:] - self.w_min_CT)/(self.w_max_CT-self.w_min_CT)
        img[0,:,:,:] = np.clip(img[0,:,:,:],self.i_min_CT,self.i_max_CT)  
        
        
        img[1,:,:,:] = (img[1,:,:,:] - self.w_min_DS)/(self.w_max_DS-self.w_min_DS)
        img[1,:,:,:] = np.clip(img[1,:,:,:],self.i_min_DS,self.i_max_DS)

        return img 
   

class Cropping(object):
    """
    Cropping also aplies translation by selecting a random position to start the crop from
    
    """
    
    def __init__(self, output_size = [96,96,96],centered= False):
        self.output_size = output_size
        self.centered = centered

    def __call__(self, img):
        
        h,w,d = img.shape[1],img.shape[2],img.shape[3]
        new_h,new_w,new_d = self.output_size[0],self.output_size[1],self.output_size[2]
        
        if self.centered:
            x = (h - new_h + 1)//2
            y = (w - new_w + 1)//2
            z = (d - new_d + 1)//2        
        else:
            x = np.random.randint(0, h - new_h + 1)
            y = np.random.randint(0, w - new_w + 1)
            z = np.random.randint(0, d - new_d + 1)
        

        img = img[:,x: x + new_h,
                  y: y + new_w,
                  z: z + new_d]
        
        return img

class Flipping(object):
    
    def __init__(self, flip_odds = 1):
        self.flip_odds = flip_odds

    def __call__(self, img):
        flip = np.random.rand()
        if flip < self.flip_odds:
            img[0,:,:,:] = torch.flip(img[0,:,:,:],[1,-1])
            img[1,:,:,:] = torch.flip(img[1,:,:,:],[1,-1])
            img[2,:,:,:] = torch.flip(img[2,:,:,:],[1,-1])
        return img


class Zooming(object):
    def __init__(self, output_size = [96,96,96], zoom_strength = 4):
        self.output_size = output_size
        self.zoom_strength = zoom_strength

    def __call__(self, img):
        
        h,w,d =img.shape[1],img.shape[2],img.shape[3] 
        
        new_h,new_w,new_d = self.output_size[0],self.output_size[1],self.output_size[2]
        
        zoom = np.random.randint(-self.zoom_strength,self.zoom_strength)
        
         
        x = np.random.randint(0, -new_h + h - zoom + 1)
        y = np.random.randint(0, -new_w + w - zoom + 1)
        z = np.random.randint(0, -new_d + d - zoom + 1)
        
        img = img[:,x: x + new_h + zoom,
                  y: y + new_w + zoom,
                  z: z + new_d + zoom]    
        
        img = img.unsqueeze(0)
        img = torch.nn.functional.interpolate(img,self.output_size[0])
        img = img.squeeze(0)
        
        return img
    
class Rotating(object):
    def __init__(self, angle = 50):
        self.angle = angle

    def __call__(self, img):
        rot_angle = np.random.uniform(-self.angle,self.angle)
        img = img
        return img

class Intensity_shift(object):
    
    def __init__(self, intense_max=5):
        self.intense_max = intense_max

    def __call__(self, img):
        I_max = torch.max(img[0])
        I_min = torch.min(img[0])
        I_range = I_max - I_min
        gamma = np.random.random()*(self.intense_max-.5) + .5
        img[0] = (((img[0]-I_min)/I_range)**gamma)*I_range + I_min

        return img
    
#%% 

def get_patiens(x,y=[0]):
    
    volumes = []
    
    for pat in x:
        out = Data_loader([pat],y,transform = transforms.Compose([Cropping(centered = True),windowing(),segmentation_adjustment()]))
        out = out[0][0]
        
        volumes.append(out)
    
    return np.array(volumes)
    
    
#%% Combined function

def get_data_loader(batch_size,num_works = 12,time_point_idx = 11, 
                    imputed = True,binary = True, tox = 'xerostomia', 
                    shuffle = True, label_eq = True, label_eq_meth = 'down', seed = 8,features = False):
    
    seg_adj = segmentation_adjustment()
    window = windowing()
    crop = Cropping()
    flip = Flipping()
    zoom = Zooming()
    intens_shift = Intensity_shift()
    rotate = Rotating()
    composed = transforms.Compose([window,flip,zoom,intens_shift,seg_adj])
    crop_only = transforms.Compose([crop,window,seg_adj])
    
    umcg_n = get_umcg_n()
    
    
    train,validation,test = Data_split(umcg_n, seed = seed)
    
    X = train
    X,y = get_endpoint_data(X,time_point_idx = time_point_idx, imputed = imputed, binary = binary, tox = tox)
    
    if label_eq:
        X,y = label_equalizer(X,y,label_eq_meth)
    
    
    if features:
        Train_data = Data_loader_features(X,y,transform = crop_only)
    else:
        Train_data = Data_loader(X,y,transform = crop_only)
        
    
    # Train_data = Data_loader(X,y,transform = crop_only)
    Train_data = DataLoader(Train_data,batch_size = batch_size, shuffle = shuffle, num_workers= num_works)
    
    X_val = validation
    X_val,y_val = get_endpoint_data(X_val,time_point_idx = time_point_idx, imputed = False, binary = binary, tox = tox)
    # Val_data = Data_loader(X_val,y_val,transform = crop_only)
    if features:
        Val_data = Data_loader_features(X_val,y_val,transform = crop_only)
    else:
        Val_data = Data_loader(X_val,y_val,transform = crop_only)
    
    
    Val_data = DataLoader(Val_data,batch_size = batch_size, num_workers= num_works)
    
    X_test = test
    X_test,y_test = get_endpoint_data(X_test,time_point_idx = time_point_idx, imputed = False, binary = binary, tox = tox)
    
    if features:
        Test_data = Data_loader_features(X_test,y_test,transform = crop_only)
    else:
        Test_data = Data_loader(X_test,y_test,transform = crop_only)
    
    # Test_data = Data_loader(X_test,y_test,transform = crop_only)
    Test_data = DataLoader(Test_data,batch_size = batch_size, num_workers= num_works) 
    
    
    
    return Train_data,Val_data,Test_data

if __name__ == '__main__':
    
    batch_size = 1
    
    # train,validation,test = get_data_loader(batch_size,imputed=True, time_point_idx = 0,label_eq= True)
    
    
    num_works = 4
    time_point_idx = 0
    imputed = True
    binary = False
    tox = 'xerostomia'
    shuffle = True
    seg_adj = segmentation_adjustment()
    window = windowing()
    crop = Cropping()
    flip = Flipping()
    zoom = Zooming()
    intens_shift = Intensity_shift()
    rotate = Rotating()
    composed = transforms.Compose([window,zoom,seg_adj])
    
    umcg_n = get_umcg_n()
    
    train,validation,test = Data_split(umcg_n)
    
    X = train
    X,y = get_endpoint_data(X,time_point_idx = time_point_idx, imputed = imputed, binary = binary, tox = tox)
    Train_data = Data_loader_features(X,y,transform = composed)
    Train_data = DataLoader(Train_data,batch_size = batch_size, shuffle = shuffle, num_workers= num_works)
    
    X_val = validation
    X_val,y_val = get_endpoint_data(X_val,time_point_idx = time_point_idx, imputed = False, binary = binary, tox = tox)
    Val_data = Data_loader(X_val,y_val,transform = None)
    Val_data = DataLoader(Val_data,batch_size = batch_size, num_workers= num_works)
    
    X_test = test
    X_test,y_test = get_endpoint_data(X_test,time_point_idx = time_point_idx, imputed = False, binary = binary, tox = tox)
    Test_data = Data_loader(X_test,y_test,transform = None)
    Test_data = DataLoader(Test_data,batch_size = batch_size, num_workers= num_works)  
    
    
    for a,b in enumerate(Train_data):
        print(b)
        # if a == 2:
        break
