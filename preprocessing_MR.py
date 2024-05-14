# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:38:21 2023

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:23:40 2023

@author: admin
"""

# N4 bias field correction /resampling isotropic voxels of 1 mm3 using B-spline interpolation/  standardization of intensity values

import numpy as np
import SimpleITK as sitk
import os
#硬盘dicom转nii.gz到本地
def dcm2nii(dcms_path, nii_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
    # 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)   

def dicom2Nii(folderPath,savefolder):   
    b=os.listdir(folderPath)
    for every_patient in b[1:124]:#遍历所有的患者列表b
        tmp_MR_path = os.path.join(folderPath,every_patient)
        _save_path =os.path.join(savefolder,every_patient)
        if not os.path.exists(_save_path):
            os.makedirs(_save_path) #创建所有的患者列表
     
        for every_time in os.listdir(tmp_MR_path):# 每位患者下面的每次MR
            time_MRI_path = os.path.join(tmp_MR_path,every_time)
            time_save_path = os.path.join(_save_path,every_time)
            if not os.path.exists(time_save_path):
                os.makedirs(time_save_path)#创建患者的每次MR文件夹
    
            for every_MRI in os.listdir(time_MRI_path):# 每次MR下的序列                           
                tmp_MRI_path = os.path.join(time_MRI_path,every_MRI)                
                path_save = time_save_path + '\\'+every_MRI+'.nii.gz'
                dcm2nii(tmp_MRI_path, path_save)

if __name__ =="__main__":
    folderPath ='F:\\'  
    savepath = 'C:\\Users\\admin\\Desktop\\yyl'
    dicom2Nii(folderPath,savepath)

#MR重采样+归一化+N4          
def img_resample(ori_data, new_spacing=[1.2, 1.2, 1.2]):

    original_spacing = ori_data.GetSpacing()
    original_size = ori_data.GetSize()
    
    transform = sitk.Transform()
    transform.SetIdentity()   
    
    new_shape = [
        int(np.round(original_spacing[0] * original_size[0] / new_spacing[0])),
        int(np.round(original_spacing[1] * original_size[1] / new_spacing[1])),
        int(np.round(original_spacing[2] * original_size[2] / new_spacing[2])),
    ]
    # print("新的形状大小为",new_shape)
    resmaple = sitk.ResampleImageFilter()  # 初始化 #初始化一个图像重采样滤波器resampler
    resmaple.SetInterpolator(sitk.sitkBSpline) #B-spline interpolation
   
    resmaple.SetTransform(transform)
    resmaple.SetDefaultPixelValue(0)
   
    resmaple.SetSize(new_shape)
    resmaple.SetOutputSpacing(new_spacing)
    resmaple.SetOutputOrigin(ori_data.GetOrigin())
    resmaple.SetOutputDirection(ori_data.GetDirection())
   
    data = resmaple.Execute(ori_data)
    return data
 
def zScoring(image):
    image = (image - np.mean(image)) / np.std(image)  # z-scoring归一化
    return image

out_path = 'D:\\MRI\\'
imgs_path= 'E:\\LVI\\LVI -\\'
files = os.listdir(imgs_path)
for every_patient in files:#遍历所有的患者列表b
     every_patient_path = os.path.join(imgs_path,every_patient)
     save_path =os.path.join(out_path,every_patient)
     if not os.path.exists(save_path):
         os.makedirs(save_path) 
     a=os.listdir(every_patient_path)
     for every_time in a: # 序列文件夹 
         every_time_path = os.path.join(every_patient_path,every_time)
         time_save_path = os.path.join(save_path,every_time)
         if not os.path.exists(time_save_path):
             os.makedirs(time_save_path)
         b=os.listdir(every_time_path)#序列列表             
         for every_img in  b:
             if every_img[-13]=='6'or every_img[-13]=='8':                
                    every_img_path = os.path.join( every_time_path,every_img)
              # N4 bias field correction
                    input_image = sitk.ReadImage( every_img_path,sitk.sitkFloat64)
                    output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)                                
              #resampling isotropic voxels
                    Resample_img = img_resample(output_image)                 
              #归一化          
                    image_array = sitk.GetArrayFromImage( Resample_img)       
                    img=zScoring(image_array)#归一化                       
                    image_out = sitk.GetImageFromArray(img)                    
                    image_out.SetOrigin( Resample_img.GetOrigin()) 
                    image_out.SetSpacing( Resample_img.GetSpacing())
                    image_out.SetDirection( Resample_img.GetDirection())
                    Proes_path =out_path + every_img
                    sitk.WriteImage(image_out,  Proes_path)






             


         
