# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:35:16 2023

@author: admin
"""

from radiomics import featureextractor
import pandas as pd
import os

#生境特征,MRI_img需要先归一化
out_path = 'E:/knee/'
imgs_path= 'E:/knee/niiDataOA/'
maskDir ='E:/knee/maskOA/'
extractor = featureextractor.RadiomicsFeatureExtractor()#初始化feature extractor
extractor.loadParams('radiomics.yaml')
df = pd.DataFrame()  #创建一个空表
#modal = ' wt2f'
files = os.listdir(maskDir)
i = 0
for file in files:
    pName = file[0:7]
    image_path = imgs_path + pName  + '.nii.gz'
    mask_path = maskDir + file
    #fileName = pName + modal
    featureVector = extractor.execute( image_path, mask_path) #提取特征
    df_add = pd.DataFrame.from_dict(featureVector.values()).T #获取特征值，转为表格形式，并转置
    df_add.columns = featureVector.keys() #获得特征名
    patients_ID=pd.DataFrame({'patients_ID':[pName]})#转为表格形式
    df_add=pd.concat([patients_ID,df_add],axis=1) #以左右的方式拼接patients_ID和df_add。
    df = pd.concat([df,df_add])#将所有病例放入一个表格中
    print('++++++++++++++++++++++++++'+str(i)+'++++++++++++++++++++++++++++++++')
    i= i+1
df.to_csv(out_path +'results'+'_Label_4'+'.csv')



#瘤周特征
# workRootDir = 'D:/data_YL/LungCa_LVI/LCa_LVI/'
# dataDir = workRootDir + 'yinxingData/'
# mask_all = workRootDir + 'yinxingMaskRevised/'
# mask_dorder= os.listdir(mask_all)

# extractor = featureextractor.RadiomicsFeatureExtractor()
# extractor.loadParams('radiomics.yaml')

# #label = extractor.settings['label']
# #modal = ' wt2f'

# for c in range(len(mask_dorder)):   
#      df = pd.DataFrame()   #创建一个空表
#      maskDir=mask_all + mask_dorder[c] #找到相应瘤周文件夹
#      files = os.listdir(maskDir)
#      for file in files:
#          pName = file[0:-21] #提取患者名字
#          imageName = dataDir + pName  + '.nii.gz' #找到img路径
#          maskName = maskDir +'/'+ file  #找到mask路径   
#          featureVector = extractor.execute(imageName,maskName) #提取特征
#          df_add = pd.DataFrame.from_dict(featureVector.values()).T #获取特征值，转为表格形式，并转置
#          df_add.columns = featureVector.keys() #获得特征名
#          patients_ID=pd.DataFrame({'patients_ID':[pName]})#转为表格形式
#          df_add=pd.concat([patients_ID,df_add],axis=1) #以左右的方式拼接patients_ID和df_add。
#          df = pd.concat([df,df_add])#将所有病例放入一个表格中
#      df.to_csv(workRootDir +'results_' +mask_dorder[c]+'.csv')
# print('+++++++++++++++++++++++++All_subjects_Finished++++++++++++++++++++++++++++++++++')


