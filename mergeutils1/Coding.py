

from ast import Str
from pickle import NONE
from random import sample
from typing import Literal
import numpy as np
from osgeo import gdal
import sys
from mergeutils.Image_preprocessing import*
from mergeutils.Sample import Sample_methods
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
def overlay(inputdir:Str,files:Literal,options:Literal):
    '''
    Description:叠加图层,将各个因子图层叠加为三维的矩阵
    inputdir:输入文件的文件夹
    outdir:输出文件夹
    options:投影参数
    return:叠加好的三维栅格图层
    '''
    print('overlay')
    i=0
    layers=np.array([])
    for file in files:
        i=i+1
    layers=np.zeros((i,options['height'],options['width']))
    #插在前面快很多
    i=0
    for file in files:
        inputfile=inputdir+file
        dataset = gdal.Open(inputfile)
        # 判断是否读取到数据
        if dataset is None:
            print('Unable to open *.tif')
            sys.exit(1)  # 退出
        # 直接读取dataset
        img_array = dataset.ReadAsArray()
        binayuan=0
        if(np.count_nonzero(img_array==img_array.max())>np.count_nonzero(img_array==0)):
            bianyuan=img_array==img_array.max()
        img_array=np.where(img_array==bianyuan,np.NaN,img_array)
        print((~np.isnan(img_array)).sum())
        print(img_array.shape)
        layers[i,:,:]=img_array
        i=i+1
    del img_array
    return layers

def Remove_Nan(inputdir:Str,files:Literal,sample=NONE):
    '''
    Description:各个因子图层相同位置可能有缺失,缺失的地方默认值维最大值,将其转换为NAN值进行相加,由于NAN值
                和常值相加为NAN,即可提取出各个图层都有值的栅格坐标。
    inputdir:输入文件的文件夹
    files:输入文件集,输入的文件的栅格值的最大最小值应该是一样的
    sample:样本点
    return:所有因素和样本点叠加的位置
    '''
    print('Remove_Nan')
    i=0
    layers=np.array([])
    for file in files:
        inputfile=inputdir+file
        dataset = gdal.Open(inputfile)
        # 判断是否读取到数据
        if dataset is None:
            print('Unable to open *.tif')
            sys.exit(1)  # 退出
        # 直接读取dataset
        img_array = dataset.ReadAsArray()
        print(img_array.shape)
        binayuan=0
        if(np.count_nonzero(img_array==img_array.max())>np.count_nonzero(img_array==0)):
            bianyuan=img_array==img_array.max()
        img_array=np.where(img_array==bianyuan,np.NaN,img_array)
        if(i==0 and sample!=NONE):
            layers=img_array+sample
        if(i==0 and sample==NONE):
            layers=img_array
        if(i>0):
            layers=layers+img_array
            print((~np.isnan(layers)).sum())
        i=i+1
    position=np.argwhere(~np.isnan(layers))#获取不为Nan的位置
    position=np.array(position)
    del img_array,layers
    return position

def create_canvas(inputdir:Str,files:Literal,options:Literal,sample=NONE):
    '''
    Descriptuon:通过overlay以及Remove_NAN之后的操作得到的三维矩阵以及栅格坐标创建画布,用于获得到模型后的
    出图。
    inputdir:输入文件的文件夹
    files:输入文件集
    options:投影参数
    '''
    canvas_position=Remove_Nan(inputdir,files)
    layers=overlay(inputdir,files,options)
    canvas=np.zeros([len(canvas_position[:,0]),len(layers[:,0,0])],dtype=np.int16)
    for i in range(len(canvas_position[:,0])):
        canvas[i]=layers[:,canvas_position[i,0],canvas_position[i,1]]
    result=np.zeros((len(canvas[:,0]),len(canvas[0])+2),dtype=np.int16)
    result[:,2:]=canvas
    result[:,:2]=canvas_position
    del canvas_position,layers,canvas
    # 前两项是坐标
    return result


def get_sample(featuredir:Str,outputdir:Str,featurename:Str,options:Literal,distance:float):
    '''
    Description:输入要素文件得到栅格化后的矩阵
    featuredir:输入要素文件的文件夹
    outdir:输出文件夹
    featurename:滑坡点文件名
    options:投影参数
    distance:缓冲距离
    '''
    #考虑出一个格式框输入输出文件名
    #求得缓冲区
    print('get_sample')
    buffer(featuredir,featurename, outputdir,distance)
    rasterize(outputdir+featurename+'Buffer.shp', outputdir+featurename+'Buffer.tif',options)
    # 滑坡点合集
    rasterize(featuredir+featurename+'.shp',outputdir+featurename+'Raster.tif',options)
    hua_array=gdal.Open(outputdir+featurename+'Raster.tif').ReadAsArray()
    hua_array=np.where(hua_array==0,np.NaN,hua_array)
    print((~np.isnan(hua_array)).sum())
    print((np.isnan(hua_array)).sum())
    #求非滑坡点合集
    feihua_array=gdal.Open(outputdir+featurename+'Buffer.tif').ReadAsArray()
    print(feihua_array)
    feihua_array=disbuffer(feihua_array)
    print(feihua_array)
    print((~np.isnan(feihua_array)).sum())
    print((np.isnan(feihua_array)).sum())
    sample=np.array([feihua_array,hua_array])
    del hua_array,options,feihua_array
    return sample


def merge(position:Literal,layers:Literal,lable:int):
    '''
    Description:通过已经处理好要素进行叠加,得到一个与输入栅格图层(长宽相同)的三维矩阵(1,2层为该点的栅格位置)
                (最后一层为标签),其余层为要素因子
    position:所有因素和样本点叠加的位置
    layers:通过overlay得到的三维因子矩阵
    lable:0或1分类
    return:滑坡点或非滑坡点对应位置和其特征及标签
    '''
    #通过Position获取滑坡点特征
    print('merge')
    features=np.zeros([len(position[:,0]),len(layers[:,0,0])],dtype=int)
    print(features.shape)
    print(len(layers[:,0,0]))
    for i in range(len(position[:,0])):
        features[i]=layers[:,position[i,0],position[i,1]]
    #将滑坡点特征和坐标以及标签合并
    result=np.zeros((len(features[:,0]),len(features[0])+3))
    result[:,2:-1]=features
    result[:,:2]=position
    result[:,-1]=lable
    del position,lable,features
    #features中前两个元素为X,y坐标,最后一个为标签
    return result


def Get_trainsample(inputdir:Str,files:Literal,featuredir:Str,featurename:Str,outdir:Str,options:Literal,distance:float,methods:Str):
    '''
    inputdir:输入文件的文件夹
    files:输入文件集
    featuredir:滑坡点所在文件夹
    featurename:滑坡点文件名
    outdir:输出文件夹
    options:投影参数
    distance:缓冲距离
    methods:采样方法
    '''
    print('Get_trainsample')
    sample=get_sample(featuredir,outdir,featurename,options,distance)
    layers=overlay(inputdir,files,options)
    position_hua=Remove_Nan(inputdir,files,sample[1])
    position_feihua=Remove_Nan(inputdir,files,sample[0])
    hua=(position_hua,layers,1)
    feihua=merge(position_feihua,layers,0)
    if(methods=='equal'):
        train_Sample=Sample_methods.equal(hua,feihua)
    if(methods=='smote'):
        train_Sample=Sample_methods.smote(hua,feihua)
    if(methods=='randomUnderSampler'):
        train_Sample=Sample_methods.randomUnderSampler(hua,feihua)
    if(methods=='all'):
        train_Sample=Sample_methods.allSampler(hua,feihua)
    if(methods=='Eeb'):
        train_Sample=Sample_methods.EasyEnsembleSampler(hua,feihua)
    del hua,feihua
    return train_Sample

def factorsort(train_sample:Literal,trainsize:float):
    X=train_sample[0][:,2:]
    y=train_sample[1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=trainsize,random_state=42)
    random_model=RandomForestClassifier(random_state=100)
    clt=random_model.fit(X_train,y_train)
    importances = clt.feature_importances_
    for f in range(X_train.shape[1]):
        print(importances[f])
    return importances