
from pickle import NONE
from turtle import st
from typing import Literal
from ast import Str
import numpy as np
from sklearn.metrics import roc_curve, auc
from osgeo import gdal
import matplotlib.pyplot as plt
def DisplayAndSave_Figue(pred:Literal,canvas_position:Literal,options:Literal,outputdir:Str):
    '''
    pred:模型预测结果
    canvas_position:canvas中不为Nan的位置
    options:投影参数
    outputdir:输出文件位置
    '''
    canvas_pred=pred[:,1]
    del pred
    canvas=np.zeros((options['height'],options['width']),dtype= np.float32)
    length=len(canvas_position[:,0])
    for i in range(length):
        x,y=canvas_position[i,0],canvas_position[i,1]
        canvas[x,y]=canvas_pred[i]
    print(np.isnan(canvas).sum())
    write_tiff(canvas,outputdir,options)
    # reclass_canvas=result_classify(canvas)
    # colors = ['white', 'green', 'yellow', 'green','red'] 
    # cmap = matplotlib.colors.ListedColormap(colors)
    # plt.imshow(reclass_canvas, cmap=cmap)
    # plt.title("predict figure",loc="center")
    # plt.show()
    # del reclass_canvas
   

def draw_scatter(X,y):
    '''
    X:X坐标
    y:y坐标
    '''
    plt.figure(figsize=(8, 4))
    colors = ['blue','red' ]  # 建立颜色列表
    labels = ['zero','one']  # 建立标签类别列表
    # 2.绘图
    for i in range(2):  # shape[] 类别的种类数量(2)
        plt.scatter(X[y == i, 0],  # 横坐标
                    X[y == i, 1],  # 纵坐标
                    c=colors[i],  # 颜色
                    label=labels[i])  # 标签

    # 3.展示图形
    plt.legend()  # 显示图例
    plt.show()  # 显示图片


def display_roc(y_test,pre_y,figure_name:Str):
    '''
    y_test:验证y
    pre_y:训练y
    '''
    fpr_Nb, tpr_Nb, _ = roc_curve(y_test, pre_y[:,1])
    aucval = auc(fpr_Nb, tpr_Nb)    # 计算auc的取值
    plt.figure(figsize=(10,8))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_Nb, tpr_Nb,"r",linewidth = 3,label='ROC curve (area = %0.2f)' % aucval)
    plt.grid()
    plt.xlabel("False Positive Rate'")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(figure_name+" Curve")
    plt.text(0.15,0.9,"AUC = "+str(round(aucval,4)))
    plt.legend(loc="lower right")
    # plt.savefig("./ROC曲线")
    plt.show()


def write_tiff(arr:Literal, raster_file:Literal,options:Literal):
    """
    Description:将数组转成栅格文件写入硬盘
    :param arr: 输入的mask数组 ReadAsArray()
    :param raster_file: 输出的栅格文件路径
    :param prj: gdal读取的投影信息 GetProjection()，默认为空
    :param trans: gdal读取的几何信息 GetGeoTransform()，默认为空
    :return:
    """
    print(arr.shape)
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(raster_file, options['width'], options['height'], 1, gdal.GDT_Float32)
 
    if options['crs']:
        dst_ds.SetProjection(options['crs'])
    if options['transform']:
        dst_ds.SetGeoTransform(options['transform'])
    print(arr)
    # 将数组的各通道写入图片
    dst_ds.GetRasterBand(1).WriteArray(arr)
    dst_ds.FlushCache()
    dst_ds = None
    print("successfully convert array to raster")
 
 
def plot_feature_importances(feature_importances,title,feature_names,change):
    print('feature_importances',feature_importances)
    print('names',feature_names)
#     feature_importances = 100.0*(feature_importances/max(feature_importances))

#     print('feature_importances',feature_importances)
#     将得分从小到大排序
    index_sorted = np.argsort(feature_importances)
    print('index_sorted',index_sorted)
   #特征名称排序
    chara= change
    i=len(change)
    for col in range(0,i):
        chara[col] = feature_names[index_sorted[col]]
    print(chara)
    
    
#     让y坐标轴上的标签居中显示
    pos = np.arange(index_sorted.shape[0])+0.5
    print(pos)
    plt.figure(figsize=(16,16))
    #0.9的分割数据
    index1 = [i] * i
    index2 = [i] * i
    feature_importances = np.append(feature_importances,0)
   
    print('feature_importances',feature_importances)
    sum = 0
    
    for col in range(0,i):
        k = feature_importances[index_sorted[i-col-1]]
        sum =sum+ k
        index1[col] = index_sorted[i-col-1]
        if (sum >= 0.9):
            break
        
    s =0       
    for col in range(0,i):
        k = feature_importances[index_sorted[col]]
        print(k)
        s =s+ k
        index2[col] = index_sorted[col]
        if (s >= 0.1):
            break
            
    print('小于0.1',index2)
    index1 = np.flipud(index1)
    print('大于0.9',index1)
    
   
    plt.barh(pos,feature_importances[index2],align='center')
    plt.barh(pos,feature_importances[index1],align='center',color="red")   
    plt.yticks(pos,chara)
    plt.xlabel('Relative Importance')
    plt.title(title)
    
    xlabel = feature_importances[index_sorted]
    ylabel = pos
    for x1, y1 in zip(xlabel,ylabel):
        # 添加文本时添加偏移量使其显示更加美观
        x1 = np.around(x1, decimals=3)
#         print("坐标",y1,x1)
        plt.text(x1+0.00005, y1, '%.3f' % x1)
    plt.show() 
