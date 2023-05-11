from typing import Literal,Any
from osgeo import gdal
from osgeo import gdalconst
from ast import Str
from osgeo import ogr
import numpy as np
from pathlib import Path
from osgeo.gdal import Dataset
def align(files:Literal,inputdir:Str,outputdir:Str,options:Literal,methods=gdalconst.GRA_Average):
    '''
    Description:对不同上个图像进行栅格对齐
    files:输入文件
    inputdir:输入文件夹
    outputdir:输出文件夹
    options:参考方法
    methods:重采样方法
            gdalconst.GRA_NearestNeighbour：near
            gdalconst.GRA_Bilinear:bilinear
            gdalconst.GRA_Cubic:cubic
            gdalconst.GRA_CubicSpline:cubicspline
            gdalconst.GRA_Lanczos:lanczos
            gdalconst.GRA_Average:average
            gdalconst.GRA_Mode:mode
    '''
    for i in files:
        # 打开tif文件
        in_ds  = gdal.Open(inputdir+i, gdalconst.GA_ReadOnly) # 输入文件
         # 参考文件与输入文件的的地理仿射变换参数与投影信息
        in_proj = in_ds.GetProjection()
        ref_trans = options['transform']
        ref_proj =options['crs']
        
        # 输入文件的行列数
        x = options['width']
        y = options['height']
        
        # 创建输出文件
        driver= gdal.GetDriverByName('GTiff')
        output = driver.Create(outputdir+i, x, y, 1, gdalconst.GDT_Int16)
        # 设置输出文件地理仿射变换参数与投影
        output.SetGeoTransform(ref_trans)
        output.SetProjection(ref_proj)
        
        # 重投影，插值方法为双线性内插法
        gdal.ReprojectImage(in_ds, output, in_proj, ref_proj, methods)
        
    # 关闭数据集与driver
    in_ds = None
    driver  = None
    output = None


def buffer(inShp,inname, fname, bdistance=100):
    """
    Discription:对要素图层进行缓冲操作
    :param inShp: 输入的矢量路径
    :param fname: 输出的矢量路径
    :param bdistance: 缓冲区距离
    :return:
    """
    ogr.UseExceptions()
    in_ds = ogr.Open(inShp+inname+'.shp')
    in_lyr = in_ds.GetLayer()
    # 创建输出Buffer文件
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if Path(fname).exists():
        driver.DeleteDataSource(fname)
    # 新建DataSource，Layer
    out_ds = driver.CreateDataSource(fname)
    out_lyr = out_ds.CreateLayer(inname+'buffer', in_lyr.GetSpatialRef(), ogr.wkbPolygon)
    def_feature = out_lyr.GetLayerDefn()
    # 遍历原始的Shapefile文件给每个Geometry做Buffer操作
    for feature in in_lyr:
        geometry = feature.GetGeometryRef()
        buffer = geometry.Buffer(bdistance)
        out_feature = ogr.Feature(def_feature)
        out_feature.SetGeometry(buffer)
        out_lyr.CreateFeature(out_feature)
        out_feature = None
    out_ds.FlushCache()
    del in_ds, out_ds


def disbuffer(buffer:np.ndarray):
    '''
    Description:缓冲区取反获取非滑坡点集
    buffer:缓冲区的栅格矩阵
    return:非缓冲区的栅格矩阵
    '''
    disbuffer=np.where(buffer!=0,np.NAN,buffer)
    print((np.isnan(disbuffer)).sum())
    print(np.count_nonzero(disbuffer==0))
    return disbuffer


def rasterize(inputfile:Literal, outputfile:Literal, options:dict[Str:Any]):
    '''
    Description:栅格化矢量图层
    inputfile:输入文件
    outputfile:输出文件
    options:投影参数
    '''
    inputfilePath = inputfile
    outputfile = outputfile
    
    width=options.get("width")
    height=options.get("height")
    crs=options.get("crs")
    transform=options.get("transform")

    #读取矢量文件
    vector = ogr.Open(inputfilePath)
    layer = vector.GetLayer()
    
    #创建新栅格文件并设置参数
    targetDataset = gdal.GetDriverByName('GTiff').Create(outputfile,width, height, 1,gdal.GDT_Byte)
    targetDataset.SetGeoTransform(transform)
    targetDataset.SetProjection(crs)

    #设置空值
    band = targetDataset.GetRasterBand(1)
    NoData_value = 0
    band.SetNoDataValue(NoData_value)
    band.FlushCache()

    #栅格化 
    gdal.RasterizeLayer(targetDataset, [1], layer)
    targetDataset=None

def getRasterizeOptions(templatefile:Literal)->dict[str,Any]:
    """获取栅格化参数
    Parameters
    ----------
    templatefile:Literal
        作为模板的参数文件
    Returns
    -------
    返回参数字典，矢量图形像栅格转换
    """
    data :Dataset= gdal.Open(templatefile, gdal.GA_ReadOnly)
    options={
        'crs':data.GetProjection() ,
        'transform': data.GetGeoTransform(),
        "width":data.RasterXSize,
        "height":data.RasterYSize
    }
    del data
    return options