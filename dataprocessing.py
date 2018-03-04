# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:23:03 2018

@author: Frank
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd

os.chdir("H:\\BaiduYunDownload\\北京空气质量")

filedir = os.listdir()
filedir = filedir[::2]
filedir = filedir[1:5]

file_2014 = os.listdir(filedir[0])
file_2015 = os.listdir(filedir[1])
file_2016 = os.listdir(filedir[2])
file_2017 = os.listdir(filedir[3])

file_2014 = file_2014[0:358]
file_2015 = file_2015[0:364]
file_2016 = file_2016[0:364]
file_2017 = file_2017[0:365]

urban = ['东四', '天坛', '官园', '万寿西宫', '奥体中心', '农展馆', '万柳', '北部新区',
         '植物园', '丰台花园','云岗', '古城']
suburbs = [ '房山', '大兴', '亦庄', '通州', '顺义', '昌平', '门头沟', '平谷', '怀柔',
           '密云','延庆',]
contrast = ['定陵', '八达岭', '密云水库', '东高村', '永乐店', '榆垡', '琉璃河']
transport = ['前门', '永定门内','西直门北', '南三环', '东四环']


def getaqi(year,n,type = 'AQI'):
    aqi_table = pd.Series()
    for file in year:
        join_item = '\\'
        myfile = join_item.join([filedir[n],file])
        with open(myfile,encoding = 'utf-8') as air_condition:
            print(air_condition.name)
            temp = pd.read_csv(air_condition,index_col = ['date','hour','type']
            ,parse_dates = True) 
            
            temp = temp.query('type == "%s"'%type)
            #mean函数对于缺失值是如何处理的？
            aqi_table = pd.concat([aqi_table,temp.mean(axis = 0)],axis = 1)   
    aqi_table = aqi_table.loc[:,1:]
    columns = [datetime.strptime(item[-12:-4],'%Y%m%d') for item in year]
    aqi_table.columns = columns
    aqi_table = aqi_table.T
    return aqi_table

def concat2017(*item):
    df = pd.concat(item,axis = 0)
    return df

def fill_aqi(data,year):
    part_date = data.index
    full_date = pd.date_range('201%d0101'%year,'201%d1231'%year)
    for item in full_date:
        if item not in part_date:
            data.loc[item,:] = [np.nan]*35
    data = data.sort_index()
    month = [item.month for item in data.index]
    month_mean = data.groupby(month).mean
    
    data.fillna()
    return data

def aqi_tocsv():
    aqi_2014.to_csv('aqi_2014.csv')
    aqi_2015.to_csv('aqi_2015.csv')
    aqi_2016.to_csv('aqi_2016.csv')
    aqi_2017.to_csv('aqi_2017.csv')
    
def get_data(file):    
    os.chdir('F:\\研究生\\应用数理统计\\数统期末大作业')
    myfile = open(file,encoding = 'utf-8')
    aqi_2014_1 = pd.read_csv(myfile,index_col ='日期',parse_dates= True)
    month = [item.month for item in aqi_2014_1.index]
    month_2014 = aqi_2014_1.groupby(month).mean()
    month_2014 = month_2014.mean(axis = 1)
    return month_2014

def get_hour_data(year,n,type):
    aqi_table = pd.DataFrame()
    for file in year:
        join_item = '\\'
        myfile = join_item.join([filedir[n],file])
        with open(myfile,encoding = 'utf-8') as air_condition:
            print(air_condition.name)
            temp = pd.read_csv(air_condition,index_col = ['date','hour','type']
            ,parse_dates=True)   
            temp = temp.query('type == "%s"'%type)
            #mean函数对于缺失值是如何处理的？
            aqi_table = pd.concat([aqi_table,temp],axis = 0)   

    return aqi_table

def get_aqi_splitby_distict(aqi_2014,aqi_2015,aqi_2016,aqi_2017):
    #按照地区划分之后，建立四个时间序列模型
    for i in range(4,8):
        exec('month = aqi_201%d.index.get_level_values(0).month'%i)
        
        exec('data_201%d  = aqi_201%d.groupby([month,]).mean()'%(i,i))
      
        exec('data_201%d["urban"] = data_201%d.apply(lambda x:np.mean(x[urban]),axis = 1)'%(i,i))
        exec('data_201%d["suburbs"] = data_201%d.apply(lambda x:np.mean(x[suburbs]),axis = 1)'%(i,i))
        exec('data_201%d["contrast"] = data_201%d.apply(lambda x:np.mean(x[contrast]),axis = 1)'%(i,i))
        exec('data_201%d["transport"] = data_201%d.apply(lambda x:np.mean(x[transport]),axis = 1)'%(i,i))
        exec('data_201%i = data_201%d[["urban","suburbs","contrast","transport"]]'%(i,i))
    
        data = pd.concat([data_2014,data_2015,data_2016,data_2017],axis = 0)
        data.index = pd.period_range('201401','201712',freq = 'M')
        data.to_csv('monthly_aqi_splitby_district.csv')
        return data
    
if __name__ == '__main__':
        
    aqi_2014 = getaqi(file_2014,0)
    aqi_2015 = getaqi(file_2015,1)
    aqi_2016 = getaqi(file_2016,2)
    #2017年5月19日到30日的数据有缺失index=[138-149]    
    #2017年7月2日到9日的数据有缺失index = [182,189]
    file_2017_before = file_2017[0:138]
    file_2017_mid = file_2017[150:182]
    file_2017_after = file_2017[190:]
    aqi_2017_before = getaqi(file_2017_before,3)
    aqi_2017_mid = getaqi(file_2017_mid,3)
    aqi_2017_after = getaqi(file_2017_after,3)
    item = [aqi_2017_before,aqi_2017_mid,aqi_2017_after]
    aqi_2017 = concat2017(*item)

    aqi_2014 = fill_aqi(aqi_2014,4)
    aqi_2015 = fill_aqi(aqi_2015,5)
    aqi_2016 = fill_aqi(aqi_2016,6)
    aqi_2017 = fill_aqi(aqi_2017,7)
    
    month = [item.month for item in aqi_2014.index]
    month_mean = aqi_2014.groupby(month).mean()
    
    month_2014 = get_data('aqi_2014.csv')
    month_2015 = get_data('aqi_2015.csv')
    month_2016 = get_data('aqi_2016_1.csv')
    month_2017 = get_data('aqi_2017_1.csv')
    
    month_aiq = pd.concat([month_2014,month_2015,month_2016,month_2017],axis = 0)
    month_aiq.index = pd.period_range('201401','201712',freq = 'M')
    
    
    data = get_aqi_splitby_distict(aqi_2014,aqi_2015,aqi_2016,aqi_2017)
#---------------------------------------------------------------

    
    file_2017_before = file_2017[0:138]
    file_2017_mid = file_2017[150:182]
    file_2017_after = file_2017[190:]
    
    aqi_2017_before = get_hour_data(file_2017_before,3,'aqi')
    aqi_2017_mid = get_hour_data(file_2017_mid,3,'aqi')
    aqi_2017_after = get_hour_data(file_2017_after,'aqi')
    
    item = [aqi_2017_before,aqi_2017_mid,aqi_2017_after]
    aqi_2017 = concat2017(*item)
    
    month = aqi_2017.index.get_level_values(0).month
    hour = aqi_2017.index.get_level_values(1)
    data  = aqi_2017.groupby([month,hour]).mean()
  
    
    data['urban'] = data.apply(lambda x:np.mean(x[urban]),axis = 1)
    data['suburbs'] = data.apply(lambda x:np.mean(x[suburbs]),axis = 1)
    data['contrast'] = data.apply(lambda x:np.mean(x[contrast]),axis = 1)
    data['transport'] = data.apply(lambda x:np.mean(x[transport]),axis = 1)
    
    data = data[['urban','suburbs','contrast','transport']]
    
    month_1_aqi = data.loc[1,:]
    month_2_aqi = data.loc[2,:]
    month_3_aqi = data.loc[3,:]
    month_4_aqi = data.loc[4,:]
    month_5_aqi = data.loc[5,:]
    month_6_aqi = data.loc[6,:]
    month_7_aqi = data.loc[7,:]
    month_8_aqi = data.loc[8,:]
    month_9_aqi= data.loc[9,:]
    month_10_aqi = data.loc[10,:]
    month_11_aqi = data.loc[11,:]
    month_12_aqi = data.loc[12,:]
    
    for i in range(12):
        exec('month_%d_aqi.to_csv("month_%d_aqi.csv")'%(i+1,i+1))
    
    ###------------------------------------------------
    aqi_2017_before = get_hour_data(file_2017_before,3,'PM2.5')
    aqi_2017_mid = get_hour_data(file_2017_mid,3,'PM2.5')
    aqi_2017_after = get_hour_data(file_2017_after,'PM2.5')
    
    item = [aqi_2017_before,aqi_2017_mid,aqi_2017_after]
    aqi_2017 = concat2017(*item)
    
    month = aqi_2017.index.get_level_values(0).month
    hour = aqi_2017.index.get_level_values(1)
    data  = aqi_2017.groupby([month,hour]).mean()
    
    data['urban'] = data.apply(lambda x:np.mean(x[urban]),axis = 1)
    data['suburbs'] = data.apply(lambda x:np.mean(x[suburbs]),axis = 1)
    data['contrast'] = data.apply(lambda x:np.mean(x[contrast]),axis = 1)
    data['transport'] = data.apply(lambda x:np.mean(x[transport]),axis = 1)
    
    data = data[['urban','suburbs','contrast','transport']]
    
    month_1_PM25 = data.loc[1,:]
    month_2_PM25 = data.loc[2,:]
    month_3_PM25 = data.loc[3,:]
    month_4_PM25 = data.loc[4,:]
    month_5_PM25 = data.loc[5,:]
    month_6_PM25 = data.loc[6,:]
    month_7_PM25 = data.loc[7,:]
    month_8_PM25 = data.loc[8,:]
    month_9_PM25= data.loc[9,:]
    month_10_PM25 = data.loc[10,:]
    month_11_PM25 = data.loc[11,:]
    month_12_PM25 = data.loc[12,:]
  
    for i in range(12):
        exec('month_%d_PM25.to_csv("month_%d_pm25.csv")'%(i+1,i+1))
    