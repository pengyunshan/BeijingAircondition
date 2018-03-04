# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:19:04 2018

@author: Frank
"""

import pandas as pd
import numpy as np

# TSA from Statsmodels
import statsmodels.api as sm

from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller

from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.diagnostic import HetGoldfeldQuandt

from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import matplotlib.pylab as plt

import warnings
warnings.filterwarnings("ignore")
import os 
os.chdir('F:\\研究生\\应用数理统计\\数统期末大作业')

def display(ts):
    """display and plotting"""
    ts.plot(figsize=(12,8))
    #plt.legend(bbox_to_anchor=(1.25, 0.5))
    plt.title("Monthly AQI in Beijing ")
    #plt.savefig('beijing.jpg')  
    plt.show()

def test_stationarity(ts):
    """滚动平均#差分#标准差"""
    rolmean = pd.rolling_mean(ts,window=12)
    ts_diff = ts - ts.shift()
    rolstd = pd.rolling_std(ts,window=12)
    ts.plot(color = 'blue',label='Original',figsize=(12,8))
    rolmean.plot(color = 'red',label='Rolling 12 Mean')
    rolstd. plot(color = 'black',label='Rolling 12 Std')
    ts_diff.plot(color = 'green',label = 'Diff 1')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation and Diff 1')
    plt.axhline(y=0,  linewidth=1, color='yellow')
    plt.savefig('stationarity.jpg')
    plt.show(block=False)
    #adf--AIC检验
    print ('Result of Augment Dickry-Fuller test--AIC')
    dftest=adfuller(ts,autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','Lags Used',
    'Number of observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical value(%s)'%key]=value
    print (dfoutput)
    #adf--BIC检验
    print('-------------------------------------------')
    print ('Result of Augment Dickry-Fuller test--BIC')
    dftest=adfuller(ts,autolag='BIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','Lags Used',
    'Number of observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical value(%s)'%key]=value
    print (dfoutput)

def season_compose(ts):
    """季节分解"""
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(ts)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    plt.subplot(411)
    plt.plot(ts, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('stationary_detect.jpg',dpi=300)
    plt.show()
    return trend,seasonal,residual

def seasonal_detect(ts,trend,seasonal,residual):  
    """直接对残差进行分析，我们检查残差的稳定性"""
    ts_decompose = residual
    ts_decompose.dropna(inplace=True)
    test_stationarity(ts_decompose)
    print('---------------------------------------------')
    fig = plt.figure
    fig = qqplot(residual, line='q', fit=True)
    fig.title('qqplot of residual')
    plt.show()
    
    fig = plt.figure(figsize=(12,8))
    #ts
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts, lags=40,ax=ax1)
    ax1.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts, lags=40, ax=ax2)
    ax2.xaxis.set_ticks_position('bottom')
    plt.savefig('ts_aacf_pacf.jpg',dpi=300)
    plt.show()
    fig.tight_layout();
    print('-----------------------------------------------')
    #trend
    fig = plt.figure(figsize=(12,8))
    
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(trend, lags=40,ax=ax1)
    ax1.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(trend, lags=40, ax=ax2)
    ax2.xaxis.set_ticks_position('bottom')
    plt.savefig('trend_acf_pacf.jpg',dpi=300)
    plt.show()
    fig.tight_layout()
    print('-----------------------------------------------')
    #seasonal
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(seasonal, lags=40,ax=ax1)
    ax1.xaxis.set_ticks_position('bottom')
    fig.tight_layout()    
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(seasonal, lags=40, ax=ax2)
    ax2.xaxis.set_ticks_position('bottom')
    plt.savefig('season_acf_pacf.jpg',dpi=300)
    plt.show()
    fig.tight_layout()

def season_model(ts,s_order = (0,0,0,12)):
    """季节模型"""
    model = SARIMAX(
            endog=ts,order=(1,0,1),seasonal_order=s_order,trend='ct',
            enforce_invertibility=False)
    results=model.fit()
    
    predict_result = results.predict(0,50)     
    print(predict_result)
    print(results.summary())
    return predict_result

def model_selection(ts):
    """模型定阶"""
    import itertools
    p = q = range(0, 3)
    pdq = list(itertools.product(p,q))
    for param in pdq:
        try:
            model = ARMA(ts,order=param)
            result_ARIMA=model.fit(disp=-1)
            print('ARMA{}-- AIC:{} -- BIC:{} --HQIC:{}'.format(
                    param,result_ARIMA.aic,result_ARIMA.bic,result_ARIMA.hqic))
        except:
            continue
    
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts, lags=40, ax=ax2)
    plt.savefig('ts_acf_pacf.jpg',dpi=300)
    plt.show()
    
    #大致尝试0-1，0-2，1-1模型   
    #AR(0,1) model
    model=ARMA(ts,order=(0,1))
    result_AR=model.fit(disp=-1)
    plt.plot(ts)
    plt.plot(result_AR.fittedvalues,color='blue')
    plt.title('RSS:%.4f'%sum(result_AR.fittedvalues-ts)**2)
    plt.savefig('ARMA01.jpg',dpi=300)
    plt.show()
    
    #MA model
    model=ARMA(ts,order=(0,2))
    result_MA=model.fit(disp=-1)
    plt.plot(ts)
    plt.plot(result_MA.fittedvalues,color='blue')
    plt.title('RSS:%.4f'%sum(result_MA.fittedvalues-ts)**2)
    plt.savefig('ARMA02.jpg',dpi=300)
    plt.show()
    
    #ARMA
    model=ARMA(ts,order=(1,1))
    result_ARMA = model.fit(disp=-1)
    plt.plot(ts)
    plt.plot(result_ARMA.fittedvalues,color='blue')
    plt.title('RSS:%.4f'%sum(result_ARMA.fittedvalues-ts)**2)
    plt.savefig('ARMA11.jpg',dpi=300)
    plt.show()
    return result_ARMA    

def model_detect(result):
    """模型检验"""
    import statsmodels.api as sm
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(result.resid.values.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(result.resid, lags=40, ax=ax2)
    plt.show()
    
    print(sm.stats.durbin_watson(result.resid.values))
    #检验结果是1.93206697832，说明不存在自相关性。
     
    resid = result.resid#残差
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    fig = qqplot(resid, line='q', ax=ax, fit=True)
    plt.show()

def model_predict(month_aqi,result_ARIMA):
    """模型预测"""
    #model=ARIMA(ts_log,order=(2,1,2))
    model_results=ARMA(month_aqi,order=(1,1))
    result_ARIMA=model_results.fit(disp=-1)
    result_ARIMA.plot_predict()
    result_ARIMA.forecast()
    plt.show()    
    predict_result = result_ARIMA.predict(48,50)      
    predictions_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues, copy=True)
    print (predictions_ARIMA_diff.head())
    return predict_result

if __name__ == '__main__':
    
    ts = pd.read_csv('monthly_aqi_splitby_district.csv',index_col = 0)        
    ts.reset_index(inplace=True)
    #When we reset the index, the old index is added as a column, and a new sequential index is used
    ts['date'] = pd.period_range('201401','201712',freq = 'M')
    ts['date'] = ts['date'].astype(np.datetime64)
#    ts['date'] = pd.to_datetime(ts['date'])
    #Convert argument to datetime.
    ts = ts.set_index('date')
    
    #--------------------------------------------------------------------------
    display(ts['urban'])
    test_stationarity(ts['urban'])
    trend,seasonal,residual = season_compose(ts['urban'])
    season_model(ts['urban'])
    result_ARIMA = model_selection(ts['urban'])
    print(result_ARIMA.summary())
    model_detect(result_ARIMA)
    x = model_predict(ts['urban'],result_ARIMA)    

    #--------------------------------------------------------------------------
    display(ts['suburbs'])
    test_stationarity(ts['suburbs'])
    trend,seasonal,residual = season_compose(ts['suburbs'])
    seasonal_detect(ts['suburbs'],trend,seasonal,residual)  
    season_model(ts['suburbs'])
    result_ARIMA = model_selection(ts['suburbs'])
    print(result_ARIMA.summary())
    model_detect(result_ARIMA)
    #--------------------------------------------------------------------------
    display(ts['contrast'])
    test_stationarity(ts['contrast'])
    trend,seasonal,residual = season_compose(ts['contrast'])
    seasonal_detect(ts['contrast'],trend,seasonal,residual)  
    season_model(ts['contrast'])
    result_ARIMA = model_selection(ts['contrast'])
    print(result_ARIMA.summary())
    model_detect(result_ARIMA)
    #--------------------------------------------------------------------------
    display(ts['transport'])
    test_stationarity(ts['transport'])
    trend,seasonal,residual = season_compose(ts['transport'])
    seasonal_detect(ts['transport'],trend,seasonal,residual)  
    season_model(ts['transport'])
    result_ARIMA = model_selection(ts['transport'])
    print(result_ARIMA.summary())
    model_detect(result_ARIMA)
    x = model_predict(ts['urban'],result_ARIMA)    
