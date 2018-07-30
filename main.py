# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:53:05 2018

@author: fabdellah
"""

# Import libraries

import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
from datetime import timedelta, date, datetime
from scipy.optimize import differential_evolution
from dateutil.relativedelta import relativedelta
from scipy import optimize
from numpy import linalg as LA
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import RidgeCV
import warnings
warnings.filterwarnings("ignore")
from time import time
import math
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from scipy.optimize import minimize
from numpy import random



# Functions

def load_data(start_date_train, end_date_train, start_date_test, end_date_test ):
    """Load the data."""    
    file = 'External_Data.xls'
    df = pd.read_excel(file)
    df_subset = df.dropna(how='any')
    df_subset = df_subset[3:df_subset.shape[0]]
    russia_Index = df_subset[['Data Type','USD.20']].reset_index()[2:df_subset.shape[0]]
    russia_Index.drop('index', axis=1, inplace=True)
    russia_Index = russia_Index.reset_index()
    russia_Index.drop('index', axis=1, inplace=True)
    russia_Index.columns = ['date', 'index']
    start_date_data = '1997-04-01 00:00:00'    
    base =  datetime.strptime(start_date_data,"%Y-%m-%d %H:%M:%S")
    nbr_months_per_data = 243
    date_list = [base + relativedelta(months=x) for x in range(0, nbr_months_per_data)]
    russia_Index['date'] = date_list
    
    mask_train = (russia_Index['date'] >= start_date_train) & (russia_Index['date'] <= end_date_train)
    russia_Index_train = russia_Index.loc[mask_train].reset_index()  
    russia_Index_train.drop('level_0', axis=1, inplace=True) 
    y_train = russia_Index_train['index']
    
    start_date_index_test = datetime.strptime(start_date_test,"%Y-%m-%d %H:%M:%S") 
    end_date_index_test = datetime.strptime(end_date_test,"%Y-%m-%d %H:%M:%S") 
    mask_test = (russia_Index['date'] >= start_date_index_test) & (russia_Index['date'] <= end_date_index_test)
    russia_Index_test = russia_Index.loc[mask_test].reset_index()  
    russia_Index_test.drop('level_0', axis=1, inplace=True) 
    y_test = russia_Index_test['index']
           
    df = pd.read_excel('spot_prices.xls')
    df_oil = df[['date_oil', 'oil']]
    df_oil.columns = ['date', 'oil']      
    df_power = df[['date_power', 'power']]
    df_power.columns = ['date', 'power']     
    df_coal = df[['date_coal', 'coal']]
    df_coal.columns = ['date', 'coal']      
    df_gas = df[['date_gas', 'gas']]
    df_gas.columns = ['date', 'gas']
      
    start_date_gas_market_test = datetime.strptime('2015-12-30 00:00:00'  ,"%Y-%m-%d %H:%M:%S") 
    end_date_gas_market_test = datetime.strptime('2017-01-31 00:00:00',"%Y-%m-%d %H:%M:%S") 
    df_gas['date'] = pd.to_datetime(df_gas['date'])  
    mask_gas = ( df_gas['date'] >= start_date_gas_market_test) & ( df_gas['date'] <= end_date_gas_market_test)
    df_gas_market =  df_gas.loc[mask_gas].reset_index()  
    df_gas_market.drop('index', axis=1, inplace=True) 
    df_gas_market = df_gas_market.set_index('date')
    df_gas_market_monthly = df_gas_market.resample("M", how='mean').reset_index()  
    y_gas_market_2016 = df_gas_market_monthly['gas']
         
    file = 'eur_usd.xlsx'
    currency_conversion = pd.read_excel(file)
    eur_usd_vector = currency_conversion['eur/usd']
        
    return russia_Index, y_train, russia_Index_train, y_test, russia_Index_test, df_oil, df_power, df_coal, df_gas, y_gas_market_2016, eur_usd_vector



def standardize(x):
    """Standardize the original data set."""    
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def calculate_mse(e):
    """Calculate the mse for vector e."""   
    return 1/2*np.mean(e**2)


def compute_mse_func(y, x1,x2,x3,x4, coef1,coef2,coef3,coef4):
    """Compute the loss by mse."""
    e = y - (x1*coef1 + x2*coef2 + x3*coef3 + x4*coef4)
    mse = e.dot(e) / (2 * len(y))
    return mse


def compute_mse(y, x, coef):
    """Compute the loss by mse."""
    e = y - np.dot(x,coef)
    mse = e.dot(e) / (2 * len(y))
    return mse


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - np.dot(tx,w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def predict(tx,coef):
    """Compute the product X * coef."""
    return np.dot(tx,coef)
    
   
def polynomial(X):    
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    return poly.fit_transform(X)  


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for linear regression."""
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
    return loss, w


def score(X_train,y_train, X_test, y_test,coef):
    """Compute R2 for the training data and the testing data."""
    y_pred_train = np.dot(X_train,coef)     
    r2 = r2_score(y_test, np.dot(X_test,coef))  
    r2_train = r2_score(y_train, y_pred_train)
    return r2,r2_train



def ridge_regression(X_train,y_train, X_test, y_test):    
    """Ridge regression algorithm."""
    # select the best alpha with RidgeCV (cross-validation)
    # alpha=0 is equivalent to linear regression
    alpha_range = 10.**np.arange(-2, 3)
    ridgeregcv = RidgeCV(alphas=alpha_range, fit_intercept=True, normalize=False, scoring='mean_squared_error') 
    ridgeregcv.fit(X_train, y_train)
    #print('best alpha=',ridgeregcv.alpha_)
    #print('ridgeregcv.coef: ',ridgeregcv.coef_)
    # predict method uses the best alpha value
    y_pred = ridgeregcv.predict(X_test)
    err = metrics.mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)  
    r2_train = r2_score(y_train, ridgeregcv.predict(X_train))
    score = ridgeregcv.score
    return ridgeregcv.coef_ , ridgeregcv.intercept_ , err, r2, r2_train, score
 

def linear_regression(X_train,y_train, X_test, y_test):
    """Linear regression algorithm."""
    regr = linear_model.LinearRegression() 
    regr.fit(X_train, y_train)  
    y_pred = regr.predict(X_test)    
    err =  mean_squared_error(y_test,y_pred)  
    # Explained variance score: 1 is perfect prediction
    r2 = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, regr.predict(X_train))
    return regr.coef_ , err, r2, r2_train
    


def OLS_stat(X,y):
    """Summary statistics for OLS."""
    est = sm.OLS(y, X)
    est2 = est.fit()
    print(est2.summary())





def MA_func_vect_out(nbr_months_per_data, df1, start_date_str,end_date_str, lag, ma_period,reset_period,vect): #vect=np.empty(0) 
    """Compute the moving average matrix using the optimal lag, averaging period and reset period."""
    start_date = datetime.strptime(start_date_str,"%Y-%m-%d %H:%M:%S") 
    end_date = datetime.strptime(end_date_str,"%Y-%m-%d %H:%M:%S") 
    start_date_x = start_date - relativedelta(months=int(round(lag))) - relativedelta(months=int(round(ma_period) ))            
    end_date =  end_date + relativedelta(months=int(round(lag))) + relativedelta(months=int(round(ma_period) ))+ relativedelta(months=1) 
    df1['date'] = pd.to_datetime(df1['date'])  
    mask = (df1['date'] >= start_date_x) & (df1['date'] <= end_date)
    df_x = df1.loc[mask].reset_index()  
    df_x.drop('index', axis=1, inplace=True)    
    df_x.iloc[:, [1]]= df_x.iloc[:, [1]].astype(float)   
    ma_monthly = pd.rolling_mean(df_x.set_index('date').resample('1BM'),window=int(round(ma_period))).dropna(how='any').reset_index().iloc[0:nbr_months_per_data, [1]].values
    ma_vect = [ ma_monthly[i] for i in range(0,nbr_months_per_data,int(round(reset_period))) ]
    nbr_reset_periods = int(math.ceil(nbr_months_per_data/int(round(reset_period))))  
    vect = np.empty(0)
    for i in range(0,nbr_reset_periods):
        vect = np.append(vect , ma_vect[i]*np.ones(int(round(reset_period))))    
    vect = vect[0:nbr_months_per_data]    
    return vect
     

def MA_plot(df1, start_date_str,end_date_str, lag, ma_period,reset_period):    
    nbr_months_per_data = 24
    vect = MA_func_vect_out(df1, start_date_str,end_date_str, lag, ma_period,reset_period,np.empty(0))
    time = np.arange(0,nbr_months_per_data)
    plt.plot(time, vect)   
    plt.title('Moving average: Lag = '+str(lag)+ ' MA period = '+ str(ma_period)+ ' Reset period = '+ str(reset_period)  )
    plt.xlabel('Time')
    plt.ylabel('Average level')
    plt.show()


def plot_predictions(russia_Index_x, russia_Index_test, y, y_test, y_pred_test ,y_train_model):
    """Plot the actual, estimated and predicted prices."""
    plt.rcParams['figure.figsize']=(10,5)
    plt.style.use('ggplot') 
    d1 = {'date' : russia_Index_x['date'],  'y_train' : y.astype(float)}
    df_y_train = pd.DataFrame(d1) 
    d2 = {'date' : russia_Index_test['date'],  'y_test' :y_test}
    df_y_test = pd.DataFrame(d2) 
    d3 = {'date' : russia_Index_test['date'],  'y_pred' : y_pred_test}
    df_y_pred = pd.DataFrame(d3) 
    d4 = {'date' : russia_Index_x['date'],  'y_train_model' : y_train_model}
    df_y_train_model = pd.DataFrame(d4) 
    other = df_y_test.set_index('date').join(df_y_pred.set_index('date'))
    other = other.reset_index()
    y_curve1 = df_y_train.append(other)
    y_curve = df_y_train_model.append(y_curve1)
    
    fig, ax1 = plt.subplots()
    ax1.plot(y_curve.date , y_curve.y_train , color='red', label = 'Actual prices')
    ax1.plot(y_curve.date , y_curve.y_test , color='orange' , label = 'Actual prices')
    ax1.plot(y_curve.date , y_curve.y_pred , color='black', linestyle=':', label = 'Forecast prices')
    ax1.plot(y_curve.date , y_curve.y_train_model , color='blue', linestyle='--', label = 'Model')
    plt.title('Russian gas prices: Actual vs forecast')
    ax1.set_ylabel('USD')
    ax1.set_xlabel('Date')
    plt.legend()
 


class class_alternate(object):
    """Class that defines the formula specifications and computes its optimal parameters."""
    def __init__(self, df_oil, df_power, df_coal, df_gas, y, start_date_str,end_date_str, nbr_months_per_data, nbr_iterations, max_lag, max_ma_period, max_reset_period ,init_coef):
            self.max_lag = max_lag
            self.max_ma_period = max_ma_period
            self.max_reset_period = max_reset_period
            
            self.nbr_iterations = nbr_iterations
 
            self.nbr_months_per_data = nbr_months_per_data
            self.bounds = [(0, self.max_lag),(0, self.max_lag),(0, self.max_lag),(0, self.max_lag), (1, self.max_ma_period), (1, self.max_ma_period),(1, self.max_ma_period),(1, self.max_ma_period),(1, self.max_reset_period),(1, self.max_reset_period),(1, self.max_reset_period),(1, self.max_reset_period)]
              
            self.df_oil = df_oil
            self.df_power = df_power
            self.df_coal = df_coal
            self.df_gas = df_gas
            self.y = y
            
            self.start_date_str = start_date_str                                             #start_date_str = '2016-01-31 00:00:00'
            self.start_date = datetime.strptime(self.start_date_str,"%Y-%m-%d %H:%M:%S") 
            self.end_date_str = end_date_str                                                 #end_date_str   = '2017-01-31 00:00:00'
            self.end_date = datetime.strptime(self.end_date_str,"%Y-%m-%d %H:%M:%S") 
            
            self.init_coef = init_coef 
            
            
    def MA_func_vect(self, lag_oil, lag_power, lag_coal, lag_gas, ma_period_oil, ma_period_power, ma_period_coal, ma_period_gas, reset_period_oil, reset_period_power, reset_period_coal, reset_period_gas ,vect): #vect=np.empty(0) # pour reset_period=1 len(ma_vect)=11 au lieu de 12 je c pas pk
        """Returns the input matrix computed with the optimal lag, ma_period and reset_period. This matrix is used for the regression in process 2."""
        start_date_oil = self.start_date - relativedelta(months=int(round(lag_oil))) - relativedelta(months=int(round(ma_period_oil) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_oil))) + relativedelta(months=int(round(ma_period_oil) ))+ relativedelta(months=1) 
        self.df_oil['date'] = pd.to_datetime(self.df_oil['date'])  
        mask = (self.df_oil['date'] >= start_date_oil) & (self.df_oil['date'] <= end_date)
        df_xoil = self.df_oil.loc[mask].reset_index()  
        df_xoil.drop('index', axis=1, inplace=True)    
        df_xoil.iloc[:, [1]] = df_xoil.iloc[:, [1]].astype(float)          
                
        start_date_power = self.start_date - relativedelta(months=int(round(lag_power))) - relativedelta(months=int(round(ma_period_power) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_power))) + relativedelta(months=int(round(ma_period_power) ))+ relativedelta(months=1) 
        self.df_power['date'] = pd.to_datetime(self.df_power['date'])  
        mask = (self.df_power['date'] >= start_date_power) & (self.df_power['date'] <= end_date)
        df_xpower = self.df_power.loc[mask].reset_index()  
        df_xpower.drop('index', axis=1, inplace=True)    
        df_xpower.iloc[:, [1]] = df_xpower.iloc[:, [1]].astype(float)            
    
        start_date_coal = self.start_date - relativedelta(months=int(round(lag_coal))) - relativedelta(months=int(round(ma_period_coal) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_coal))) + relativedelta(months=int(round(ma_period_coal) ))+ relativedelta(months=1) 
        self.df_coal['date'] = pd.to_datetime(self.df_coal['date'])  
        mask = (self.df_coal['date'] >= start_date_coal) & (self.df_coal['date'] <= end_date)
        df_xcoal = self.df_coal.loc[mask].reset_index()  
        df_xcoal.drop('index', axis=1, inplace=True)    
        df_xcoal.iloc[:, [1]] = df_xcoal.iloc[:, [1]].astype(float)          
      
        start_date_gas = self.start_date - relativedelta(months=int(round(lag_gas))) - relativedelta(months=int(round(ma_period_gas) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_gas))) + relativedelta(months=int(round(ma_period_gas) ))+ relativedelta(months=1) 
        self.df_gas['date'] = pd.to_datetime(self.df_gas['date'])  
        mask = (self.df_gas['date'] >= start_date_gas) & (self.df_gas['date'] <= end_date)
        df_xgas = self.df_gas.loc[mask].reset_index()  
        df_xgas.drop('index', axis=1, inplace=True)    
        df_xgas.iloc[:, [1]] = df_xgas.iloc[:, [1]].astype(float)             
        
        ma_monthly_oil = pd.rolling_mean(df_xoil.set_index('date').resample('1BM'),window=int(round(ma_period_oil))).dropna(how='any').reset_index().iloc[0:self.nbr_months_per_data, [1]].values
        ma_monthly_power = pd.rolling_mean(df_xpower.set_index('date').resample('1BM'),window=int(round(ma_period_power))).dropna(how='any').reset_index().iloc[0:self.nbr_months_per_data, [1]].values
        ma_monthly_coal = pd.rolling_mean(df_xcoal.set_index('date').resample('1BM'),window=int(round(ma_period_coal))).dropna(how='any').reset_index().iloc[0:self.nbr_months_per_data, [1]].values
        ma_monthly_gas = pd.rolling_mean(df_xgas.set_index('date').resample('1BM'),window=int(round(ma_period_gas))).dropna(how='any').reset_index().iloc[0:self.nbr_months_per_data, [1]].values
          
        
        ma_vect_oil = [ ma_monthly_oil[i] for i in range(0,self.nbr_months_per_data,int(round(reset_period_oil ))) ]
        ma_vect_power = [ ma_monthly_power[i] for i in range(0,self.nbr_months_per_data,int(round(reset_period_power ))) ]
        ma_vect_coal = [ ma_monthly_coal[i] for i in range(0,self.nbr_months_per_data,int(round(reset_period_coal ))) ]
        ma_vect_gas = [ ma_monthly_gas[i] for i in range(0,self.nbr_months_per_data,int(round(reset_period_gas ))) ]
       
        nbr_reset_periods_oil = int(math.ceil(self.nbr_months_per_data/int(round(reset_period_oil))))  
        nbr_reset_periods_power = int(math.ceil(self.nbr_months_per_data/int(round(reset_period_power))))
        nbr_reset_periods_coal = int(math.ceil(self.nbr_months_per_data/int(round(reset_period_coal))))
        nbr_reset_periods_gas = int(math.ceil(self.nbr_months_per_data/int(round(reset_period_gas))))
               
        vect_oil = np.empty(0)
        for i in range(0,nbr_reset_periods_oil):
            vect_oil = np.append(vect_oil , ma_vect_oil[i]*np.ones(int(round(reset_period_oil))))      
        
        vect_power = np.empty(0)
        for i in range(0,nbr_reset_periods_power):
            vect_power = np.append(vect_power , ma_vect_power[i]*np.ones(int(round(reset_period_power))))      
       
        
        vect_coal = np.empty(0)
        for i in range(0,nbr_reset_periods_coal):
            vect_coal = np.append(vect_coal , ma_vect_coal[i]*np.ones(int(round(reset_period_coal))))      

        
        vect_gas = np.empty(0)
        for i in range(0,nbr_reset_periods_gas):
            vect_gas = np.append(vect_gas , ma_vect_gas[i]*np.ones(int(round(reset_period_gas))))      

        vect_oil = vect_oil[0:self.nbr_months_per_data]
        vect_power = vect_power[0:self.nbr_months_per_data]
        vect_coal = vect_coal[0:self.nbr_months_per_data]
        vect_gas = vect_gas[0:self.nbr_months_per_data]
        
        vect = np.c_[np.array(vect_oil) ,np.array(vect_power) , np.array(vect_coal) , np.array(vect_gas) ]
        
        return vect
    
        
    def func_lag_period(self, parameters, *data):
        """Objective function to minimize."""
        lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas = parameters
        df_oil , df_power , df_coal , df_gas , coef, values = data
        values = compute_mse(self.y.astype(float) ,np.c_[np.ones(self.nbr_months_per_data), preprocessing.scale(self.MA_func_vect(lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas, np.empty(0)))]   , coef) 
        return values


    def de_optimization(self, coef):
        """Differential evolution for the lag, ma_period and reset_period for oil, power, coal and gas."""        
        args = (self.df_oil,self.df_power,self.df_coal,self.df_gas, coef, np.empty(1))
        result =differential_evolution(self.func_lag_period,  self.bounds, args=args)
        lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas = result.x                          
         
        return lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas
    
    
    def alternate(self):
        """Alternate between process 1 and process 2."""    
        max_iters = 100
        gamma = 0.1
        coef =  self.init_coef
        gradient_w = self.init_coef      
        
        for itera in range(self.nbr_iterations):            
            print('///////////////////////////////////////')
            print('Iteration: ', itera )
            
            #process 1: optimal lag, optimal ma_period and optimal reset period for a given coefficient           
            t00 = time()
            lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas = self.de_optimization(coef)
            t11 = time()
            d1 = t11-t00
            lag = np.array([lag_oil, lag_power, lag_coal, lag_gas])
            ma_period = np.array([period_oil, period_power, period_coal, period_gas])
            reset_period = np.array([reset_oil, reset_power, reset_coal, reset_gas])
            print ("Duration of process 1 in Seconds %6.3f" %d1)      
            
            #process2: optimal coefficient for a given lag, ma_period and reset period
            t02 = time()
            X_df = self.MA_func_vect(lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas, np.empty(0))     
            XX_stand = np.c_[np.ones(X_df.shape[0]), preprocessing.scale(X_df).reshape(self.nbr_months_per_data,4)] 
            w_initial = gradient_w
            X_train, X_test, y_train, y_test = train_test_split(XX_stand, self.y.astype(float), random_state=1)
            
            gradient_loss, gradient_w = gradient_descent(y_train, X_train, w_initial, max_iters, gamma)  
            res_ridge = ridge_regression(X_train, y_train, X_test, y_test)
             
           # update coef

            coef = np.concatenate([[res_ridge[1]],res_ridge[0][1:5]])
            y_pred_GD = np.dot(X_test,gradient_w)
            print('--------- OLS ---------')
            print('Coef OLS:', gradient_w )
            print('Error OLS:', metrics.mean_squared_error(y_test, y_pred_GD))
            print('R2_train OLS', r2_score(y_train, np.dot(X_train,gradient_w))  )
            print('R2_test OLS', r2_score(y_test, y_pred_GD)  )
                                
            print('--------- Ridge regression ---------')
            
            print('Coef RR:', coef )
            print('Error RR: ', res_ridge[2])
            print('R2_train RR: ', res_ridge[4])
            print('R2_test RR: ', res_ridge[3])
            
            
            t12 = time()
            d2 = t12-t02
            print ("Duration of process 2 in Seconds %6.3f" % d2)        
                
        return coef , lag , ma_period, reset_period ,X_df, XX_stand
           


def MA_func_test(nbr_months_per_testing_data, X_df, df_oil, df_power, df_coal, df_gas, start_day, end_day, lag, ma_period, reset_period):
    """Computes the moving average matrix for the testing data."""
    mean_oil = X_df.mean(axis=0)[0]
    mean_power = X_df.mean(axis=0)[1]
    mean_coal = X_df.mean(axis=0)[2]
    mean_gas = X_df.mean(axis=0)[3]
    std_oil = X_df.std(axis=0)[0]
    std_power = X_df.std(axis=0)[1]
    std_coal = X_df.std(axis=0)[2]
    std_gas = X_df.std(axis=0)[3]
    
    oil_test = MA_func_vect_out(nbr_months_per_testing_data, df_oil, start_day , end_day , lag[0] , ma_period[0] , reset_period[0] , np.empty(0))
    power_test = MA_func_vect_out(nbr_months_per_testing_data, df_power, start_day , end_day , lag[1],ma_period[1], reset_period[1] , np.empty(0))
    coal_test = MA_func_vect_out(nbr_months_per_testing_data, df_coal, start_day , end_day , lag[2], ma_period[2], reset_period[2] , np.empty(0))
    gas_test = MA_func_vect_out(nbr_months_per_testing_data,df_gas, start_day , end_day , lag[3],ma_period[3], reset_period[3] , np.empty(0))
    X_test = np.c_[(oil_test-mean_oil)/std_oil , (power_test-mean_power)/std_power , (coal_test-mean_coal)/std_coal , (gas_test-mean_gas)/std_gas ]
    X_test_stand = np.c_[np.ones(X_test.shape[0]), X_test.reshape(nbr_months_per_testing_data,4)]  
    
    return X_test_stand



def model_calibration_schwartz(nbr_simulations, fwd, sigma_market = 0.4, theta = 1):
    """Adjust the parameters of the 1-factor Schwartz model to meet the market specifications ."""  
    fwd_curve = fwd
    fwd_curve.columns = ['time', 'price']
   
    ethetats_1y = np.exp(1 * theta)
    sigma = np.sqrt(sigma_market * sigma_market * theta * 2.0 / (1.0 - 1.0 / (ethetats_1y * ethetats_1y)))
      
    ts = fwd_curve.index.values
    ethetats = np.exp(ts * theta)
    
    prices = fwd_curve['price'].astype(float)
    vrs = sigma * sigma / theta / 2.0 * (1.0 - 1.0 / (ethetats * ethetats))
    mius = np.log(prices) - vrs / 2.0
    
    # Calibrate miu means.
    dethetats = np.diff(ethetats)
    miumeans = []
    for i in range(1, prices.shape[0]):
       miumeans.append((mius[i] * ethetats[i] - mius[0] - np.inner(dethetats[:i - 1], miumeans)) / (dethetats[i - 1]))
        
    mean = np.concatenate([[mius[0]] , miumeans] )
    variance = sigma_market*sigma_market*np.ones(len(mean))
    covariance = np.diag(variance)
    
    # simulations
    seed = 1
    np.random.seed(seed)
    
    log_S = np.random.multivariate_normal(mean, covariance, int( nbr_simulations))  
    #S =  np.concatenate(( prices[0]*np.ones(nbr_simulations).reshape((nbr_simulations,1)), np.exp(log_S).reshape( (nbr_simulations,len(mean)) ) ), axis=1 ) 
    S = np.exp(log_S)
    simulated_price_matrix =S.T
    
    return simulated_price_matrix




def unit_conversion(price_usd_mmbtu, energy_unit, currency_unit):
    """Convert the gas prices from USD/mmbtu to EUR/MWh ."""  
    price_usd_mwh = price_usd_mmbtu/energy_unit
    price_eur_mwh = np.multiply(price_usd_mwh, currency_unit)   # ptet prob ici np.dot
    return price_eur_mwh



def unit_volume_conversion(price_usd_mmbtu, energy_unit):
    """Convert the gas prices from USD/mmbtu to USD/MWh ."""  
    price_usd_mwh = price_usd_mmbtu/energy_unit
    return price_usd_mwh


def forward_curve_diff_func(russia_Index_test, y_pred_test, y_gas_market_2016_eur_mwh, usd_to_eur_vector):
    """Estimate the difference between the predicted gas prices and the market gas prices."""  
    energy_unit = 0.29329722222222
    y_pred_test_eur_mwh = unit_conversion(y_pred_test, energy_unit, usd_to_eur_vector)
   # df = pd.concat([russia_Index_test['date'].to_frame(), pd.DataFrame(np.abs(y_pred_test_eur_mwh - y_gas_market_2016_eur_mwh) )  ], axis=1)
    
    df = pd.concat([russia_Index_test['date'].to_frame(), pd.DataFrame((y_pred_test_eur_mwh - y_gas_market_2016_eur_mwh) )  ], axis=1)
    
    df.columns = ['time', 'price']
    print('y_pred_test_eur_mwh',y_pred_test_eur_mwh)
    print('y_gas_market_2016_eur_mwh', y_gas_market_2016_eur_mwh)
    print('y_russia_test_true_value ',  unit_conversion(russia_Index_test['index'] , energy_unit, usd_to_eur_vector)  )
    print('diff prediction - market', y_pred_test_eur_mwh-y_gas_market_2016_eur_mwh )
    print('diff true values - market', y_pred_test_eur_mwh-y_gas_market_2016_eur_mwh )
    return df


def forward_curve_spot_func(russia_Index_test, y_pred_test, y_gas_market_2016_eur_mwh, usd_to_eur_vector):
    """Estimate the difference between the predicted gas prices and the market gas prices."""  
#    energy_unit = 0.29329722222222
#    y_pred_test_usd_mwh =  unit_volume_conversion(y_pred_test, energy_unit)    
#    df = pd.concat([russia_Index_test['date'].to_frame(), pd.DataFrame(y_pred_test_usd_mwh)  ], axis=1)
    df = pd.concat([russia_Index_test['date'].to_frame(), pd.DataFrame(y_pred_test)  ], axis=1)
    df.columns = ['time', 'price']
#    print('y_pred_test_eur_mwh',y_pred_test_eur_mwh)
#    print('y_gas_market_2016_eur_mwh', y_gas_market_2016_eur_mwh)
#    print('y_russia_test_true_value ',  unit_conversion(russia_Index_test['index'] , energy_unit, usd_to_eur_vector)  )
#    print('diff prediction - market', y_pred_test_eur_mwh-y_gas_market_2016_eur_mwh )
#    print('diff true values - market', y_pred_test_eur_mwh-y_gas_market_2016_eur_mwh )
    return df



def volume_level_func(decision_rule, steps, nbr_simulations, volume_start, alpha):
    """Returns the optimal volume stored at each time given a volume at the starting time"""
    volume_level_stored = np.zeros((steps+1, nbr_simulations))  
    volume_level_stored[0,:] = volume_start
    for b in range(nbr_simulations):   
        for t  in range(steps): 
            m_level = ( (volume_level_stored[t,b])/alpha).astype(int) 
            volume_level_stored[t+1,b] = volume_level_stored[t,b] + decision_rule[t,b,m_level]
    volume_level_avg =    np.zeros(steps+1)     
    volume_level_avg = (np.sum(volume_level_stored,axis =1)/nbr_simulations).astype(int)
    return volume_level_stored, volume_level_avg


def decision_volume(decision_rule, steps, nbr_simulations, volume_start, alpha):
    """Returns the optimal behaviour (inject or withdraw) at each time given a volume at the final time"""
    inj = np.zeros((steps+1, nbr_simulations))  
    withd = np.zeros((steps+1, nbr_simulations))  
    volume_level_stored, volume_level_avg = volume_level_func(decision_rule, steps, nbr_simulations, volume_start, alpha) 
    
    for b in range(nbr_simulations):   
        for t in range(steps):               
            m_level = ( (volume_level_stored[t,b])/alpha).astype(int)     
            if decision_rule[t,b,m_level]>0 :
                inj[t,b] = decision_rule[t,b,m_level]
            else:
                withd[t,b] = np.abs(decision_rule[t, b, m_level])
    inj_avg = np.zeros(steps+1)     
    inj_avg = (np.sum(inj,axis =1)/nbr_simulations).astype(int)
    withd_avg = np.zeros(steps+1)     
    withd_avg = (np.sum(withd,axis =1)/nbr_simulations).astype(int)
    return withd_avg, inj_avg


    

 


class gas_storage(object):
    """ 
    simulated_price_matrix_fwd : simulated gas prices from the predicted prices 
    T : time to maturity   
    r : riskless discount rate (constant)
    sigma_GBM :  volatility of prices for simulating Geometric brownian motion prices
    nbr_simulation : number of simulated paths
    
    vMax : maximum capacity [MWh]
    vMin : minimum capacity [MWh]
    max_rate : maximum injection rate [MWh/day]
    min_rate : maximum withdrawal rate [MWh/day]
    
    injection_cost :cost of injection [EUR/MWh]
    withdrawal_cost : cost of withdrawal [EUR/MWh]
    
    steps : number of discrete times (delta_t = T/steps)
    M : number of units of the maximum capacity  
    alpha : fixed width of the discretized volume
    
    """

    def payoff(self,S, dv, injection_cost, withdrawal_cost):   
        """specify the payoff."""
        return S*dv*np.where(dv > 0, -injection_cost, -withdrawal_cost )    
       # return -S*dv
    
    def penalty(self,v,v_target):    
        """specify the penalty function."""       
        penalty = -np.abs(v-v_target)*100
        return penalty
    

    def __init__(self, simulated_price_matrix_fwd, prices_choice,  T, steps, M, r, sigma_GBM, vMax, vMin, inj_rate, with_rate, injection_cost, withdrawal_cost, v_end):
     
            self.simulated_price_matrix_fwd = simulated_price_matrix_fwd     
            self.prices_choice = prices_choice                                   # prices_choice = 'example' or 'simulations'                  
            self.nbr_simulations = self.nbr_simulations()
            self.T = T
            self.r = r
            self.sigma_GBM = sigma_GBM
        
            self.vMax = vMax                                    
            self.vMin = vMin                               
            self.inj_rate = inj_rate             
            self.with_rate = with_rate             
            
            self.injection_cost =injection_cost                 
            self.withdrawal_cost = withdrawal_cost
        
            if T <= 0 or r < 0  or sigma_GBM < 0  :
              raise ValueError('Error: Negative inputs not allowed')
 
            self.steps = steps  
            self.M = M
            self.alpha = self.vMax/(self.M-1)
            self.delta_t = self.T / float(self.steps)
            self.discount = np.exp(-self.r * self.delta_t)
              
            self.v_end = v_end
        
        
    def nbr_simulations(self):
        """ Returns the number of simuations according to price choice """
        if self.prices_choice == 'example':
            nbr_simulations = 2
        elif self.prices_choice == 'simulations':
            nbr_simulations = 50
        return nbr_simulations
    
    
    def possible_values(self):
        """ Returns the possible actions according to price choice: allow injections and withdrawal if "example", allow only injection if "simulations" """
        if self.prices_choice == 'example':
            possible_values = [-self.alpha,0,self.alpha]
        elif self.prices_choice == 'simulations':
            possible_values = [0,self.alpha]
        return possible_values
       
                   
    def simulated_price_matrix_GBM(self, seed = 1):
        """ Returns Monte Carlo simulated prices (matrix) with a Geometric Brownian Motion
            rows: time
            columns: price-path simulation """
    
        np.random.seed(seed)
        simulated_price_matrix_GBM = np.zeros((self.steps + 2, self.nbr_simulations), dtype=np.float64)
        S0 = 3
        simulated_price_matrix_GBM[0,:] = S0
        for t in range(1, self.steps + 2):
            brownian = np.random.standard_normal( int(self.nbr_simulations / 2))
            brownian = np.concatenate((brownian, -brownian))        
            simulated_price_matrix_GBM[t, :] = (simulated_price_matrix_GBM[t - 1, :]      
                                  * np.exp((self.r - self.sigma_GBM ** 2 / 2.) * self.delta_t
                                  + self.sigma_GBM * brownian * np.sqrt(self.delta_t)))
            #needs to be specified according to the corresponding 2-factor model
        return simulated_price_matrix_GBM           
    

    def example_price_matrix(self):
        """ Returns a contango curve (just for testing the model) """
        example_price_matrix = np.zeros((self.steps + 2, self.nbr_simulations), dtype=np.float64)
        for b in range(0, self.nbr_simulations):
            example_price_matrix[:,b] = np.linspace(1, 14, self.steps + 2)      #contango
#            example_price_matrix[:,b] = np.linspace(14, 1, self.steps + 2)      #backwardation
#            example_price_matrix[:,b] = np.array([7,6,5,4,3,2,1,2,3,4,5,6,7,8])

        return example_price_matrix
      
        
    def price_matrix(self):
        """ Returns the prices for the example model or for the simulated model according to your price choice """
        if self.prices_choice == 'example':
            price_matrix = self.example_price_matrix()
        elif self.prices_choice == 'simulations':
            price_matrix = self.simulated_price_matrix_fwd
        return price_matrix
        
       
    def max_inj_rates(self,vol=1):
        #return min(self.vMax - vol , self.inj_rate )
        return self.inj_rate


    def max_wit_rates(self,vol=1):
        #return max(self.vMin - vol ,self.with_rate )
        return self.with_rate
        
       
    def f(self,x,t,b,m):
        return ( self.payoff(self.price_matrix()[t, b], x ,self.injection_cost,self.withdrawal_cost ) + self.continuation_value[t,b, (m-1) + int(x/self.alpha) ]  )
     
    
    def contract_value(self):
        """Returns the value of the contract at time 0, the accumulated cashflows, the decision rule and the final price of the storage."""   
        
        levels_volume = np.linspace(0,self.alpha*100,101)
        acc_cashflows = np.zeros((self.steps+1, self.nbr_simulations, self.M))  # time, path , volume level
        acc_cashflows[-1,:, :] = self.penalty(levels_volume, self.v_end)
        
        decision_rule = np.empty((self.steps+1, self.nbr_simulations, self.M))  # time, path , volume level
        decision_rule[:] = np.nan
        
        self.continuation_value = np.zeros((self.steps, self.nbr_simulations, self.M)) 
        
        
        
        for t in range(self.steps-1 , -1  , -1):
        
            print ('-----------')
            print ('Time: %5.3f, Spot price: %5.1f ' % (t, self.price_matrix()[t, 1]))           
            for m in range(1,self.M):                      
                                   
                    regression = np.polyfit(self.price_matrix()[t, :], acc_cashflows[t+1, :, m-1] * self.discount, 2)
                    self.continuation_value[t,:,m-1] = np.polyval(regression, self.price_matrix()[t, :])
            
            for b in range(self.nbr_simulations):
                for m in range(1,self.M): 
                                      
                     possible_values = self.possible_values()
                     out = np.array([self.f(x,t,b,m) for x in possible_values])
                     
                     arg_maximum = np.argmax(out)
                     
                     dv_opt = possible_values[arg_maximum]
                                              
                     decision_rule[t,b,m-1] = dv_opt  
                                         
                     acc_cashflows[t,b,m-1] = self.payoff(self.price_matrix()[t, b],
                                     decision_rule[t,b,m-1] , self.injection_cost, self.withdrawal_cost) + acc_cashflows[t+1 , b,  (m-1) + int((decision_rule[t,b , m-1])/self.alpha)  ]*self.discount
                        
                  
        contract_value = acc_cashflows[1,:,:] * self.discount             # at time 0
        
        price = round((np.sum(contract_value,axis=0) /  float(self.nbr_simulations))[0] , 2)
        
        return contract_value, acc_cashflows, decision_rule, price



def print_optimal_actions(inj,withd):
    print("\n")
    for t in range(13):
        print( '%5s  %5s  %5s  %5s  %5s %5s' % ('Time (month): ', t, '   Inject: ', inj[t], '   Withdraw: ' , withd[t]) )


def plot_price_matrix(facility):
    time = np.arange(0,14)  
    plt.plot(time, facility.price_matrix()[:,0]  ) 
    plt.title('Forward curve')
    plt.xlabel('Time (month)')
    plt.ylabel('Price [USD/MMBT]')
    plt.show()


def plot_simulated_prices(facility):
    time = np.arange(0,14)  
    plt.plot(time, facility.simulated_price_matrix_fwd) 
    plt.ylim(ymin = 3, ymax = 8) 
    plt.title('Simulated prices')
    plt.xlabel('Time (month)')
    plt.ylabel('Price [USD/MMBT]')
    plt.show()



def plot_optimal_volume(volume_level_avg, withd, inj, ymini = 80000, ymaxi = 120000):
    """ Plot the optimal volume stored """
    time = np.arange(0,13)
    plt.rcParams['figure.figsize']=(10,5)
    plt.style.use('ggplot') 
    pal = ["#9b59b6", "#e74c3c", "#34495e", "#2ecc71"]
    plt.bar(time, withd+ymini, label = 'Withdrawal' )
    plt.bar(time, inj+ymini , label = 'injection' )
    plt.stackplot(time, volume_level_avg, colors=pal, alpha=0.4, labels = 'Volume stored')
    plt.ylim(ymin = ymini, ymax = ymaxi) 
    plt.legend(loc='upper right')
    plt.title('Volume of natural gas imported')
    plt.xlabel('Time (months)')
    plt.ylabel('Volume [MMBT]')
    plt.show()




########### Main


#### Step 1: Load the data  
start_date_train = '2014-01-01 00:00:00'    
end_date_train = '2015-12-01 00:00:00'  
start_date_test = '2015-12-01 00:00:00'   
end_date_test = '2017-01-01 00:00:00' 
 
russia_Index, y_train, russia_Index_train, y_test, russia_Index_test, df_oil, df_power, df_coal, df_gas, y_gas_market_2016, eur_usd_vector = load_data(start_date_train,end_date_train,start_date_test, end_date_test )



#### Step 2: Predict Russia's gas price  

# Training the model 
start_date_model = '2014-01-31 00:00:00'
end_date_model = '2016-01-31 00:00:00'
optimization = class_alternate(df_oil, df_power ,df_coal ,df_gas, y_train ,start_date_model , end_date_model, 24, 1, 7 , 7, 5, np.array([0,0,0,0,0]))

t0 = time()
coef , lag , ma_period, reset_period , X_df, XX_stand = optimization.alternate()    
t1 = time()
d = t1 - t0
print ("Total duration in Seconds %6.3f" % d)               
print('final coef: ', coef)

OLS_stat(XX_stand , y_train.astype(float))

# Testing the model
nbr_months_per_testing_data = 14 
start_day = '2015-12-31 00:00:00'
end_day = '2017-01-31 00:00:00'

X_test_stand = MA_func_test(nbr_months_per_testing_data, X_df, df_oil, df_power, df_coal, df_gas, start_day, end_day, lag, ma_period, reset_period)

OLS_stat(X_test_stand , y_test.astype(float))
  
# Plot & Metrics
y_train_model = np.dot(XX_stand , coef) 
y_pred_test = np.dot(X_test_stand, coef) 
plot_predictions(russia_Index_train , russia_Index_test , y_train, y_test, y_pred_test , y_train_model)

R2 = score(XX_stand, y_train.astype(float), X_test_stand, y_test.astype(float), coef )
error_test = compute_mse(y_test.astype(float), X_test_stand, coef) 
mae = mean_absolute_error(y_pred_test, y_test)     



#### Step 3: Model calibration

y_pred_test = pd.read_excel('y_pred_test.xlsx').iloc[:,0]
nbr_simulations = 50
forward_curve = forward_curve_diff_func(russia_Index_test, y_pred_test, y_gas_market_2016, eur_usd_vector)  
forward_curve_spot = forward_curve_spot_func(russia_Index_test, y_pred_test, y_gas_market_2016, eur_usd_vector)  
simulated_price_matrix = model_calibration_schwartz(nbr_simulations, forward_curve, sigma_market = 0.08, theta = 12)
simulated_price_matrix_spot = model_calibration_schwartz(nbr_simulations, forward_curve_spot, sigma_market = 0.08 , theta = 12)

pd.DataFrame(simulated_price_matrix).to_excel('simulated_price_matrix.xlsx', sheet_name='simulated_price_matrix', index=False)
pd.DataFrame(simulated_price_matrix_spot).to_excel('simulated_price_matrix_spot.xlsx', sheet_name='simulated_price_matrix_spot', index=False)



#### Step 4: Monte Carlo for gas storage

facility =  gas_storage(simulated_price_matrix_spot, 'simulations' , 1, 12, 101, 0.035, 0.08 , 250000, 0 , 2500, -2500  , 0.1, 0.1 , 100000)  
contract_value, acc_cashflows, decision_rule, price = facility.contract_value() 


volume_start = 80000

volume_level,  volume_level_avg = volume_level_func(decision_rule, facility.steps, facility.nbr_simulations, volume_start, facility.alpha)
withd, inj = decision_volume(decision_rule, facility.steps, facility.nbr_simulations, volume_start, facility.alpha)

print_optimal_actions(inj,withd)

plot_price_matrix(facility)

plot_optimal_volume(volume_level_avg, withd, inj, ymini = 80000, ymaxi = 120000)
 
plot_simulated_prices(facility)












import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

plt.rcParams['figure.figsize']=(15,6)
plt.style.use('ggplot') 
fig, ax = plt.subplots()
ax.plot(russia_Index['date'], russia_Index['index'])

# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years...
datemin = np.datetime64(russia_Index['date'][0], 'Y')
datemax = np.datetime64(russia_Index['index'][-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)


# format the coords message box
def price(x):
    return '$%1.2f' % x
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = price
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

plt.show()