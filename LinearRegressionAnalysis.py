# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:22:52 2018

@author: KSinghrao
"""

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 2. Define paths
# Windows machine path
#path = 'G:/My Drive/UCLA/Lewis_lab/Projects/MultiSequence MR-CT phantom/#Manuscript June272018/Results/Cleaned Data'

# Laptop path
path = '/Volumes/GoogleDrive/My Drive/UCLA/Lewis_lab/Projects/MultiSequence MR-CT phantom/Manuscript 6_27_2018/Results/Cleaned Data/'

os.chdir(path)

# 3. Import data
dataset = pd.read_csv('CleanData.csv')

# Select CaC03 data
#dataset_CaCO3 = dataset[dataset.cSi ==0]
#dataset2 = dataset_CaCO3
dataset2 = dataset.copy()

# Create 1/T1, 1/T2 data
def square(list):
    return [i ** 2 for i in list]


l = [None]*len(dataset)

dataset2.meanT1 = 1./dataset2.meanT1
dataset2.errT1 = square(dataset2.meanT1) * dataset2.errT1
dataset2.meanT2 = 1./dataset2.meanT2
dataset2.errT2 = square(dataset2.meanT2) * dataset2.errT2
                             
    
# 4. Visuialize data
# Plot T1 vs Gd
#for i in range(len(dataset2)): 
#    fmat = ptype(dataset2,i)
#    plt.errorbar(dataset2.cGd[i],dataset2.meanT1[i], xerr = dataset2.cGd_err[i], yerr= dataset2.errT1[i], fmt=fmat)
#
#plt.show()

# 5. Fit data
# Multiparametric linear regression
# Clean up data
dataset2 = dataset2.drop(dataset2.columns[0],axis=1)
Gdc = dataset2.cGd.copy()
Agc = dataset2.cAg.copy()
Cac = dataset2.cCa.copy()
Sic = dataset2.cSi.copy()

T1v = dataset2.meanT1
T2v = dataset2.meanT2
HUv = dataset2.meanHU

X = pd.DataFrame([Gdc,Agc,Cac,Sic])
X = X.transpose()

y = T1v


from fitT1 import T1fit
fz = T1fit(X,y)


# 2. Model training: Fitting the multiple linear regression model
from sklearn.linear_model import BayesianRidge
reg = BayesianRidge()
reg.fit(X,y)
#y_pred = reg.predict(X)

Q = reg.coef_.transpose().tolist()
Q.insert(0,reg.intercept_.tolist())

# Try polynomial fit for T1
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)

# Keep statistically significant varaibles
# Remove variable with highest p value and repeat step

import statsmodels.formula.api as sm
# Optimal model for 2 parameter fit
X_opt = X_poly[:,[0,1,4,5,6,14]]
regressor_OLS = sm.OLS(endog =y, exog = X_opt).fit()
regressor_OLS.summary() 


# Remove those particular powers from the fit model and refit
from sklearn.linear_model import LinearRegression 
polyreg_ = LinearRegression()
polyreg_.fit(X_poly,y)

interC_ = polyreg_.intercept_

coef_ = list([None])
sups_ = list([None])

for i in [0,1,4,5,6,14]:
      coef_.append(polyreg_.coef_[i])
      sups_.append(poly_reg.powers_[i,:].tolist())

# X_ = np.delete(X_,(1),axis=1)

# To get which power variable
# print poly.powers_


# Plot fits with data
# Plot T1 vs Gd
lineN = 7

x = np.linspace(0,max(Gdc),11).tolist()
n=4
lists = [[1]*len(x) for j in range(n)]

lists[0] =  np.transpose(x)
lists[1] =  [X.cAg[5*lineN]*d for d in lists[1]]
lists[2] =  [X.cCa[5*lineN]*d for d in lists[2]]
lists[3] =  [X.cSi[5*lineN]*d for d in lists[3]]

xVal = pd.DataFrame(data=lists,dtype=np.float64)
xVal = xVal.transpose()

# Polynomial fit
X_poly_val = poly_reg.fit_transform(xVal)
xw = set(range(0,15))
b = set([0,1,4,5,6,14])
remCols = list(xw^b)

for jk in remCols:
      X_poly_val[:,jk]  = 0*X_poly_val[:,jk] 

polyfit_ = polyreg_.predict(X_poly_val)

# Linear fit
fitLin = reg.predict(xVal)

# Create plots
plt.errorbar(dataset2.cGd[5*lineN:5*lineN+5],dataset2.meanT1[5*lineN:5*lineN+5],xerr=dataset2.cGd_err[5*lineN:5*lineN+5],yerr = dataset2.errT1[5*lineN:5*lineN+5],fmt='.k')
plt.xlabel('Gd (umol/ml)',fontsize=14)
plt.ylabel('1/T1 (1/ms)',fontsize=14)
plt.plot(x,fitLin,'-k')
# plt.plot(x,polyfit_,'-b')
plt.show()

# Plot T2 vs Agarose
plt.errorbar(dataset2.cAg,dataset2.meanT2,xerr=dataset2.cAg_err,yerr = dataset2.errT2,fmt='.k')    
plt.xlabel('Ag (g)',fontsize=14)
plt.ylabel('1/T2 (1/ms)',fontsize=14)

# Plot HU vs SiO2
dataset_SiO2 = dataset2[dataset2.cCa ==0]
plt.errorbar(dataset_SiO2.cSi,dataset_SiO2.meanHU,xerr=dataset_SiO2.cSi_err,yerr = dataset_SiO2.errHU,fmt='.k')    
plt.xlabel('SiO2 (g)',fontsize=14)
plt.ylabel('HU',fontsize=14)

# Plot HU vs SiO2
dataset_CaCO3 = dataset2[dataset2.cSi ==0]
plt.errorbar(dataset_CaCO3.cCa,dataset_CaCO3.meanHU,xerr=dataset_CaCO3.cSi_err,yerr = dataset_CaCO3.errHU,fmt='.k')    
plt.xlabel('CaCO3 (g)',fontsize=14)
plt.ylabel('HU',fontsize=14)

