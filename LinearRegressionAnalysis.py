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
#path = '/Volumes/GoogleDrive/My Drive/UCLA/Lewis_lab/Projects/MultiSequence MR-CT phantom/Manuscript 6_27_2018/Results/Cleaned Data/'
path = '/Volumes/GoogleDrive/My Drive/UCLA/Lewis_lab/Projects/MultiSequence MR-CT phantom/Manuscript 6_27_2018/'

os.chdir(path+'Results/Cleaned Data/')

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


os.chdir(path+'Fit code')
from fitT1 import T1fit

fT1 = []
for lineN in range(0,18):
      fT1info = T1fit(X,y,lineN)
      xR = fT1info.x
      fT1.append(fT1info.fitLin)
      

# Create plots
nrow = 6
ncol = 3
fig, axs = plt.subplots(nrow, ncol)
for i, ax in enumerate(fig.axes):
      #ax.set_ylabel(str(i))
      ax.errorbar(dataset2.cGd[5*i:5*i+5],dataset2.meanT1[5*i:5*i+5],xerr=dataset2.cGd_err[5*i:5*i+5],yerr = dataset2.errT1[5*i:5*i+5],fmt='.k') 
      ax.set_xlabel('Gd (umol/ml)',fontsize=7)
      ax.set_ylabel('1/T1 (1/ms)',fontsize=7)
      ax.tick_params(labelsize=6)
      ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
      fig.tight_layout(rect=[0, 0, 1, 1.5])
      
# Create indivual plots      
lineN = 18      
plt.errorbar(dataset2.cGd[5*lineN:5*lineN+5],dataset2.meanT1[5*lineN:5*lineN+5],xerr=dataset2.cGd_err[5*lineN:5*lineN+5],yerr = dataset2.errT1[5*lineN:5*lineN+5],fmt='.k')
plt.xlabel('Gd (umol/ml)',fontsize=14)
plt.ylabel('1/T1 (1/ms)',fontsize=14)
plt.plot(xR,fT1[lineN],'-k')
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

