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
path = '/Volumes/GoogleDrive/My Drive/UCLA/Lewis_lab/Projects/MultiSequence MR-CT phantom/Manuscript 6_27_2018/'

os.chdir(path+'Results/Cleaned Data/')

# 3. Import data
dataset = pd.read_csv('CleanData.csv')
dataset2 = dataset.copy()

# Create 1/T1, 1/T2 data
def square(list):
    return [i ** 2 for i in list]


l = [None]*len(dataset)

dataset2.meanT1 = 1./dataset2.meanT1
dataset2.errT1 = square(dataset2.meanT1) * dataset2.errT1
dataset2.meanT2 = 1./dataset2.meanT2
dataset2.errT2 = square(dataset2.meanT2) * dataset2.errT2

# 4. Create T1 vs Gd plots and fits                             
# Reformat data
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


# Create plot data
dset = dataset2.copy()
dset = dset.reset_index(drop=True)
      
xB = dset[['cGd', 'cAg', 'cCa','cSi']].copy()

# Fit data
y = dataset2.meanT1

os.chdir(path+'Fit code')
from sklearn.linear_model import BayesianRidge
reg = BayesianRidge()
reg.fit(xB,y)

# Save fit parameters
fT1_par = [reg.intercept_] + reg.coef_.tolist()

# Create plots
nrow = 3
ncol = 6
fig, axs = plt.subplots(nrow, ncol,figsize=(10, 10))
for i, ax in enumerate(fig.axes):
      #xQ = [dset.cGd[5*i:5*i+5],dset.cAg[5*i:5*i+5],dset.cCa[5*i:5*i+5],dset.cSi[5*i:5*i+5]]      

      xQ = xB[5*i:5*i+5]
      yQ = dset.meanT1[5*i:5*i+5]
      xQerr = dset.cGd_err[5*i:5*i+5]
      yQerr = dset.errT1[5*i:5*i+5]
      
      
      #ax.set_ylabel(str(i))
      ax.errorbar(xQ.cGd,yQ,xerr=xQerr,yerr = yQerr,fmt='.k') 
      ax.plot(xQ.cGd,reg.predict(xQ).tolist(),'-g')
      
      title_ = 'Agarose = ' + str(dataset2.cAg[5*i]) + 'g \n CaCO$_3$ = ' + str(dataset2.cCa[5*i]) + 'g \n GM = ' + str(dataset2.cSi[5*i]) + 'g'
      ax.set_title(title_,fontsize = 7,y=1.08)
      ax.set_xlabel('Gd (umol/ml)',fontsize=7)
      ax.set_ylabel('1/T1 (1/ms)',fontsize=7)
      ax.tick_params(labelsize=6)
      ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
      ax.yaxis.offsetText.set(size=7)
      fig.tight_layout(rect=[0, 0, 1, 1])
      ax.set_xlim([0,0.6])
      ax.set_ylim([0,10e-3])
            
# 4. Create T2 vs Ag plots and fits         
# 4a.CaCO3 data
# Try combo fit      
regT = BayesianRidge(compute_score=True)
regT.fit(X,T2v)
fT2_par = [regT.intercept_] + regT.coef_.tolist()

# Reformat CaCO3 data  
dCaCO3 = dataset2[(dataset2.cSi ==0)]

dAg_CaCO3_di = []

for h in [0,0.005,0.05,0.25,0.5]:
      dAg_CaCO3_di.append(dCaCO3.loc[(dCaCO3.cGd==h)])
      
dAg_CaCO3_rf = pd.concat(dAg_CaCO3_di)

# Create plot data
dset = dAg_CaCO3_rf.copy()
dset = dset.reset_index(drop=True)
xB = dset[['cGd', 'cAg', 'cCa','cSi']].copy()

# Fit data      
y = dset.meanT2

os.chdir(path+'Fit code')
from sklearn.linear_model import BayesianRidge
reg = BayesianRidge(compute_score=True)
reg.fit(xB,y)

# Save fit parameters
fT2_CaCO3_par = [reg.intercept_] + reg.coef_.tolist()
      
# Create plots
nrow = 4
ncol = 4
fig, axs = plt.subplots(nrow, ncol,figsize=(10, 10))
for i, ax in enumerate(fig.axes):   
      xQ = xB[4*i:4*i+4]
      yQ = dset.meanT2[4*i:4*i+4]
      xQerr = dset.cAg_err[4*i:4*i+4]
      yQerr = dset.errT2[4*i:4*i+4]

      ax.errorbar(xQ.cAg,yQ,xerr=xQerr,yerr = yQerr,fmt='.k') 
      #ax.plot(xQ.cAg,reg.predict(xQ).tolist(),'-k')
      ax.plot(xQ.cAg,regT.predict(xQ).tolist(),'-g')
            
      title_ = 'Gd = ' + str(dset.cGd[4*i]) + 'g \n CaCO$_3$ = ' + str(dset.cCa[4*i]) + 'g \n GM = ' + str(dataset2.cSi[4*i]) + 'g'
      ax.set_title(title_,fontsize = 7,y=1.08)
      ax.set_xlabel('Ag (g)',fontsize=7)
      ax.set_ylabel('1/T2 (1/ms)',fontsize=7)
      ax.tick_params(labelsize=6)
      ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
      ax.yaxis.offsetText.set(size=7)
      ax.set_xlim([0,4])
      ax.set_ylim([0,3.5e-2])

      fig.tight_layout(rect=[0, 0, 1, 1])
      
      
axs[-1,-1].axis('off')            
plt.show()

# 4b.SiO2 data  
# Reformat data
dSiO2 = dataset2[60:len(dataset2)].copy()
dA_0 = dataset2[(dataset2.cCa==0) & (dataset2.cSi==0) & (dataset2.cAg ==0)]
dA_1 = dataset2[(dataset2.cCa==0) & (dataset2.cSi==0) & (dataset2.cAg ==1)]
dset_0 = dSiO2[dSiO2.cAg==0]
dset_1 = dSiO2[dSiO2.cAg==1]

dset = pd.concat([dA_0,dset_0,dA_1,dset_1])
dset = dset.reset_index(drop=True)

dAg_SiO2_di = []

for h in [0,0.005,0.05,0.25,0.5]:
      dAg_SiO2_di.append(dSiO2.loc[(dSiO2.cGd==h)])
      
dAg_SiO2_rf = pd.concat(dAg_SiO2_di)
dAg_SiO2_rf = dAg_SiO2_rf.reset_index(drop=True)

# Create plot data 
dset = dAg_SiO2_rf.copy()
dset = dset.reset_index(drop=True)
xB = dset[['cGd', 'cAg', 'cCa','cSi']].copy()

# Fit data      
y = dset.meanT2

os.chdir(path+'Fit code')
from sklearn.linear_model import BayesianRidge
reg = BayesianRidge(compute_score=True)
reg.fit(xB,y)

# Save fit parameters
fT2_SiO2_par = [reg.intercept_] + reg.coef_.tolist()


# Create plots
nrow = 3
ncol = 4
fig, axs = plt.subplots(nrow, ncol,figsize=(5, 10))
for i, ax in enumerate(fig.axes): 
      xQ = xB[3*i:3*i+3]     
      yQ = dset.meanT2[3*i:3*i+3]
      xQerr = dset.cAg_err[3*i:3*i+3]
      yQerr = dset.errT2[3*i:3*i+3]
      ax.errorbar(xQ.cAg,yQ,xerr=xQerr,yerr = yQerr,fmt='.k') 
      #ax.plot(xQ.cAg,reg.predict(xQ).tolist(),'-k')
      ax.plot(xQ.cAg,regT.predict(xQ).tolist(),'-g')

      ax.set_xlabel('Ag (g)',fontsize=7)
      ax.set_ylabel('1/T2 (1/ms)',fontsize=7)
      ax.tick_params(labelsize=6)
      ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
      ax.yaxis.offsetText.set(size=7)
      fig.tight_layout(rect=[0, 0, 1, 1])
      title_ = 'Gd = ' + str(dset.cGd[3*i]) + 'g \n CaCO$_3$ = ' + str(dset.cCa[3*i]) + 'g \n GM = ' + str(dset.cSi[3*i]) + 'g'
      #print(title_)
      ax.set_title(title_,fontsize = 7,y=1.08)
      ax.set_xlim([-0.1,1.1])
      ax.set_ylim([0,1.5e-1])

# 5. Create HU vs CaCO3 and SiO2 plots and fits  
# 5a. Plot CaCO3 vs HU
# Try combo fit      
regT = BayesianRidge(compute_score=True)
regT.fit(X,HUv)
fHU_par = [regT.intercept_] + regT.coef_.tolist() 
      
# Reformat CaCO3 data      
dHU_CaCO3_di = []

for h in [0,0.005,0.05,0.25,0.5]:
      for j in [0,1,2.5,4]:
            dHU_CaCO3_di.append(dCaCO3.loc[(dCaCO3.cGd==h) & (dCaCO3.cAg==j)])

dHU_CaCO3_rf = pd.concat(dHU_CaCO3_di)
dHU_CaCO3_rf = dHU_CaCO3_rf.reset_index(drop=True)

# Create plot data 
dset = dHU_CaCO3_rf.copy()
dset = dset.reset_index(drop=True)
xB = dset[['cGd', 'cAg', 'cCa','cSi']].copy()

# Fit data      
y = dset.meanHU

os.chdir(path+'Fit code')
from sklearn.linear_model import BayesianRidge
reg = BayesianRidge(compute_score=True)
reg.fit(xB,y)

# Save fit parameters
fHU_CaCO3_par = [reg.intercept_] + reg.coef_.tolist()

# Plot data
nrow = 5
ncol = 4
fig, axs = plt.subplots(nrow, ncol,figsize=(10, 10))
for i, ax in enumerate(fig.axes): 
      xQ = xB[3*i:3*i+3]
      yQ = dset.meanHU[3*i:3*i+3]
      xQerr = dset.cCa_err[3*i:3*i+3]
      yQerr = dset.errHU[3*i:3*i+3]
      ax.errorbar(xQ.cCa,yQ,xerr=xQerr,yerr = yQerr,fmt='.k') 
      #ax.plot(xQ.cCa,reg.predict(xQ).tolist(),'-k')
      ax.plot(xQ.cCa,regT.predict(xQ).tolist(),'-g')

      title_ = 'Gd = ' + str(dset.cGd[3*i]) + 'g \n Agarose = ' + str(dset.cAg[3*i]) + 'g \n GM = ' + str(dset.cSi[3*i]) + 'g'
      ax.set_title(title_,fontsize = 7,y=1.08)
      ax.set_xlabel('CaCO$_3$ (g)',fontsize=7)
      ax.set_ylabel('CT# (HU)',fontsize=7)
      ax.tick_params(labelsize=6)
      ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
      ax.yaxis.offsetText.set(size=7)
      fig.tight_layout(rect=[0, 0, 1, 1])
      ax.set_xlim([-0.1,5.1])
      ax.set_ylim([0,220])

      
# Plot SiO2 data
# Reformat data   
# Add SiO2 =0 datapoint
dSiO2 = dataset2[60:len(dataset2)].copy()
dA_0 = dataset2[(dataset2.cCa==0) & (dataset2.cAg ==0)]
dA_1 = dataset2[(dataset2.cCa==0) & (dataset2.cAg ==1)]

dSiO2_HU = pd.concat([dA_0,dA_1])
dSiO2_HU = dSiO2_HU.reset_index(drop=True)

dHU_SiO2_di = []

for h in [0,0.005,0.05,0.25,0.5]:
      for j in [0,1,2.5,4]:
            dHU_SiO2_di.append(dSiO2_HU.loc[(dSiO2_HU.cGd==h) & (dSiO2_HU.cAg==j)])

dHU_SiO2_rf = pd.concat(dHU_SiO2_di)
dHU_SiO2_rf = dHU_SiO2_rf.reset_index(drop=True)

# Create plot data 
dset = dHU_SiO2_rf.copy()
dset = dset.reset_index(drop=True)
      
xB = dset[['cGd', 'cAg', 'cCa','cSi']].copy()

# Fit data      
y = dset.meanHU

os.chdir(path+'Fit code')
from sklearn.linear_model import BayesianRidge
reg = BayesianRidge(compute_score=True)
reg.fit(xB,y)

# Save fit parameters
fHU_SiO2_par = [reg.intercept_] + reg.coef_.tolist()

# Plot data
nrow = 2
ncol = 5

fig, axs = plt.subplots(nrow, ncol,figsize=(10, 10))
for i, ax in enumerate(fig.axes):
      xQ = xB[3*i:3*i+3]
      yQ = dset.meanHU[3*i:3*i+3]
      xQerr = dset.cCa_err[3*i:3*i+3]
      yQerr = dset.errHU[3*i:3*i+3]
      ax.errorbar(xQ.cSi,yQ,xerr=xQerr,yerr = yQerr,fmt='.k') 
      #ax.plot(xQ.cSi,reg.predict(xQ).tolist(),'-k')
      ax.plot(xQ.cSi,regT.predict(xQ).tolist(),'-g')
 
      title_ = 'Gd = ' + str(dset.cGd[3*i]) + 'g \n Agarose = ' + str(dset.cAg[3*i]) + 'g \n CaCO$_3$ = ' + str(dset.cCa[3*i]) + 'g'
      ax.set_title(title_,fontsize = 7,y=1.08)
      ax.set_xlabel('SiO$_2$ (g)',fontsize=7)
      ax.set_ylabel('CT# (HU)',fontsize=7)
      ax.tick_params(labelsize=6)
      ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
      ax.yaxis.offsetText.set(size=7)
      fig.tight_layout(rect=[0, 0, 1, 1])      
      ax.set_xlim([-0.1,5.1])
      ax.set_ylim([-200,20])
      

# Save fit parameters
a = np.asarray([fT1_par, fT2_par, fHU_par ])
np.savetxt("fitParameters.csv", a, delimiter=",")
      