#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 13:02:18 2018

@author: kamalsinghrao
"""

# 1. Import libraries
import pandas as pd
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


def cPlots(vCa,vSi):
      datasetT = dataset[(dataset.cCa==vCa) & (dataset.cSi==vSi)]
      # Create Gd dataset
      dset_Gd = datasetT.copy()
      dset_Gd = dset_Gd.reset_index(drop=True)
      
      # Create Ag data
      dAg_di = []
      
      for h in [0,0.005,0.05,0.25,0.5]:
            dAg_di.append(datasetT.loc[(datasetT.cGd==h)])
            
      dAg_rf = pd.concat(dAg_di)
      
      dset_Ag = dAg_rf.copy()
      dset_Ag = dset_Ag.reset_index(drop=True)
      
      
      # Plot T1 vs T2      
      nrow = 1
      ncol = 3
      fig, axs = plt.subplots(nrow, ncol,figsize=(20, 10))
      
      plt.suptitle('CaCO$_3$ concentration = ' + str(100*dset_Ag.cCa[0]/50) + ' w/w% \n GM concentration = ' + str(100*dset_Ag.cSi[0]/50) + ' w/w%')
      
      ax = plt.subplot(1,3,1)
      
      for i in range(5): 
            ax.plot(dset_Gd[5*i:5*i+5].meanT2,dset_Gd[5*i:5*i+5].meanT1,'ok-',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
            ax.plot(dset_Ag[4*i:4*i+4].meanT2,dset_Ag[4*i:4*i+4].meanT1,'ok--',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
            
      ax.set_xlabel('T2 (ms)',fontsize=12)
      ax.set_ylabel('T1 (ms)',fontsize=12)
      ax.tick_params(labelsize=12)
      #ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
      ax.yaxis.offsetText.set(size=12)
      
      # Plot T1 vs HU
      ax = plt.subplot(1,3,2)
      
      for i in range(5): 
            ax.plot(dset_Gd[5*i:5*i+5].meanHU,dset_Gd[5*i:5*i+5].meanT1,'ok-',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
            ax.plot(dset_Ag[4*i:4*i+4].meanHU,dset_Ag[4*i:4*i+4].meanT1,'ok--',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
            
      ax.set_xlabel('CT number (HU)',fontsize=12)
      ax.set_ylabel('T1 (ms)',fontsize=12)
      ax.tick_params(labelsize=12)
      #ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
      ax.yaxis.offsetText.set(size=12)
      
      # Plot T2 vs HU     
      ax = plt.subplot(1,3,3) 
      
      for i in range(5): 
            ax.plot(dset_Gd[5*i:5*i+5].meanHU,dset_Gd[5*i:5*i+5].meanT2,'ok-',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
            ax.plot(dset_Ag[4*i:4*i+4].meanHU,dset_Ag[4*i:4*i+4].meanT2,'ok--',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
            
      ax.set_xlabel('CT number (HU)',fontsize=12)
      ax.set_ylabel('T2 (ms)',fontsize=12)
      ax.tick_params(labelsize=12)
      #ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
      ax.yaxis.offsetText.set(size=12)
      
      plt.subplots_adjust(wspace=0.3)
      
      return fig

