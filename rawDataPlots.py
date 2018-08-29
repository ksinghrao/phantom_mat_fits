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
import numpy as np
from scipy.spatial import ConvexHull

# 2. Define paths
# Windows machine path
#path = 'G:/My Drive/UCLA/Lewis_lab/Projects/MultiSequence MR-CT phantom/#Manuscript June272018/Results/Cleaned Data'

# Laptop path
path = '/Volumes/GoogleDrive/My Drive/UCLA/Lewis_lab/Projects/MultiSequence MR-CT phantom/Manuscript 6_27_2018/'

os.chdir(path+'Results/Cleaned Data/')

# 3. Import data
dataset = pd.read_csv('CleanData.csv')


# vCa = 0
# vSi =0
def preProcess(dataset,vCa,vSi):
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
      
      return {'dset_Gd':dset_Gd,'dset_Ag':dset_Ag}
      
# 0. Create verticies for real tissue values
# T1l, T1h, T2l, T2h, HUl, HUh      
muscle = [898,1509,35,53,20,200]    
fat = [253,450,41,371,-200,-80]
bone = [214,278,0,20,200,3000]
bmarrow = [106,365,40,160,200,600]
wm = [728,1735,65,75,30,80]
gm =[968,1717,83,109,30,80]

# Define colors for each tissue type
colT = ['red','yellow','gray','cyan','blue','green']
colTkey = ['Muscle','Fat','Bone','Bone Marrow','White Matter','Grey Matter']

from shapely.geometry import Polygon

polygonM_T1T2 = Polygon([[muscle[0],muscle[2]], [muscle[1],muscle[2]], [muscle[1],muscle[3]], [muscle[0],muscle[3]]])


# Make muscle observables 
y_M_T1T2=[muscle[0],muscle[1],muscle[1],muscle[0]]
x_M_T1T2=[muscle[2],muscle[2],muscle[3],muscle[3]]

y_M_T2HU=[muscle[2],muscle[3],muscle[3],muscle[2]]
x_M_T2HU=[muscle[4],muscle[4],muscle[5],muscle[5]]

y_M_T1HU=[muscle[0],muscle[1],muscle[1],muscle[0]]
x_M_T1HU=[muscle[4],muscle[4],muscle[5],muscle[5]]

# Make fat observables
y_F_T1T2=[fat[0],fat[1],fat[1],fat[0]]
x_F_T1T2=[fat[2],fat[2],fat[3],fat[3]]

y_F_T2HU=[fat[2],fat[3],fat[3],fat[2]]
x_F_T2HU=[fat[4],fat[4],fat[5],fat[5]]

y_F_T1HU=[fat[0],fat[1],fat[1],fat[0]]
x_F_T1HU=[fat[4],fat[4],fat[5],fat[5]]

# Make bone observables
y_B_T1T2=[bone[0],bone[1],bone[1],bone[0]]
x_B_T1T2=[bone[2],bone[2],bone[3],bone[3]]

y_B_T2HU=[bone[2],bone[3],bone[3],bone[2]]
x_B_T2HU=[bone[4],bone[4],bone[5],bone[5]]

y_B_T1HU=[bone[0],bone[1],bone[1],bone[0]]
x_B_T1HU=[bone[4],bone[4],bone[5],bone[5]]

# Make wm observables
y_WM_T1T2=[wm[0],wm[1],wm[1],wm[0]]
x_WM_T1T2=[wm[2],wm[2],wm[3],wm[3]]

y_WM_T2HU=[wm[2],wm[3],wm[3],wm[2]]
x_WM_T2HU=[wm[4],wm[4],wm[5],wm[5]]

y_WM_T1HU=[wm[0],wm[1],wm[1],wm[0]]
x_WM_T1HU=[wm[4],wm[4],wm[5],wm[5]]

# Make gm observables
y_GM_T1T2=[gm[0],gm[1],gm[1],gm[0]]
x_GM_T1T2=[gm[2],gm[2],gm[3],gm[3]]

y_GM_T2HU=[gm[2],gm[3],gm[3],gm[2]]
x_GM_T2HU=[gm[4],gm[4],gm[5],gm[5]]

y_GM_T1HU=[gm[0],gm[1],gm[1],gm[0]]
x_GM_T1HU=[gm[4],gm[4],gm[5],gm[5]]

# Make bmarrow observables
y_BM_T1T2=[bmarrow[0],bmarrow[1],bmarrow[1],bmarrow[0]]
x_BM_T1T2=[bmarrow[2],bmarrow[2],bmarrow[3],bmarrow[3]]

y_BM_T1HU=[bmarrow[0],bmarrow[1],bmarrow[1],bmarrow[0]]
x_BM_T1HU=[bmarrow[4],bmarrow[4],bmarrow[5],bmarrow[5]]

y_BM_T2HU=[bmarrow[2],bmarrow[3],bmarrow[3],bmarrow[2]]
x_BM_T2HU=[bmarrow[4],bmarrow[4],bmarrow[5],bmarrow[5]]

# Legend colors
colZ = ['blue','green','red','cyan','olive']
colZid = ['0g CaCO$_3$ \n 0g GM','2.5g CaCO$_3$ \n 0g GM','5g CaCO$_3$ \n 0g GM','0g CaCO$_3$ \n 2.5g GM','0g CaCO$_3$ \n 5g GM'];

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=colZ[0], lw=4),
                Line2D([0], [0], color=colZ[1], lw=4),
                Line2D([0], [0], color=colZ[2], lw=4),
                Line2D([0], [0], color=colZ[3], lw=4),                
                Line2D([0], [0], color=colZ[4], lw=4)]


# 1a. Plot T1 vs T2      
gridsize = (4, 2)
fig = plt.figure(figsize=(10, 10))

ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid(gridsize, (2, 0))
ax3 = plt.subplot2grid(gridsize, (2, 1))
ax4 = plt.subplot2grid(gridsize, (3, 0))
ax5 = plt.subplot2grid(gridsize, (3, 1))
axes = [ax1,ax2,ax3,ax4,ax5]

plt.subplots_adjust(wspace=0.3,hspace = 0.6)

plt.legend(custom_lines, [colZid[0], colZid[1], colZid[2],colZid[3],colZid[4]],loc="upper right",bbox_to_anchor=(0.97, 5.7))

rCa = [0,2.5,5,0,0,10,15,25]
rSi = [0,0,0,2.5,5,0,0,0]
plotLims = [600,200,200,50,50]
plotFt = [12,8,8,8,8]

for ai in range(5):
      Q = preProcess(dataset,rCa[ai],rSi[ai])
      dset_Ag = Q['dset_Ag']
      dset_Gd = Q['dset_Gd']      
      
      ax = axes[ai]
      for i in range(5):
            ax.plot(dset_Gd[5*i:5*i+5].meanT2,dset_Gd[5*i:5*i+5].meanT1,'ok-',markersize =5,markeredgecolor ='k',markerfacecolor ='w',linewidth=0.9)
            ax.plot(dset_Ag[4*i:4*i+4].meanT2,dset_Ag[4*i:4*i+4].meanT1,'ok--',markersize =5,markeredgecolor ='k',markerfacecolor ='w',linewidth=0.9)
      
      # Create convex hull
      zD = zip(dset_Gd.meanT2.tolist(),dset_Gd.meanT1.tolist())
      zDvert = list(zD)
      zDvert = np.asarray(zDvert)
      hull = ConvexHull(zDvert)                  
      #for simplex in hull.simplices:
      #      ax.plot(zDvert[simplex, 0], zDvert[simplex, 1], 'b-') 
      ax.fill(zDvert[hull.vertices,0], zDvert[hull.vertices,1],colZ[ai],alpha=0.2)      
      ax.set_xlabel('T2 (ms)',fontsize=plotFt[ai])
      ax.set_ylabel('T1 (ms)',fontsize=plotFt[ai])
      ax.tick_params(labelsize=plotFt[ai])
      #ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
      ax.yaxis.offsetText.set(size=plotFt[ai])
      ax.set_xlim([0,plotLims[ai]])
      ax.set_ylim([0,2500])

# 1b. Create a zoom out plot for complete HU range
fig = plt.figure(figsize=(10, 10))
for ai in range(8):
      Q = preProcess(dataset,rCa[ai],rSi[ai])
      #dset_Ag = Q['dset_Ag']
      dset_Gd = Q['dset_Gd']      

      if ai in range(5):
            for i in range(5): 
                  plt.plot(dset_Gd[5*i:5*i+5].meanT2,dset_Gd[5*i:5*i+5].meanT1,'ok',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
      else:
            for i in range(1): 
                  plt.plot(dset_Gd[5*i:5*i+5].meanT2,dset_Gd[5*i:5*i+5].meanT1,'ok',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
            
      # Fill tissue regions
      plt.fill(x_M_T1T2, y_M_T1T2,colT[0],alpha=0.1)
      plt.fill(x_F_T1T2, y_F_T1T2,colT[1],alpha=0.1)
      plt.fill(x_B_T1T2, y_B_T1T2,colT[2],alpha=0.1)
      plt.fill(x_BM_T1T2, y_BM_T1T2,colT[3],alpha=0.1)
      plt.fill(x_WM_T1T2, y_WM_T1T2,colT[4],alpha=0.1)
      plt.fill(x_GM_T1T2, y_GM_T1T2,colT[5],alpha=0.1)
      #plt.xlim([-200,1000])      
      
plt.xlabel('T1 (ms)',fontsize=12)
plt.ylabel('T2 (ms)',fontsize=12)
#plt.ylim([0,500])
plt.tick_params(labelsize=12)

# Legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=colT[0], lw=4),
                Line2D([0], [0], color=colT[1], lw=4),
                Line2D([0], [0], color=colT[2], lw=4),
                Line2D([0], [0], color=colT[3], lw=4),                
                Line2D([0], [0], color=colT[4], lw=4),
                Line2D([0], [0], color=colT[5], lw=4)]
plt.legend(custom_lines, [colTkey[0], colTkey[1],colTkey[2],colTkey[3],colTkey[4],colTkey[5]])

# 2a. Plot T1 vs HU
fig1 = plt.figure(figsize=(10, 10))
for ai in range(5):
      Q = preProcess(dataset,rCa[ai],rSi[ai])
      dset_Ag = Q['dset_Ag']
      dset_Gd = Q['dset_Gd']      

      for i in range(5): 
            plt.plot(dset_Gd[5*i:5*i+5].meanHU,dset_Gd[5*i:5*i+5].meanT1,'ok-',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
            plt.plot(dset_Ag[4*i:4*i+4].meanHU,dset_Ag[4*i:4*i+4].meanT1,'ok--',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
            
            # Create convex hull
            zD = zip(dset_Gd.meanHU.tolist(),dset_Gd.meanT1.tolist())
            zDvert = list(zD)
            zDvert = np.asarray(zDvert)
            
            hull = ConvexHull(zDvert)                  
            #for simplex in hull.simplices:
            #      ax.plot(zDvert[simplex, 0], zDvert[simplex, 1], 'b-') 
            plt.fill(zDvert[hull.vertices,0], zDvert[hull.vertices,1],colZ[ai],alpha=0.2)

plt.xlabel('CT number (HU)',fontsize=12)
plt.ylabel('T1 (ms)',fontsize=12)
plt.ylim([0,2500])
plt.tick_params(labelsize=12)
##plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
#plt.yaxis.offsetText.set(size=12)

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=colZ[0], lw=4),
                Line2D([0], [0], color=colZ[1], lw=4),
                Line2D([0], [0], color=colZ[2], lw=4),
                Line2D([0], [0], color=colZ[3], lw=4),                
                Line2D([0], [0], color=colZ[4], lw=4)]
plt.legend(custom_lines, [colZid[0], colZid[1], colZid[2],colZid[3],colZid[4]])

# 2b. Create a zoom out plot for complete HU range
fig2 = plt.figure(figsize=(10, 10))
for ai in range(8):
      Q = preProcess(dataset,rCa[ai],rSi[ai])
      #dset_Ag = Q['dset_Ag']
      dset_Gd = Q['dset_Gd']      

      if ai in range(5):
            for i in range(5): 
                  plt.plot(dset_Gd[5*i:5*i+5].meanHU,dset_Gd[5*i:5*i+5].meanT1,'ok',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
      else:
            for i in range(1): 
                  plt.plot(dset_Gd[5*i:5*i+5].meanHU,dset_Gd[5*i:5*i+5].meanT1,'ok',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
            
      # Fill tissue regions
      plt.fill(x_M_T1HU, y_M_T1HU,colT[0],alpha=0.1)
      plt.fill(x_F_T1HU, y_F_T1HU,colT[1],alpha=0.1)
      plt.fill(x_B_T1HU, y_B_T1HU,colT[2],alpha=0.1)
      plt.fill(x_BM_T1HU, y_BM_T1HU,colT[3],alpha=0.1)
      plt.fill(x_WM_T1HU, y_WM_T1HU,colT[4],alpha=0.1)
      plt.fill(x_GM_T1HU, y_GM_T1HU,colT[5],alpha=0.1)
      plt.xlim([-200,1000])      
      
plt.xlabel('CT number (HU)',fontsize=12)
plt.ylabel('T1 (ms)',fontsize=12)
#plt.ylim([0,500])
plt.tick_params(labelsize=12)

# Legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=colT[0], lw=4),
                Line2D([0], [0], color=colT[1], lw=4),
                Line2D([0], [0], color=colT[2], lw=4),
                Line2D([0], [0], color=colT[3], lw=4),                
                Line2D([0], [0], color=colT[4], lw=4),
                Line2D([0], [0], color=colT[5], lw=4)]
plt.legend(custom_lines, [colTkey[0], colTkey[1],colTkey[2],colTkey[3],colTkey[4],colTkey[5]])

# 3a. Plot T2 vs HU
fig2 = plt.figure(figsize=(10, 10))
for ai in range(5):
      Q = preProcess(dataset,rCa[ai],rSi[ai])
      dset_Ag = Q['dset_Ag']
      dset_Gd = Q['dset_Gd']      

      for i in range(5): 
            plt.plot(dset_Gd[5*i:5*i+5].meanHU,dset_Gd[5*i:5*i+5].meanT2,'ok-',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
            plt.plot(dset_Ag[4*i:4*i+4].meanHU,dset_Ag[4*i:4*i+4].meanT2,'ok--',markersize =5,markeredgecolor ='k',markerfacecolor ='w')

            # Create convex hull
            zD = zip(dset_Gd.meanHU.tolist(),dset_Gd.meanT2.tolist())
            zDvert = list(zD)
            zDvert = np.asarray(zDvert)

            hull = ConvexHull(zDvert)                  
            #for simplex in hull.simplices:
            #      ax.plot(zDvert[simplex, 0], zDvert[simplex, 1], 'b-') 
            plt.fill(zDvert[hull.vertices,0], zDvert[hull.vertices,1],colZ[ai],alpha=0.2)

plt.xlabel('CT number (HU)',fontsize=12)
plt.ylabel('T2 (ms)',fontsize=12)
plt.ylim([0,500])
plt.tick_params(labelsize=12)
##plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0)) 
#plt.yaxis.offsetText.set(size=12)

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=colZ[0], lw=4),
                Line2D([0], [0], color=colZ[1], lw=4),
                Line2D([0], [0], color=colZ[2], lw=4),
                Line2D([0], [0], color=colZ[3], lw=4),                
                Line2D([0], [0], color=colZ[4], lw=4)]
plt.legend(custom_lines, [colZid[0], colZid[1], colZid[2],colZid[3],colZid[4]])

# 3b. Create a zoom out plot for complete HU range
fig2 = plt.figure(figsize=(10, 10))
for ai in range(8):
      Q = preProcess(dataset,rCa[ai],rSi[ai])
      #dset_Ag = Q['dset_Ag']
      dset_Gd = Q['dset_Gd']      

      if ai in range(5):
            for i in range(5): 
                  plt.plot(dset_Gd[5*i:5*i+5].meanHU,dset_Gd[5*i:5*i+5].meanT2,'ok',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
      else:
            for i in range(1): 
                  plt.plot(dset_Gd[5*i:5*i+5].meanHU,dset_Gd[5*i:5*i+5].meanT2,'ok',markersize =5,markeredgecolor ='k',markerfacecolor ='w')
            
      # Fill tissue regions
      plt.fill(x_M_T2HU, y_M_T2HU,colT[0],alpha=0.1)
      #plt.fill(x_F_T2HU, y_F_T2HU,colT[1],alpha=0.1)
      plt.fill(x_B_T2HU, y_B_T2HU,colT[2],alpha=0.1)
      plt.fill(x_BM_T2HU, y_BM_T2HU,colT[3],alpha=0.1)
      plt.fill(x_WM_T2HU, y_WM_T2HU,colT[4],alpha=0.1)
      plt.fill(x_GM_T2HU, y_GM_T2HU,colT[5],alpha=0.1)
      plt.xlim([-200,1000])      
      
plt.xlabel('CT number (HU)',fontsize=12)
plt.ylabel('T2 (ms)',fontsize=12)
#plt.ylim([0,500])
plt.tick_params(labelsize=12)

# Legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=colT[0], lw=4),
                #Line2D([0], [0], color=colT[1], lw=4),
                Line2D([0], [0], color=colT[2], lw=4),
                Line2D([0], [0], color=colT[3], lw=4),                
                Line2D([0], [0], color=colT[4], lw=4),
                Line2D([0], [0], color=colT[5], lw=4)]
#plt.legend(custom_lines, [colTkey[0], colTkey[1],colTkey[2],colTkey[3],colTkey[4],colTkey[5]])
plt.legend(custom_lines, [colTkey[0],colTkey[2],colTkey[3],colTkey[4],colTkey[5]])
