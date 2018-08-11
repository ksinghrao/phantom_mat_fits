#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 17:42:43 2018

@author: kamalsinghrao
"""

# Plot HU vs CaCO3
dataset_CaCO3 = dataset2[dataset2.cSi ==0]
plt.errorbar(dataset_CaCO3.cCa,dataset_CaCO3.meanHU,xerr=dataset_CaCO3.cCa_err,yerr = dataset_CaCO3.errHU,fmt='.k')    
plt.xlabel('CaCO3 (g)',fontsize=14)
plt.ylabel('HU',fontsize=14)

# Plot T1 vs T2
plt.errorbar(dataset.meanT2,dataset.meanT1,xerr=dataset.errT2,yerr = dataset.errT1,fmt='.k')
plt.xlabel('T2 (ms)',fontsize=14)
plt.ylabel('T1 (ms)',fontsize=14)

# Plot T1 vs HU
plt.errorbar(dataset.meanHU,dataset.meanT1,xerr=dataset.errHU,yerr = dataset.errT1,fmt='.k')
plt.xlabel('HU',fontsize=14)
plt.ylabel('T1 (ms)',fontsize=14)

# Plot T2 vs HU
plt.errorbar(dataset.meanHU,dataset.meanT2,xerr=dataset.errHU,yerr = dataset.errT2,fmt='.k')
plt.xlabel('HU',fontsize=14)
plt.ylabel('T2 (ms)',fontsize=14)


def ptype(list,i):
    if list.cCa[i] == 2.5:
        if list.cAg[i] ==0.0:
            pointT = '-b'   
        elif list.cAg[i] ==1.0:
            pointT = '-g'
        elif list.cAg[i] ==2.5:
            pointT = '-r'
        elif list.cAg[i] ==4.0:
            pointT = '-k'            
    elif list.cCa[i] == 5.0:
        if list.cAg[i] ==0.0:
            pointT = ':b'   
        elif list.cAg[i] ==1.0:
            pointT = ':g'
        elif list.cAg[i] ==2.5:
            pointT = ':r'
        elif list.cAg[i] ==4.0:
            pointT = ':k'
    else: 
        if list.cAg[i] ==0.0:
            pointT = '--b'   
        elif list.cAg[i] ==1.0:
            pointT = '--g'
        elif list.cAg[i] ==2.5:
            pointT = '--r'
        elif list.cAg[i] ==4.0:
            pointT = '--k'
    return pointT
