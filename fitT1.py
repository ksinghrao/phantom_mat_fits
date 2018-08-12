      #!/usr/bin/env python3
      # -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 14:54:49 2018

@author: kamalsinghrao
"""
class ReturnValue(object):
      def __init__(self, x, fitLin):
            self.x = x
            self.fitLin = fitLin

def T1fit(X,y,line):      
      import pandas as pd

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
      
      
      coef_ = list([None])
      sups_ = list([None])
      
      for i in [0,1,4,5,6,14]:
            coef_.append(polyreg_.coef_[i])
            sups_.append(poly_reg.powers_[i,:].tolist())
      
      
      # To get which power variable
      # print poly.powers_
      
      
      # Plot fits with data
      # Plot T1 vs Gd

      import numpy   
      x = numpy.linspace(0,max(X.cGd),11).tolist()
      n=4
      lists = [[1]*len(x) for j in range(n)]
      
      lists[0] =  numpy.transpose(x)
      lists[1] =  [X.cAg[5*line]*d for d in lists[1]]
      lists[2] =  [X.cCa[5*line]*d for d in lists[2]]
      lists[3] =  [X.cSi[5*line]*d for d in lists[3]]
      
      xVal = pd.DataFrame(data=lists,dtype=numpy.float64)
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
      return ReturnValue(x, fitLin)
