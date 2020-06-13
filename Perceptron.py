from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import csv as csv
import math 
import scipy.interpolate as spin 
#Function to train a perceptron
def PerceptronTrain(X, Y):
 d = np.shape(X)[1] 
 n = np.shape(X)[0]
 theta = np.zeros([1, d])
 alltheta=[]
 allError=[]  
 allError.append(1)
 alltheta.append(theta[0])
 firstItr=0    
 k = 0
 i = 0
 while True:
  for row in X:    
   mu = Y[i]*np.dot(theta, row)
   if mu < 0 or firstItr==0:
    tempRow=[]
    for elm in row:
        multip=float(Y[i])
        tempRow.append(elm*multip)
    theta=theta+tempRow
    alltheta.append(theta[0])
    allError.append(perceptron_test(theta,X,Y))  
    firstItr=firstItr+1
    k = k+1
   if i==n-1:
    i=-1
   i = i+1  
  if allError[k]==0:
   break
 if d==2: 
  errorSurfacePlot(alltheta,allError)
 return theta, k
#plot the error surface for 2-dim inputs only
def errorSurfacePlot(alltheta,allError):
 xaxis=[]
 yaxis=[]
 zaxis=allError     
 for elm in alltheta:  
  xaxis.append(elm[0])
  yaxis.append(elm[1])
 old_indicesX = np.arange(0,len(xaxis))
 old_indicesy = np.arange(0,len(yaxis))
 old_indicesz = np.arange(0,len(zaxis))
 new_indicesX = np.linspace(0,len(xaxis)-1,10000)
 new_indicesy = np.linspace(0,len(yaxis)-1,10000)
 new_indicesz = np.linspace(0,len(zaxis)-1,10000)
 splx = spin.UnivariateSpline(old_indicesX,xaxis,k=3,s=0)
 sply = spin.UnivariateSpline(old_indicesy,yaxis,k=3,s=0)
 splz = spin.UnivariateSpline(old_indicesz,zaxis,k=3,s=0)
 newX = splx(new_indicesX) 
 newy = sply(new_indicesy)
 newZ = splz(new_indicesz)  
 fig = plt.figure()
 ax = plt.axes(projection='3d')
 surf=ax.plot_trisurf(np.array(newy), np.array(newX), np.array(newZ), cmap='plasma',antialiased=True,shade=True) 
 ax.set_xlabel('Theta(i)', fontsize=10, rotation=150)
 ax.set_ylabel('Theta(J)', fontsize=10, rotation=130)
 ax.set_zlabel('||Error||', fontsize=10, rotation=-60)
 fig.colorbar(surf, shrink=0.5, aspect=10)
 plt.show()
#Function to Count errors
def perceptron_test(theta, X_test, y_test):
 theta=theta[0]      
 y=[]
 for row in X_test:
     y.append(math.copysign(1,np.dot(row,theta)))
 error = []
 zip_object = zip(y, y_test)
 for list1_i, list2_i in zip_object:
       if list1_i==list2_i:
         error.append(0)  
       if list1_i!=list2_i:
         error.append(1)  
 magOferror=sum(error)/len(error) 
 return magOferror 
#Decision boundary plot
def Decision_boundary_Plot(theta,X):
 theta=theta[0]      
 y=[]  
 plt.figure(figsize=(10,6))
 plt.grid(True)    
 for row in X:
  y.append(math.copysign(1,np.dot(row,theta)))
 for X,Y in zip(X,y):
  plt.plot(X[0],X[1],'r*' if (Y == 1.0) else 'b*',)
 slope=-1*theta[1]/theta[0] 
 i = np.linspace(-100*np.amin(X),100*np.amax(X))
 z=slope*i
 plt.plot(i, z,) 
 plt.show()    
#Test Script
with open('p1_b_X.dat') as f:
 datax = f.readlines()
 X = csv.reader(datax, delimiter=' ')
 xd = []
 floatRow=[]
for row in X:
 while '' in row:
        row.remove('') 
 for item in row:
        floatRow.append(float(item))       
 xd.append(floatRow)
 floatRow=[]
with open('p1_b_y.dat') as z:
 datay = z.readlines()
 Y = csv.reader(datay, delimiter=' ')
 yd = []
for row in Y:
 while '' in row:
  row.remove('') 
 for item in row:
        floatRow.append(float(item))    
 yd.append(floatRow[0])
 floatRow=[]
out=PerceptronTrain(xd,yd)
Decision_boundary_Plot(out[0],xd)







