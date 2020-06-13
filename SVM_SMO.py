#By: Belal El-Shinnawey
#If you have useful comments Email: belalshinnawey@gmail.com
from mpl_toolkits import mplot3d
from mlxtend.data import loadlocal_mnist
import scipy.interpolate as spin 
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter
import random
import math
import csv
import cv2
class SVM_SMO():
    def __init__(self,maxNumberOfPasses=1000,C=1,tol=0.001,kernel='linear'):
         self.kernelTypes={'quadratic': self.QuadraticKernel,'linear': self.LinearKernel}
         self.c=C
         self.is2D=0
         self.W=None
         self.tol=tol
         self.ThetaO=0
         self.allError=[]
         self.alltheta=[]
         self.alltheta=[]
         self.isTrained=0
         self.allOffsets=[]
         self.kernelType=kernel
         self.maxNumberOfPasses=maxNumberOfPasses
         self.kernelFunction=self.kernelTypes[self.kernelType]        
 ##################################################  Training Function  ################################
    def trainingFunction(self,Y,X):
        n, d = X.shape[0], X.shape[1]                                                    #n: number of data points, d: dimension of each data point
        if d==2: self.is2D=1
        Theta=np.zeros([n])                                                              #initial value for multipliers
        currentNumberOfPasses=0
        isFirstTime=0
        while(currentNumberOfPasses < self.maxNumberOfPasses):
            numberOfOptimiztions=0           
            for i in range(0,n):
                ThetaOld=Theta
                j=i
                while j==i : j=random.randint(0,n-1)                                      #Make sure j!=i
                Xi=np.asarray(X[i])
                Xj=np.asarray(X[j])
                Yi=Y[i]
                Yj=Y[j]
                Thetai=Theta[i]
                Thetaj=Theta[j]                                                           #Save the old values 
                (L,H)=self.CalculateLH(self.c,Thetai,Thetaj,Yi,Yj)                        #Calculate SMO parameters 
                Eta=self.CalculateEta(Xi,Xj)
                if Eta==0: continue
                self.W= self.CalculateW(X,Y,Theta)                                        #Update the weight vector                                                                                              
                self.ThetaO=self.CalculateOffset(X,Y,self.W)                              #Update Offset bias               
                Ei=self.CalculateEn(Xi,Yi,self.W,self.ThetaO)                             #Find the ith and jth Error
                Ej=self.CalculateEn(Xj,Yj,self.W,self.ThetaO)
                Theta[j]= self.UpdateThetaj(Eta,Ei,Ej,Yj,H,L,Thetaj)                      #Update the ith and jth multipliers
                Theta[i]= self.UpdateThetai(Thetai,Thetaj,Theta[j],Yi,Yj)                 
                if np.linalg.norm(ThetaOld-Theta)<self.tol and isFirstTime!=0: continue   #If the optimiztion is insignificant, it doesn't count
                self.alltheta.append(self.W)                                              #keeping track of all weight vectors
                self.allOffsets.append(self.ThetaO)
                if self.is2D==1:                                                          #This was added for graphical representation of 2-d data, remove if your data is more than 2d
                   errorCount=0
                   for item,label in zip(X,Y):
                      if self.F(item,self.W,self.ThetaO)!=label:errorCount+=1                              
                   self.allError.append(errorCount) 
                numberOfOptimiztions+=1
            isFirstTime=1
            if numberOfOptimiztions==0:currentNumberOfPasses+=1
            else:currentNumberOfPasses=0
        self.isTrained=1    
 ################################################ Calculate Weight vector W ############################
    def CalculateW(self,X,Y,Theta):
        return np.asarray(self.kernelFunction(X.T,np.multiply(Theta,Y)))
 ################################################ Calculate Offset bias ThetaO ######################### 
    def CalculateOffset(self,X,Y,W):
        return np.mean(Y-self.kernelFunction(X,W.T))         
 ##################################################### Update Thetai ###################################
    def UpdateThetai(self,Thetai,Thetaj,ThetajUpdated,Yi,Yj):
        return Thetai+Yi*Yj*(Thetaj-ThetajUpdated)
 ##################################################### Update Thetaj ###################################
    def UpdateThetaj(self,Eta,Ei,Ej,Yj,H,L,Thetaj):
        ThetajUpdated=Thetaj+( (Yj*(Ei-Ej)) / Eta )
        if ThetajUpdated<L:return L
        if ThetajUpdated>H:return H
        return ThetajUpdated
 ##################################################### L & H Function ##################################
    def CalculateLH(self,C,Thetai,Thetaj,Yi,Yj):
        if(Yi != Yj): return (max(0,Thetaj-Thetai)  ,  min(C,C-Thetai+Thetaj))
        return               (max(0,Thetai+Thetaj-C),  min(C,Thetai+Thetaj))    
 ###################################################### Eta Function  ##################################
    def CalculateEta(self,Xi,Xj):
       return self.kernelFunction(Xi,Xi)+self.kernelFunction(Xj,Xj)-2*self.kernelFunction(Xi,Xj)
 ##################################################### Output Function #################################
    def F(self,X,W,thetaO):
        return np.sign(self.kernelFunction(X,W)+thetaO)
 ###################################################### Error Function #################################             
    def CalculateEn(self,Xn,Yn,W,ThetaO):
        return self.F(Xn,W,ThetaO)-Yn
 ################################################## Types of Kernel functions ##########################
    def QuadraticKernel(self,Xi,Xj):                                                      #quadratic
        return (np.dot(Xi,Xj)+1)**2
    def LinearKernel(self,Xi,Xj):                                                         #linear
        return np.dot(Xi,Xj)
 ################################################## classify ###########################################
    def classify(self,X):
         return self.F(X,self.W,self.ThetaO)
 #################################### Probability of (Y=1|<W,X>+thetaO) ################################
    def ProbabilityOfOne(self,X):
         return (1+math.exp(-1*(self.kernelFunction(self.W,X)+self.ThetaO)))**-1,self.classify(X)
 ##################################### A function for ploting the data in 2-d only ###########################
    def Decision_boundary_Plot(self,theta,thetaO,X):      
       if self.is2D==0: return
       y=[]  
       plt.figure(figsize=(10,6))
       plt.grid(True)    
       for row in X:
          y.append(np.sign(np.dot(row,theta)+thetaO))
       for X,Y in zip(X,y):
          plt.plot(X[0],X[1],'r*' if (Y == 1.0) else 'b*',)
       slope=-1*(theta[1]/theta[0]) 
       i = np.linspace(-100*np.amin(X),100*np.amax(X))
       z=slope*i+thetaO
       plt.plot(i, z) 
       plt.show()  
 #################################### Plot the error surface for 2-dim only ###################################
    def errorSurfacePlot(alltheta,allError,allOffsets):
       xaxis=[]
       yaxis=[]
       zaxis=np.multiply(allError,1/max(allError))     
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
 ################################################ Get Functions ########################################
    def getW(self):
       return self.W
    def getOffset(self):
       return self.ThetaO
    def getkernelType(self):
       return self.kernelType 
    def getAllError(self):
       return self.allError
    def getAllOffsets(self):
       return self.allOffsets
    def getAllTheta(self):
       return self.alltheta
    def is2D(self):
       return self.is2D    
    def isTrained(self):
       return self.isTrained            
 ################################################### Set Functions #####################################
    def setTol(self,tol):
       self.tol=tol
       self.isTrained=0 
    def setC(self,C):
       self.c=C
       self.isTrained=0
    def setKernel(self,kernelType):
       self.kernelType=kernelType
       self.kernelFunction=self.kernelTypes[kernelType]
       self.isTrained=0                
    def setMaxNumberofPasses(self,maxNumberOfPasses):
       self.maxNumberOfPasses=maxNumberOfPasses
       self.isTrained=0
###################################################### Test Function #################################### 
#The rest of the code is an example on how to use the function to classify
def main():
 ##################### Replace this part with your method of reading data into vectors ##################      
    X, unfilteredLabels = loadlocal_mnist(
        images_path='MNIST/train-images.idx3-ubyte', 
        labels_path='MNIST/train-labels.idx1-ubyte')
    np.multiply(X,1/255)
    X=X+0.0001    
    Y0=[]
    Y1=[]
    Y2=[]
    Y3=[]
    Y4=[]
    Y5=[]
    Y6=[]
    Y7=[]
    Y8=[]
    Y9=[]
    for item in unfilteredLabels:
          if item==0: 
                Y0.append(1)
                Y1.append(-1)
                Y2.append(-1)
                Y3.append(-1)
                Y4.append(-1)
                Y5.append(-1)
                Y6.append(-1)
                Y7.append(-1)
                Y8.append(-1)
                Y9.append(-1)
          elif item==1:
                Y0.append(-1)
                Y1.append(1)
                Y2.append(-1)
                Y3.append(-1)
                Y4.append(-1)
                Y5.append(-1)
                Y6.append(-1)
                Y7.append(-1)
                Y8.append(-1)
                Y9.append(-1)
          elif item==2:
                Y0.append(-1)
                Y1.append(-1)
                Y2.append(1)
                Y3.append(-1)
                Y4.append(-1)
                Y5.append(-1)
                Y6.append(-1)
                Y7.append(-1)
                Y8.append(-1)
                Y9.append(-1)
          elif item==3:
                Y0.append(-1)
                Y1.append(-1)
                Y2.append(-1)
                Y3.append(1)
                Y4.append(-1)
                Y5.append(-1)
                Y6.append(-1)
                Y7.append(-1)
                Y8.append(-1)
                Y9.append(-1)
          elif item==4:
                Y0.append(-1)
                Y1.append(-1)
                Y2.append(-1)
                Y3.append(-1)
                Y4.append(1)
                Y5.append(-1)
                Y6.append(-1)
                Y7.append(-1)
                Y8.append(-1)
                Y9.append(-1)
          elif item==5:
                Y0.append(-1)
                Y1.append(-1)
                Y2.append(-1)
                Y3.append(-1)
                Y4.append(-1)
                Y5.append(1)
                Y6.append(-1)
                Y7.append(-1)
                Y8.append(-1)
                Y9.append(-1)    
          elif item==6:
                Y0.append(-1)
                Y1.append(-1)
                Y2.append(-1)
                Y3.append(-1)
                Y4.append(-1)
                Y5.append(-1)
                Y6.append(1)
                Y7.append(-1)
                Y8.append(-1)
                Y9.append(-1)
          elif item==7:
                Y0.append(-1)
                Y1.append(-1)
                Y2.append(-1)
                Y3.append(-1)
                Y4.append(-1)
                Y5.append(-1)
                Y6.append(-1)
                Y7.append(1)
                Y8.append(-1)
                Y9.append(-1)
          elif item==8:
                Y0.append(-1)
                Y1.append(-1)
                Y2.append(-1)
                Y3.append(-1)
                Y4.append(-1)
                Y5.append(-1)
                Y6.append(-1)
                Y7.append(-1)
                Y8.append(1)
                Y9.append(-1)
          elif item==9:
                Y0.append(-1)
                Y1.append(-1)
                Y2.append(-1)
                Y3.append(-1)
                Y4.append(-1)
                Y5.append(-1)
                Y6.append(-1)
                Y7.append(-1)
                Y8.append(-1)
                Y9.append(1)            
    ##################################### Example on Training The Classifier ###############################  
    classifierFor0=SVM_SMO()                                                              #Example on how to use the training function 
    classifierFor0.setC(100)                                                              #Train 10 classifiers on 10 handwritten numbers
    classifierFor0.setTol(0.001)
    classifierFor0.setKernel('linear')                                                    #A higher order kernel doesn't mean better classifier 
    classifierFor0.setMaxNumberofPasses(100)
    classifierFor0.trainingFunction(np.asarray(Y0),np.asarray(X))                         #Train The Classifier
    w0=classifierFor0.getW()                                                              #Output
    offsetFor0=classifierFor0.getOffset()

    classifierFor1=SVM_SMO()                                                                     
    classifierFor1.setC(100)
    classifierFor1.setTol(0.001)                                                                      
    classifierFor1.setKernel('linear')                                                           
    classifierFor1.setMaxNumberofPasses(100)
    classifierFor1.trainingFunction(np.asarray(Y1),np.asarray(X))                                 
    w1=classifierFor1.getW()                                                                     
    offsetFor1=classifierFor1.getOffset()

    classifierFor2=SVM_SMO()                                                                     
    classifierFor2.setC(100)
    classifierFor2.setTol(0.001)                                                                      
    classifierFor2.setKernel('linear')                                                           
    classifierFor2.setMaxNumberofPasses(100)
    classifierFor2.trainingFunction(np.asarray(Y2),np.asarray(X))                                 
    w2=classifierFor2.getW()                                                                     
    offsetFor2=classifierFor2.getOffset()
    
    classifierFor3=SVM_SMO()                                                                     
    classifierFor3.setC(100)
    classifierFor3.setTol(0.001)                                                                      
    classifierFor3.setKernel('linear')                                                           
    classifierFor3.setMaxNumberofPasses(100)
    classifierFor3.trainingFunction(np.asarray(Y3),np.asarray(X))                                 
    w3=classifierFor3.getW()                                                                     
    offsetFor3=classifierFor3.getOffset()
    
    classifierFor4=SVM_SMO()                                                                     
    classifierFor4.setC(100)
    classifierFor4.setTol(0.001)                                                                      
    classifierFor4.setKernel('linear')                                                           
    classifierFor4.setMaxNumberofPasses(100)
    classifierFor4.trainingFunction(np.asarray(Y4),np.asarray(X))                                 
    w4=classifierFor4.getW()                                                                     
    offsetFor4=classifierFor4.getOffset()

    classifierFor5=SVM_SMO()                                                                     
    classifierFor5.setC(100)
    classifierFor5.setTol(0.001)                                                                      
    classifierFor5.setKernel('linear')                                                           
    classifierFor5.setMaxNumberofPasses(100)
    classifierFor5.trainingFunction(np.asarray(Y5),np.asarray(X))                                 
    w5=classifierFor5.getW()                                                                     
    offsetFor5=classifierFor5.getOffset()

    classifierFor6=SVM_SMO()                                                                     
    classifierFor6.setC(100)
    classifierFor6.setTol(0.001)                                                                      
    classifierFor6.setKernel('linear')                                                           
    classifierFor6.setMaxNumberofPasses(100)
    classifierFor6.trainingFunction(np.asarray(Y6),np.asarray(X))                                 
    w6=classifierFor6.getW()                                                                     
    offsetFor6=classifierFor6.getOffset()

    classifierFor7=SVM_SMO()                                                                     
    classifierFor7.setC(100)
    classifierFor7.setTol(0.001)                                                                      
    classifierFor7.setKernel('linear')                                                           
    classifierFor7.setMaxNumberofPasses(100)
    classifierFor7.trainingFunction(np.asarray(Y7),np.asarray(X))                                 
    w7=classifierFor7.getW()                                                                     
    offsetFor7=classifierFor7.getOffset()

    classifierFor8=SVM_SMO()                                                                     
    classifierFor8.setC(100) 
    classifierFor8.setTol(0.001)                                                                     
    classifierFor8.setKernel('linear')                                                           
    classifierFor8.setMaxNumberofPasses(100)
    classifierFor8.trainingFunction(np.asarray(Y8),np.asarray(X))                                 
    w8=classifierFor8.getW()                                                                     
    offsetFor8=classifierFor8.getOffset()

    classifierFor9=SVM_SMO()                                                                     
    classifierFor9.setC(100)
    classifierFor9.setTol(0.001)                                                                      
    classifierFor9.setKernel('linear')                                                           
    classifierFor9.setMaxNumberofPasses(100)
    classifierFor9.trainingFunction(np.asarray(Y9),np.asarray(X))                                 
    w9=classifierFor9.getW()                                                                     
    offsetFor9=classifierFor9.getOffset() 

    workbook = xlsxwriter.Workbook('W0.xlsx') 
    worksheet = workbook.add_worksheet()   
    row = 0
    column = 0 
    for item in w0 : 
       worksheet.write(row, column, item) 
       row += 1
    worksheet.write(0,1,offsetFor0)         
    workbook.close() 

    workbook = xlsxwriter.Workbook('W1.xlsx') 
    worksheet = workbook.add_worksheet()   
    row = 0
    column = 0 
    for item in w1 : 
       worksheet.write(row, column, item) 
       row += 1
    worksheet.write(0,1,offsetFor1)         
    workbook.close()

    workbook = xlsxwriter.Workbook('W2.xlsx') 
    worksheet = workbook.add_worksheet()   
    row = 0
    column = 0 
    for item in w2 : 
       worksheet.write(row, column, item) 
       row += 1
    worksheet.write(0,1,offsetFor2)         
    workbook.close()
    
    workbook = xlsxwriter.Workbook('W3.xlsx') 
    worksheet = workbook.add_worksheet()   
    row = 0
    column = 0 
    for item in w3 : 
       worksheet.write(row, column, item) 
       row += 1
    worksheet.write(0,1,offsetFor3)         
    workbook.close()

    workbook = xlsxwriter.Workbook('W4.xlsx') 
    worksheet = workbook.add_worksheet()   
    row = 0
    column = 0 
    for item in w4 : 
       worksheet.write(row, column, item) 
       row += 1
    worksheet.write(0,1,offsetFor4)         
    workbook.close()

    workbook = xlsxwriter.Workbook('W5.xlsx') 
    worksheet = workbook.add_worksheet()   
    row = 0
    column = 0 
    for item in w5 : 
       worksheet.write(row, column, item) 
       row += 1
    worksheet.write(0,1,offsetFor5)         
    workbook.close()
    
    workbook = xlsxwriter.Workbook('W6.xlsx') 
    worksheet = workbook.add_worksheet()   
    row = 0
    column = 0 
    for item in w6 : 
       worksheet.write(row, column, item) 
       row += 1
    worksheet.write(0,1,offsetFor6)         
    workbook.close()

    workbook = xlsxwriter.Workbook('W7.xlsx') 
    worksheet = workbook.add_worksheet()   
    row = 0
    column = 0 
    for item in w7 : 
       worksheet.write(row, column, item) 
       row += 1
    worksheet.write(0,1,offsetFor7)         
    workbook.close()    

    workbook = xlsxwriter.Workbook('W8.xlsx') 
    worksheet = workbook.add_worksheet()   
    row = 0
    column = 0 
    for item in w8 : 
       worksheet.write(row, column, item) 
       row += 1
    worksheet.write(0,1,offsetFor8)         
    workbook.close()

    workbook = xlsxwriter.Workbook('W9.xlsx') 
    worksheet = workbook.add_worksheet()   
    row = 0
    column = 0 
    for item in w9 : 
       worksheet.write(row, column, item) 
       row += 1
    worksheet.write(0,1,offsetFor9)         
    workbook.close()    
if __name__ == "__main__":                                                                
    main()        