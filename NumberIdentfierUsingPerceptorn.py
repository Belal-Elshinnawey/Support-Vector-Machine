from PIL import Image
import numpy as np
from scipy import misc
from pandas import *
import cv2
import csv
import xlsxwriter
import math  
#Note: This model is just for illustration. If stronger results are required, The training set size must be increased
#Training Functions
def SVMTrain(X,Y,gamma):    
 Totalerror=[]
 n=np.shape(X)[0]
 d=np.shape(X)[1]
 theta=np.zeros([1,d])[0]
 k=0
 firstItr=0
 error=1
 while error!=0:
  i=0     
  for row in X:
   mu= Y[i]*np.dot(row,theta)
   if mu<gamma or firstItr==0:
    k+=1
    firstItr=1     
    theta=theta+Y[i]*row
    errorIndx=0
    error=0
    for item in X:
     yt=np.copysign(1,np.dot(theta,item))  
     if Y[errorIndx]!=yt:
      error+=1
     errorIndx+=1 
   Totalerror.append(error)  
   i+=1
 return theta, k, Totalerror, gamma

def PerceptronTrain(X, Y):
 Totalerror=[]     
 n=np.shape(X)[0]
 d=np.shape(X)[1]
 theta=np.zeros([1,d])[0]
 k=0
 firstItr=0
 error=1
 while error!=0:
  i=0     
  for row in X:
   mu= Y[i]*np.dot(row,theta)
   if mu<0 or firstItr==0:
    k+=1
    firstItr=1     
    theta=theta+Y[i]*row
    errorIndx=0
    error=0
    for item in X:
     yt=np.copysign(1,np.dot(theta,item))  
     if Y[errorIndx]!=yt:
      error+=1
     errorIndx+=1 
   Totalerror.append(error)  
   if error==0:
     break
   i+=1
  gamma=0
  j=0
  for item in X:
    psy=abs(np.dot(item,theta)/np.linalg.norm(theta))
    j+=1    
    if psy<gamma or gamma==0:
     gamma=psy    
 return SVMTrain(X,Y,gamma)   
    
#Reading 350 images each one is [28x28] from the file trainSample
#Mapping the read images into a variable 'X'[350,784] 
#Each row in X is one image converted to dim of [1,784]
i=0
x=np.zeros([350,784])
while i<350:
 img = np.asarray(cv2.imread(r"Numbers\trainSample\img_"+str(i+1)+".jpg",0))
 k=0
 for elm in img:
  for item in elm:    
   x[i][k]=item
   k=k+1
 i=i+1

#write all the values to an xlsx file
#This part was used during test and debug & can be removed
workbook = xlsxwriter.Workbook('ImageMat.xlsx') 
worksheet = workbook.add_worksheet() 
row = 0
column = 0 
for item in x :
 column=0     
 for elm in item:   
  worksheet.write(row, column, elm) 
  column += 1 
 row+=1   
workbook.close()

# Create 10 identifiers each on corresponds to a number in [0-9]
# Each iden. is a row in theta
i=0
targetsi=[]
y=[]
while i<=9:
 #Load the labels corresponding to each number from the file TrainingSample_fori where i:[0-9]     
 with open('Numbers\TargetLists\TrainingSample_for'+str(i)+'.txt') as file:
  tempData = file.readlines()
  Reader = csv.reader(tempData)
  for item in Reader:
   targetsi.append(float(item[0]))    
  y.append(targetsi)
  targetsi=[]
 i+=1
theta=[]
gamma=[]
for row in y:
 output=PerceptronTrain(x, row)    
 theta.append(output[0])
 gamma.append(output[3])

# Test the trained vectors in theta with any image in the file: testSet
img = np.asarray(cv2.imread(r"Numbers\testSet\img_485.jpg",0)) #Change the number in img_#.jpg to test an image from the file testSet 
testInput=[] 
probOutput=[]
for elm in img:
 for item in elm:
   testInput.append(item)
for identifier,margin in zip(theta,gamma):
 probOutput.append(math.acos(np.dot(identifier,testInput)/(np.linalg.norm(identifier)*np.linalg.norm(testInput))))
print('The Number is '+str(probOutput.index(min(probOutput))))



   
