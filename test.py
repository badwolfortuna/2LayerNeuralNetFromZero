import numpy as np
#import denemee as den
import thirdtry as thr
##***train edilecek neural network ile test edilecek dataset uzunlugu aynı olmalı(bias weightlerden dolayı)
#import denemee as dn
f = open(r'C:\Users\ASUS\Desktop\test-data\ann-test1.txt')#neural networkün çalıstıgını göstermek için aynı set kullanıldı

line = f.readline()
dataarray = []
i=0
for line in f:
    dataarray.append(line)
      # print(dataarray[i])3771
    i+=1
   # print(i)
f.close()
####################################################
inputarray=[]
labelarray=[] 
labelmat=[]
setlen =30 #test datasının uzunlugu
hiddenlayer=thr.hiddenlay
#np.random.seed(42)
#weighthid = np.random.rand(21,5)
#weightout = np.random.rand(5,3)









#lamda=0.01  #learningrate

testweight1=np.array(thr.nweighthid)


testweight2=np.array(thr.nwweightout)
bias1=np.array(thr.nwbiashid, order='C')#resize biases
bias1.resize((setlen,hiddenlayer))

bias2=np.array(thr.nwbiasout, order='C')
print(bias2.shape)
bias2.resize((3,setlen))
print(bias2.shape)

#print(testweight2)
####################################################
labelrow=[]
for y in range(0,setlen):
    A=[]
    for z in range(0,3):
        A.append(0)
    labelrow.append(A)  

def inputarr():
    for i in range(setlen):

        
        datarowarray=[]
       
        datarow =dataarray[i].split() 
        m=0
        for temp in datarow:
            datarowarray.append(float(temp))    
          
            m+=1
            if(m==21):
                break
            
        
        inputarray.append(datarowarray)

    return inputarray


def inputlabel ():#bu sütuna kadar tüm satırları döndürür
    for i in range(setlen):#ilk 10 sütunun labellarını döndüren array 
#3371
        
        datarowarray=[]
       
        datarow =dataarray[i].split() #her bir hücre 
        
        for temp in datarow:
            datarowarray.append(float(temp))    
          
           # inputs= np.array(datarowarray)
       # print(datarowarray)
       #sadece satırları yazıyor 
        
        labelarray.append(datarowarray[21])#for normalizing 

    return labelarray

inputarr()
inputlabel()
#print("labelarray",labelarray)
#print("inputarray",inputarray)

def labelmatrix():
    x=0
    for i in range (len(labelarray)):
        if(round(labelarray[i])==1):
            labelrow[x][0]=1
        if(round(labelarray[i])==2):
            labelrow[x][1]=1
        if(round(labelarray[i])==3):
            labelrow[x][2]=1
            #print(labelrow[x])
        x+=1 
    labelmat.append(labelrow)
    return labelmat

labelmatrix()
#######################################################
npinput=np.array(inputarray)
nplabel=np.array(labelrow)





def sigmoid(x):
    return np.power((np.add(1,np.exp(-x))),(-1))

ZH = np.dot(testweight1,npinput.T)+bias1.T
zh=sigmoid(ZH)
#print(zh.shape)



ZO=np.dot(testweight2,zh)+bias2
zo=sigmoid(ZO)
print("realclasses","\n",nplabel)
#print("predicted value",zo.T)

           


for i in range (setlen):
    if(np.round(zo.T.max(axis=1)[i])==nplabel[i][0]):
        print("predicted class1")
    if(np.round(zo.T.max(axis=1)[i])==nplabel[i][1]):
        print("predicted class2")
    if(np.round(zo.T.max(axis=1)[i])==nplabel[i][2]):
        print("predicted class3")
    




"""
for i in range (setlen):
   x=np.abs(np.subtract(1,zo[i][0]))
    y=np.abs(np.subtract(1,zo[i][1]))
    z=np.abs(np.subtract(1,zo[i][2]))
    m=min(x,y,z)

    #if(min(x,y,z)==x):
        #print("1")
    #if(min(x,y,z)==y):
        #print("2")
    #if(min(x,y,z)==z):
        #print("3")

"""



