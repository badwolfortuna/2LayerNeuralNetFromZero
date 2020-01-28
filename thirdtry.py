import numpy as np
f = open(r'C:\Users\ASUS\Desktop\Neural network\ann-train1.txt')

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
setlen =3000
np.random.seed(42)
hiddenlay=5
weighthid = np.random.rand(hiddenlay,21)
weightout = np.random.rand(3,hiddenlay)
lamda=10  #learningrate





biasrow=[]
for y in range(0,setlen):
    A=[]
    for z in range(0,hiddenlay):
        A.append(1)
    biasrow.append(A)  
biasrowout=[]
for y in range(0,setlen):
    A=[]
    for z in range(0,3):
        A.append(1)
    biasrowout.append(A)  

biashid=np.array(biasrow)
biasoutt=np.array(biasrowout)
biasout=biasoutt.T
#print(biasout.shape)

#print(biashid)
#print(biasout)
mout=setlen*3


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

#######################################################
def sigmoid(x):
    return np.power((np.add(1,np.exp(-x))),(-1))
def sigmoder(x):
    return np.multiply( sigmoid(x),(np.subtract(1,sigmoid(x))))
def meansquare (gerçek,rastgele):
    return  np.square(np.subtract(gerçek, rastgele)).mean()
def logistic(gerçek,rastgele):
    return -np.mean(np.multiply(gerçek,np.log(rastgele))+np.multiply(np.subtract(1,gerçek),np.log(np.subtract(1,rastgele))))






for i in range(5):
#########################################Feedforward
    ZH = np.dot(weighthid,npinput.T)+biashid.T#hiddenweight[hiddenlay,21].input[10,21] biass=[setlen,hiddenlay]
    zh=sigmoid(ZH)
    #print("zhshape",zh.shape)
    ZO=np.dot(weightout,zh)+biasout #ouputweight[3,5].zh[5,10]+biasoutput[3,10]
    zo=sigmoid(ZO)#[3,10]son katman prediction değer
#print(zo)
########################################Feedforward
    labelT=nplabel.T
#print(labelT)
    #print(meansquare(zo,labelT))
##################################################### Error calculation
    print("cost in ",i,"epoc",logistic(labelT,zo))#loss fonksiyonu
    error=(zo-labelT)##normalde labelT-zo ama -1/m deki - isareti konulmasın diye böyle yazıldı
    errordagılım=error/mout
    errorsigmo=sigmoder(error) #son katmanda errorun nekadarının yansıtılıcagı rate
    costhidden=np.multiply(errorsigmo,errordagılım)##0.37 #yansıtılacak error
#####################################################Error calculation
#####################################################Outputbiasdeltacalculation
   # print("shshape",costhidden.shape)
    biascostrate=np.multiply(costhidden,biasout)##
   # print("biascostrate",biascostrate.shape)
    biasrowsumx=np.sum(biascostrate, axis=1)
   # print("biasrowsum",biasrowsumx)
    biasoutrowsum=biasrowsumx.T #(1,3)olmalı
    #print(biasoutrowsum)
    somethingout=[]
    for i in range(setlen):
        somethingout.append(biasoutrowsum)
    
    somethingoutbias=np.array(somethingout)
    somethingoutbiasT=somethingoutbias.T
    #print(somethingoutbiasT)## outputbiasta kullanılcakbiascost
    #print(biasout.shape)
##################################################### Outputbiasdeltacalculation
##################################################### Outputlayerweightlerin update edilmesi    
    weightoutrate=np.dot(costhidden,zh.T)##3,3 bu 0,024 outputlayerweightleri update ederken kullanılcak cost
##################################################### Outputlayerweightlerin update edilmesi 
#print(weightout.T.shape)
#print(costhidden.shape)
    costweight=np.dot(weightout.T,costhidden) #0,004 upside gradient
    hiddensigmo=sigmoder(zh)#0.25
    hiddenz=np.multiply(costweight,hiddensigmo)#0.001
    costweight1=np.dot(hiddenz,npinput)##ilklayerweightleriupdateederken kullanılcak cost
#print(costweight1)


    costbiashid=np.dot(hiddenz,biashid)
    costbiashidsum=np.sum(costbiashid,axis=1) #0.02,0.04,0.07



    somethinghid=[]
    for i in range(setlen):
        somethinghid.append(costbiashidsum)
    
    somethinghidbias=np.array(somethinghid)#hidden biasları update ederken kullanılcak biascost


######################################################### WEİGHT UPDATE
    weightout=np.subtract(weightout,np.multiply(weightoutrate,lamda))

    biasout=np.subtract(biasout,np.multiply(somethingoutbiasT,lamda))

    weighthid=np.subtract(weighthid,np.multiply(costweight1,lamda))

    biashid=np.subtract(biashid,np.multiply(somethinghidbias,lamda))

######################################################### WEİGHT UPDATE
    

nwweightout=weightout#en son çıkan weightler test sınıfında kullanılacaklar 
#print("nwweightout",nwweightout.shape)
nwbiasout=biasout
#print("nwbiasout",nwbiasout.shape)
nweighthid=weighthid
#print("nweighthid",nweighthid.shape)
nwbiashid=biashid#en son test sınıfında kullanılacak weight 
#print("nwbiashid",nwbiashid.shape)
#print("biashid",biashid.shape)
#print("inputshape",npinput.shape)
#print("weightoutshape",weightout.shape)
#print("biasoutput",biasout.shape)

















