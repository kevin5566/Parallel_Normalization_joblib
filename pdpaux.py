import numpy as np
from joblib import Parallel, delayed
from math import sqrt

def readdata(f):
    tmp=[]
    X=[]
    Xf=np.zeros((17,1))
    index=0
    
    for line in f:
        index=index+1
        if (index % 18)==11:
            continue
        tmp=line.strip().split(',')
        del tmp[0]
        del tmp[0]
        del tmp[0]
        tmp=[float(i) for i in tmp]
        X.append(tmp)
        if (index % 18)==0:
            X=np.asarray(X)
            Xf=np.column_stack((Xf,X))
            X=[]

    Xf=Xf[:,1:]
    return Xf

def calmean(i,X,p,q):
    x_mean=np.zeros((p,1))				#overhead
    for j in range(q):
        x_mean[i,0]=x_mean[i,0]+X[i,j]
    return x_mean[:,0]

def calsd(i,X,x_mean,p,q):
    x_sd=np.zeros((p,1))				#overhead
    for j in range(q):
        x_sd[i,0]=x_sd[i,0]+(X[i,j]-x_mean[i,0])*(X[i,j]-x_mean[i,0])
    return x_sd[:,0]

def processNormal(i,X,x_mean,x_sd,p,q):
    x_tmp=np.zeros((1,q))				#overhead
    for j in range(0,q):
        x_tmp[0,j]=(X[i,j]-x_mean[i,0])/x_sd[i,0]
    return x_tmp

def normalizeDataParallel(X,core):
    (p,q)=X.shape						##non-parallel part
    x_mean=np.zeros((p,1))				##non-parallel part
    x_sd=np.zeros((p,1))				##non-parallel part

    x_mean=x_mean+Parallel(n_jobs=core)( 			##parallel part
        delayed(calmean)(i,X,p,q)		 			##parallel part
        for i in range(p))				 			##parallel part
    x_mean=np.diag(x_mean) 			##overhead
    x_mean.shape=(p,1)	   			##overhead
	
    for i in range(p):					##non-parallel part
        x_mean[i,0]=x_mean[i,0]/q		##non-parallel part

    x_sd=x_sd+Parallel(n_jobs=core)(	 			##parallel part
        delayed(calsd)(i,X,x_mean,p,q)	 			##parallel part
        for i in range(p))				 			##parallel part
    x_sd=np.diag(x_sd)     			##overhead
    x_sd.shape=(p,1)       			##overhead
	
    for i in range(p):					##non-parallel part
        x_sd[i,0]=x_sd[i,0]/q			##non-parallel part
        x_sd[i,0]=sqrt(x_sd[i,0])		##non-parallel part

    tmp=Parallel(n_jobs=core)(			 			##parallel part
        delayed(processNormal)(i,X,x_mean,x_sd,p,q) ##parallel part
        for i in range(p))				 			##parallel part
    X=np.zeros((1,q))	   	        ##overhead
    for i in range(len(tmp)):	    ##overhead
        X=np.row_stack((X,tmp[i]))  ##overhead
    X=X[1:,:]						##overhead

    return X, x_mean, x_sd

def normalizeData(X):
    (p,q)=X.shape
    x_mean=np.zeros((p,1))
    x_sd=np.zeros((p,1))
    
    for i in range(p):
        for j in range(q):
            x_mean[i,0]=x_mean[i,0]+X[i,j]

    for i in range(p):
        x_mean[i,0]=x_mean[i,0]/q
    
    for i in range(p):
        for j in range(q):
            x_sd[i,0]=x_sd[i,0]+(X[i,j]-x_mean[i,0])*(X[i,j]-x_mean[i,0])

    for i in range(p):
        x_sd[i,0]=x_sd[i,0]/q
        x_sd[i,0]=sqrt(x_sd[i,0])

    for i in range(0,p):
        for j in range(0,q):
            X[i,j]=X[i,j]-x_mean[i,0]
            X[i,j]=X[i,j]/x_sd[i,0]

    return X, x_mean, x_sd

