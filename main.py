import pdpaux
import numpy as np
import time

f=open('./data.csv','r')
f.readline()
X=pdpaux.readdata(f)         #(17, 5760)
f.close()
X=np.column_stack((X,X))
X=np.column_stack((X,X))
X=np.column_stack((X,X))
X=np.column_stack((X,X))
Y=X                       #(17, 92160)
(p,q)=X.shape

print 'data dimension:\t',
print X.shape

t1=time.time()
(x1,x2,x3)=pdpaux.normalizeDataParallel(X,4)
t2=time.time()-t1

t3=time.time()
(y1,y2,y3)=pdpaux.normalizeData(Y)
t4=time.time()-t3

print 'Parallel execution time:\t',
print t2
print 'Non-Parallel execution time:\t',
print t4

print 'result checking:\t\t',
if np.asarray(np.where(x1-x2)).shape==(2, p*q):
    print 'pass'
else:
    print 'fail'

#python -W ignore main.py