import numpy as np
import os
import pylab as plt
from scipy.stats import scoreatpercentile as sap
plt.ion()

def dump(stuff,labels,fn='test.out'):
    f = open(fn,'w')
    for i in range(len(stuff)):
        if np.array(stuff[i]).ndim==0:
            f.write(labels[i]+' <- '+str(stuff[i])+'\n')
        elif np.array(stuff[i]).ndim==1:
            f.write(labels[i]+' <-\nc(')
            for j in range(stuff[i].shape[0]-1):
                if np.isnan(stuff[i][j]): f.write('NA, ')
                else: f.write(str(stuff[i][j])+', ')
            if np.isnan(stuff[i][-1]): f.write('NA)\n')
            else: f.write(str(stuff[i][-1])+')\n')
        else:
            f.write(labels[i]+' <-\nstructure(c(')
            for j in range(stuff[i].shape[1]):
                for k in range(stuff[i].shape[0]):
                    if k+1==stuff[i].shape[0] and j+1==stuff[i].shape[1]: suf=')'
                    else: suf=', '
                    if np.isnan(stuff[i][k][j]): f.write('NA'+suf)
                    else: f.write(str(stuff[i][k][j])+suf)
                if j+1==stuff[i].shape[1]: f.write(', .Dim=c(%dL, %dL))\n'%np.array(stuff[i]).shape)
                else: f.write('\n')
           
            
    f.close()

def loadCoda(pname=''):
    def getIndex(a):
        if (a.count('[')>0 and a.count(',')==0):
            i=int(a.rstrip(']').rsplit('[')[1]); j=-1
        elif a.count('[')>0 and a.count(',')>0:
            i,j = a.rstrip(']').rsplit('[')[1].rsplit(',')
            i=int(i); j=int(j)
        else: i=-1;j=-1
        return i,j
    def reorder(indices):
        inds=[]
        i=0
        while i < indices.shape[0]:
            if indices[i,0]==-1 and indices[i,1]==-1:
                inds.append(indices[i,[2,3]]);i+=1
            elif indices[i,1]==-1:
                #print i, indices[i,[2,3]]
                temp=[indices[i,[2,3]]]
                i+=1
                while i < indices.shape[0] and indices[i,0]!=1 and indices[i,0]!=-1 and i < indices.shape[0]:
                    #print i, indices[i,[2,3]]
                    temp.append(indices[i,[2,3]])
                    i+=1
                inds.append(np.array(temp).T);
            else:
                k=0
                #print k,i, indices[i,[2,3]]
                temp=[[indices[i,[2,3]]]];i+=1
                while i < indices.shape[0] and indices[i,0]!=1 and indices[i,0]!=-1:
                    #print k,i, indices[i,[2,3]]
                    temp[0].append(indices[i,[2,3]]); i+=1
                k+=1
                while i < indices.shape[0] and indices[i,1]!=1 and indices[i,1]!=-1:
                    #print k,i, indices[i,[2,3]]
                    temp.append([indices[i,[2,3]]]);i+=1
                    while i < indices.shape[0] and indices[i,0]!=1 and indices[i,0]!=-1:
                        #print k,i, indices[i,[2,3]]
                        temp[k].append(indices[i,[2,3]]); i+=1
                    k+=1
                inds.append(np.array(temp).T);
                #bla
        return inds
    chain=1
    f=open('%sCODAindex.txt'%pname,'r')
    inds=f.readlines()
    f.close()
    indices=[]
    for ind in inds:
        a,b,c=ind.rstrip('\n').rsplit(' ')
        i,j=getIndex(a)
        indices.append([i,j,int(b),int(c)])
    indices=np.array(indices)
    T=indices[0,3]-indices[0,2]+1
    indices=reorder(indices)
    #print indices
    f=open('%sCODAchain%d.txt'%(pname,chain),'r')
    D=[]
    i=0
    for m in indices:   
        if m.ndim==1:
            data=np.ones(T)
            for k in range(data.size):
                data[k]=float(f.readline().rstrip('\n').rsplit(' ')[2])
                i+=1
        elif m.ndim==2:
            data=np.ones((m.shape[1],T))
            for g in range(data.shape[0]):
                #print i, m[:,g]
                for k in range(data.shape[1]):
                    data[g,k]=float(f.readline().rstrip('\n').rsplit(' ')[2])
                    i+=1
        elif m.ndim==3:
            data=np.ones((m.shape[1],m.shape[2],T))
            for g in range(data.shape[1]):
                for h in range(data.shape[0]):
                    #print i, m[:,h,g]
                    for k in range(data.shape[2]):
                        data[h,g,k]=float(f.readline().rstrip('\n').rsplit(' ')[2])
                        i+=1
        D.append(data)
    return D

def jags(pname,indata,indatlabels,outdatlabels,modelSpec,inpars=[],inparlabels=[],
         nrChains=1,burnIn=5000,chainLen=15000, thin=1):
    if len(inpars)>0:
        dump(inpars,inparlabels,fn=pname+'.inpar')
    #print 'bla',indata[1].shape
    dump(indata,indatlabels,fn=pname+'.indat')
    #bla
    f=open(pname+'.bug','w')
    f.write(modelSpec)
    f.close()
    ss=''
    for s in outdatlabels:
        ss+= 'monitor '+s+', thin('+str(thin)+')\n'
        
    script='''model in "%s.bug"
    data in "%s.indat"
    compile, nchains(%d)
    '''%(pname,pname,nrChains)
    if len(inpars)>0: script+='parameters in "%s.inpar"\n'%pname
    script+='''initialize
    update %d
    %supdate %d
    coda *
    '''%(burnIn,ss, chainLen)

    f=open(pname+'.script','w')
    f.write(script)
    f.close()
    nfo= os.system('jags '+pname+'.script')
    print nfo
    if nfo!=256:return loadCoda()

def plotNode(data,thin=1,burn=0):
    plt.close('all')
    data=data[range(burn,data.size,thin)]
    plt.subplot(2,1,1)
    plt.plot(data)
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.subplot(2,2,3)
    plt.acorr(data,maxlags=data.size/2)
    plt.xlabel('lag')
    plt.ylabel('autocorrelation')
    plt.subplot(2,2,4)
    plt.hist(data,min(data.size/30,40),range=[sap(data,1), sap(data,99)])

    m=data.mean()
    ax=plt.gca()
    plt.plot([m,m],ax.get_ylim(),'k--')
    m=sap(data,5)
    plt.plot([m,m],ax.get_ylim(),'k--')
    m=sap(data,95)
    plt.plot([m,m],ax.get_ylim(),'k--')
    
    
    

#D=loadCoda()
#for d in D:
#    print d.shape
