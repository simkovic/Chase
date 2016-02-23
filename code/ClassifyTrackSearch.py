from ipyparallel import Client
rc = Client(profile='chase')
dview= rc[:]
print 'number of detected engines:',len(dview)
import pickle
import numpy as np
path='/home/matus/Desktop/chase/evaluation/'


def svmPreprocessData(data,vp,suf,subsample=1):
    # make list to ndarray
    data=np.concatenate(data)
    assert len(filter(lambda x: len(x)!=39,data))==0
    assert np.all(data[:,0]==vp)
    data=data[:,1:];# discard vp id
    # remove nans/missing values
    #data=np.concatenate([data,np.roll(data[:,2:],1,axis=0)],axis=1)
    sel=np.isnan(data).sum(1)==0
    print 'vp %d proportion of removed samples (mis. values):%.3f'%(vp,1-sel.mean())
    data=data[sel,:]
    data=data[data[:,0]==np.roll(data[:,0],1),:]
    X=data[:,2:];y=data[:,1];N=X.shape[0]  
    # whitening
    X-=np.array(X.mean(0),ndmin=2)
    X/=2*np.array(X.std(0),ndmin=2)
    # randomize
    inds=np.random.permutation(N)
    X=X[inds,:];y=y[inds]
    # subsample
    N=int(N*subsample)
    X=X[:N,:]; y=y[:N]
    print 'vp %d nr of samples ',N 
    return X,y
    
def svmTrain(args,kernelm=0,ncv=5):
    X,y,kernelvar,penalty=args
    from commands import getstatusoutput
    import numpy as np
    from sklearn.svm import SVC
    N=X.shape[0];N-=N%ncv
    X=X[:N,:]; y=y[:N]
    print 'compute kernel, kernelvar=',np.round(kernelvar,3)
    assert N%ncv==0
    spl=int(N/float(ncv))
    assert spl*ncv==N
    S1=np.zeros((N-spl,N-spl),order='C')*np.nan
    S2=np.zeros((spl,N),order='C')*np.nan
    for row in range(N):
        D=np.atleast_2d(X[row,:])-X
        D=np.square(D).sum(1)
        temp=np.exp(-np.exp(kernelvar)*(D-kernelm))
        assert not np.any(np.isnan(temp))
        assert not np.any(temp==np.inf)
        if row>=spl:S1[row-spl,:]=temp[spl:]
        else:S2[row,:]=temp
    y1=y[spl:];y2=y[:spl]
    del D
    print 'train, kernelvar=',np.round(kernelvar,3)
    out=[]
    for r in range(ncv):
        if r:
            temp=np.copy(S2)
            for k in range(1,ncv):
                if k==r:
                    temp[:,:spl]=S1[spl*(r-1):spl*(r),spl*(k-1):spl*(k)]
                    S1[spl*(r-1):spl*(r),spl*(k-1):spl*(k)]=S2[:,:spl]
                else:
                    temp[:,spl*k:spl*(k+1)]=S1[spl*(r-1):spl*(r),spl*(k-1):spl*(k)]
                    S1[spl*(r-1):spl*(r),spl*(k-1):spl*(k)]=S2[:,spl*k:spl*(k+1)]
                    S1[spl*(k-1):spl*(k),spl*(r-1):spl*(r)]=S2[:,spl*k:spl*(k+1)]
            S2=temp
            tempy=np.copy(y1)
            y1[spl*(r-1):spl*(r)]=y2
            y2=tempy[spl*(r-1):spl*(r)]
        for pi in range(len(penalty)): 
            if not r: out.append([kernelvar,penalty[pi]])
            svm=SVC(C=np.exp(penalty[pi]),cache_size=12000,verbose=False,
                kernel='precomputed',max_iter=1e4)
            svm.fit(S1,y1)
            res=svm.predict(S2[:,spl:])
            acc=np.abs(res-y2).mean()
            out[pi].append(1-acc)
    return out
def gridSearchEach(suf=''):  
    kernelvars=np.linspace(-20,0,21)
    penalty=np.linspace(-10,5,21)
    fargs=[];vpids=[]
    #pool=Pool(4)
    for vp in range(1,5):
        f=open(path+'vp%03d/preds%s.pickle'%(vp,suf),'r')
        d=pickle.load(f)
        f.close()
        X,y=svmPreprocessData(d,vp,suf)
        for kv in kernelvars:
            fargs.append([X,y,kv,penalty])
            vpids.append(vp)
    bla
    out=dview.map(svmTrain,fargs,block=True)
    #out=map(svmTrain,fargs)
    for vp in range(1,5):
        temp=[]
        for oi in range(len(out)):
            if vpids[oi]==vp: temp.append(out[oi])
        np.save(path+'vp%03d/svm%s.npy'%(vp,suf),temp) 
 
def gridSearchAll(N,suf=''):
    ''' TODO '''  
    kernelvars=np.linspace(-15,5,21)
    penalty=np.linspace(-5,10,21)
    #pool=Pool(4)
    d=[];vpn=range(1,5)
    for vp in vpn:
        f=open(path+'vp%03d/preds%s.pickle'%(vp,suf),'r')
        dd=pickle.load(f)
        f.close()
        dd=np.concatenate(dd)
        dd=dd[np.random.permutation(dd.shape[0])[:N/len(vpn)],:]
        assert len(filter(lambda x: len(x)!=38,dd))==0
        assert np.all(dd[:,0]==vp)
        dd=dd[:,1:];
        d.append(dd)
    d=np.concatenate(d)
    
    fargs=[]
    for kv in kernelvars:
        fargs.append([d,kv,penalty])
    #out=dview.map(svmTrain,fargs,block=True)
    out=map(svmTrain,fargs)
    path='/home/matus/Desktop/chase/evaluation/vpall/'
    np.save(path+'svm%s.npy'%s,out)   
    
if __name__ == '__main__':
    gridSearchEach(suf='MA')
    #gridSearchAll(N=30000)
