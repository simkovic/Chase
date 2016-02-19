#from ipyparallel import Client
#rc = Client(profile='chase')
#dview= rc[:]

import pickle
def svmTrain(args,fn='svm',kernelm=0,subsample=0.5,ncpu=8,ncv=5):
    data,kernelvar,penalty=args
    from commands import getstatusoutput
    import numpy as np
    from sklearn.svm import SVC
    #remove nans
    data=data[np.isnan(data).sum(1)==0,:]
    X=data[:,1:];y=data[:,0];N=X.shape[0]
    #whitening
    X-=np.array(X.mean(0),ndmin=2)
    X/=2*np.array(X.std(0),ndmin=2)
    # randomize
    inds=np.random.permutation(N)
    X=X[inds,:];y=y[inds]
    # subsample
    if subsample<1:
        N=int(N*subsample)
        X=X[:N,:]; y=y[:N]
    print 'compute kernel'
    kv=np.round(kernelvar,3)
    S=np.zeros((N,N),order='C')
    for row in range(N):
        D=np.atleast_2d(X[row,:])-X
        D=np.square(D).sum(1)
        S[row,:]=np.exp(-np.exp(kernelvar)*(D-kernelm))
    del D
    print 'train','kernelvar=',kv
    out=[]
    for r in range(ncv):
        spl=int(N-N/float(ncv))
        S1=np.copy(S[:spl,:spl]);S2=np.copy(S[spl:,:spl])
        for p in penalty:
            if not r: out.append([kernelvar,p])
            

            svm=SVC(C=np.exp(p),cache_size=12000,verbose=True,kernel='precomputed')
            svm.fit(S1,y[:spl])
            res=svm.predict(S2)
            acc=np.abs(res-y[spl:]).mean()
            out[-1].append(acc)
        print acc
        bla
    return out
if __name__ == '__main__':
    import numpy as np
    kernelvars=np.linspace(-10,10,21)
    penalty=np.linspace(-10,10,21)
    for vp in range(1,5):
        fargs=[] 
        path='/home/matus/Desktop/chase/evaluation/vp%03d/'%vp
        f=open(path+'preds.pickle','r')
        d=pickle.load(f)
        f.close()
        d=np.concatenate(d)
        assert len(filter(lambda x: len(x)!=38,d))==0
        assert np.all(d[:,0]==vp)
        d=d[:,1:]
        for kv in kernelvars:
            fargs.append([d,kv,penalty])
        out=map(svmTrain,fargs) 

