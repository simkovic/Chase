#from ipyparallel import Client
#import numpy as np
#rc = Client(profile='chase')
#dview= rc[:]

import pickle
def svmTrain(args,fn='svm',kernelm=0,subsample=0.01,ncpu=8):
    data,kernelvar,penalty=args
    from commands import getstatusoutput
    import numpy as np
    #remove nans
    data=data[np.isnan(data).sum(1)==0,:]
    X=data[:,1:];y=data[:,0];N=X.shape[0]
    #whitening
    X-=np.array(X.mean(0),ndmin=2)
    X/=2*np.array(X.std(0),ndmin=2)
    if subsample<1:
        r=np.random.permutation(N)
        N=int(N*subsample)
        sel=r[:N]
        X=X[sel,:]; y=y[sel]
    print 'export'
    kv=np.round(kernelvar,3)
    f=open(fn+'%.3f.in'%kv,'w')
    for row in range(N):
        D=np.atleast_2d(X[row,:])-X
        D=np.square(D).sum(1)
        s='%d 0:%d'%(int(y[row]),row+1)
        S=np.exp(-np.exp(kernelvar)*(D-kernelm))
        for col in range(N):
            s+=' %d:%.6f'%(col+1,S[col])
        s+='\n'
        f.write(s)
    f.close()
    del D,S
    print 'train','kernelvar=',kv
    out=''
    for p in penalty:
        # assure that worker has svm-train compiled with 
        out+='kernelvar=%.1f\npenalty=%.1f\n'%(kv,p)
        s,o=getstatusoutput('export OMP_NUM_THREADS=%d'%ncpu+' && svm-train -s'+
            ' 0 -v 5 -t 4 -c %f -m 12000 %s%.3f.in'%(np.exp(p),fn,kv))
        if s: out+=o+'\nCross Validation Accuracy = 0%\n'
        else: out+=o+'\n' 
    getstatusoutput('rm '+fn+'%.3f.in'%np.round(kernelvar,3))
    print 'finished' 
    return out
if __name__ == '__main__':
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
        out=dview.map(svmTrain,fargs)    
        f=open(path+'svm.log','w')
        for ot in out: f.write(ot)
        f.close()

