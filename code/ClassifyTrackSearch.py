from ipyparallel import Client
#rc = Client(profile='chase')
#dview= rc[:]
#print 'number of detected engines:',len(dview)
import pickle, os
import numpy as np

EVALPATH = os.getcwd().rstrip('code')+'evaluation'+os.path.sep
FIGPATH = os.getcwd().rstrip('code')+'figures'+os.path.sep

def svmPreprocessData(data,vp,suf,subsample=1,whitening=True,expNrPrs=45,
    returnSel=False):
    # make list to ndarray
    data=np.concatenate(data)
    assert len(filter(lambda x: len(x)!=expNrPrs,data))==0
    assert np.all(data[:,0]==vp)
    data=data[:,2:];# discard vp id and block id
    sel=np.arange(data.shape[0])
    # remove nans/missing values
    data=np.concatenate([data,np.roll(data[:,2:],1,axis=0)],axis=1)
    nns=np.isnan(data).sum(1)==0
    print 'vp %d proportion of removed samples (mis. values):%.3f'%(vp,1-sel.mean())
    data=data[nns,:]
    data=data[data[:,0]==np.roll(data[:,0],1),:]
    X=data[:,2:];y=data[:,1];N=X.shape[0];sel=sel[nns] 
    if whitening:
        X-=np.array(X.mean(0),ndmin=2)
        X/=2*np.array(X.std(0),ndmin=2)
    # randomize
    inds=np.random.permutation(N)
    X=X[inds,:];y=y[inds];sel=sel[inds]
    # subsample
    N=int(N*subsample)
    X=X[:N,:]; y=y[:N];sel=sel[:N]
    print 'vp %d nr of samples '%vp,N,X.shape[1]
    if returnSel: return X,y,sel
    else: return X,y
    
def svmTrainOld(args,kernelm=0,ncv=5):
    '''This is faster, unfortunately does not work properly
        needs debugging
    '''
    X,y,kernelvar,penalty=args
    from commands import getstatusoutput
    import numpy as np
    from sklearn.svm import SVC
    N=X.shape[0];N-=N%ncv
    X=X[:N,:]; y=y[:N]
    print 'compute kernel, kernelvar=%.3f'%kernelvar
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
    print 'train, kernelvar=%.3f'%kernelvar
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

    
def svmTrain(args,kernelm=0,ncv=5):
    def clastats(y,t):
        ''' y - predicted values
            t - target values
            where positives are coded as 1, negatives as 0
            returns array with 4 values:
            1. hit rate, true positive rate, p(P|P)
            2. correct rejection, true negative rate, p(N|N)
            3. total proportion of correct 
            4. matthews correlation coefficient '''
        assert y.size==t.size and y.ndim==1 and t.ndim==1
        N=y.size
        assert np.all(np.logical_or(y==1,y==0))
        assert np.all(np.logical_or(t==1,t==0))
        TP= np.logical_and(y==1,t==1).sum(); TN=np.logical_and(y==0,t==0).sum()
        FP= np.logical_and(y==0,t==1).sum(); FN=np.logical_and(y==1,t==0).sum()
        a=TP/np.float32(TP+FP); b=TN/np.float32(TN+FN); c=(TN+TP)/float(N)
        r= (TP*TN-FP*FN)/((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP))**0.5
        return [a,b,c,r]
    X,y,kernelvar,penalty=args
    import numpy as np
    from sklearn.svm import SVC
    N=X.shape[0];N-=N%ncv
    X=X[:N,:]; y=y[:N]
    print 'compute kernel, kernelvar=%.3f'%kernelvar
    assert N%ncv==0
    spl=int(N/float(ncv))
    assert spl*ncv==N
    print 'train, kernelvar=%.3f'%kernelvar
    out=[[kernelvar,penalty,0,0]]
    for r in range(ncv):
            svm=SVC(C=np.exp(penalty),cache_size=7000,verbose=False,
                kernel='rbf',gamma=np.exp(kernelvar),max_iter=1e4)
            sel=np.random.permutation(N)
            svm.fit(X[sel[spl:],:],y[sel[spl:]])
            res=svm.predict(X[sel[:spl],:])
            stats=clastats(res,y[sel[:spl]])
            out.append(stats)
    return out
    
#X=np.array([[1,-1],[1,1],[0,1]])
#y=np.array([1,2,3])
#svmTrain([X,y,0,np.arange(5)],ncv=3)
#bla


def gridSearchEach(suf=''):  
    kernelvars=np.linspace(-20,0,21)
    penalties=np.linspace(-5,15,21)
    fargs=[];vpids=[]
    #pool=Pool(8)
    for vp in range(1,5):
        f=open(EVALPATH+'vp%03d/predictors%s.pickle'%(vp,suf),'r')
        d=pickle.load(f)
        f.close()
        X,y=svmPreprocessData(d,vp,suf,subsample=1)
        for kv in kernelvars:
            for pen in penalties:
                fargs.append([X,y,kv,pen])
                vpids.append(vp)
    out=dview.map(svmTrain,fargs,block=True)
    #out=map(svmTrain,fargs)
    for vp in range(1,5):
        temp=[]
        for oi in range(len(out)):
            if vpids[oi]==vp: temp.append(out[oi])
        np.save(EVALPATH+'vp%03d/svm%s.npy'%(vp,suf),temp) 
 
def gridSearchAll(N,suf=''):  
    kernelvars=np.linspace(-20,0,21)
    penalties=np.linspace(-5,15,21)
    Xs=[];ys=[];vpn=range(1,5)
    for vp in vpn:
        f=open(EVALPATH+'vp%03d/predictors%s.pickle'%(vp,suf),'r')
        data=pickle.load(f)
        f.close()
        X,y=svmPreprocessData(data,vp,suf,subsample=1,whitening=False)
        Xs.append(X[:N/len(vpn),:])
        ys.append(y[:N/len(vpn)])
    Xs=np.concatenate(Xs)  
    ys=np.concatenate(ys)
    Xs-=np.array(Xs.mean(0),ndmin=2)
    Xs/=2*np.array(Xs.std(0),ndmin=2)
    print 'gridSearchAll: Xs.shape',Xs.shape,'ys.shape',ys.shape 
    fargs=[]
    for kv in kernelvars:
        for pen in penalties:
            fargs.append([Xs,ys,kv,pen])
    out=dview.map(svmTrain,fargs,block=True)
    #out=map(svmTrain,fargs)
    np.save(EVALPATH+'vpall/svm%s.npy'%suf,out)  
    
def plotGridSearch(suf=''):
    import pylab as plt
    plt.figure(figsize=(12,8))
    k=1
    for vp in ['all']:#['001','002','003','004','all']:
        path=EVALPATH+'vp%s'%vp+os.path.sep
        plt.subplot(2,3,k);k+=1
        dat=np.load(path+'svm%s.npy'%suf)
        kernelvars=np.unique(dat[:,0,0])
        penalties=np.unique(dat[:,0,1])
        fun=np.reshape(dat[:,1:,2].mean(1), (kernelvars.size,penalties.size))
        plt.pcolor(kernelvars,penalties,fun.T,cmap='hot',vmin=0.5,vmax=np.max(fun))
        plt.xlabel('kernelvar');plt.ylabel('penalty')
        plt.xlim([kernelvars[0],kernelvars[-1]]);
        plt.ylim([penalties[0],penalties[-1]])
        plt.colorbar();
        plt.gca().set_aspect(1)
    mn=(fun==np.max(fun)).nonzero()
    topKV=kernelvars[mn[0][0]]
    topPN=penalties[mn[1][0]] 
    plt.plot(topKV,topPN,'xk')
    plt.savefig(FIGPATH+'svmGridSearch');
    out=[topKV, topPN,np.max(fun)]
    print 'top: kernelvar %.1f, penalty %.1f, performance %.3f'%tuple(out)
    np.save(path+'opt',out)  
    
def trainAll(N,suf='',vpn=range(1,5)):
    from sklearn.svm import SVC
    from sklearn.externals import joblib
    Xs=[];ys=[];
    opt=np.load(EVALPATH+'vpall/opt.npy')
    kv,pn,perf=opt.tolist()
    for vp in vpn:
        f=open(EVALPATH+'vp%03d/predictors%s.pickle'%(vp,suf),'r')
        data=pickle.load(f)
        f.close()
        X,y=svmPreprocessData(data,vp,suf,subsample=1,whitening=False)
        Xs.append(X[:N/len(vpn),:])
        ys.append(y[:N/len(vpn)])
    path=EVALPATH+'vpall'+os.path.sep
    Xs=np.concatenate(Xs)
    ys=np.concatenate(ys)
    pm=np.array(Xs.mean(0),ndmin=2)
    psd=2*np.array(Xs.std(0),ndmin=2)
    Xs-=pm;Xs/=psd
    np.save(path+'preprocessMean',pm)
    np.save(path+'preprocess2sd',psd)
    print 'trainAll: Xs.shape',Xs.shape,'ys.shape',ys.shape  
    svm=SVC(C=np.exp(pn),cache_size=28000,verbose=False,
                kernel='rbf',gamma=np.exp(kv),max_iter=1e5)
    svm.fit(Xs,ys)
    joblib.dump(svm,path+'trainedSvm')
    #res=svm.predict(Xnew,:])
    
def predictAll(suf='',vpn=range(20,70)):
    from sklearn.svm import SVC
    from sklearn.externals import joblib
    svm=joblib.load(EVALPATH+'vpall/trainedSvm')
    pm=np.load(EVALPATH+'vpall/preprocessMean.npy')
    psd=np.load(EVALPATH+'vpall/preprocess2sd.npy')
    for vp in vpn:
        print 'predicting vp',vp
        try:
            f=open(EVALPATH+'vp%03d/predictors%s.pickle'%(vp,suf),'r')
        except IOError: continue
        data=pickle.load(f)
        f.close()
        if not len(data): 
            print 'no ET data'
            continue
        X,y,sel=svmPreprocessData(data,vp,suf,subsample=1,
            whitening=False,returnSel=True)
        X-=pm;X/=psd
        y=svm.predict(X)
        np.save(EVALPATH+'vp%03d/predicted%s'%(vp,suf),y)
        np.save(EVALPATH+'vp%03d/sel%s'%(vp,suf),sel)
    
def exportForEval(suf='',vpn=range(20,70)):
    '''for each subject and sac event compute
        id based on vp block and trial, si
        median of gaze pos, gaze speed, gaze angle, distance gaze-to-agent,
        angle deviation gaze-to-agent, angles in range
        
    '''
    for vp in vpn:
        try:
            f=open(EVALPATH+'vp%03d/coords%s.pickle'%(vp,suf),'r')
        except IOError: continue
        coords=pickle.load(f);f.close()
        coords=np.concatenate(coords)
        y=np.load(EVALPATH+'vp%03d/predicted%s.npy'%(vp,suf))
        sel=np.load(EVALPATH+'vp%03d/sel%s.npy'%(vp,suf))
        sel=sel[y==1]
        print len(coords),y.size,sel.size
        out=[]
        for i in range(len(coords)):
            ii=(i==sel).nonzero()[0]
            if not len(ii):continue
            cd=coords[ii[0]]
            out.append([np.median(np.linalg.norm(cd[:,0,:],axis=1))])
            #df=np.diff(cd[:,0,:],0)
            #out[-1].append(np.median(np.linalg.norm(cd,axis=1)))
            bla
        
        
        
        
    
    
if __name__ == '__main__':
    #gridSearchEach()
    #gridSearchAll(N=13440*4)
    #plotGridSearch()
    #trainAll(N=13440*4,suf='')
    predictAll()
    #exportForEval()
