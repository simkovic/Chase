import numpy as np
import pylab as plt
import os
plt.ion()

from jagstools import jags
from pymc.Matplot import plot
import scipy.stats as stats

def loadData(vpn, verbose=False):

    D=[]
    for vp in vpn:
        path = os.getcwd()
        path = path.rstrip('code')
        dat=np.loadtxt(path+'behavioralOutput/vp%03d.res'%vp)
        if verbose: print vp, dat.shape
        D.append(dat[dat[:,1]>0,:])
    D=np.concatenate(D,0)
    return D

vpn=range(20,70)
vpn.remove(27)

D=loadData(vpn)
acc=np.zeros((len(vpn),2))*np.nan

# check accuracy
for i in range(len(vpn)):
    sel2= np.logical_and(D[:,0]==vpn[i],D[:,6]!=30)
    acc[i,0]= np.sum(D[sel2,-1])
    acc[i,1]=np.sum(sel2)
p=acc[:,0]/acc[:,1];rem=[] 
for i in (p<0.50).nonzero()[0].tolist():
    print 'subject id ',vpn[i],' accuracy %.3f '%p[i],'< .5 , removed from analysis'
    rem.append(vpn[i])
for r in rem: vpn.remove(r)

stat=np.ones((len(vpn),4,2))*np.nan
rts=np.zeros((len(vpn),160))*np.nan
acc=np.zeros((len(vpn),2))*np.nan
N=np.zeros(len(vpn))
for i in range(len(vpn)):
    for b in range(4):
        sel=np.logical_and(D[:,0]==vpn[i],D[:,1]==b+1)
        if sel.sum()>0:
            N[i]+=1
            stat[i,b,0]= (D[sel,6]==30).mean()
            sel2= np.logical_and(sel,~(D[:,6]==30))
            stat[i,b,1]=np.median(D[sel2,6])
            rts[i,b*40:(b+1)*40]=D[sel,6]
    sel2= np.logical_and(D[:,0]==vpn[i],D[:,6]!=30)
    acc[i,0]= np.sum(D[sel2,-1])
    acc[i,1]=np.sum(sel2)

def modelACC(acc):
    pname='chaseACC'
    indata=[acc[:,0],acc[:,1],acc.shape[0]]
    indatlabels=['cor','tot','vpnr']
    outdatlabels=['p','mu','K']
    inpars=[]
    inparlabels=[]
    model='''
    model{
        for (i in 1:vpnr){
            cor[i]~ dbin(p[i],tot[i])
            p[i]~dbeta(1,1)
        }
        
    }
    '''
    from jagstools import jags
    D=jags(pname,indata,indatlabels,outdatlabels,model,
        inpars,inparlabels,chainLen=50000,burnIn=5000,thin=5)
    p=D[0]
    for vp in range(acc.shape[0]): plt.plot(vp,p[vp,:].mean(),'ok')
    return p
from jagstools import loadCoda
D=loadCoda('chaseRT1')
rtpred=D[0]
p=modelACC(acc)
perc=25
dt=np.zeros((p.shape[0],6))
for i in range(p.shape[0]):
    dt[i,0]= np.median(p[i,:])
    dt[i,1]=dt[i,0]-stats.scoreatpercentile(p[i,:],perc)
    dt[i,2]=stats.scoreatpercentile(p[i,:],100-perc)-dt[i,0]
    dt[i,3]= np.median(rtpred[i,:])
    dt[i,4]=dt[i,3]-stats.scoreatpercentile(rtpred[i,:],perc)
    dt[i,5]=stats.scoreatpercentile(rtpred[i,:],100-perc)-dt[i,3]
plt.figure()
b=np.sort(dt[:,:3],axis=0)
plt.errorbar(range(p.shape[0]),b[:,0],yerr=[b[:,1],b[:,2]],fmt='o')
plt.xlim([-1,46])
plt.ylim([0.5,1])
plt.xlabel('Subjects (Sorted by Median)')
plt.ylabel('Accuracy')
plt.figure()
b=np.sort(dt[:,3:],axis=0)
plt.errorbar(range(p.shape[0]),b[:,0],yerr=[b[:,1],b[:,2]],fmt='o')
plt.xlim([-1,46])
plt.ylim([0,30])
plt.xlabel('Subjects (Sorted by Median)')
plt.ylabel('Search Time')
        
def modelSingleVP(vp=2):
    rts[rts==30]=np.nan    
    pname='chaseRT'
    dat=rts[vp,:]
    censored=np.float32(np.isnan(dat))
    censored[N[vp]*40:]=np.nan
    shift=np.nanmin(dat)-0.001
    indata=[dat-shift, censored,shift]
    indatlabels=['rt','rtcens','shift']
    outdatlabels=['rtpred','mean','sdev']
    rtInit=np.ones(160)*np.nan
    rtInit[censored==1]=31
    inpars=[rtInit]
    inparlabels=['rt']
    model='''
    model{
        for (t in 1:160){
            rtcens[t]~ dinterval(rt[t],30-shift)
            rt[t] ~ dgamma(pow(mean,2)*pow(sdev,-2),mean*pow(sdev,-2))
        }
        mean ~ dunif(0,30)
        sdev ~ dunif(0,30)
        
        rtpred<-rttemp+shift
        rttemp~ dgamma(pow(mean,2)*pow(sdev,-2),mean*pow(sdev,-2))
    }
    '''
    from jagstools import jags
    from pymc.Matplot import plot
    import scipy.stats as stats
    plt.close('all')
    rtPred,mean,sdev=jags(pname,indata,indatlabels,outdatlabels,model,
        inpars,inparlabels,chainLen=20000,burnIn=5000,thin=5)

    plt.hist(rts[vp,censored==0],bins=range(0,30,2))
    plot(rtPred,'rtPred')
    #plt.figure()
    #stats.probplot(dat[~censored],dist='gamma',plot=plt,
    #    sparams=(shape.mean(),shift,scale.mean()))
    #print np.abs((rtPred>30).mean()-censored.mean())/censored.mean()
    #print 'binomtest ', stats.binom_test(censored.sum(),N[vp]*40,(rtPred>30).mean() )

    #plot(shape,'shape')
    #plot(scale,'scale')
    plot(mean,'mean')
    plot(sdev,'sd')

def modelRT(rts):
    rts[rts==30]=np.nan    
    pname='chaseRT'
    dat=np.copy(rts)
    shift=np.nanmin(rts,axis=1)-0.001
    censored=np.float32(np.isnan(dat))
    for vp in range(N.size):
        dat[vp,:]-= shift[vp]
        censored[vp,(N[vp]*40):]=np.nan


    indata=[dat, censored,shift,N.size]
    indatlabels=['rt','rtcens','shift','n']
    rtInit=np.ones((N.size,160))*np.nan
    rtInit[censored==1]=31
    inpars=[rtInit]
    inparlabels=['rt']

    modelgrand='''
    model{
        for (vp in 1:n){
            for (t in 1:160){
                rtcens[vp,t]~ dinterval(rt[vp,t],30-shift[vp])
                rt[vp,t] ~ dgamma(pow(mu,2)*pow(sdev,-2),mu*pow(sdev,-2))
            }
        }

        mu ~ dunif(0,30)
        sdev ~ dunif(0,30)
    }
    '''


    modelunpooled='''
    model{
        for (vp in 1:n){
            for (t in 1:160){
                rtcens[vp,t]~ dinterval(rt[vp,t],30-shift[vp])
                rt[vp,t] ~dgamma(a[vp],b[vp])
            }

            rtpred[vp]<-rttemp[vp]+shift[vp]
            rttemp[vp]~ dgamma(a[vp],b[vp])
            a[vp]<-pow(mu[vp],2)*pow(sdev[vp],-2)
            b[vp]<-mu[vp]*pow(sdev[vp],-2)
            mu[vp] ~ dunif(0,30)
            sdev[vp] ~ dunif(0,30)
        }
    }
    '''

    modelhier='''
    model{
        for (vp in 1:n){
            for (t in 1:160){
                rtcens[vp,t]~ dinterval(rt[vp,t],30-shift[vp])
                rt[vp,t] ~dgamma(a[vp],b[vp])
            }

            rtpred[vp]<-rttemp[vp]+shift[vp]
            rttemp[vp]~ dgamma(a[vp],b[vp])
            a[vp]<-pow(mu[vp],2)*pow(sdev[vp],-2)
            b[vp]<-mu[vp]*pow(sdev[vp],-2)
            mu[vp] ~ dt(mumu,pow(musd,-2),tdf)
            sdev[vp] ~ dgamma(pow(sdmu,2)*pow(sdsd,-2),sdmu*pow(sdsd,-2))
        }
        mumu~dunif(0,30)
        musd~dunif(0,30)
        sdmu~dunif(0,30)
        sdsd~dunif(0,30)
        tdf <- 1 - tdfGain * log(1-udf)
        udf ~ dunif(0,1)
        tdfGain <- 1
    }
    '''


    plt.close('all')
    outdatlabels=['rtpred','mu','sdev','mumu','musd','sdmu','sdsd','tdf']
    rtpred,mean,sdev,mumu,musd,sdmu,sdsd,tdf=jags(pname,indata,indatlabels,outdatlabels,modelhier,
        inpars,inparlabels,chainLen=50000,burnIn=5000,thin=5)

    plot(mumu,'mumu')
    plot(musd,'musd')
    plot(sdmu,'sdmu')
    plot(sdsd,'sdsd')
    plot(tdf,'tdf')
    plt.figure()
    for vp in range(len(vpn)): plt.plot(vp,mean[vp,:].mean(),'ok')
    plt.title('mean')
    plt.figure()
    bp=[];r2=[]
    for vp in range(len(vpn)):
        plt.plot(vp,sdev[vp,:].mean(),'ok')
        shape=mean[vp,:].mean()**2 / sdev[vp,:].mean()**2
        scale=sdev[vp,:].mean()**2 / mean[vp,:].mean()
        e,f=stats.probplot(rts[vp,~np.isnan(rts[vp,:])],dist='gamma',
            sparams=(shape,shift[vp],scale))
        r2.append(f[-1])
        #print '\t',(rtpred[vp,:]>30).mean(),(censored[vp,:]==1).sum()/N[vp]/40
        bp.append(stats.binom_test((censored[vp,:]==1).sum(),
                                N[vp]*40,(rtpred[vp,:]>30).mean() ))
    plt.title('sdev')
    plt.figure()        
    plt.plot(r2,'ok')
    plt.title('R^2 coefs')
    plt.figure()        
    plt.plot(bp,'ok')
    plt.ylim([0,1])
    plt.title('binom test tail')
    print 'Mean R2',np.array(r2).mean()
    return rtpred



def modelPymc():
    import pymc
    from pymc import MCMC
    from pymc.Matplot import plot
    from scipy.stats import weibull_min as weib
    from scipy.stats import gamma
    import scipy.stats as stats

    plt.close('all')
    vp=4
    N=N[:5]
    dat=rts#[vp,:N[vp]*40]
    censored= dat==30
    shift=np.nanmin(dat,axis=1)-0.001#pymc.Uniform('shift',lower=0,upper=np.nanmin(dat))
    scale=pymc.Gamma('scale',1*np.ones(N.size),10*np.ones(N.size),value=5*np.ones(N.size),size=N.size)
    shape=pymc.Gamma('shape',1*np.ones(N.size),10*np.ones(N.size),value=1*np.ones(N.size),size=N.size)


    def rtloglik(value,shift,scale,shape):
        if value<shift: return -np.inf
        if value>30: return np.log(1-gamma.cdf(30,shape,loc=shift,scale=scale))
        p=gamma.pdf(value,shape,loc=shift,scale=scale)
        if p==0: return -np.inf
        return np.log(p)


    rt=np.empty((N.size,160),dtype=object)
    for vp in range(N.size):
        for t in range(int(N[vp])*40):
            rt[vp,t]=pymc.Stochastic(name='rt%d'%vp,logp=rtloglik,
                parents={'shift':shift[vp],'scale':scale[vp],'shape':shape[vp]},
                doc='censored gamma',observed=True,value=dat[vp,t])

    model=[rt,shift,scale,shape]
    M=pymc.MCMC(model,db='pickle', dbname='chaseRT.pickle')
    M.sample(20000,10000,10)

    @pymc.observed
    def rt(value=dat,shift=shift,scale=scale,shape=shape):
        loglik=0
        for vp in range(N.size):
            if np.any(value[vp,:N[vp]*40]<shift[vp]): return -np.inf
            p=1-gamma.cdf(30,shape[vp],loc=shift[vp],scale=scale[vp])
            if p==0: return -np.inf
            loglik+=censored[vp,:N[vp]*40].sum()*np.log(p)
            loglik+= np.log(gamma.pdf(value[vp,~censored[vp,:N[vp]*40]],
                        shape[vp],loc=shift[vp],scale=scale[vp])).sum()
        return loglik
    rtPredInit=np.ones(N.size)*10
    @pymc.stochastic
    def rtPred(value=rtPredInit,shift=shift,scale=scale,shape=shape):
        def logp(value=rtPredInit,shift=shift,scale=scale,shape=shape):
            loglik=0
            if np.any(value-shift<0): return -np.inf
            sel=value==30
            if sel.sum()>0:
                loglik+=np.log(1-gamma.cdf(30,shape[sel],
                    loc=shift[sel],scale=scale[sel])).sum()
            if not np.all(sel):
                loglik+= np.log(gamma.pdf(value[~sel],shape[~sel],
                            loc=shift[~sel],scale=scale[~sel])).sum()
            return loglik
        
        def random(shift=shift,scale=scale,shape=shape):
            return np.minimum(gamma.rvs(shape,loc=shift,scale=scale,size=N.size),np.ones(N.size)*30)

    model=[rt,shift,scale,shape]
    M=pymc.MCMC(model,db='pickle', dbname='chaseRT.pickle')
    M.sample(20000,10000,10)
    M.db.close()
    plot(shift)
    plot(scale)
    plot(shape)
    plt.figure()
    plt.hist(rtPred.trace())
    plt.figure()
    plt.hist(dat)
    plt.figure()
    stats.probplot(dat[~censored],dist='weibull_min',plot=plt,
        sparams=(shape.stats()['mean'],shift,scale.stats()['mean']))
    print (rtPred.trace()==30).mean(), censored.mean()
            
