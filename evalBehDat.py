import numpy as np
import pylab as plt
import os
plt.ion()

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

vpn=range(20,51)
vpn.remove(27)
D=loadData(vpn)

stat=np.ones((len(vpn),4,2))*np.nan
rts=np.zeros((len(vpn),160))*np.nan
# omissions
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

import pymc
from pymc import MCMC
from pymc.Matplot import plot
from scipy.stats import weibull_min as weib
import scipy.stats as stats

plt.close('all')
vp=2
dat=rts[vp,:N[vp]*40]
censored= dat==30
shift=np.nanmin(dat)#pymc.Uniform('shift',lower=0,upper=np.nanmin(dat))
scale=pymc.Gamma('scale',1,10,value=5)
shape=pymc.Gamma('shape',1,10,value=1)
@pymc.observed
def rt(value=dat,shift=shift,scale=scale,shape=shape):
    if np.any(value<shift): return -np.inf
    p=1-weib.cdf(30,shape,loc=shift,scale=scale)
    if p==0: return -np.inf
    loglik=censored.sum()*np.log(p)
    loglik+= np.log(weib.pdf(value[~censored],shape,loc=shift,scale=scale)).sum()
    
    #loglik+= np.log(lik[lik>0]).sum()
    #loglik+= np.log(1e-12)*(lik==0).sum()
    return loglik

@pymc.stochastic
def rtPred(value=10,shift=shift,scale=scale,shape=shape):
    def logp(value=10,shift=shift,scale=scale,shape=shape):
        if value-shift<0: return -np.inf
        if value==30: return np.log(1-weib.cdf(30,shape,loc=shift,scale=scale))
        return np.log(weib.pdf(value,shape,loc=shift,scale=scale))
    
    def random(shift=shift,scale=scale,shape=shape):
        return np.minimum(weib.rvs(shape,loc=shift,scale=scale),30)
model=[rt,rtPred,shift,scale,shape]
M=pymc.MCMC(model,db='pickle', dbname='chaseRT.pickle')
M.sample(20000,10000,5)
M.db.close()
#plot(shift)
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
##rts[rts==30]=np.nan    
##pname='chaseRT'
##vp=4;b=3
##censored=np.isnan(rts[vp,:])
##indata=[rts[vp,:], np.int32(censored),N[vp]*40]
##indatlabels=['rt','rtcens','n']
##outdatlabels=['scale','shape','rtpred']
##rtInit=np.ones(160)*np.nan
##rtInit[censored]=31
##inpars=[rtInit]
##inparlabels=['rt']
##model='''
##model{
##    for (t in 1:n){
##        rtcens[t]~ dinterval(rt[t],30)
##        rt[t] ~ dweib(shape,scale)
##    }
##    rtpred~dweib(shape,scale)
##    shape~dgamma(1,10)
##    scale~dgamma(1,10)
##    shift~dunif(0,30)
##}
##'''
##from jagstools import jags
##from pymc.Matplot import plot
##import scipy.stats as stats
##
##D=jags(pname,indata,indatlabels,outdatlabels,model,
##    inpars,inparlabels,chainLen=20000,burnIn=5000,thin=5)
##
##plt.hist(rts[vp,~censored],bins=range(0,30,2))
##plot(D[2],'rtPred')
##plot(D[0],'shape')
##plot(D[1],'scale')
##shape=D[0].mean()
##scale=D[1].mean()**(-2/shape)
