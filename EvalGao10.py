import numpy as np
import pylab as plt
from Settings import Q
plt.close('all')
X=0;Y=1
vp=301
block=0
D= np.loadtxt(Q.outputPath+'vp%03d.res'%vp)
T=D.shape[0]

# gao10 eval
quadMaps=[[1,1,0,0],[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,1,0]]
prop=np.zeros(T)
for t in range(T):  
    ms=np.load(Q.inputPath+'vp%03d'%vp+Q.delim+
        'chsVp%03db%dtrial%03d.npy'%(vp,block,t))
    qmap=quadMaps[int(D[t,4])]
    prop[t]+=np.logical_and(ms[:,X]>0, ms[:,Y]>0).mean()*qmap[0]
    prop[t]+=np.logical_and(ms[:,X]<=0, ms[:,Y]>0).mean()*qmap[1]
    prop[t]+=np.logical_and(ms[:,X]>0, ms[:,Y]<=0).mean()*qmap[2]
    prop[t]+=np.logical_and(ms[:,X]<=0, ms[:,Y]<=0).mean()*qmap[3]
print (1-prop.mean())*17, 1-prop.mean()

# rep. momentum
posA=np.zeros((T,2))*np.nan
phiA=np.zeros(T)*np.nan
posC=np.zeros((T,2))*np.nan
for t in range(T):
    fname='vp300trial%03d.npy' % D[t,3]
    traj=np.load(Q.inputPath+'vp300'+Q.delim+fname)
    posA[t,:]=traj[-1,D[t,6],:2]
    phiA[t]=(traj[-1,D[t,6],2]-180)/180.0*np.pi
    
    ms=np.load(Q.inputPath+'vp%03d'%vp+Q.delim+
        'chsVp%03db%dtrial%03d.npy'%(vp,block,t))
    posC[t,:]=ms[-1,:]
phiRM=D[:,7]   
phiC=np.arctan2(posA[:,1]-posC[:,1],posA[:,0]-posC[:,0])
d=posA-D[:,[8,9]]
phiD= np.arctan2(d[:,1],d[:,0])
dist=np.sqrt(np.power(d,2).sum(1))
sel=dist<2
print 'proportion of utilized sample is', sel.mean()


k=0; titles=['Absolute Discrepancy','Motion Momentum','Chasee Gravity','Eyes Orientation Momentum']
for dd in [0,phiA,phiC,phiRM]:
    plt.subplot(2,2,k+1)
    phi=phiD-dd
    x=np.cos(phi)*dist
    y=np.sin(phi)*dist
    plt.plot(x[sel],y[sel],'o')
    plt.plot(x[sel].mean(),y[sel].mean(),'xk',markersize=6)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.title(titles[k])
    plt.plot([0,2],[0,0],'k')
    ax=plt.gca()
    ax.set_aspect('equal')
    plt.grid()
    k+=1


x=np.cos([np.zeros(T),phiA,phiC,phiRM])
y=np.sin([np.ones(T)*np.pi/2.0,phiA,phiC,phiRM])
# frequentist evaluation
print np.linalg.lstsq(x[:,sel].T,d[sel,0])
print np.linalg.lstsq(y[:,sel].T,d[sel,1])
# bayesian estimation
from jagstools import jags
from pymc.Matplot import plot
import os
opath= os.getcwd().rstrip('code')+'evaluation/jags/'
pname='gao10e3'

indata=[x[:,sel],y[:,sel],d[sel,0],d[sel,1],sel.sum()]
indatlabels=['px','py','dx','dy','N']
outdatlabels=['sdev','b0','b1','b2','b3']
inpars=[]
inparlabels=[]

model='''
model{
    for (t in 1:N){
        dx[t] ~ dnorm(mx[t], pow(sdev,-2))
        dy[t] ~ dnorm(my[t], pow(sdev,-2))
        mx[t] <- b1*px[2,t]+b2*px[3,t]+b3*px[4,t]
        my[t] <- b0*py[1,t]+b1*py[2,t]+b2*py[3,t]+b3*py[4,t]
    }
    b3 ~ dnorm(0,1/1000)
    b2 ~ dnorm(0,1/1000)
    b1 ~ dnorm(0,1/1000)
    b0 ~ dnorm(0,1/1000)
    sdev ~ dunif(0,5)
}
'''


R=jags(pname,indata,indatlabels,outdatlabels,model,
    inpars,inparlabels,chainLen=20000,burnIn=5000,thin=5,path='')
for k in range(len(R)):plot(R[k],outdatlabels[k])



plt.show()
