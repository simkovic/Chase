import numpy as np
import pylab as plt
from scipy import stats
from matustools.matusplotlib import *
import os
plt.close('all')
plt.ion()
event=1
for vp in range(1,5):
    plt.figure(0)
    sw=-400; ew=400;hz=85.0 # start, end (in ms) and sampling frequency of the saved window
    fw=int(np.round((abs(sw)+ew)/1000.0*hz))
    bnR=np.arange(0,15,0.5)
    bnS=np.diff(np.pi*bnR**2)
    bnd= bnR+(bnR[1]-bnR[0])/2.0;bnd=bnd[:-1]
    path=os.getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
    figpath=os.getcwd().rstrip('code')+'figures/Analysis/'
    subplot(2,2,1)
    I=np.load(path+'E%d/agdens.npy'%event)
    I/=float(bnR.size)
    print I.shape
    m=I.shape[2]/2
    plt.plot(bnd,I[0,:,m,0])
    if vp==4:
        #plt.plot(bnd,I[2,:,m,0])
        # show histogram
        x=np.concatenate([bnd,bnd[::-1]])
        ci=np.concatenate([I[2,:,m,2],I[2,:,m,1][::-1]])
        plt.gca().add_patch(plt.Polygon(np.array([x,ci]).T,
                    alpha=0.2,fill=True,fc='red',ec='red'))
    plt.xlabel('Radial Distance')
    plt.ylabel('Agent Density')
    plt.legend(['S1','S2','S3','S4','RT'],loc=1)
    plt.grid();plt.ylim([0,0.15]);plt.xlim([0,15])

    subplot(2,2,2)
    plt.grid()
    x=np.linspace(sw,ew,I[0].shape[1])/1000.
    hhh=0
    plt.plot(x,I[0,hhh,:,0])
    if vp==4: plt.plot(x,I[2,hhh,:,0])
    #plt.ylim([1,5])
    plt.xlim([-0.4,0.4])
    #plt.title('Radius=5 deg')
    plt.ylabel('Agent Density')#[a per deg^2]
    plt.xlabel('Time to Saccade Onset')

    #direction change
    K=np.load(path+'E%d/dcK.npy'%event)
    nK=np.load(path+'E%d/dcnK.npy'%event)
    bn=np.arange(0,20,0.5)
    d= bn+(bn[1]-bn[0])/2.0;d=d[:-1]
    c=np.diff(np.pi*bn**2)
    # this code fragment shows the interaction
    #plt.figure(1)
    #plt.subplot(2,2,vp)
    #plt.imshow(np.nansum(K[0],2)/np.sum(nK[0],2),origin='lower',extent=[sw,ew,bn[0],bn[-1]],aspect=20,cmap='hot',vmax=11,vmin=2)     
    #plt.colorbar()
    ################
    plt.figure(0)
    subplot(2,2,3)
    hz=85.0
    plt.plot(d,np.nansum(K[0][:,0,:],1)/ nK[0][:,0,:].sum(1))
    if vp==4:# confidence band assuming iid binomial
        p=np.nansum(K[2][:,0,:],1)/ nK[2][:,0,:].sum(1)/hz
        ci=1.96*np.sqrt(p*(1-p)/nK[2][:,0,:].sum(1))
        l=(p-ci)*hz;h=(p+ci)*hz
        x=np.concatenate([d,d[::-1]])
        ci=np.concatenate([h,l[::-1]])
        plt.gca().add_patch(plt.Polygon(np.array([x,ci]).T,
                    alpha=0.2,fill=True,fc='red',ec='white'))
        
    plt.ylabel('Direction Changes')
    plt.xlabel('Radial Distance')
    plt.xlim([0,15]);plt.grid()
    subplot(2,2,4)
    x=np.linspace(sw,ew,K[0].shape[1])/1000.
    kk=0
    plt.plot(x,np.nansum(K[0][kk,:,:],1)/ nK[0][kk,:,:].sum(1))
    if vp==4:
        #plt.plot(x,np.nansum(K[2][kk,:,:],1)/ nK[2][kk,:,:].sum(1))
        p=np.nansum(K[2][kk,:,:],1)/ nK[2][kk,:,:].sum(1)/hz
        ci=1.96*np.sqrt(p*(1-p)/nK[2][kk,:,:].sum(1))
        l=(p-ci)*hz;h=(p+ci)*hz
        x=np.concatenate([x,x[::-1]])
        ci=np.concatenate([h,l[::-1]])
        
        plt.gca().add_patch(plt.Polygon(np.array([x,ci]).T,
                    alpha=0.2,fill=True,fc='red',ec='white'))
    
    plt.ylabel('Direction Changes')
    plt.xlabel('Time to Saccade Onset')
    #plt.title('Radius = 10 deg')
    plt.grid()
    plt.savefig(figpath+'agdensE%d.png'%event)

#plt.show()
