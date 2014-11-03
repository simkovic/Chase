import numpy as np
import pylab as plt
from scipy import stats
from matustools.matusplotlib import *
import os
plt.close('all')
plt.ion()
event=-1
lim=[[0.0025,0.025],[0.0037,0.025]]
for vp in range(1,5):
    plt.figure(0)
    if event>0: sw=-400; ew=400
    else: sw=-800; ew=0
    hz=85.0 # start, end (in ms) and sampling frequency of the saved window
    fw=int(np.round((abs(sw)+ew)/1000.0*hz))
    bnR=np.arange(0,30,0.5)
    bnS=np.diff(np.pi*bnR**2)
    bnd= bnR+(bnR[1]-bnR[0])/2.0;bnd=bnd[:-1]
    path=os.getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
    figpath=os.getcwd().rstrip('code')+'figures/Analysis/'
    subplot(4,2,1)
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
    plt.legend(['S1','S2','S3','S4','RT'],loc=1,fontsize=7)
    plt.grid();plt.ylim([0,lim[event][0]]);
    plt.xlim([0,14])

    subplot(4,2,2)
    plt.grid()
    x=np.linspace(sw,ew,I[0].shape[1])/1000.
    x=np.reshape(x,[x.size/2,2]).mean(1)
    hhh=0
    plt.plot(x,np.reshape(I[0,hhh,:,0],[x.size,2]).mean(1))
    if vp==4: plt.plot(x,np.reshape(I[2,hhh,:,0],[x.size,2]).mean(1))
    plt.ylim([0,lim[event][0]]);
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
    subplot(4,2,3)
    hz=85.0
    mm=K[0].shape[1]/2
    plt.plot(d,np.nansum(K[0][:,mm,:],1)/ nK[0][:,mm,:].sum(1)*hz)
    if vp==4:# confidence band assuming iid binomial
        p=np.nansum(K[2][:,mm,:],1)/ nK[2][:,mm,:].sum(1)
        ci=1.96*np.sqrt(p*(1-p)/nK[2][:,mm,:].sum(1))
        l=(p-ci)*hz;h=(p+ci)*hz
        x=np.concatenate([d,d[::-1]])
        ci=np.concatenate([h,l[::-1]])
        plt.gca().add_patch(plt.Polygon(np.array([x,ci]).T,
                    alpha=0.2,fill=True,fc='red',ec='white'))
        
    plt.ylabel('Direction Changes')
    plt.xlabel('Radial Distance')
    plt.xlim([0,14]);plt.grid();plt.ylim([0,12])
    subplot(4,2,4)
    x=np.linspace(sw,ew,K[0].shape[1])/1000.
    ss=2
    x=np.reshape(x,[x.size/ss,ss]).mean(1)
    kk=0
    y=np.nansum(K[0][kk,:,:],1)/ nK[0][kk,:,:].sum(1)*hz
    plt.plot(x,np.reshape(y,[x.size,ss]).mean(1))
    if vp==4:
        #plt.plot(x,np.nansum(K[2][kk,:,:],1)/ nK[2][kk,:,:].sum(1))
        p=np.nansum(K[2][kk,:,:],1)/nK[2][kk,:,:].sum(1)
        ci=1.96*np.sqrt(p*(1-p)/nK[2][kk,:,:].sum(1))
        l=(p-ci)*hz;h=(p+ci)*hz
        l=np.reshape(l,[x.size,ss]).mean(1);h=np.reshape(h,[x.size,ss]).mean(1)
        x=np.concatenate([x,x[::-1]])
        ci=np.concatenate([h,l[::-1]])
        
        plt.gca().add_patch(plt.Polygon(np.array([x,ci]).T,
                    alpha=0.2,fill=True,fc='red',ec='white'))
    
    plt.ylabel('Direction Changes')
    plt.xlabel('Time to Saccade Onset')
    plt.xlim([-0.4,0.4])
    plt.ylim([0,12])
    #plt.title('Radius = 10 deg')
    plt.grid()
    
    hh=-1
    for chan in ['SOintensity','COmotion']:
        hh+=1
        K=np.load(path+'E%d/grd%s.npy'%(event,chan))
        I=np.load(path+'E%d/rad%s.npy'%(event,chan)).T
        if vp==4:IR=np.load(path+'E%d/radT%s.npy'%(event,chan)).T
            
        subplot(4,2,5+2*hh)
        plt.plot(np.arange(1,15),I[:,I.shape[1]/2])
        if vp==4: plt.plot(np.arange(1,15),IR[:,IR.shape[1]/2])
        plt.xlabel('Radial Distance')
        plt.ylim([0.008,lim[event][1]])
        plt.ylabel(['Light Contrast','Motion'][hh]+' Saliency')
        plt.grid();plt.xlim([0,14])
        subplot(4,2,6+2*hh)
        x=np.linspace(sw,ew,I.shape[1])/1000.
        hhh=3
        #plt.plot(x,np.nanmean(I[:hhh,:],0))
        plt.plot(x,I[0,:])
        if vp==4: plt.plot(x,IR[0,:]) 
        plt.xlabel('time')
        plt.ylabel(['Light Contrast','Motion'][hh]+' Saliency')
        plt.xlabel('Time to Saccade Onset')
        plt.ylim([0.008,lim[event][1]])
        plt.xlim([-0.4,0.4])
        plt.grid()
    #plt.legend(['gaze','rand time','rand agent','rand pos'],loc=2)
    #plt.show()
plt.savefig(figpath+'agdensE%d.png'%event)

##    plt.figure()
##    radbins=np.arange(1,15)
##    plt.imshow(I,extent=[sw,ew,radbins[0],radbins[-1]],
##        aspect=30,origin='lower',vmin=0.008, vmax=0.021)
##    plt.ylabel('radial distance from sac target')
##    plt.xlabel('time from sac onset')
##    plt.title('saliency')
##    plt.grid()
##    plt.colorbar();#plt.set_cmap('hot')
##    plt.show()
    
##    plt.figure()
##    plt.imshow(K[34,:,:]);plt.grid()
##    plt.show()
