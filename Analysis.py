import numpy as np
import pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import nanmean

plt.close('all')
#plt.ion()
from os import getcwd
path=getcwd()
path=path.rstrip('code')
path+='evaluation/'
vp=1

bs=range(1,22)
sw=-400; ew=400;hz=85.0 # start, end (in ms) and sampling frequency of the saved window
fw=int(np.round((abs(sw)+ew)/1000.0*hz))

def plotSearchInfo(plot=True):
    D=np.load(path+'SIvp%03d.npy'%(vp))
    if plot:
        plt.close('all')
        plt.plot(D[:,6],D[:,7],'.k')
        plt.title('Target positions')
        ax=plt.gca()
        ax.set_aspect(1)

        plt.figure()
        plt.plot(D[:,2],D[:,3],'.k')
        plt.title('Disengagement positions')
        ax=plt.gca()
        ax.set_aspect(1)

        plt.figure()
        reloc=np.sqrt(np.sum(np.power(D[:,[6,7]]-D[:,[2,3]],2),axis=1))
        plt.title('Distance')
        plt.hist(reloc,50)

        plt.figure()
        dur = D[:,5]-D[:,1]
        plt.hist(dur,50)
        plt.title('Duration')

        plt.figure()
        vel=reloc/dur
        plt.hist(vel[~np.isnan(vel)]*1000,50)
        plt.title('Velocity')
    return D


def loadData():
    si=plotSearchInfo(plot=False)
    inpath=getcwd().rstrip('code')+'/input/'
    sf=int(np.round((si[:,1]+sw)/1000.0*hz))
    ef=sf+fw
    valid= si[:,1]+sw>=0 and si[:,1]+ew <= si[:,-3]
    print 'proportion of utilized samples is', valid.mean()
    DG=np.zeros((valid.sum(),fw,14,2))*np.nan
    DT=np.zeros((valid.sum(),fw,14,2))*np.nan
    DP=np.zeros((valid.sum(),fw,14,2))*np.nan
    DA=np.zeros((valid.sum(),fw,14,2))*np.nan
    k=0; lastt=-1;
    rt=np.random.permutation(valid.nonzero()[0])
    for h in range(si.shape[0]):
        if not valid[h]: continue
        if si[h,-1]!=lastt:
            order = np.load(inpath+'/vp%03d/ordervp%03db%d.npy'%(vp,vp,si[h,-2]))
            traj=np.load(inpath+'/input/vp%03d/vp%03db%dtrial%03d.npy'%(vp,vp,si[h,-2],order[si[h,-1]]) )
        DG[k,:,:,:]=traj[sf[h]:ef[h],:,:2]
        DT[k,:,:,:]=traj[sf[rt[k]]:ef[rt[k]],:,:2]
        k+=1; lastt=si[h,-2]
    rp=np.random.permutation(valid.nonzero()[0])
    #sel=si[:,8]>0#np.logical_not(np.all(np.all(np.all(np.isnan(D[:,:14]),3),2),1))
    #DG=DG[sel,:,:,:]
    for a in range(14):
        for f in range(D.shape[1]):
            for q in range(2):
                DG[:,f,a,q]-= si[:,6+q]
                DT[:,f,a,q]-= si[rt,6+q] # check whether the index works ok !!!
                DA[:,f,a,q]= DG[:,f,a,q] - DG[:,DG.shape[1]/2,2,q]
                DP[:,f,a,q]= DG[:,f,a,q] - si[rp,6+q]
    return DG,DP,DA,DT

def agentDensity(D):
    #np.random.seed(1234)
    
    shape=D[0].shape
    plt.close('all');I=[]
    for i in range(len(D)): I.append(np.zeros((bnS.size,shape[1])))
    bnR=np.arange(0,30,0.5)
    bnS=np.diff(np.pi*bnR**2)
    bnd= bnR+(bnR[1]-bnR[0])/2.0;bnd=bnd[:-1]
    plot=False
    g=1;figs=range(1,100);fig1=figs.pop(0);fig2=figs.pop(0)
    r=np.random.permutation(D.shape[0])
    sir=np.copy(si);sir=sir[r,:]
    for f in range(0,D.shape[1]):
        H=[]
        for q in range(len(D)):
            H.append(D[q][:,f,:,:].reshape([shape[0]*shape[2],1,1,2]).squeeze())
        if plot:
            if g>12: fig1=figs.pop(0); fig2=figs.pop(0); g=1
            plt.figure(fig1)
            plt.subplot(3,4,g)
            plt.title('Frame %.1f'%(f*1000/hz+sw))
            bn=np.linspace(-20,20,41)
            print H[0][0,0],H[0][0,1], D[0][0,f,0,0], D[0][0,f,0,1]
            b1,c,d=np.histogram2d(H[0][:,0],H[0][:,0],bins=[bn,bn])
            b2,c,d=np.histogram2d(H[1][:,0],H[1][:,0],bins=[bn,bn])
            plt.imshow(b1-b2, extent=[-20,20,-20,20],origin='lower',
                       interpolation='nearest')#,vmin=-140,vmax=40)
            plt.set_cmap('hot')
            plt.colorbar()
            plt.figure(fig2)
            plt.subplot(3,4,g)
            plt.title('Frame %.1f'%(f*1000/hz+sw))
            g+=1
        for q in range(len(D)): 
            a,b=np.histogram(np.sqrt(np.power(H[q],2).sum(axis=1)),bins=bnR)
            I[q][:,f]=a/bnS/n*(bnR.size-1)
            if q==2: I[q][0,f]=np.nan
            if plot: plt.plot(bnd,I[q][:,f])
##            blb=np.copy(bnS);blb[0]=0
##            if plot:
##               plt.plot(bnd,a/blb/n*(bnR.size-1))
        if plot:
            plt.ylim([0,2])
            plt.xlabel('Distance from Saccade Target Location')
            plt.ylabel('Agent Density [a per deg^2]') #[# agents per deg^2]
            if g==2:plt.legend(['rand pos','gaze','rand ag','rand time'])
    plt.figure()
    plt.imshow((I[0]-I[1]),extent=[sw,ew,bnR[0],bnR[-1]],aspect=10,origin='lower')
    plt.ylabel('radial distance from sac target')
    plt.xlabel('time from sac onset')
    plt.colorbar();plt.set_cmap('hot')
    plt.figure()
    for q in I: plt.plot(bnd,q.mean(1))
    plt.xlabel('radial distance')
    plt.ylabel('Agent Density [a per deg^2]')
    plt.legend(['gaze','rand pos','rand agent','rand time'])
    plt.figure()
    x=np.linspace(sw,ew,I[0].shape[1])
    hhh=10
    for q in I: plt.plot(x[:hhh],q[:hhh,:].mean(0))
    plt.xlabel('time')
    plt.ylabel('Agent Density [a per deg^2]')
    plt.legend(['gaze','rand pos','rand agent','rand time'],loc=2)
    



def dirChanges(D):
    plt.close('all'); P=[];J=[];K=[];nK=[]
    shape=D[0].shape
    bn=np.arange(0,30,0.5)
    d= bn+(bn[1]-bn[0])/2.0;d=d[:-1]
    c=np.diff(np.pi*bn**2)
    for q in range(len(D)):
        J.append(np.zeros((shape[0],shape[1]-2,shape[2])))
        P.append(np.zeros((D.shape[0],D.shape[1]-2,D.shape[2])))
        K.append(np.zeros((d.size,J.shape[1],J.shape[2])))
        nK.append(np.zeros((d.size,J.shape[1],J.shape[2])))
    tol=0.0001
    for a in range(D.shape[2]):
        for q in range(len(D)):
            for f in range(1,D.shape[1]-1):
                P[q][:,f-1,a]=np.sqrt(np.power(D[q][:,f,a,:],2).sum(1))
        J[q][:,:,a]= np.logical_or(np.abs(np.diff(D[q][:,:,a,1],2))>tol,
                np.abs(np.diff(D[q][:,:,a,0],2))>tol)
    for i in range(d.size):
        for f in range(shape[1]-2):
            for a in range(shape[2]):
                sel=[]
                for q in range(len(D)):
                    if i==len(bn)-1:
                        sel.append(P[q][:,f,a]>bn[i])
                    else:
                        sel.append(np.logical_and(P[q][:,f,a]>bn[i],dH[:,f,a]<bn[i+1]))
                    K[q][i,f,a]=np.sum(J[q][sel[q],f,a],axis=0)*hz
                    nK[q][i,f,a]=np.sum(sel[q])
    sig=[0.5,0.5]
    for q in range(len(D)):
        plt.imshow(np.nansum(K[q],2)/np.sum(nK[q],2),origin='lower',extent=[sw,ew,bn[0],bn[-1]],aspect=20,cmap='hot',vmax=11,vmin=2)     
        plt.colorbar()
        plt.figure()
    for q in range(len(D)):
        plt.plot(d,np.nansum(np.nansum(K[q],2),1)/ nK[q].sum(2).sum(1))
    plt.legend(['gaze','rand pos','rand ag','rand time'])
    plt.figure()
    x=np.linspace(sw,ew,KH.shape[1])
    kk=10
    for q in range(len(D)):
        plt.plot(x,np.nansum(np.nansum(K[q][:kk,:,:],2),0)/ nK[q][:kk,:,:].sum(2).sum(0))
    plt.legend(['gaze','rand pos','rand ag','rand time'])


### different method for extracting radial density, gives similar results
##H,C,G,bn=agentDensity(D,si)
##def radialHist(M,bn,nn):
##    from scipy.interpolate import interp1d
##    b,c,d=np.histogram2d(M[:,0],M[:,1],bins=[bn,bn])
##    # find bin center
##    d= bn+(bn[1]-bn[0])/2.0
##    d=d[:-1]
##    # get unique data points
##    xa,xb=np.meshgrid(d,d)
##    x=np.sqrt(xa**2+xb**2)
##    x=np.reshape(x,x.size)
##    y=np.reshape(b,b.size)
##    nx=np.unique(x)
##    ny=[]
##    for k in nx.tolist():
##        ny.append(y[x==k].mean())
##    ny=np.array(ny)
##    # interpolate
##    f=interp1d(nx,ny)
##    plt.plot(nn,f(nn))
##
##plt.figure()
##radialHist(H,bn,range(1,26))
##radialHist(C,bn,range(1,26))
##radialHist(G,bn,range(1,26))
##plt.legend(['gaze','rand pos','rand ag'])

###ax = plt.gca(projection='3d')
##for g in [0]:#range(0,1000,100):
##    plt.figure()
##    ax=plt.gca()
##    for a in range(14):
##        x=D[g,:,a,0]-si[g,1,2]
##        y=D[g,:,a,1]-si[g,1,3]
##        plt.plot(x, y,'k')
##        arr=plt.arrow(x[-2], y[-2],x[-1]-x[-2],y[-1]-y[-2],width=0.1,fc='k')
##        ax.add_patch(arr)
##        #arr=plt.arrow(x[49], y[49],x[50]-x[49],y[50]-y[49],width=0.1,fc='k')
##        #ax.add_patch(arr)
##        plt.plot(x[50],y[50],'ok')
##    rect=plt.Rectangle([-2.5,-2.5],5,5,fc='white')
##    ax.add_patch(rect)
##    plt.plot(0,0,'or')
##    plt.xlim([-10,10])
##    plt.ylim([-10,10])
##    ax.set_aspect('equal')
##    #ax.legend()


if __name__ == '__main__':
    from matplotlib.mlab import PCA
    from test import princomp
    E,D,si=loadData()
    path= path.rstrip('G/')+'R/'
    bla,Dr,blah=loadData()
    #J=dirChanges(Dr)
    agentDensity(Dr)

    #plt.close('all')
    
##    F=np.zeros((E.shape[0],E.shape[3]*E.shape[1]))
##    for k in range(E.shape[0]):
##        F[k,:]= E[k,:,0,:].reshape([F.shape[1],1]).squeeze()
##    pcs,score,lat=princomp(F)
##    plt.plot(lat)
##    plt.figure()
##    for i in range(6):
##        plt.subplot(2,3,i+1)
##        b=pcs[:,i].reshape([E.shape[1],E.shape[3]])
##        plt.plot(b[:,0],b[:,1]);
    plt.show()

