import numpy as np
import pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import nanmean
from scipy.stats import scoreatpercentile as sap
from os import getcwd
plt.close('all')
plt.ion()


def initVP(vpl=1):
    global vp
    global path
    global figpath
    vp=vpl
    path=getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
    figpath=getcwd().rstrip('code')+'/figures/Analysis/'


bs=range(1,22)
sw=-400; ew=400;hz=85.0 # start, end (in ms) and sampling frequency of the saved window
fw=int(np.round((abs(sw)+ew)/1000.0*hz))
radbins=np.arange(1,15)
event=1 # 0 - search saccades; 1- 1st tracking saccade, 2-second tracking saccade
dtos=['G','A','T','P']

def saveFigures(name):
    for i in range(1,plt.gcf().number-1):
        plt.figure(i)
        plt.savefig(figpath+'E%d'%event+name+'%02dvp%03d.png'%(i,vp))
        
def plotSearchInfo(plot=True):
    D=np.load(path+'si.npy')
    D=D[D[:,14]==event,:]
    if event==1:
        # discard tracking events initiated by tracking or blink
        sel=np.logical_or(np.isnan(D[:,11]),D[:,11]-D[:,4]>=0)
        D=D[sel,:]
    if plot:
        #plt.close('all')

        plt.figure()
        lim=12
        bn=np.linspace(-lim,lim,21)
        bb,c,d=np.histogram2d(D[:,6],D[:,7],bins=[bn,bn])
        plt.imshow(bb, extent=[-lim,lim,-lim,lim],origin='lower',
                       interpolation='bicubic')#,vmin=-140,vmax=40)
        plt.colorbar()
        plt.grid()
        plt.title('Saccade Target Positions')
        
        plt.figure()
        reloc=np.sqrt(np.sum(np.power(D[:,[6,7]]-D[:,[2,3]],2),axis=1))
        plt.xlabel('Distance in Degrees')
        plt.hist(reloc,bins=np.linspace(0,20,41),normed=True)
        plt.xlim([0,20])
        plt.ylim([0,0.3])

        plt.figure()
        dur = D[:,5]-D[:,1]
        plt.hist(dur,bins=np.linspace(0,150,41),normed=True)
        plt.xlabel('Saccade Duration in ms')
        plt.xlim([0,150])
        plt.ylim([0,0.05])
        
        plt.figure()
        vel=reloc/dur
        plt.hist(vel[~np.isnan(vel)]*1000,50,normed=True)
        plt.xlabel('Velocity deg/ms')
        plt.xlim([0,200])
        plt.ylim([0,0.035])
        saveFigures('si')
    return D


def extractTrajectories():
    si=plotSearchInfo(plot=False)
    inpath=getcwd().rstrip('code')+'input/'
    sf=np.int32(np.round((si[:,1]+sw)/1000.0*hz))
    ef=sf+fw
    Np=100;rp=[]; # number of random replications for DP CI calculation
    valid= np.logical_and(si[:,1]+sw>=0, si[:,1]+ew <= si[:,12])
    print 'proportion of utilized samples is', valid.mean()
    D=np.zeros((valid.sum(),fw,14,2))*np.nan
    DG=np.zeros((valid.sum(),fw,14,2))*np.nan
    DT=np.zeros((valid.sum(),fw,14,2))*np.nan
    DP=np.zeros((valid.sum(),fw,14,2))*np.nan
    DA=np.zeros((valid.sum(),fw,14,2))*np.nan
    temp=np.zeros((valid.sum(),Np))*np.nan
    k=0; lastt=-1;
    rt=np.random.permutation(valid.nonzero()[0])
    for h in range(si.shape[0]):
        if not valid[h]: continue
        if si[h,-1]!=lastt:
            order=np.load(inpath+'vp%03d/ordervp%03db%d.npy'%(vp,vp,si[h,-2]))
            traj=np.load(inpath+'vp001/vp001b%dtrial%03d.npy'%(si[h,-2],order[si[h,-1]]) )
        D[k,:,:,:]=traj[sf[h]:ef[h],:,:2]
        DT[k,:,:,:]=traj[sf[rt[k]]:ef[rt[k]],:,:2]
        k+=1; lastt=si[h,-2]
    for k in range(Np): rp.append(np.random.permutation(valid.nonzero()[0]))
    #sel=si[:,8]>0#np.logical_not(np.all(np.all(np.all(np.isnan(D[:,:14]),3),2),1))
    #DG=DG[sel,:,:,:]
    for a in range(14):
        for f in range(fw):
            for q in range(2):
                DG[:,f,a,q]= D[:,f,a,q] - si[valid,6+q]
                DT[:,f,a,q]-= si[rt,6+q]
                DA[:,f,a,q]= D[:,f,a,q] - D[:,D.shape[1]/2,2,q]
    for k in range(Np):
        for a in range(14):
            for f in range(fw):
                for q in range(2):
                    DP[:,f,a,q]= D[:,f,a,q] - si[rp[k],6+q]
        np.save(path+'E%d/DP%02d.npy'%(event,k),DP)
    D=[DG,DA,DT]       
    for d in range(len(D)): np.save(path+'E%d/D%s.npy'%(event,dtos[d]),D[d])
def extractDensity():
    from matustools.matusplotlib import histCI
    D=[]
    for d in range(len(dtos)): D.append(np.load(path+'E%d/D%s.npy'%(event,dtos[d])))
    shape=D[0].shape
    #plt.close('all');
    I=[]
    bnR=np.arange(0,30,0.5)
    bnS=np.diff(np.pi*bnR**2)
    bnd= bnR+(bnR[1]-bnR[0])/2.0;bnd=bnd[:-1]
    for i in range(len(D)+3): I.append(np.zeros((bnS.size,shape[1],3)))
    g=1;n=shape[0]*shape[2]
    for f in range(0,shape[1]):
        print f
        H=[]
        for q in range(len(D)):
            H.append(D[q][:,f,:,:].reshape([n,1,1,2]).squeeze())
            a,b,l,u=histCI(np.sqrt(np.power(H[q],2).sum(axis=1)),bins=bnR,plot=False)
            I[q][:,f,0]=a/np.float32(shape[0]*bnS)
            I[q][:,f,1]=l/np.float32(shape[0]*bnS)
            I[q][:,f,2]=u/np.float32(shape[0]*bnS)
            #if q==2: I[q][0,f]=np.nan
    np.save(path+'E%d/agdens'%event,I)


def extractDirC():
    ''' extract direction change information
    '''
    D=[]
    #plt.close('all');
    #print D[0].shape
    K=[];nK=[]
    bn=np.arange(0,20,0.5)
    d= bn+(bn[1]-bn[0])/2.0;d=d[:-1]
    c=np.diff(np.pi*bn**2)
    tol=0.0001
    for q in range(len(dtos)):
        D=np.load(path+'E%d/D%s.npy'%(event,dtos[q]))
        shape=D.shape
        J=np.zeros((shape[0],shape[1]-2,shape[2]))
        P=np.zeros((shape[0],shape[1]-2,shape[2]))
        K.append(np.zeros((d.size,shape[1]-2,shape[2])))
        nK.append(np.zeros((d.size,shape[1]-2,shape[2])))
        for a in range(shape[2]):
            for f in range(1,shape[1]-1):
                # compute distance from saccade center
                P[:,f-1,a]=np.sqrt(np.power(D[:,f,a,:],2).sum(1))
            # compute 2nd order differential of position for each agent, event
            # non-zero 2nd order diff - direction change 
            J[:,:,a]= np.logical_or(np.abs(np.diff(D[:,:,a,1],n=2,axis=1))>tol,
                np.abs(np.diff(D[:,:,a,0],n=2,axis=1))>tol)
        # compute 
        for f in range(shape[1]-2):
            print f,q
            for i in range(d.size):
                for a in range(shape[2]):
                    if i==len(bn)-1:
                        sel=P[q][:,f,a]>bn[i]
                    else:
                        sel=np.logical_and(P[:,f,a]>bn[i],P[:,f,a]<bn[i+1])
                    K[q][i,f,a]=np.sum(J[sel,f,a],axis=0)
                    nK[q][i,f,a]=np.sum(sel)
    np.save(path+'E%d/dcK'%event,K)
    np.save(path+'E%d/dcnK'%event,nK)

def extractSaliency(channel='intensity'):
    ''' the saliency for each trial and position has been precomputed and saved
    '''
    print vp, channel
    try:
        from ezvisiontools import Mraw
        si=plotSearchInfo(plot=False)
        inpath=getcwd().rstrip('code')+'input/'
        sf=np.int32(np.round((si[:,1]+sw)/1000.0*hz))
        ef=sf+fw
        valid= np.logical_and(si[:,1]+sw>=0, si[:,1]+ew <= si[:,12])
        si=si[valid,:]
        sf=sf[valid]
        ef=ef[valid]
        lastt=-1;dim=64;k=0
        gridG=np.zeros((fw,dim,dim));radG=np.zeros((fw,radbins.size))
        gridT=np.zeros((fw,dim,dim));radT=np.zeros((fw,radbins.size))
        #gridA=np.zeros((fw,dim,dim));radA=np.zeros((fw,radbins.size))
        #gridP=np.zeros((fw,dim,dim));radP=np.zeros((fw,radbins.size))
        rt=np.random.permutation(si.shape[0])
        rp=np.random.permutation(si.shape[0])
        #print si.shape[0]
        for h in range(si.shape[0]):
            if si[h,-1]!=lastt:
                order = np.load(inpath+'vp%03d/ordervp%03db%d.npy'%(vp,vp,si[h,-2]))
                #traj=np.load(inpath+'vp%03d/vp%03db%dtrial%03d.npy'%(vp,vp,si[h,-2],order[si[h,-1]]))
                fname=path.rstrip('vp%03d/'%vp)+'/saliency/vp001b%dtrial%03d%s-.%dx%d.mgrey'%(int(si[h,-2]),order[si[h,-1]],channel,dim,dim)
                try: vid=Mraw(fname)
                except: print 'missing saliency file',vp,int(si[h,-2]),order[si[h,-1]],fname
            temp1,temp2=vid.computeSaliency(si[h,[6,7]],[sf[h],ef[h]],rdb=radbins)
            gridG+=temp1; radG+=temp2.T; lastt=si[h,-2];
            if vp==1:
                temp1,temp2=vid.computeSaliency(si[rt[h],[6,7]],[sf[rt[h]],ef[rt[h]]],rdb=radbins)
                gridT+=temp1; radT+=temp2.T;
            #temp1,temp2=vid.computeSaliency(traj[sf[h]+fw/2,2,:2],[sf[h],ef[h]],rdb=radbins)
            #gridA+=temp1; radA+=temp2.T;
            #temp1,temp2=vid.computeSaliency(si[rp[h],[6,7]],[sf[h],ef[h]],rdb=radbins)
            #gridP+=temp1; radP+=temp2.T;k+=1

    ##    grid=[gridG,gridT,gridP,gridA]
    ##    rad=[radG,radT,radP,radA]
    ##    for i in range(len(grid)):
    ##        grid[i]/=float(k);rad[i]/=float(k)
    ##        np.save(path+'E%d/grd%s%s.npy'%(event,dtos[i],channel),grid[i])
    ##        np.save(path+'E%d/rad%s%s.npy'%(event,dtos[i],channel),rad[i])
        gridG/=float(k);radG/=float(k)
        np.save(path+'E%d/grd%s.npy'%(event,channel),gridG)
        np.save(path+'E%d/rad%s.npy'%(event,channel),radG)
        if vp==1:
            gridT/=float(k);radT/=float(k)
            np.save(path+'E%d/grdRT%s.npy'%(event,channel),gridT)
            np.save(path+'E%d/radRT%s.npy'%(event,channel),radT)
    except:
        print k,si[h,-2],si[h,-1]
        

def plotSaliency():
    K=[];I=[]
    for i in range(len(dtos)):
        K.append(np.load(path+'E%d/grd%sS.npy'%(event,dtos[i])))
        I.append(np.load(path+'E%d/rad%sS.npy'%(event,dtos[i])).T)

    plt.figure();plot=False
    if plot:
        for q in I:
            plt.imshow(q,extent=[sw,ew,radbins[0],radbins[-1]],
                aspect=30,origin='lower',vmin=0.008, vmax=0.021)
            plt.ylabel('radial distance from sac target')
            plt.xlabel('time from sac onset')
            plt.title('saliency')
            plt.colorbar();plt.set_cmap('hot')
            plt.figure()
    for q in I: plt.plot(radbins,q.mean(1))
    plt.xlabel('radial distance')
    plt.ylabel('Average Saliency [0,1]')
    plt.legend(['gaze','rand time','rand agent','rand pos'])
    plt.figure()
    x=np.linspace(sw,ew,I[0].shape[1])
    hhh=3
    for q in I: plt.plot(x,nanmean(q[:hhh,:],0))
    plt.xlabel('time')
    plt.title('Radius=%d deg'%hhh)
    plt.ylabel('Average Saliency [0,1]')
    plt.xlabel('time to saccade onset')
    plt.legend(['gaze','rand time','rand agent','rand pos'],loc=2)
    saveFigures('sa')
    #plt.show()

def pcaMotionSingleAgent(D,ag=3,plot=False):
    for q in range(len(D)):
        E=D[q]
        F=np.zeros((E.shape[0],E.shape[3]*E.shape[1]))
        for k in range(E.shape[0]):
            F[k,:]= E[k,:,ag,:].reshape([F.shape[1],1]).squeeze()
        pcs,score,lat=princomp(F)
        plt.figure(1)
        plt.plot(lat)
        plt.legend(['gaze','rand pos','rand ag','rand time'])
        plt.figure(2)
        for i in range(6):
            plt.subplot(2,3,i+1)
            b=pcs[:,i].reshape([E.shape[1],E.shape[3]])
            plt.plot(b[:,0],b[:,1]);
        plt.legend(['gaze','rand pos','rand ag','rand time'])
    if plot:
        # items
        plt.figure()
        plt.plot(score[0,:],score[1,:],'.');plt.gca().set_aspect(1);
        plt.figure()
        plt.plot(score[2,:],score[3,:],'.');plt.gca().set_aspect(1);
        plt.figure()
        plt.plot(score[4,:],score[5,:],'.');plt.gca().set_aspect(1);
        # traj reconstraction with help of first few components
        for k in range(10):
            res=0
            for i in range(6): res+=pcs[:,i]*score[i,k]
            res=res.reshape([E.shape[1],E.shape[3]])
            plt.figure()
            plt.plot(E[k,:,0,0],E[k,:,0,1]);
            plt.plot(res[:,0],res[:,1]);

def pcaMotionMultiAgent(D):
    q=1
    E=D[q]
##    F=np.zeros((E.shape[0],E.shape[3]*E.shape[1]*E.shape[2]))
##    for k in range(E.shape[0]):
##        B=np.copy(E[k,:,:,:])
##        r=np.random.permutation(range(E.shape[2]))
##        #B=B[:,r,:]
##        F[k,:]= B.reshape([F.shape[1],1]).squeeze()
##    pcs,score,lat=princomp(F)
##    np.save('pcs%d'%q,pcs)
##    np.save('lat%d'%q,lat)
##    np.save('score%d'%q,score)
    
    pcs=np.load('pcs%d.npy'%q)
    score=np.load('score%d.npy'%q)
    lat=np.load('lat%d.npy'%q)
    plt.figure(1)
    plt.plot(lat)
    plt.xlim([0,30])
    #plt.legend(['gaze','rand pos','rand ag','rand time'])
    plt.figure(2)
    for i in range(30):
        plt.subplot(5,6,i+1)
        b=pcs[:,i].reshape([E.shape[1],E.shape[2],E.shape[3]])
        plotTraj(b);
    
    
    
def plotTraj(traj):
    ax=plt.gca()
    clr=['g','r','k']
    for a in range(traj.shape[1]):
        x=traj[:,a,0];y=traj[:,a,1]
        plt.plot(x,y,clr[min(a,2)]);
        #arr=plt.arrow(x[-2], y[-2],x[-1]-x[-2],y[-1]-y[-2],width=0.2,fc='k')
        plt.plot(x[-1],y[-1],'.'+clr[min(a,2)])
        #ax.add_patch(arr)
        midf=traj.shape[0]/2
        #arr=plt.arrow(x[midf], y[midf],x[midf+1]-x[midf],y[midf+1]-y[midf],width=0.1,fc='g')
        #ax.add_patch(arr)

    #rect=plt.Rectangle([-5,-5],10,10,fc='white')
    #ax.add_patch(rect)
    plt.plot(0,0,'ob')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    #plt.xlim([-20,20])
    #plt.ylim([-20,20])
    ax.set_aspect('equal')
##def pcaScript():
##    from matplotlib.mlab import PCA
##    from test import princomp
##
##    q=1
##    E=D[q]
##    F=[]
##    for q in range(2):
##        F.append(np.zeros((E.shape[0],E.shape[3]*E.shape[1]*E.shape[2])))
##        for k in range(E.shape[0]):
##            B=np.copy(E[k,:,:,:])
##            F[q][k,:]= B.reshape([F[q].shape[1],1]).squeeze()
##    x=np.concatenate(F,axis=0)
##    x=np.concatenate([-np.ones((x.shape[0],1)), x],axis=1)
##    y=np.zeros((F[q].shape[0]*2))
##    y[:F[q].shape[0]]=1
##    del D, E, F





   

if __name__ == '__main__':
##    for event in [0]:
##        for vpl in [1]:#range(1,5):
##            initVP(vpl=vpl)
##            #extractTrajectories()
##            #agentDensity()
##            #extractDensity()
##            #extractDirC()
    initVP(vpl=4)
    extractSaliency(channel='SOintensity')
    


    
    
    
    

    




