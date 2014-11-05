import numpy as np
import pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import nanmean
from scipy.stats import scoreatpercentile as sap
import pickle, os
plt.close('all')
plt.ion()
dtos=['G','A','T']
radbins=np.arange(1,15)
bs=range(1,22);hz=85.0 #sampling frequency 

def initVP(vpl=1,evl=1):
    '''event - 0: search saccades,1: 1st tracking saccade,2:second ...
            -1: -250 sec prior to button press, -2: -300 sec to BP ...'''
    global vp, path, figpath, event,bs,sw,ew,hz,fw,radbins
    vp=vpl; event=evl
    path=os.getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
    figpath=os.getcwd().rstrip('code')+'/figures/Analysis/'
    if event>=0: sw=-400; ew=400;# start and end (in ms) of the window
    else: sw=-800; ew=0
    fw=int(np.round((abs(sw)+ew)/1000.0*hz))
    

def saveFigures(name):
    for i in range(1,plt.gcf().number-1):
        plt.figure(i)
        plt.savefig(figpath+'E%d'%event+name+'%02dvp%03d.png'%(i,vp))


        
def plotSearchInfo(plot=True):
    D=np.load(path+'si.npy')
    if event<0:
        d=np.load(path+'finalev.npy')
        print 'finalev len:',d.shape
        if plot:
            plt.plot(d[:,event,1],d[:,event,2],'.')
            plt.grid(b=False);plt.xlim([-15,15])
            plt.gca().set_aspect(1)
        d=d[:,[0,event],:]
        sel=np.all(~np.isnan(d[:,1,:]),axis=1 )
        d=d[sel,:,:]
        print 'finalev len2:',d.shape
        si=np.zeros((d.shape[0],D.shape[1]))*np.nan
        si[:,[-2,-1]]=d[:,0,:2]
        si[:,[1,6,7]]=d[:,1,:]
        si[:,12]=np.inf # set to something greater than d[:,1,0]
        return si,[]
    inds=D[D[:,14]>0,14]
    phi=np.load(path+'phiTrack.npy')
    phi=phi[inds==event,-1,1] # return median values
    D=D[D[:,14]==event,:]
    if event>0: assert phi.shape[0]==D.shape[0]
    if event==1:
        # discard tracking events initiated by tracking or blink
        sel=np.logical_and(~np.isnan(D[:,11]),D[:,11]-D[:,4]>=0)
        D=D[sel,:];phi=phi[sel]
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
        #saveFigures('si')
    return D,phi


def extractTrajectories():
    si,phi=plotSearchInfo(plot=False)
    inpath=os.getcwd().rstrip('code')+'input/' 
    sf=np.int32(np.round((si[:,1]+sw)/1000.0*hz))
    ef=sf+fw
    Np=100;rp=[]; # number of random replications for DP CI calculation
    valid= np.logical_and(si[:,1]+sw>=0, si[:,1]+ew <= si[:,12])
    print 'proportion of utilized samples is', valid.mean(),' total = ',valid.sum()
    if not os.path.exists(path+'E%d'%(event)):
        os.makedirs(path+'E%d'%(event))
    np.save(path+'E%d/si.npy'%(event),si[valid])
    if event>0:np.save(path+'E%d/phi2.npy'%(event),phi[valid])
    return
    D=np.zeros((valid.sum(),fw,14,3))*np.nan
    DG=np.zeros((valid.sum(),fw,14,3))*np.nan
    DT=np.zeros((valid.sum(),fw,14,2))*np.nan
    #DP=np.zeros((valid.sum(),fw,14,2))*np.nan
    DA=np.zeros((valid.sum(),fw,14,2))*np.nan
    temp=np.zeros((valid.sum(),Np))*np.nan
    k=0; lastt=-1;
    rt=np.random.permutation(valid.nonzero()[0])
    for h in range(si.shape[0]):
        if not valid[h]: continue
        if si[h,-1]!=lastt:
            order=np.load(inpath+'vp%03d/ordervp%03db%d.npy'%(vp,vp,si[h,-2]))
            traj=np.load(inpath+'vp001/vp001b%dtrial%03d.npy'%(si[h,-2],order[si[h,-1]]) )
        D[k,:,:,:]=traj[sf[h]:ef[h],:,:]
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
            DG[:,f,a,2]= D[:,f,a,2]
# DP was used to compute random level with bootstrapping
# was abandoned because it takes too long to compute the CI from samples
##    for k in range(Np):
##        for a in range(14):
##            for f in range(fw):
##                for q in range(2):
##                    DP[:,f,a,q]= D[:,f,a,q] - si[rp[k],6+q]
##        np.save(path+'E%d/DP%02d.npy'%(event,k),DP)
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
        H=[]
        for q in range(len(D)):
            H.append(D[q][:,f,:,:2].reshape([n,1,1,2]).squeeze())
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
                P[:,f-1,a]=np.sqrt(np.power(D[:,f,a,:2],2).sum(1))
            # compute 2nd order differential of position for each agent, event
            # non-zero 2nd order diff - direction change 
            J[:,:,a]= np.logical_or(np.abs(np.diff(D[:,:,a,1],n=2,axis=1))>tol,
                np.abs(np.diff(D[:,:,a,0],n=2,axis=1))>tol)
        # compute 
        for f in range(shape[1]-2):
            #print f,q
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
        inpath=os.getcwd().rstrip('code')+'input/'
        sf=np.int32(np.round((si[:,1]+sw)/1000.0*hz))
        ef=sf+fw
        valid= np.logical_and(si[:,1]+sw>=0, si[:,1]+ew <= np.minimum(si[:,12],30000))
        print si.shape, valid.sum()
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
        for h in range(si.shape[0]):
            if si[h,-2]>21: continue
            if si[h,-1]!=lastt: # avoid reloading saliency map for the same trial 
                order = np.load(inpath+'vp%03d/ordervp%03db%d.npy'%(vp,vp,si[h,-2]))
                #traj=np.load(inpath+'vp%03d/vp%03db%dtrial%03d.npy'%(vp,vp,si[h,-2],order[si[h,-1]]))
                fname=path.rstrip('vp%03d/'%vp)+'/saliency/vp001b%dtrial%03d%s-.%dx%d.mgrey'%(int(si[h,-2]),order[si[h,-1]],channel,dim,dim)
                try: vid=Mraw(fname)
                except IOError:
                    print 'missing file: '+ fname
            
            #print sf[h],ef[h],ef[h]-sf[h],fw,si[h,12]
            temp1,temp2=vid.computeSaliency(si[h,[6,7]],[sf[h],ef[h]],rdb=radbins)
            if not temp1 is None: gridG+=temp1; radG+=temp2.T;k+=1
            else: print 'saccade ignored ',h,si[h,-2],si[h,-1]
            lastt=si[h,-1]
            if vp==4:
                temp1,temp2=vid.computeSaliency(si[rt[h],[6,7]],[sf[rt[h]],ef[rt[h]]],rdb=radbins)
                if not temp1 is None: gridT+=temp1; radT+=temp2.T;
            #temp1,temp2=vid.computeSaliency(traj[sf[h]+fw/2,2,:2],[sf[h],ef[h]],rdb=radbins)
            #gridA+=temp1; radA+=temp2.T;
            #temp1,temp2=vid.computeSaliency(si[rp[h],[6,7]],[sf[h],ef[h]],rdb=radbins)
            #gridP+=temp1; radP+=temp2.T;
            

    ##    grid=[gridG,gridT,gridP,gridA]
    ##    rad=[radG,radT,radP,radA]
    ##    for i in range(len(grid)):
    ##        grid[i]/=float(k);rad[i]/=float(k)
    ##        np.save(path+'E%d/grd%s%s.npy'%(event,dtos[i],channel),grid[i])
    ##        np.save(path+'E%d/rad%s%s.npy'%(event,dtos[i],channel),rad[i])
        gridG/=float(k);radG/=float(k)
        np.save(path+'E%d/grd%s.npy'%(event,channel),gridG)
        np.save(path+'E%d/rad%s.npy'%(event,channel),radG)
        if vp==4:
            gridT/=float(k);radT/=float(k)
            np.save(path+'E%d/grdT%s.npy'%(event,channel),gridT)
            np.save(path+'E%d/radT%s.npy'%(event,channel),radT)
    except:
        raise
        print 'Error', h,si[h,-2],si[h,-1]
        gridG/=float(k);radG/=float(k)
        np.save(path+'E%d/grd%s.npy'%(event,channel),gridG)
        np.save(path+'E%d/rad%s.npy'%(event,channel),radG)
        raise
        

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

##########################################
#
#       extract angles
#
##########################################

def extractPhi(PLOT=False):
    for vp in range(1,5):
        initVP(vp)
        f=open(path+'trdir.pickle','rb')
        dt=pickle.load(f)
        f.close()
        #sds=[]
        alp=0.05;clr='k'
        phis=np.zeros((len(dt),35,2))*np.nan
        for i in range(len(dt)):
            tr=np.reshape(dt[i],(len(dt[i])/2,2))
            tr-=tr[0]
            phis[i,:min(34,tr.shape[0]),:]=tr[:34,:]
            if PLOT:
                plt.figure(1,figsize=(10,10));plt.subplot(2,2,vp)
                plt.plot(tr[:34,0],tr[:34,1],'k',alpha=alp)
                plt.grid(b=False)
            temp=np.diff(tr,axis=0)
            if PLOT:
                plt.figure(2,figsize=(10,10));plt.subplot(2,2,vp)
                plt.plot(np.linalg.norm(temp,axis=1),'k',alpha=alp)
                plt.grid(b=False)
            phi=np.arctan2(temp[:,1],temp[:,0])
            
            pold=np.copy(phi)
            df=np.diff(phi)
            df=-np.sign(df)*(np.abs(df)>np.pi)
            phi[1:]+=np.cumsum(df)*2*np.pi
            
            if np.any(np.diff(phi)>np.pi):stop
            phis[i,-1,0]=phi[:34].mean()
            phis[i,-1,1]=np.median(phi)
            phi-=phis[i,-1,0]
            #sds.append(np.std(phi))
            # is exclusion of outliers needed?
            #if np.std(phi)>0.3: continue
            if PLOT:
                plt.figure(3,figsize=(10,10));plt.subplot(2,2,vp)
                plt.plot(np.linspace(0,phi.size,phi.size),phi/np.pi*180,clr,alpha=alp)
                plt.grid(b=False)
                plt.ylim([-60,60])
                c=np.cos(dd);s=np.sin(dd)
                plt.figure(1,figsize=(10,10));plt.subplot(2,2,vp)
                plt.plot([0,c],[0,s],'g',alpha=alp)
                plt.gca().set_aspect(1)
                plt.grid(b=False);plt.xlim([-1,1]);plt.ylim([-1,1])
        np.save(path+'phiTrack',phis)
        #print (np.array(sds)>0.3).mean()

if __name__ == '__main__':
    extractPhi()
    for event in range(-6,0):
    #for event in range(1,3):
        for vpl in range(1,5):
            initVP(vpl=vpl,evl=event)
            print vp, event
            extractTrajectories()
            #extractDensity()
            #extractDirC()
            #extractSaliency(channel='COmotion')
            #extractSaliency(channel='SOintensity')
    


    
    
    
    

    




