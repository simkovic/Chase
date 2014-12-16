import numpy as np
import pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import nanmean,norm,nanmedian
from scipy.stats import scoreatpercentile as sap
from scipy.interpolate import interp1d
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
    figpath=os.getcwd().rstrip('code')+'figures/analysis/'
    if event>=0: sw=-400; ew=400;# start and end (in ms) of the window
    else: sw=-800; ew=0
    fw=int(np.round((abs(sw)+ew)/1000.0*hz))

##########################################
#
#       compute rotation angles
#
##########################################

def computePhi(path,PLOT=False):
    f=open(path+'purxy.pickle','rb')
    purxy=pickle.load(f)
    f.close()
    phis=np.zeros(len(purxy))*np.nan
    alp=0.9; f1lim=2
    for i in range(1000):#len(purxy)):
        pxy=np.reshape(purxy[i],(len(purxy[i])/2,2))
        pxy-=pxy[pxy.shape[0]/2]
        if PLOT:
            plt.figure(1,figsize=(10,10));plt.subplot(2,2,vp)
            plt.plot(pxy[:,0],pxy[:,1],'k',alpha=alp)
            plt.grid(b=False);plt.title('centered trajectory')
        temp=np.diff(pxy,axis=0)
        if PLOT:
            plt.figure(2,figsize=(10,10));plt.subplot(2,2,vp)
            plt.plot(np.linalg.norm(temp,axis=1),'k',alpha=alp)
            plt.grid(b=False);plt.title('velocity')
        phi=np.arctan2(temp[:,1],temp[:,0])
        
        pold=np.copy(phi)
        df=np.diff(phi)
        df=-np.sign(df)*(np.abs(df)>np.pi)
        phi[1:]+=np.cumsum(df)*2*np.pi
        
        assert np.all(np.diff(phi)<np.pi)
        phis[i]=np.median(phi)
        phi-=phis[i]

        if PLOT:
            plt.figure(3,figsize=(10,10));plt.subplot(2,2,vp)
            plt.plot(np.linspace(0,phi.size,phi.size),
                     phi/np.pi*180,'k',alpha=alp)
            plt.grid(b=False)
            plt.ylim([-60,60])
            c=np.cos(phis[i])*f1lim;s=np.sin(phis[i])*f1lim
            plt.figure(1,figsize=(10,10));plt.subplot(2,2,vp)
            #plt.plot([-c,c],[-s,s],'g',alpha=alp)
            plt.gca().set_aspect(1);plt.grid(b=False);
            plt.xlim([-f1lim,f1lim]);plt.ylim([-f1lim,f1lim])
            plt.figure(4,figsize=(10,10));plt.subplot(2,2,vp);
            plt.grid(b=False)
            plt.plot(np.linspace(0,phi.size,phi.size),
                     np.sort(phi)/np.pi*180,'k',alpha=alp)
    return phis


    

def saveFigures(name):
    for i in range(1,plt.gcf().number-1):
        plt.figure(i)
        plt.savefig(figpath+'E%d'%event+name+'%02dvp%03d.png'%(i,vp))

        
def plotSearchInfo(plot=True,suf=''):
    si=np.load(path+'si.npy')
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
        si=np.zeros((d.shape[0],si.shape[1]))*np.nan
        si[:,[-2,-1]]=d[:,0,:2]
        si[:,[1,6,7]]=d[:,1,:]
        si[:,12]=np.inf # set to something greater than d[:,1,0]
        return si,[]
    if suf=='3':
        from Constants import OLPUR,CLPUR
        phi=computePhi(path)
        sel=np.logical_or(si[:,10]==OLPUR,si[:,10]==CLPUR)
        sel=np.logical_and(si[:,14]>0,sel)
        si=si[sel,:]
        phi=phi[si[:,14]==event]
    else: pass
    
    si=si[si[:,14]==event,:]
    if event==1:
        # discard tracking events initiated not by saccade (tracking or blink)
        sel=np.logical_and(~np.isnan(si[:,11]),si[:,11]-si[:,4]>=0)
        si=si[sel,:];
        if suf=='3':phi=phi[sel]
    if suf=='3' and event>0: assert phi.shape[0]==si.shape[0]
    if plot:
        #plt.close('all')
        plt.figure()
        lim=12
        bn=np.linspace(-lim,lim,21)
        bb,c,d=np.histogram2d(si[:,6],si[:,7],bins=[bn,bn])
        plt.imshow(bb, extent=[-lim,lim,-lim,lim],origin='lower',
                       interpolation='bicubic')#,vmin=-140,vmax=40)
        plt.colorbar()
        plt.grid()
        plt.title('Saccade Target Positions')
        
        plt.figure()
        reloc=np.sqrt(np.sum(np.power(si[:,[6,7]]-si[:,[2,3]],2),axis=1))
        plt.xlabel('Distance in Degrees')
        plt.hist(reloc,bins=np.linspace(0,20,41),normed=True)
        plt.xlim([0,20])
        plt.ylim([0,0.3])

        plt.figure()
        dur = si[:,5]-D[:,1]
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
    if suf=='3': return si,phi
    else: return si


def extractTrajectories(suf=''):
    si,phi=plotSearchInfo(plot=False,suf=suf)
    inpath=os.getcwd().rstrip('code')+'input/' 
    sf=np.int32(np.round((si[:,1]+sw)/1000.0*hz))
    ef=sf+fw
    Np=100;rp=[]; # number of random replications for DP CI calculation
    valid= np.logical_and(si[:,1]+sw>=0, si[:,1]+ew <= si[:,12])
    print 'proportion of utilized samples is', valid.mean(),' total = ',valid.sum()
    if not os.path.exists(path+'E%d'%(event)):
        os.makedirs(path+'E%d'%(event))
    np.save(path+'E%d/si.npy'%(event),si[valid])
    if suf=='3':np.save(path+'E%d/phi%s.npy'%(event,suf),phi[valid])
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
    for d in range(len(D)): np.save(path+'E%d/D%s%s.npy'%(event,dtos[d],suf),D[d])


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


#############################################################
#
#                       TRACKING
#
#############################################################

def _computePhi(g,sd=0.05,hz=500):
    ''' compute the movement direction 
        does basic smoothing with gaussian filter
        g - Tx2 array with gaze position
        sd - standard deviation of  the guass filter
        hz - rate of g
        output - vector Tx1 with direction in radians
    '''
    sdf=int(sd*hz) # standard deviation as nr of frames ( ms times hz)
    t=np.linspace(-2,2,4*sdf-1)
    gf=norm.pdf(t);gf/= gf.sum()
    temp=np.diff(g[:,1:],axis=0)
    phi=np.arctan2(temp[:,1],temp[:,0])
    df=np.diff(phi)
    df=-np.sign(df)*(np.abs(df)>np.pi)
    phi[1:]+=np.cumsum(df)*2*np.pi
    res=np.median(phi)
    if g.shape[0]<=(gf.size+2): return res*np.ones(g.shape[0]),g[:,0]
    phismooth=np.convolve(phi,gf,'valid')
    n=int(gf.size-1)/2
    return phismooth,g[(n+1):-n,0]
    #phi[n:-n]=phismooth
    #plt.plot(g[1:,0],phi*180/np.pi)

def _computeAgTime(trackxy,ti):
    txy=[]
    for ii in range(ti.shape[0]):
        txy.append([])
        for k in range(len(trackxy[ii][1]))[::-1]:
            s=max(ti[ii,:][4],trackxy[ii][1][k][0])
            e=min(ti[ii,:][5],trackxy[ii][1][k][1])
            if e>s: txy[-1].append([trackxy[ii][0][k],s,e])
    return txy

def plotDur():
    evs=([1,-1,-2,-3],range(1,5))
    plt.figure(0,figsize=(10,6))
    meddur=[]
    for vp in range(1,5):
        initVP(vp,1)
        ti=np.load(path+'ti.npy')
        dur=(ti[:,5]-ti[:,4])*2
        meddur.append([[],[]])

        for i in range(2):
            for k in range(len(evs[i])+1):
                if k==len(evs[i]): 
                    if i: sel=evs[i][-1]<ti[:,i]
                    else: sel=evs[i][-1]>ti[:,i]
                else: sel= evs[i][k]==ti[:,i]
                #print vp,k,sel.sum() ,np.std(dur[sel]), np.std(dur[sel])/float(sel.sum()**0.5)
                y,x=np.histogram(dur[sel],np.linspace(0,1000,21),normed=True)
                plt.subplot(2,4,i*4+vp)
                plt.plot(x[:-1],y)
                plt.grid(b=False)
                if np.any(sel):
                    meddur[-1][i].append(np.median(dur[sel]))
                    #print vp,k,int(np.median(dur[sel]))       
            plt.legend(map(str,evs[i])+['other'])
    plt.figure(1,figsize=(10,10))
    plt.subplot(221);plt.xlim([0,1200]);plt.grid(False)
    plt.hist(dur,bins=np.linspace(0,1200,50))
    for i in range(4):
        plt.subplot(223)
        x=range(len(evs[1])+1)
        plt.plot(x,meddur[i][1],'o-');plt.ylim([200,370])
        plt.xlim([x[0]-0.5,x[-1]+0.5]);plt.grid(b=False,axis='x')
        plt.gca().set_xticks(x);plt.ylabel('Ereignisdauer [ms]')
        plt.gca().set_xticklabels(['E1','E2','E3','E4','rest'])
        plt.legend(['VP1','VP2','VP3','VP4'],loc=2)
        
        plt.subplot(224)
        x=range(len(evs[0]))[::-1]
        clr=next(plt.gca()._get_lines.color_cycle)
        plt.plot(x,meddur[i][0][1:],'o-',color=clr);plt.ylim([200,370])
        plt.plot(x[0],meddur[i][0][0],'x',mew=2,color=clr)
        plt.xlim([x[-1]-0.5,x[0]+0.5]);plt.grid(b=False,axis='x')
        plt.gca().set_xticks(x);plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels(['E-1','E-2','E-3','rest'])
    plt.savefig(figpath+'trackDur')

def plotTimeVsAgcount():
    evs=([1,-1,-2,-3],range(1,5))
    plt.figure(0,figsize=(10,6))
    plt.figure(1,figsize=(10,6))

    for vp in range(1,5):
        initVP(vp,1)
        f=open(path+'trackxy.pickle','rb')
        trackxy=pickle.load(f);f.close()
        ti=np.load(path+'ti.npy') 
        txy=_computeAgTime(trackxy,ti)

        for i in range(2):
            for k in range(len(evs[i])+1):
                if k==len(evs[i]): sel=evs[i][-1]<ti[:,1]
                else: sel= evs[i][k]==ti[:,i]
                count=np.zeros(500)
                totc=np.zeros(500)
                countN=np.zeros(200)
                for j in sel.nonzero()[0]:
                    temp=np.zeros(500)
                    for ag in txy[j]:
                        count[(ag[1]-ti[j,:][4]):(ag[2]-ti[j,:][4])]+=1
                        tot=float(ti[j,:][5]-ti[j,:][4])
                        s=int(round(countN.size*(ag[1]-ti[j,:][4])/tot))
                        e=int(round(countN.size*(ag[2]-ti[j,:][4])/tot))
                        countN[s:e]+=1
                        temp[:tot]+=1
                    totc+=np.int32(temp>0)
                plt.figure(0);plt.subplot(2,4,i*4+vp);plt.grid(False)
                plt.plot(np.linspace(0,1000,500),count/totc)
                if vp==4:plt.legend(map(str,evs[i])+['other'],loc=4)
                plt.figure(1);plt.subplot(2,4,i*4+vp);plt.grid(False)
                plt.plot(np.linspace(0,1000,200),countN/float(sel.sum()))
    plt.figure(0)
    initVP(1,1);plt.savefig(figpath+'trackTimeCount')
    
def computeTrackInfo(allAgents=False):
    told=999;dist=[];dirch=[];
    gs=[];trajs=[]
    for vp in range(1,5):
        initVP(vp,1)
        inpath=os.getcwd().rstrip('code')+'input/' 
        f=open(path+'trackxy.pickle','rb')
        trackxy=pickle.load(f);f.close()
        ti=np.load(path+'ti.npy') 
        txy=_computeAgTime(trackxy,ti)
        for some in [gs,trajs,dist,dirch]:some.append([])
        for j in range(len(txy)):
            for some in [gs,trajs,dist,dirch]:some[-1].append([])
            txyall= np.array([range(14),[ti[j,4]]*14,[ti[j,5]]*14]).T.tolist()
            for ag in [txy[j],txyall][int(allAgents)]:
                g=np.array(trackxy[j][2])
                g=g.reshape([g.size/3,3])
                s=int(round(g[0,0]*hz/1000.0))+1; e=int(round(g[-1,0]*hz/1000.0))-1
                if ti[j,3]!=told:
                    order=np.load(inpath+'vp%03d/ordervp%03db%d.npy'%(vp,vp,ti[j,2]))
                    traj=np.load(inpath+'vp001/vp001b%dtrial%03d.npy'%(ti[j,2],order[ti[j,3]]) )
                traj=traj[s:e,ag[0],:2]
                dirch[-1][j].append(np.int32(np.linalg.norm(
                            np.diff(traj,2,axis=0),axis=1)>0.0001))
                ttraj=np.linspace(s*1000/hz,e*1000/hz,e-s)
                temp=np.zeros((e-s,2))
                temp[:,0]=interp1d(g[:,0],g[:,1])(ttraj)
                temp[:,1]=interp1d(g[:,0],g[:,2])(ttraj)
                gs[-1][j].append(temp)
                traj-=temp[:traj.shape[0],:]
                dist[-1][j].append(np.linalg.norm(traj,axis=1))
                phiold,tphi= _computePhi(g)
                phi=interp1d(tphi,phiold,bounds_error=False)(ttraj)
                inds=np.isnan(phi).nonzero()[0]
                for i in inds:
                    if i<phi.size/2: phi[i]=phiold[0]
                    else: phi[i]=phiold[-1]
                M=traj.shape[0]
                temp=np.zeros((M,2))
                c=np.cos(phi[:M]);s=np.sin(phi[:M])
                temp[:,0]=c*traj[:,0]+s*traj[:,1]
                temp[:,1]= -s*traj[:,0]+c*traj[:,1]
                trajs[-1][j].append(temp)
    return dist,trajs,dirch,gs
def plotAgdist():
    dist,discard,the,rest=computeTrackInfo()
    del discard,the,rest
    plt.figure(0,figsize=(10,8))
    for vp in range(1,5):
        xlim=500
        ys=dist[vp-1]
        dat=np.zeros((len(ys),int(hz*xlim/1000.0),2))*np.nan
        datrev=np.zeros((len(ys),int(hz*500/1000.0),2))*np.nan
        #datN=np.zeros((len(ys),xlim/20))
        for i in range(len(ys)):
            ao=np.argsort(map(np.median,ys[i]))
            if len(ys[i])==0:continue
            N=ys[i][ao[0]].size
            if N==0:continue
            dat[i,:min(dat.shape[1],N),0]=ys[i][ao[0]][:min(dat.shape[1],N)]
            datrev[i,-min(datrev.shape[1],N):,0]=ys[i][ao[0]][-min(datrev.shape[1],N):]
            N=ys[i][ao[-1]].size
            dat[i,:min(dat.shape[1],N),1]=ys[i][ao[-1]][:min(dat.shape[1],N)]
            datrev[i,-min(datrev.shape[1],N):,1]=ys[i][ao[-1]][-min(datrev.shape[1],N):]
        nrags=np.array(map(len,ys))
        ylims=[[[1,2.5]]*3,[[],[3,4],[3,5]]]
        for a in range(3)[::-1]:
            if a==2: sel=nrags>=(a+1)
            else: sel = nrags==(a+1)
            for i in range(2):
                if a==0 and i==1:continue
                plt.subplot(4,4,i*8+vp);plt.grid(b=False);#plt.ylim(ylims[i][a])
                plt.plot(np.linspace(0,xlim/1000.,dat.shape[1]),nanmedian(dat[sel,:,i],0));
                plt.subplot(4,4,i*8+vp+4);plt.grid(b=False);#plt.ylim(ylims[i][a])
                ss=datrev.shape[1]/hz
                plt.plot(np.linspace(-ss,0,datrev.shape[1]),nanmedian(datrev[sel,:,i],0));
    plt.subplot(441)
    plt.legend(['> 2','2','1'],loc=4)
    initVP(1,1)
    plt.savefig(figpath+'trackAgdist')

def computeMeanPF(P=129,T=85,pvar=0.3**2,allAgents=False,PLOT=False):
    ''' compute mean pf with gaussian kernel
        out - mean pf of size 4x3xPxPxT
        pvar - determines the sd of smoothing across pixels
        no smoothing along the time axis

        math formula: $$f(x,y,t)= \frac{1}{N}\sum_n^N w_n(\left[x-x_{nt},y-y_{nt}\right])$$
$$w_n(\mathbf{h},\mathbf{\Sigma}) \sim \exp \left(-\frac{1}{2}\mathbf{h^T} \mathbf{\Sigma^{-1}} \mathbf{h} \right)$$
    '''
    p=np.linspace(-5,5,P)
    pp=np.meshgrid(p,p)
    x,y=np.meshgrid(p,p)
    H=np.zeros((4,3,P,P,T))
    G=np.zeros((4,3,P,P,T))
    F=np.zeros((4,3,P,P,T))
    J=np.zeros((4,3,T,3))
    discard,trajs,the,rest=computeTrackInfo()
    for vp in range(4):
        for nrags in [[0,1,2],[14]][int(allAgents)]:
            ind=[nrags,0][int(allAgents)]
            print vp, nrags
            for traj in trajs[vp]:
                if nrags==2 and len(traj)<3: continue
                if nrags<2 and len(traj)!=(nrags+1): continue
                for atraj in traj:
                    t=0
                    for pos in atraj.tolist():
                        if t>=T: break
                        if np.any(np.isnan(pos)): continue
                        d=np.square(x-pos[0])+np.square(y-pos[1])
                        H[vp,ind,:,:,t]+= np.exp(-d/pvar/2.)
                        J[vp,ind,t,0]+=1;t+=1
                    t=0
                    for pos in atraj.tolist()[::-1]:
                        if t>=T: break
                        if np.any(np.isnan(pos)): continue
                        dist=np.square(x-pos[0])+np.square(y-pos[1])
                        G[vp,ind,:,:,-t]+= np.exp(-dist/pvar/2.)
                        J[vp,ind,-t,1]+=1;t+=1
                    btraj=atraj.tolist()
                    S=len(btraj)
                    for s in range(S):
                        pos=btraj[s]
                        #if t>=T: break
                        if np.any(np.isnan(pos)): continue
                        t=s+(T-S)/2
                        if t<0 or t>=T:continue
                        dist=np.square(x-pos[0])+np.square(y-pos[1])
                        F[vp,ind,:,:,t]+= np.exp(-dist/pvar/2.)
                        J[vp,ind,t,2]+=1
            for t in range(T):
                H[vp,ind,:,:,t]/=J[vp,ind,t,0]
                G[vp,ind,:,:,t]/=J[vp,ind,t,1]
                F[vp,ind,:,:,t]/=J[vp,ind,t,2]
    suf=['','All'][int(allAgents)]
    np.save(path+'trackPFforw'+suf,H)
    np.save(path+'trackPFback'+suf,G)
    np.save(path+'trackPFmid'+suf,F)
    np.save(path+'trackPFcount'+suf,J[:,:,0,0])
    return H,G,F

def plotMeanPF():
    initVP(4,1)
    H=np.load(path+'trackPFforw.npy')
    temp=np.load(path+'trackPFforwAll.npy')
    H=np.concatenate([H,temp],axis=1);H=H[:,[3,0,1,2],:,:,:]
    G=np.load(path+'trackPFback.npy')
    temp=np.load(path+'trackPFbackAll.npy')
    G=np.concatenate([G,temp],axis=1);G=G[:,[3,0,1,2],:,:,:]
    F=np.load(path+'trackPFmid.npy')
    F=np.concatenate([F,np.zeros(F.shape)],axis=1);F=F[:,[3,0,1,2],:,:,:]
    k=0
    for some in [H,G,F]:
        Fs=[]
        denom=[0.05,0.15,0.05,0.05]
        for i in range(some.shape[0]):
            Fs.append([])
            for j in range(some.shape[1]):
                #print np.nanmax(some[i,j,:,:,:])
                temp=some[i,j,:,:,:]/denom[j]
                temp[temp>1]=1
                Fs[-1].append(temp)
        initVP(1,1)
        from matustools.matusplotlib import plotGifGrid
        lbls=[['A',20,-10,95],['B',20,-10,235],['C',20,-10,370],
              ['D',20,-10,505],['VP1',20,65,-20],['VP2',20,200,-20],
              ['VP3',20,340,-20],['VP4',20,475,-20]]
        plotGifGrid(Fs,fn=figpath+['trackPFforw','trackPFback','trackPFmid'][k],
                    bcgclr=0.5,plottime=True,text=lbls);k+=1

    plt.figure(figsize=(10,6))
    for i in range(3):
        D=[G,F,H][i]
        for vp in range(4):
            d=D[vp,2,60:68,:,:].mean(0)
            T=d.shape[1];P=d.shape[0]
            plt.subplot(3,4,i*4+vp+1)
            t=[np.linspace(-T*1000/85,0,T),
               np.linspace(-T/2*1000/85,T/2*1000/85,T),
               np.linspace(0,T*1000/85,T)][i]
            if not i:plt.title('VP'+str(vp+1))
            else:  plt.gca().set_xticklabels([])
            p=np.linspace(-5,5,P)
            plt.pcolor(p,t,d.T,cmap='gray',vmax=0.05)
            plt.xlim([p[0],p[-1]])
            if vp: plt.gca().set_yticklabels([])
            else: 
                plt.ylabel(['A','B','C'][i],size=20,
                    rotation='horizontal',va='center',ha='right')
            if i==1: plt.ylim([-500,500])
    plt.savefig(figpath+'trackPFrail')



if __name__ == '__main__':
    #plotDur()
    #computeMeanPF(PLOT=False,allAgents=False)
    plotMeanPF()
    #for event in range(-6,0):
    #plotAgdist()
##    for event in range(1,3):
##        for vpl in range(1,5):
##            initVP(vpl=vpl,evl=event)
##            print vp, event
##            extractTrajectories(suf='3')
            #extractDensity()
            #extractDirC()
            #extractSaliency(channel='COmotion')
            #extractSaliency(channel='SOintensity')
    


    
    
    
    

    




