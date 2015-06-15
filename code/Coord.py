## The MIT License (MIT)
##
## Copyright (c) <2015> <Matus Simkovic>
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
## THE SOFTWARE.

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
bs=range(1,22);HZ=85.0 #sampling frequency 

def initVP(vpl=1,evl=1):
    '''event - 0: search saccades,1: 1st tracking saccade,2:second ...
            -1: -250 sec prior to button press, -2: -300 sec to BP ...'''
    global vp, path, figpath, event,bs,sw,ew,hz,fw,radbins
    vp=vpl; event=evl
    path=os.getcwd().rstrip('code')+'evaluation/vp%03d/'%vp
    figpath=os.getcwd().rstrip('code')+'figures/analysis/'
    if event>=0: sw=-400; ew=400;# start and end (in ms) of the window
    else: sw=-800; ew=0
    fw=int(np.round((abs(sw)+ew)/1000.0*HZ))
    return vp,event,path


def computePhi(path,PLOT=False):
    ''' compute rotation angles '''
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


#############################################################
#
#                       SALIENCY
#
#############################################################
    

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
    sf=np.int32(np.round((si[:,1]+sw)/1000.0*HZ))
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

    
def computeSaliencyMaps(vp):
    import commands
    """ Calls ezvision and get saliency maps
        put all data into a single folder INPATH. The format of the data name
        should be <name>.mpeg. Create folder OUTPATH and run

    """
    inpath='saliency/input/vp%03d/'%vp
    outpath='saliency/output/vp%03d/'%vp
    files = os.listdir(inpath)
    files.sort()

    for f in files:
        print f
        status,output=commands.getstatusoutput('ezvision '+
            '--in='+inpath+f+' --input-frames=0-MAX@85Hz --rescale-input=1024x1024 '+
            ' --crop-input=128,0,1152,1028 --rescale-output=64x64 '+
            '--out=mraw:'+outpath+f[:-5]+' --output-frames=@85Hz ' +
            '--save-channel-outputs --vc-chans=IM --sm-type=None '+
            '--nodisplay-foa --nodisplay-patch --nodisplay-traj '+
            '--nodisplay-additive --wta-type=None --nouse-random '+
            '--direction-sqrt --num-directions=8 --vc-type=Thread:Std ' +
            '-j 2 --nouse-fpe --logverb=Error')
        commands.getstatusoutput('rm '+outpath+f[:-5]+'SOdir_*')
        if status!=0:
            print output
    
def extractSaliency(channel='intensity',reps=100):
    ''' the saliency for each trial and position has been precomputed and saved
    '''
    print vp, channel
    try:
        from matustools.ezvisiontools import Mraw
        si=plotSearchInfo(plot=False)
        inpath=os.getcwd().rstrip('code')+'input/'
        sf=np.int32(np.round((si[:,1]+sw)/1000.0*HZ))
        ef=sf+fw
        valid= np.logical_and(si[:,1]+sw>=0, si[:,1]+ew <= np.minimum(si[:,12],30000))
        print si.shape, valid.sum()
        si=si[valid,:]
        sf=sf[valid]
        ef=ef[valid]
        lastt=-1;dim=64;k=0
        gridG=np.zeros((fw,dim,dim));radG=np.zeros((fw,radbins.size))
        gridT=np.zeros((reps,fw,dim,dim));radT=np.zeros((reps,fw,radbins.size))
        #gridA=np.zeros((fw,dim,dim));radA=np.zeros((fw,radbins.size))
        #gridP=np.zeros((fw,dim,dim));radP=np.zeros((fw,radbins.size))
        rt=np.zeros((reps,si.shape[0]))
        for rr in range(reps):
            rt[rr,:]=np.random.permutation(si.shape[0])
        #rp=np.random.permutation(si.shape[0])
        for h in range(si.shape[0]):
            print "finished=%.2f"%(h/float(si.shape[0])*100)
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
                for rr in range(reps):
                    temp1,temp2=vid.computeSaliency(si[rt[rr,h],[6,7]],[sf[rt[rr,h]],ef[rt[rr,h]]],rdb=radbins)
                    if not temp1 is None: gridT[rr,:,:,:]+=temp1; radT[rr,:,:]+=temp2.T;
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
        if vp==4 and reps>0:
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


#############################################################
#
#                       PCA
#
#############################################################

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
    F=np.zeros((E.shape[0],E.shape[3]*E.shape[1]*E.shape[2]))
    for k in range(E.shape[0]):
        B=np.copy(E[k,:,:,:])
        r=np.random.permutation(range(E.shape[2]))
        #B=B[:,r,:]
        F[k,:]= B.reshape([F.shape[1],1]).squeeze()
    pcs,score,lat=princomp(F)
    np.save('pcs%d'%q,pcs)
    np.save('lat%d'%q,lat)
    np.save('score%d'%q,score)
    
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
def pcaScript():
    from matplotlib.mlab import PCA
    from test import princomp

    q=1
    E=D[q]
    F=[]
    for q in range(2):
        F.append(np.zeros((E.shape[0],E.shape[3]*E.shape[1]*E.shape[2])))
        for k in range(E.shape[0]):
            B=np.copy(E[k,:,:,:])
            F[q][k,:]= B.reshape([F[q].shape[1],1]).squeeze()
    x=np.concatenate(F,axis=0)
    x=np.concatenate([-np.ones((x.shape[0],1)), x],axis=1)
    y=np.zeros((F[q].shape[0]*2))
    y[:F[q].shape[0]]=1
    del D, E, F


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
        print vp, np.mean(dur)

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
    plt.subplot(222)
    plt.hist()
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
    
def computeTrackInfo():
    def _hlpfun(atraj,gaze,phiold,tphi,ttraj):           
        dchv=np.int32(np.linalg.norm(np.diff(atraj,2,axis=0),axis=1)>0.0001)
        atraj-=gaze[:atraj.shape[0],:]
        dst=np.linalg.norm(atraj,axis=1)
        phi=interp1d(tphi,phiold,bounds_error=False)(ttraj)
        inds=np.isnan(phi).nonzero()[0]
        for i in inds:
            if i<phi.size/2: phi[i]=phiold[0]
            else: phi[i]=phiold[-1]
        M=atraj.shape[0]
        temp=np.zeros((M,2))
        cs=np.cos(phi[:M]);sn=np.sin(phi[:M])
        temp[:,0]=cs*atraj[:,0]+sn*atraj[:,1]
        temp[:,1]= -sn*atraj[:,0]+cs*atraj[:,1]
        return dchv,dst,temp
    told=999;
    for vp in range(1,5):
        initVP(vp,1)
        ti=np.load(path+'ti.npy')
        #print np.median(ti[:,-1]-ti[:,-2])*2
        inpath=os.getcwd().rstrip('code')+'input/' 
        f=open(path+'trackxy.pickle','rb')
        trackxy=pickle.load(f);f.close()
        txy=_computeAgTime(trackxy,ti)
        dist=[];dirch=[];gs=[];trajs=[]
        for j in range(len(txy)):
            for some in [trajs,dist,dirch]: some.append([[],[],[],[]])          
            g=np.array(trackxy[j][2])
            g=g.reshape([g.size/3,3])
            s=int(round(g[0,0]*HZ/1000.0))+1; e=int(round(g[-1,0]*HZ/1000.0))-1
            if ti[j,3]!=told:
                order=np.load(inpath+'vp%03d/ordervp%03db%d.npy'%(vp,vp,ti[j,2]))
                traj=np.load(inpath+'vp001/vp001b%dtrial%03d.npy'%(ti[j,2],order[ti[j,3]]) )
            ttraj=np.linspace(s*1000/HZ,e*1000/HZ,e-s)
            temp=np.zeros((e-s,2))
            temp[:,0]=interp1d(g[:,0],g[:,1])(ttraj)
            temp[:,1]=interp1d(g[:,0],g[:,2])(ttraj)
            gs.append(temp)
            phiold,tphi= _computePhi(g)
            for ag in txy[j]:
                atraj=traj[s:e,ag[0],:2]
                dchv,dst,trj = _hlpfun(atraj,temp,phiold,tphi,ttraj)
                dirch[j][min(len(txy[j]),3)].append(dchv)
                dist[j][min(len(txy[j]),3)].append(dst)
                trajs[j][min(len(txy[j]),3)].append(trj)
            txyall= np.array([range(14),[ti[j,4]]*14,[ti[j,5]]*14]).T.tolist() 
            for ag in txyall:
                atraj=traj[s:e,ag[0],:2]
                dchv,dst,trj = _hlpfun(atraj,temp,phiold,tphi,ttraj)
                dirch[j][0].append(dchv)
                dist[j][0].append(dst)
                trajs[j][0].append(trj)
        np.save(path+'trackDist.npy',dist)
        np.save(path+'trackDirch.npy',dirch)
        np.save(path+'trackGaze.npy',gs)
        np.save(path+'trackTraj.npy',trajs)

        
def plotAgdist():
    dist,discard,the,rest=computeTrackInfo()
    del discard,the,rest
    plt.figure(0,figsize=(10,8))
    for vp in range(1,5):
        xlim=500
        ys=dist[vp-1]
        dat=np.zeros((len(ys),int(HZ*xlim/1000.0),2))*np.nan
        datrev=np.zeros((len(ys),int(HZ*500/1000.0),2))*np.nan
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
                ss=datrev.shape[1]/HZ
                plt.plot(np.linspace(-ss,0,datrev.shape[1]),nanmedian(datrev[sel,:,i],0));
    plt.subplot(441)
    plt.legend(['> 2','2','1'],loc=4)
    initVP(1,1)
    plt.savefig(figpath+'trackAgdist')

def computeMeanPF(P=129,T=85,pvar=0.3**2):
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
    D=np.zeros((4,4,3,P,P,T))
    J=np.zeros((4,4,3,T))
    computeTrackInfo()
    Lout=np.zeros((4,3,T))
    for vp in range(4):
        print vp
        initVP(vp+1,1)
        gs=np.load(path+'trackGaze.npy')
        trajs=np.load(path+'trackTraj.npy')
        L=[[],[],[]];
        for i in range(3):
            for t in range(T): L[i].append([])
        for ind in range(4):
            for kk in range(len(trajs)):
                traj =trajs[kk][ind]
                for ag in range(len(traj)):
                    atraj=traj[ag]
                    for t in range(atraj.shape[0]):
                        if t>=T: break
                        pos = atraj[t]
                        if np.any(np.isnan(pos)): continue
                        dist=np.square(x-pos[0])+np.square(y-pos[1])
                        D[vp,ind,0,:,:,t]+= np.exp(-dist/pvar/2.)
                        if t and not ind and not ag:
                            ddd=gs[kk][t,:]-gs[kk][t-1,:]
                            L[0][t].append(np.linalg.norm(ddd))
                        J[vp,ind,0,t]+=1
                    for t in range(atraj.shape[0]):
                        if t>=T: break
                        pos = atraj[-t]
                        if np.any(np.isnan(pos)): continue
                        dist=np.square(x-pos[0])+np.square(y-pos[1])
                        D[vp,ind,1,:,:,-t]+= np.exp(-dist/pvar/2.)
                        if t and not ind and not ag:
                            ddd=gs[kk][-t,:]-gs[kk][-t+1,:]
                            L[1][-t].append(np.linalg.norm(ddd))
                        J[vp,ind,1,-t]+=1
                    btraj=atraj.tolist()
                    S=len(btraj)
                    for s in range(S):
                        pos=btraj[s]
                        #if t>=T: break
                        if np.any(np.isnan(pos)): continue
                        t=s+(T-S)/2
                        if t<0 or t>=T:continue
                        dist=np.square(x-pos[0])+np.square(y-pos[1])
                        D[vp,ind,2,:,:,t]+= np.exp(-dist/pvar/2.)
                        if t and not ind and not ag:
                            ddd=gs[kk][s,:]-gs[kk][s-1,:]
                            L[2][t].append(np.linalg.norm(ddd))
                        J[vp,ind,2,t]+=1
            for t in range(T):
                for hh in range(3):
                    D[vp,ind,hh,:,:,t]/=J[vp,ind,hh,t]
                    Lout[vp,hh,t]=nanmedian(L[hh][t])
    np.save(path+'trackPF.npy',D)
    np.save(path+'trackPFcount.npy',J)
    np.save(path+'trackVel',Lout)

def createIdealObserver(vpnr=999,N=5000,rseed=10):
    np.random.seed(rseed)
    initVP(vpnr,0);wdur=(ew-sw)/1000.
    inpath=os.getcwd().rstrip('code')+'input/' 
    ts=np.random.rand(N)*(30-wdur)*40*23
    ts=np.sort(ts)
    blocks=np.int32(ts/(30-wdur)/40)
    trials=np.int32((ts-blocks*(30-wdur)*40)/(30-wdur))
    frames=np.int32((ts%(30-wdur))*HZ)

    D=np.zeros((N,int(wdur*HZ),14,2))
    mdp=int(wdur*HZ)/2
    for n in range(N):
        if not n or trials[n]!=trials[n-1]:
            traj=np.load(inpath+'vp001/vp001b%dtrial%03d.npy'%(blocks[n]+1,trials[n]))
        #print frames[n],frames[n]+int(HZ*wdur)
        
        D[n,:,:,:]=traj[frames[n]:(frames[n]+int(HZ*wdur)),:,:2]
        g=(D[n,mdp,0,:]+D[n,mdp,1,:])/2.
        D[n,:,:,0]-=g[0]
        D[n,:,:,1]-=g[1]
    np.save(path+'E1/DG.npy',D)  
         
if __name__ == '__main__':
    plotDur()
    
    bla
    for event in range(0,3)+range(96,100):
        for vpl in range(1,5):
            initVP(vpl=vpl,evl=event)
            print vp, event
            extractTrajectories(suf='3')
            extractDensity()
            extractDirC()
            computeSaliencyMaps(vp)
            extractSaliency(channel='COmotion')
            extractSaliency(channel='SOintensity')
    computeMeanPF()
    computeTrackInfo()
    createIdealObserver()
