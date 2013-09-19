import numpy as np
import pylab as plt
from psychopy import visual,core
from psychopy.misc import deg2pix
from Constants import *
from Settings import Q
import random, Image,ImageFilter, os
from scipy.ndimage.filters import convolve,gaussian_filter
from ImageOps import grayscale
from psychopy import core
from images2gif import writeGif


def position2image(positions,elem=None,wind=None):
    '''transforms vector of agent positions to display snapshot
        output format is HxW matrix of light intensity values (uint8)
    '''
    if type(wind)==type(None):
        close=True; wind=Q.initDisplay()
    else: close=False
    if type(elem)==type(None):
        elem=visual.ElementArrayStim(wind,fieldShape='sqr',
            nElements=positions.shape[0], sizes=Q.agentSize,
            elementMask=RING,elementTex=None,colors='white')
    try:
        elem.setXYs(positions)      
        elem.draw()    
        wind.getMovieFrame(buffer='back')
        ret=wind.movieFrames[0]
        wind.movieFrames=[]
        visual.GL.glClear(visual.GL.GL_COLOR_BUFFER_BIT | visual.GL.GL_DEPTH_BUFFER_BIT)
        wind._defDepth=0.0
        if close: wind.close()
        return grayscale(ret)# make grey, convert to npy
    except:
        if close: wind.close()
        raise
def traj2movie(traj,width=10,outsize=64,elem=None,wind=None,rot=2,
               hz=85.0,SX=0.5,SY=0.5,ST=50):
    ''' extracts window at position 0,0 of width WIDTH deg
        from trajectories and subsamples to OUTSIZExOUTSIZE pixels
        HZ - trajectory sampling frequency
        ROT - number of rotations to output
        SX,SY,ST - standard deviation of gaussian filter in deg,deg,ms
        
    '''
    if type(wind)==type(None):
        close=True; wind=Q.initDisplay()
    else: close=False
    if type(elem)==type(None):
        elem=visual.ElementArrayStim(wind,fieldShape='sqr',
            nElements=traj.shape[1], sizes=Q.agentSize,
            elementMask=RING,elementTex=None,colors='white')
    try:
        sig=[ST/1000.0*hz]
        sig.append(deg2pix(SX,wind.monitor))
        sig.append(deg2pix(SY,wind.monitor))
        w=int(np.round(deg2pix(width,wind.monitor)/2.0))
        D=np.zeros((traj.shape[0],outsize,outsize,rot),dtype=np.uint8)
        Ims=[]
        for f in range(0,traj.shape[0]):
            Im=position2image(traj[f,:,:],wind=wind)
            bb=int(Im.size[0]/2.0)
            Im=Im.crop(np.int32((bb-1.5*w,bb-1.5*w,bb+1.5*w,bb+1.5*w)))
            Im=np.asarray(Im,dtype=np.float32)
            Ims.append(Im)
        Ims=np.array(Ims)
        Ims=gaussian_filter(Ims,sig)
        if np.any(Ims>255): print 'warning, too large'
        if np.any(Ims<0): print 'warning, too small'
        Ims=np.uint8(np.round(Ims))
        for f in range(Ims.shape[0]):
            Im=Image.fromarray(np.array(Ims[f,:,:]))
            bb=int(Im.size[0]/2.0)
            I=Im.crop((bb-w,bb-w,bb+w,bb+w))
            I=np.asarray(I.resize((outsize,outsize),Image.ANTIALIAS))
            D[f,:,:,0]=I
            for r in range(1,rot):
                I2=Im.rotate(90/float(rot)*r)
                I2=I2.crop((bb-w,bb-w,bb+w,bb+w))
                I2=np.asarray(I2.resize((outsize,outsize),Image.ANTIALIAS))
                D[f,:,:,r]=I2
        if close: wind.close()
        return D
    except:
        if close: wind.close()
        raise
def PFlist2gif(pf,fname='pf'):
    pf=np.split(pf,range(1,pf.shape[2]),axis=2)
    for k in range(len(pf)): pf[k]=pf[k].squeeze()
    pf.append(np.zeros(pf[0].shape,dtype=np.float32))
    writeGif(fname,pf,duration=0.1)
def PFextract(part=[0,1]):
    inc=E.shape[0]/part[1]
    start=part[0]*inc
    ende=min((part[0]+1)*inc,E.shape[0])
    print start,ende,E.shape
    os=64
    rot=15
    D=np.zeros((ende-start,E.shape[1],os,os,rot),dtype=np.uint8)
    try:
        wind=Q.initDisplay()
        elem=visual.ElementArrayStim(wind,fieldShape='sqr',
                nElements=E.shape[1], sizes=Q.agentSize,
                elementMask=RING,elementTex=None,colors='white')
        for i in range(ende-start):
            #print i
            D[i,:,:,:,:]=traj2movie(E[i+start,:,:,:],outsize=os,
                        elem=elem,wind=wind,rot=rot)
        wind.close()
        PF=np.rollaxis(D,1,5)
        if len(part)==2: np.save('cxx/similPix/TI/PF%d.npy'%part[0],PF)
        else: np.save('PF.npy',PF)
    except:
        wind.close()
        raise


def PFsubsampleF():
    ''' subsample PF files from 500 hz to 100hz
        to make computeSimilarity run faster'''
    for i in range(0,601):
        print i
        PF=np.load('cxx/similPix/data/PF%d.npy'%i)
        PF=PF[:,:,:,:,range(2,152,5)]
        PF=np.save('cxx/similPix/dataRedux/PF%d.npy'%i,PF)
def PFparallel():
    ''' you can setup stack by np.save('stackPF.npy',range(601))'''
    stack=np.load('stackPF.npy').tolist()

    while len(stack):
        jobid=stack.pop(0)
        np.save('stackPF.npy',stack)
        PFextract([jobid,50])
        loaded=False
        while not loaded:
            try:
                stack=np.load('stackPF.npy').tolist()
                loaded=True
            except IOError:
                print 'IOError'
                core.wait(1)
#E=np.load('tiD0.npy')
#PFparallel()
def Mcompute():
    ''' mean of the PF data'''
    inpath='cxx/similPix/TI/'
    ids=[]
    N=len(os.listdir(inpath))
    pf=None
    for i in range(N):
        pf1=np.load(inpath+'PF%d.npy'%i)
        pf1=pf1[:,:,:,0,:].mean(0).squeeze()
        if pf==None: pf=pf1
        else: pf+=pf1
        print i
    pf/=N
    pf=pf.flatten()
    np.save('cxx/similPix/PFmeanTI.npy',pf)
                
                
def Ccompute():
    ''' covariance of PF data, no rotation'''
    inpath='cxx/similPix/TI/'
    ids=[]
    M=np.load(inpath+'PFmean.npy')
    N=len(os.listdir(inpath+'PF/'))
    for i in range(N):
        pf1=np.load(inpath+'PF/PF%d.npy'%i)
        pf1=pf1[:,:,:,0,:].squeeze()
        pf1=pf1.reshape((pf1.shape[0],pf1.size/pf1.shape[0]))-M
        for j in range(i,N):
            print i,j
            pf2=np.load(inpath+'PF/PF%d.npy'%j)
            pf2=pf2[:,:,:,0,:].squeeze()
            pf2=pf2.reshape((pf2.shape[0],pf2.size/pf2.shape[0]))-M
            np.save(inpath+'C/C%03dx%03d.npy'%(i,j),pf1.T.dot(pf2))
            
#Ccompute()           
        
def Cmerge():
    inpath='cxx/similPix/TI/'
    ids=[]
    M=np.load(inpath+'PFmean.npy')
    N=len(os.listdir(inpath+'PF/'))
    C=[[0]*N]*N
    for i in range(N):
        for j in range(N):
            if i<=j: C[i][j]=np.load(inpath+'C/C%03dx%03d.npy'%(i,j))
            else: C[i][j]=np.load(inpath+'C/C%03dx%03d.npy'%(j,i)).T 
        C[i]=np.concatenate(C[i],axis=1)
    C=np.concatenate(tuple(C),axis=0)
    np.save(inpath+'C.npy',C)

#def Cpca():
##inpath='cxx/similPix/TI/'
##C=np.load(inpath+'C.npy')
##[latent,coeff] = np.linalg.eig(C)
##idcs=np.argsort(latent)[::-1]
##latent=latent[idcs]
##coeff=coeff[:,idcs]
    

def Scompute():
    '''also available as c++ code
    g++ -o simil simil.cpp -L/usr/local -lcnpy
    '''
    plt.ion()
    #vp18script()
    ##D1=np.load('cxx/similPix/PF.npy')
    D1=np.load('cxx/similPix/data/PF0.npy')
    D2=np.load('cxx/similPix/data/PF0.npy')
    #D1=D1[:10,:,:,:,:]
    n1=0
    n2=1
    #n2=D2.shape[1]
    from time import time
    P=D1.shape[1]
    R=D1.shape[3]/3
    F=D1.shape[4]
    r2=0
    f2=10
    S=np.zeros((D1.shape[0],D2.shape[0],R*4,2))*np.nan
    ori=np.zeros(8)
    mid=(P-1)/2.0
    mask=np.zeros((P,P,51),dtype=np.bool8)
    for i in range(P):
        for j in range(P):
            if np.sqrt((i-mid)**2+(j-mid)**2)<=P/2.0: mask[i,j,:]=True
    t0=time()
    for n1 in range(D1.shape[0]):
        print n1
        for n2 in range(D2.shape[0]):
            for r1 in range(R):
                for ori in range(4):
                    a=np.float32(D1[n1,:,:,r1*3,8:59])
                    b=np.float32(D2[n2,:,:,r2*3,8:59])#*mask
                    S[n1,n2,r1+R*ori,0]=np.linalg.norm((np.rot90(a,ori)-b)*mask)
                    S[n1,n2,r1+R*ori,1]=np.linalg.norm((np.fliplr(np.rot90(a,ori))-b)*mask)
    print 'time', time()-t0
    # check the correctness of both programs
    S2=np.load('cxx/similPix/S/S000x001.npy')
    print (S-S2).mean()  


def SinitParallel(N=601):
    out=[]
    for i in range(N):
        for j in range(i,N):
            out.append([i,j])
    np.save('cxx/similPix/stack',out)
def Sparallel():
    ''' parallel similarity computation'''
    os.chdir('cxx/similPix')
    stack=np.load('stack.npy').tolist()
    
    while len(stack):
        loaded=False
        while not loaded:
            try:
                stack=np.load('stack.npy').tolist()
                loaded=True
            except IOError:
                print 'IO Error'
                core.wait(1)
            except ValueError:
                print 'Value Error'
                core.wait(1)
        jobinfo=stack.pop(0)
        np.save('stack.npy',stack)
        print jobinfo
        os.system('./simil %d %d'%tuple(jobinfo))

def checkSimWideGrid():
    from Analysis import E
    wind=Q.initDisplay()
    elem=visual.ElementArrayStim(wind,fieldShape='sqr',
            nElements=E.shape[1], sizes=Q.agentSize,
            elementMask=RING,elementTex=None,colors='white')
    a=traj2movie(E[0,:,:,:],outsize=32,elem=elem,wind=wind,rot=30)
    a=np.float32(a)
    for i in range(1,10): 
        b=traj2movie(E[i,:,:,:],outsize=32,elem=elem,wind=wind,rot=2)

        b=np.float32(b)
        S=np.zeros((100,4*30))

        for r in range(4):
            for r2 in range(30):
                for f in range(100):
                    S[f,r*30+r2]= np.linalg.norm(np.rot90(a[f,:,:,r2],r)-b[f,:,:,0])
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(S)
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.plot(np.arange(120)*3,S.sum(0))
    wind.close()
def iCirc(r1,r2,d):
    """intersection area of two circles
        r-radius
        d-distance
    """
    if d>=(r2+r1): return 0.0
    if r2-r1>=d: return r1**2*np.pi
    if r2<r1: temp=r2;r2=r1;r1=temp
    d1=(d**2-r1**2+r2**2)/2.0/d
    d2=(d**2+r1**2-r2**2)/2.0/d
    A1=r1**2*np.arccos(d2/float(r1))-d2*(r1**2-d2**2)**0.5
    A2=r2**2*np.arccos(d1/float(r2))-d1*(r2**2-d1**2)**0.5
    return A1+A2

def iRing(r1,r2,d):
    ''' intersection area of two identical rings,
        r1 - radius of the smaller inner circle
        r2 - radius of the outer circle
    '''
    return (iCirc(r1,r1,d)-2*iCirc(r1,r2,d)+iCirc(r2,r2,d))
def iRingSmoothed():
    '''computes vector with intersections of smoothed rings
        conditional on the distance of the rings
    '''
    def filterGaussian2d(sd):
        #F=np.zeros((sd*6,sd*6))
        x= np.repeat(np.array(np.linspace(-3,3,sd*6),ndmin=2).T,sd*6,axis=1)
        y=np.rot90(x)
        F=np.exp(-(x**2+y**2)/2.0)
        F=F/F.sum() # normalize
        return F
    N=256
    ring=np.ones((N,N))*0
    for i in range(N):
        for j in range(N):
            if (np.sqrt((i-N/2+0.5)**2+(j-N/2+0.5)**2)>2*N/5 and
                np.sqrt((i-N/2+0.5)**2+(j-N/2+0.5)**2)<N/2):
                ring[i,j]= 1
    fsd=1/2.0
    GF=filterGaussian2d(int(fsd*N))
    from scipy import signal as sg
    rf=sg.convolve2d(ring,GF)
    D=np.linspace(0,3,301)
    F=np.zeros(D.size)*np.nan
    i=0
    for d in D:

        im = np.zeros((rf.shape[0],rf.shape[0]+N*d))
        im[:,:rf.shape[0]]=rf
        im2=np.fliplr(im)
        F[i]= (im*im2).sum()
        i+=1
    F=F/F[0]
    F=1-F
    np.save('F.npy',F)

def comparePixXyRepresentation():
    bi=8
    a=np.float32(traj2movie(E[3,:,:,:],outsize=256,rot=1)[50,:,:,0])
    a[a<140]=0; a[a>=140]=1
    b=traj2movie(E[bi,:,:,:],outsize=256,rot=1)[50,:,:,0]
    b[b<140]=0; b[b>=140]=1

    N=float(a.size)
    ##print 'Image distance ',np.abs(a-b).sum() / ((a.sum()+b.sum()))
    print 'Image Distance', 1-np.abs(a-b).sum() / float(a.sum()+b.sum())
    c=a;d=b

    a=E[3,50,:,:]
    a=a[np.power(a,2).sum(1)<25,:]
    b=E[bi,50,:,:]
    b=b[np.power(b,2).sum(1)<25,:]
    A=np.pi*(0.5**2-0.4**2)
    if a.shape[0]>b.shape[0]: K1=b; K2=a
    else: K2=a; K1=b
    D=np.zeros((K1.shape[0],K2.shape[0]))
    for i in range(K1.shape[0]):
        for j in range(K2.shape[0]):
            D[i,j]=iRing(0.4,0.5,np.linalg.norm(K1[i,:]-K2[j,:]))
    print D.max(axis=1).sum()/float((D.shape[0]+D.shape[1])*A)
    plt.imshow(c-d)

def plotTraj(D,clr='k',lim=7):
    ax=plt.gca()
    for a in range(D.shape[1]):
        plt.plot(D[:,a,0],D[:,a,1],'-'+clr,linewidth=3)
        ar=plt.arrow(D[-2,a,0],D[-2,a,1],D[-1,a,0]-D[-2,a,0],D[-1,a,1]-D[-2,a,1],
              length_includes_head=False,head_width=0.2,fc=clr)
        ax.add_patch(ar)
    ax.set_aspect(1)
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
        
def compareTrajs(D1,D2,r1,f1,m1):
    print 'r=',r1,'f=',f1, 'm=',m1
    plt.figure()
    ax=plt.gca()
    def transform(D,r,f,m):
        D=D[f:(f+D.shape[0]/2),:,:]
        R=5; print '!!!!R= ',R
        phi=r/float(R*4)*np.pi*2
        T=np.matrix([[np.cos(phi),-np.sin(phi)],
                [np.sin(phi),np.cos(phi)]])
        for f in range(D.shape[0]):
            D[f,:,:]= np.array(T*np.matrix(D[f,:,:]).T,ndmin=2).T
        if m==1: D[:,:,0]= -D[:,:,0]
        return D


    plotTraj(transform(D1,r1,f1,m1),'r')
    plotTraj(transform(D2,0,D2.shape[0]/4,0),'b')
    
    plt.title('r=%d, f=%d, m=%d'%(r1,f1,m1))
def compareScript():
    plt.close('all')
    h=0
    k=1
    for k in range(1,9):
        plt.figure()
        plt.imshow(R[k,:,:,0],aspect=0.2)
        plt.set_cmap('gray')
        plt.colorbar() 
        plt.figure()
        plt.imshow(S[k,:,:,0],aspect=0.2)
        plt.colorbar()
        bb=(R[k,:,:,0].min()==R[k,:,:,0]).nonzero()
        compareTrajs(np.copy(E[h,:,:,:]),np.copy(E[k,:,:,:]),bb[0][0],bb[1][0],0)
    #compareTrajs(np.copy(E[h,:,:,:]),np.copy(E[k,:,:,:]),2,10)
    #tt=1-bb[0][0]
    #bb=(R[tt,:,:,k].min()==R[tt,:,:,k]).nonzero()
    #compareTrajs(np.copy(E[h,:,:,:]),np.copy(E[k,:,:,:]),bb[0][0],bb[1][0],tt)

def computeSimilarityCoord():
    '''also available as c++ code
    g++ -o similCoord similCoord.cpp -L/usr/local -lcnpy
    '''
    E=np.load('cxx/similCoord/Etrimmed.npy')
    from time import time

    F=np.load('F.npy')
    #E=E[:,range(0,200,2),:,:]
    h=7
    plt.close('all')
    r=np.arange(0,360,15)/180.0*np.pi
    step=1
    R=np.zeros((E.shape[0],r.size,E.shape[1]/2/step,2))
    #DENOM=np.zeros((E.shape[0],r.size,E.shape[1]/2,2,E.shape[1]/2))*np.nan
    T=[]
    t0=time()
    f2=E.shape[1]/4
    V=(np.sqrt(np.power(E,2).sum(3)).mean(1)<5)
    for k in [8]:#E.shape[0]):
        print k
        for m in range(r.size):
            T=np.matrix([[np.cos(r[m]),-np.sin(r[m])],
                [np.sin(r[m]),np.cos(r[m])]])
            if k==h: continue
            for f in range(0,E.shape[1]/2,step):
                for fi in range(0,E.shape[1]/2):
                    I1=np.copy(E[h,fi+f,:,:])
                    #a=I1[np.power(I1,2).sum(1)<25,:]
                    a=I1[V[h,:],:]
                    if a.shape[0]==0: continue
                    a= np.array(T*np.matrix(a).T,ndmin=2).T
                    I2=np.copy(E[k,f2+fi,:,:])
                    #b=I2[np.power(I2,2).sum(1)<25,:]
                    b=I2[V[k,:],:]
                    if b.shape[0]==0: continue
                    
                    for mir in range(2):
                        if mir==1: a[:,0]= -a[:,0]
                        if a.shape[0]<b.shape[0]: K1=a; K2=b
                        else: K2=a; K1=b;
                        D=np.zeros((K1.shape[0],K2.shape[0]))
                        for i in range(K1.shape[0]):
                            for j in range(K2.shape[0]):
                                dd=np.linalg.norm(K1[i,:]-K2[j,:])
                                #D[i,j]=1-iRing(0.4,0.5,dd)
                                D[i,j]=F[int(round(min(dd,4)*100))]
                                #D[i,j]=dd
                        if a.shape[0]==b.shape[0]:
                            R[k,m,f,mir]+=min(D.min(axis=1).mean(),D.min(axis=0).mean())
                        else:
                            R[k,m,f,mir]+=D.min(axis=1).mean()
                        #DENOM[k,m,f,mir,fi]=D.shape[0]
        R[k,:,:,:]/=(E.shape[1]/2)
    plt.figure();plt.imshow(R[k,:,:,0]); plt.colorbar()
    print time()-t0
def findPars():
    """ find optimal time shift and rotation for perc fields"""
    V=np.int32(np.load('cxx/similCoord/V.npy').sum(1).T)
    S=np.load('cxx/similCoord/S.npy')
    from scipy.stats import nanmean
    #S=np.rollaxis(S3,2,1)
    N=S.shape[0]
    F=S.shape[2]
    R=S.shape[4]
    M=S.shape[5]
    np.random.seed(123)
    K=1000
    P=np.zeros((N,3),dtype=np.int32) # columns: f,r,m

    T=50
    C=np.zeros((T,N))*np.nan
    D=np.zeros((N,F,R,M))
    G=[]
    Ctop=1
    for k in range(K):
        P[:,0]=np.random.randint(0,F,N)
        P[:,1]=np.random.randint(0,R,N)
        P[:,2]=np.random.randint(0,M,N)
        Clast=np.nan
        for t in range(T):
            order=np.random.permutation(N)
            for n1 in order:#range(N):
                if V[n1]==0: continue
                denom= np.minimum(V,V[n1]).sum()

                for n2 in range(N):
                    temp=np.roll(S[n1,n2,:,P[n2,0],:,:],-P[n2,1],1)
                    D[n2,:,:,:]=np.roll(temp,P[n2,2],2)
                temp=np.nansum(D,0)/denom
                C[t,n1]=temp.min()
                m=(temp==C[t,n1]).nonzero()
                for i in range(3): P[n1,i]=m[i][0];
            # check convergence
            if (Clast-nanmean(C[t,:])==0): break
            Clast=nanmean(C[t,:])
        #plt.plot(range(t+1),nanmean(C[:t+1,:],1))
        print k, t, Clast
        if Clast<Ctop:
            Ptop=np.copy(P)
            Ctop=Clast
        G.append(nanmean(C[t,:]))
    G=np.array(G)
    #plt.plot(np.sort(G))
def compareP(P1,P2):
    #P1=P[1,:,:]
    #P2=P[2,:,:]
    plt.hist(np.abs(P1[:,0]-P2[:,0]),np.arange(0,F)-0.5)
    plt.figure()
    dat=[]
    for r1 in range(R):
        for r2 in range(r1+1,R):
            diff=np.abs(P1[r1,1]-P1[r2,1])
            temp=np.minimum(diff,R-diff)
            diff=np.abs(P2[r1,1]-P2[r2,1])
            dat.append(np.abs(np.minimum(diff,R-diff)-temp))
    dat=np.array(dat)

    plt.hist(dat,np.arange(0,R/2)-0.5)
    
#for i in range(9): plt.subplot(3,3,i+1);plt.plot(C[i,:,:]);plt.ylim([0.5,0.9])
          
#plt.figure();
#plt.figure();plt.imshow(D[:,:,0].T); plt.colorbar()
#plt.figure();plt.imshow(D[:,:,1].T); plt.colorbar()

#def mov2svmInp():
##fout=open('traj.out','w')
##def writeMov(mov,fout)
##    sample=5
##    fout.write('%d '%label)
##    i=1
##    for f in range(mov.shape[0]/sample):
##        for x in range(16):
##            for y in range(16):
##                fout.write('%d:%.4f '%(i,mov[f,x,y]/256.0));i+=1
##    fout.write('\n')
##
##fout.close()

def PtoS(P):
    P=np.load('Ptop.npy')
    #S=np.load('cxx/similCoord/S.npy')
    Sout=np.zeros((N,N))
    for n1 in range(N):
        for n2 in range(N):
            r=abs(P[n1,1]-P[n2,1])
            m= abs(P[n1,2]-P[n2,2])
            Sout[n1,n2]=S[n1,n2,P[n1,0],P[n2,0],r,m]

#########################################################
#                                                       #
#                       PCA                             #
#                                                       #
#########################################################

def PF2X(merge=1):
    ''' merge - # of pf files to merge into one x file
                by default 1 and no files are merged
    '''
    pfpath=inpath.rstrip('/X/')+'/PF/'
    fs=os.listdir(pfpath)
    fs.sort();k=0;x=[];h=0
    for f in fs:
        pf=np.load(pfpath+f)
        pf=pf[:,:,:,0,:].squeeze()
        x.append(pf.reshape((pf.shape[0],pf.size/pf.shape[0])))
        if k%merge==merge-1:
            x=np.concatenate(x,0)
            np.save(inpath+'X%d'%h,x)
            x=[];h+=1
        k+=1
    if len(x)>0:
        x=np.concatenate(x,0)
        np.save(inpath+'X%d'%h,x);h+=1
    #h=201
    #x=np.load(inpath+'X0.npy')
    global X
    X=h
    inc=int(np.ceil(x.shape[1]/float(h)))
    for g in range(h):
        out=[]
        print g
        for j in range(h):
            x=np.load(inpath+'X%d.npy'%j)
            out.append(x[:,g*inc:(g+1)*inc].copy())
            if g==h-1:
                x=np.float64(x)/255.0
                np.save(inpath+'X%d.npy'%j,x)
            del x
        out =np.concatenate(out,0).T
        out=np.float64(out)/255.0
        np.save(inpath+'XT%d'%g,out)

def split(X):
    inc=int(np.ceil(X.shape[0]/float(N)))
    for i in range(N):
        np.save(inpath+'X%d.npy'%i,X[i*inc:(i+1)*inc,:])
    inc=int(np.ceil(X.shape[1]/float(N)))
    for i in range(N):
        np.save(inpath+'XT%d.npy'%i,X[:,i*inc:(i+1)*inc].T)
        
def merge(fprefix='X',N=5):
    out=[]
    for i in range(N): out.append(np.load(inpath+fprefix+'%d.npy'%i))
    return np.concatenate(out,0)
def mergeT(fprefix='X'):
    out=[]
    for i in range(N): out.append(np.load(inpath+fprefix+'T%d.npy'%i).T)
    return np.concatenate(out,1)
def XgetColumn(k):
    inc=np.load(inpath+'XT0.npy').shape[0]
    a=k/inc;b=k%inc
    return np.matrix(np.load(inpath+'XT%d.npy'%a)[b,:]).T
    

def Xleftmult(A,transpose=False):
    ''' OUT=X*A or OUT=X.T*A '''
    out=[];A=np.matrix(A)
    for i in range(N):
        X=np.load(inpath+'X%d.npy'%i)
        if transpose: X=np.load(inpath+'XT%d.npy'%i)
        out.append(X*A)
    return np.concatenate(out,0)
def XminusOuterProduct(A,B):
    ''' X=X-A*B.T todo save XT also'''
    out=[];A=np.matrix(A);B=np.matrix(B).T
    s=0
    for i in range(N):
        X=np.load(inpath+'X%d.npy'%i)
        e=s+X.shape[0]
        X= X- A[s:e,:]*B
        np.save(inpath+'X%d.npy'%i,X)
        s=e
    s=0
    for i in range(N):
        X=np.load(inpath+'XT%d.npy'%i)
        e=s+X.shape[0]
        X= X- (A*B[:,s:e]).T
        np.save(inpath+'XT%d.npy'%i,X)
        s=e
def XmeanCenter(axis=1):
    #compute mean
    ss=['','T']
    tot=0
    for i in range(N):
        X=np.load(inpath+'X%s%d.npy'%(ss[axis],i))
        if i==0: res=X.sum(0)
        else: res+=X.sum(0)
        tot+=X.shape[0]
    res/=tot
    np.save(inpath+'X%smean.npy'%ss[axis],res.squeeze())
    for i in range(N):
        X=np.load(inpath+'X%s%d.npy'%(ss[axis],i))
        np.save(inpath+'X%s%d.npy'%(ss[axis],i),X-np.matrix(res))
    inc=int(np.ceil(X.shape[1]/float(N)))
    for i in range(N):
        X=np.load(inpath+'X%s%d.npy'%(ss[1-axis],i))
        np.save(inpath+'X%s%d.npy'%(ss[1-axis],i),X-np.matrix(res[i*inc:(i*inc+X.shape[0])]).T)  

def Xcov(transpose=False):
    transpose=int(transpose)
    XmeanCenter(1-transpose)
    ss=['','T']
    C=[[0]*N]*N
    for i in range(N):
        X1=np.load(inpath+'X%s%d.npy'%(ss[transpose],i))
        for j in range(N):
            print i,j, X1.shape
            X2=np.load(inpath+'X%s%d.npy'%(ss[transpose],j))
            C[i][j]= X1.dot(X2.T)
        C[i]=np.concatenate(C[i],1)
    return np.concatenate(C,0)/float(X1.shape[1]-1)
    
def pcaSVD(A,highdim=None):
    """ performs principal components analysis 
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to features/attributes. 

    Returns :  
    coeff :
    is a p-by-p matrix, each column containing coefficients 
    for one principal component.
    score : 
    the principal component scores; that is, the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
    latent : 
    a vector containing the normalized eigenvalues (percent variance explained)
    of the covariance matrix of A.
    Reference: Bishop, C. (2006) PRML, Chap. 12.1
    """
    n=A.shape[0];m=A.shape[1]
    A=np.array(A)
    if highdim==None: highdim=n<m
    M = (A-np.array(A.mean(1),ndmin=2).T) # mean-center data
    if highdim:
        [latent,coeff] = np.linalg.eig(np.cov(M))
        coeff=M.T.dot(coeff)
    else:
        [latent,coeff] = np.linalg.eig(np.cov(M.T))
    score = M.dot(coeff).T
    latent=np.abs(latent)
    coeff/=np.sqrt(A.shape[int(highdim==False)]*np.array(latent,ndmin=2)) #unit vector length
    latent/=latent.sum()
    # sort the data
    indx=np.argsort(latent)[::-1]
    latent=latent[indx]
    coeff=coeff[:,indx]
    score=score[indx,:]
    return coeff,score,latent
  
def pcaNIPALS(K=5,tol=1e-4,verbose=False):
    ''' Reference:
            Section 2.2 in Andrecut, M. (2009).
            Parallel GPU implementation of iterative PCA algorithms.
            Journal of Computational Biology, 16(11), 1593-1599.
            
    '''
    if verbose: print 'Mean centering columns'
    XmeanCenter(1)
    latent=[]
    for k in range(K):
        lam0=0;lam1=np.inf
        T=np.matrix(XgetColumn(k))
        if verbose: print 'Computing PC ',k
        h=0
        while abs(lam1-lam0)>tol and h<100:
            P=Xleftmult(T,True)
            P=P/np.linalg.norm(P)
            T=Xleftmult(P)
            lam0=lam1
            lam1=np.linalg.norm(T)
            if verbose: print '\t Iteration '+str(h)+', Convergence =', abs(lam1-lam0)
            h+=1
        latent.append(lam1)
        XminusOuterProduct(T,P)
        #np.save(inpath+'T%02d'%k,T)
        np.save(inpath+'coeffT%d'%k,P.T)
    np.save(inpath+'latent',latent)

def testPca():
    global inpath
    global N
    inpath=''
    N=5;M=21
    np.random.seed(1)
    X=np.random.rand(52,M)
    split(X)
    assert np.all(np.abs(X-merge())==0)
    assert np.all(np.abs(np.matrix(X[:,M/2]).T-XgetColumn(M/2))==0)
    Y1=(X-np.matrix(X.mean(1)).T)
    XmeanCenter(1)
    Y2=mergeT()
    assert np.mean(np.abs(Y1-Y2))<1e-8
    Y1=(X-np.matrix(X.mean(0)))
    split(X)
    XmeanCenter(0)
    Y2=mergeT()
    assert np.mean(np.abs(Y1-Y2))<1e-8
    Y3=merge()
    assert np.mean(np.abs(Y1-Y3))<1e-8
    split(X)
    A=np.matrix(np.random.rand(M,3))
    Y1=X*A
    Y2=Xleftmult(A)
    assert np.all(np.abs(Y1-Y2)==0)
    split(X)
    A=np.matrix(np.random.rand(52,3))
    Y1=X.T*A
    Y2=Xleftmult(A,True)
    assert np.all(np.abs(Y1-Y2)<0.001)
    A=np.matrix(np.random.rand(52,1))
    B=np.matrix(np.random.rand(M,1))
    Y1=X-A*B.T
    XminusOuterProduct(A,B)
    Y2=merge()
    assert np.all(np.abs(Y1-Y2)==0)
    Y3=mergeT()
    assert np.all(np.abs(Y1-Y3)==0)
    C1=np.cov(X.T)
    split(X)
    C2=Xcov(True)
    assert np.abs(C1-C2).mean()<1e-12
    global c1,s1,v1,c2,s2,v2
    c1,s1,v1=pcaSVD(X,True)
    c2,s2,v2=pcaSVD(X,False)
    print v1,v2

    X=np.random.rand(52,M)
    split(X)
    pcaNIPALS(K=M)
    a1=mergeT('coeff')
    c1=np.load(inpath+'latent.npy')
    a2,b2,c2=pcaSVD(X)
    print c2/c2.sum(), c2.sum()
    for k in range(a1.shape[1]):
        plt.figure()
        plt.plot(a2[:,k]);
        if np.abs((a2[:,k]-a1[:,k])).mean()> np.abs((a2[:,k]+a1[:,k])).mean() : plt.plot(-a1[:,k])
        else: plt.plot(a1[:,k])
    plt.show()

inpath='cxx/similPix/TI/X/'
N=13
##PF2X(merge=4)
##C=Xcov()
##np.save('C',C)
##[latent,coeff]=np.linalg.eig(C)
##coeff=Xleftmult(coeff[:,:100],True)
##np.save(inpath+'coeff.npy',coeff)
##latent/=latent.sum()
##np.save(inpath+'latent',latent[:100])

####pcaNipals(K=20)
##
Sparallel()
##
####latent=np.load(inpath+'latent.npy')
##N=5
##fs=os.listdir(inpath)
##N=0
##for f in fs: N+= f.startswith('coeff')
##coeff=mergeT('coeff')
##coeff=np.load(inpath+'coeff.npy')
##for h in range(coeff.shape[1]):
##    pf=coeff[:,h].reshape((64,64,68))
##    pf-= np.min(pf)
##    pf/= np.max(pf)
##    #pf*=255
##    #pf=np.uint8(pf)
##    pf=np.split(pf,range(1,pf.shape[2]),axis=2)
##    for k in range(len(pf)): pf[k]=pf[k].squeeze()
##    pf.append(np.zeros(pf[0].shape,dtype=np.float32))
##    writeGif(inpath+'pcN%d.gif'%h,pf,duration=0.1)
    

def controlSimulations():
    P=32;T=20
    I=np.random.rand(P,P,T)*1e-2
    for t in range(T):
        I[[t,t+1],P/2,t] = 1
        I[[t,t+1],P/2+1,t] = 1
        I[[t+10,t+11],P/2,t] = 1
        I[[t+10,t+11],P/2+1,t] = 1

    it=10
    D=[np.matrix((1-I).flatten())]
    for phi in range(it,360,it):
        print phi
        Im=np.copy(I)
        for t in range(T):
            M=Image.fromarray(Im[:,:,t])
            M=M.rotate(float(phi))
            Im[:,:,t]=np.asarray(M)
        D.append(np.matrix((1-Im).flatten()))
    D=np.concatenate(D)

    D=D-np.matrix(D.mean(0))

    coeff,score,latent=pcaSVD(D)
    for k in range(coeff.shape[1]):
        PFlist2gif(coeff[:,k].reshape((P,P,T)),'pc%02d'%k)



