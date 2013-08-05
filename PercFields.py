import numpy as np
import pylab as plt
from psychopy import visual,core
from psychopy.misc import deg2pix
from Constants import *
from Settings import Q
import random, Image,ImageFilter, os
from scipy.ndimage.filters import convolve,gaussian_filter
from ImageOps import grayscale



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

def PFextract(part=[]):
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
        if len(part)==2: np.save('cxx/similPix/data/PF%d.npy'%part[0],PF)
        else: np.save('PF.npy',PF)
    except:
        wind.close()
        raise


def PFsubsampleF():
    ''' subsample PF files from 500 hz to 100hz
        to make computeSimilarity run faster'''
    for i in range(400,601):
        print i
        PF=np.load('cxx/similPix/data/PF%d.npy'%i)
        PF=PF[:,:,:,:,range(2,152,5)]
        PF=np.save('cxx/similPix/dataRedux/PF%d.npy'%i,PF)
def PFparallel():
    ''' you can setup stack by np.save('stack.npy',range(601))'''
    stack=np.load('stack.npy').tolist()

    while len(stack):
        jobid=stack.pop(0)
        np.save('stack.npy',stack)
        PFextract([jobid,600])
        stack=np.load('stack.npy').tolist()
#E=np.load('D0.npy')


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
        stack=np.load('stack.npy').tolist()
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
