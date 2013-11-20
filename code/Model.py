import numpy as np
import pylab as plt
from psychopy import visual,core
from psychopy.misc import deg2pix
from Constants import *
from Settings import *
import random, Image,ImageFilter, os
from scipy.ndimage.filters import convolve
from ImageOps import grayscale


plt.ion()
#plt.set_cmap('gray')
#random.seed(3)

def traj2image(positions,elem=None,maze=None,wind=None):
    if type(wind)==type(None):
        close=True; wind=initDisplay()
    else: close=False
    if type(elem)==type(None):
        elem=visual.ElementArrayStim(wind,fieldShape='sqr',
            nElements=positions.shape[0],sizes=agentSize,
            elementMask='circle',elementTex=None)
    try:
        if type(maze)!=type(None):
            maze.draw(wind)
            maze.draw(wind)
        elem.setXYs(positions)      
        elem.draw()    
        wind.getMovieFrame(buffer='back')
        ret=wind.movieFrames[0]
        wind.movieFrames=[]
        visual.GL.glClear(visual.GL.GL_COLOR_BUFFER_BIT | visual.GL.GL_DEPTH_BUFFER_BIT)
        wind._defDepth=0.0
        if close:
            wind.close()
        return grayscale(ret)# make grey
    except:
        if close:
            wind.close()
        raise
    
def getSamples(dname,maxS=10000,rfSize=(250,250),sampleSize=(15,15,9)):
    A=(CHASEE,CHASER,DISTRACTOR,DISTRACTOR2)
    D=np.zeros((np.prod(sampleSize),maxS,len(A)))
    fnames=os.listdir(dname)
    fnames.sort()
    wind=initDisplay(sz=1100)
    elem=visual.ElementArrayStim(wind,fieldShape='sqr',
            nElements=int(dname[-2:]),sizes=agentSize,
            elementMask='circle',elementTex=None)
    S=np.zeros(sampleSize)
    try:
        i=0; dfile=0
        for fname in fnames:
            print 'Processing ',fname
            traj=np.load(dname+'/'+fname)
            # wait 3 seconds at the start until chaser
            # catches with chasee
            fend=trialDur*refreshRate/10
            
            while fend<(trialDur-3)*refreshRate:
                # during next 2 seconds choose random time
                fend+=random.randint(sampleSize[T],2*refreshRate)
                for a in A:
                    k=sampleSize[T]-1
                    for f in range(sampleSize[T]):
                        I=traj2image(traj[fend-f,:,:].squeeze(),
                                     wind=wind,elem=elem)
                        I=I.filter(ImageFilter.BLUR)
                        pos=deg2pix(traj[fend-f,a,:].squeeze(),
                                    wind.monitor)+wind.size[0]/2
                        pos[Y]=wind.size[0]-pos[Y]
                        start=pos-np.array(rfSize)/2.0
                        ende=pos+np.array(rfSize)/2.0+1
                        S[:,:,k]=np.asarray(I.transform(sampleSize[0:2],
                            Image.EXTENT,(start[X],start[Y],
                            ende[X],ende[Y]),Image.BILINEAR))
                        k-=1
                    D[:,i,a]=S.reshape((np.prod(sampleSize),1,1)).squeeze()
                i+=1
                if i>=maxS:
                    np.save(dname+'data%d.npy'% dfile,D)
                    i=0;dfile+=1
        np.save(dname+'data%d.npy'% dfile,D[:,:i,:])
        wind.close()
    except:
        wind.close()
        raise
def plotFilterResponse(D):
    def plot(x,r,ylab):
        plt.figure()
        plt.plot(x,r[:,CHASEE],'-k')
        plt.plot(x,r[:,CHASER],'--k')
        plt.plot(x,r[:,2:],':k')
        plt.legend(ANAMES,loc=2)
        plt.xlabel('Time in Seconds')
        plt.ylabel(ylab)
        
    #plt.close('all')
    x=np.arange(0,1801/60.0,1/60.0)
    xx=np.arange(0,1801,20)
    plot(x[xx],D[xx,:],'Filter Response')
    
    cs=np.cumsum(D,axis=0)
    plot(x,cs,'Cummulated Filter Response')
    m=np.repeat(np.matrix(np.arange(0,1801*D.mean(),D.mean())),11,axis=0)
    #m=np.repeat(np.matrix(cs.mean(axis=1)),11,axis=0)
    print m.shape, x.shape
    plt.plot(x,m[0,:].transpose(),'-k',lw=2)
    r=cs-m.T
    plot(x,r,'Translated Cummulated Filter Response')
    plt.plot([0,30],[0,0],'-k',lw=2)
    plt.grid()
def computeFilterResponse(fname,filt,wind,elem,rfSize=(250,250)):
    RF=np.matrix(np.reshape(filt,(filt.size,1)))

    try:
        #print 'Processing '+fname
        traj=np.load(fname)
        if traj.shape[2]>2:
            traj=traj[:,:,[X,Y]]
        D=np.zeros((traj.shape[0],traj.shape[1]))
        #D2=np.zeros((traj.shape[0],traj.shape[1]))
        #rfm=RF-RF.mean();rfsd=RF.std();
        for f in range(traj.shape[0]):
            I=traj2image(traj[f,:,:],wind=wind,elem=elem)
    ##        I2=I.resize((66,66),Image.ANTIALIAS)
    ##        I=Image.fromarray(convolve(I2,chaseeRF))
    ##        plt.close()
    ##        plt.imshow(I)
    ##        pos=deg2pix(traj[f,:,:].squeeze(),
    ##            wind.monitor)+wind.size[0]/2
    ##        #pos[:,Y]=wind.size[0]-pos[:,Y]
    ##        pos=66/1100.0*pos
    ##        plt.scatter(pos[:,X],pos[:,Y])
    ##        bla
            for a in range(traj.shape[1]):
                pos=deg2pix(traj[f,a,:].squeeze(),
                    wind.monitor)+wind.size[0]/2
                pos[Y]=wind.size[0]-pos[Y]
                start=pos-np.array(rfSize)/2.0
                ende=pos+np.array(rfSize)/2.0+1
                S=np.asarray(I.transform(filt.shape,
                    Image.EXTENT,(start[X],start[Y],
                    ende[X],ende[Y]),Image.NEAREST))/255
                S2=np.matrix(np.reshape(S,(filt.size,1)))
                #D[f,a]=RF.T*S2
                D[f,a]=S2.mean()
                #D[f,a]=rfm.T*(S2-S2.mean())/(rfsd*S2.std())/float(rfm.size)
    ##            plt.subplot(4,3,a+1)
    ##            plt.imshow(S*chaseeRF)
    ##            plt.title(str(D[f,a]/S.mean()))
                #if f==1: bla
                #D[f,a]=np.asarray(I.transform(chaseeRF.shape,
                #    Image.EXTENT,(start[X],start[Y],
                #    ende[X],ende[Y]),Image.BILINEAR)).sum()
            if f%100==0:
                print f
            #bla
        #wind.close()
    except:
        wind.close()
        raise
    return D

cond=17
dname='input/cond%02d'%cond
wind=initDisplay(sz=1100)
elem=visual.ElementArrayStim(wind,fieldShape='sqr',
    nElements=cond,sizes=agentSize,
    elementMask='circle',elementTex=None)
#getSamples(dname)
#chaseeRF=np.load(dname+'chaseeRF.npy')
filt=np.load('diffFilter.npy')
filt=(filt-np.min(filt))/(np.max(filt)-np.min(filt))#rescale
#filt=filt/filt.sum()#normalize
trajPath='/home/matus/Desktop/promotion/distractorVariation/input/'
vpn=range(14,22)
for vp in vpn:
    fnames=os.listdir(trajPath+'/vp'+str(vp))
    fnames.sort()
    D=[]
    for fname in fnames:
        if int(fname[-6:-4])==cond:
            print fname
            D.append(computeFilterResponse(trajPath+'/vp'+str(vp)+'/'+fname, filt,wind,elem))
    np.save('pilotFR/vp'+str(vp)+'.npy',D)
wind.close()      
#D=computeFilterResponse(dname+'/trial002.npy', filt)
#plotFilterResponse(D)
#print D.mean()
