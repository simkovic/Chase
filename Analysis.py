import numpy as np
import pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

path='/home/matus/Desktop/pylink/evaluation/sacTargets/'

def sacInfo(bs):
    D=[]
    for b in bs:
        D.append(np.load(path+'SIvp018b%d.npy'%b))
    D=np.concatenate(D,axis=0)
##    plt.close('all')
##    plt.plot(D[:,1,2],D[:,1,3],'.k')
##    ax=plt.gca()
##    ax.set_aspect(1)
##    plt.show()
##
##    plt.figure()
##    plt.plot(D[:,0,2],D[:,0,3],'.k')
##    ax=plt.gca()
##    ax.set_aspect(1)
##    plt.show()
##
##    plt.figure()
##    reloc=np.sqrt(np.sum(np.power(D[:,1,[2,3]]-D[:,0,[2,3]],2),axis=1))
##    plt.hist(reloc,50)
##
##    plt.figure()
##    dur = D[:,1,1]-D[:,0,1]
##    plt.hist(dur,50)
##
##    plt.figure()
##    vel=reloc/dur
##    plt.hist(vel[~np.isnan(vel)]*1000,50)
    return D
def computeD():
    bs=range(3,25)
    bs.remove(22)
    bs.remove(23)
    plt.ion()
    #plt.set_cmap('hot')
    si=sacInfo(bs)
    D=[]
    for b in bs:
        D.append(np.load(path+'vp018b%d.npy'%b))
    D=np.concatenate(D,axis=0)
    sel=np.sqrt(np.power(si[:,1,2:]-si[:,0,2:],2).sum(axis=1))>5
    D=D[sel,:,:,:]
    si=si[sel,:,:]
    return D,si
def computeE():
    D,si=computeD()
    #D=D[:D.shape[0]/2,:,:,:]
    E=D[:,:,:14,:]
    r=np.random.permutation(D.shape[0])
    for a in range(14):
        for f in range(D.shape[1]):
            E[:,f,a,0]-= si[r,1,2]
            E[:,f,a,1]-= si[r,1,3]
    E=E[:,range(0,200,5),:,:]
    return E

plt.close('all')
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

def dirChanges(D,si):
    J=np.zeros((14*D.shape[0],98))
    H=np.zeros((14*D.shape[0],2))
    C=np.zeros((14*D.shape[0],2))
    n=D.shape[0]
    f=90
    r=np.random.permutation(D.shape[0])
    for a in range(14):
        H[a*n:(a+1)*n,0]= D[:,f,a,0]-si[:,1,2]
        H[a*n:(a+1)*n,1]= D[:,f,a,1]-si[:,1,3]
        C[a*n:(a+1)*n,0]= D[:,f,a,0]-si[r,1,2]
        C[a*n:(a+1)*n,1]= D[:,f,a,1]-si[r,1,3]
        J[a*n:(a+1)*n,:]= np.logical_or(np.abs(np.diff(D[:,:,a,1],2))>0.0001,
                    np.abs(np.diff(D[:,:,a,0],2))>0.0001)
    bn=np.arange(0,20,0.5)




    
    d= bn+(bn[1]-bn[0])/2.0;d=d[:-1]
    c=np.diff(np.pi*bn**2)
    dH=np.sqrt((H*H).sum(axis=1))
    dC=np.sqrt((C*C).sum(axis=1))
    #a,b=np.histogram(dH,bins=bn)

    KH=np.zeros((len(bn)-1,J.shape[1]))
    KC=np.zeros((len(bn)-1,J.shape[1]))
    for i in range(len(bn)-1):
        sel=np.logical_and(dH>bn[i],dH<bn[i+1])
        sel2=np.logical_and(dC>bn[i],dC<bn[i+1])
        KH[i,:]=J[sel,:].mean(0)*500
        KC[i,:]=J[sel2,:].mean(0)*500

    #sel=H<5
    #ot=J[sel,:].sum(0)
    plt.imshow(KC,origin='lower',extent=[-200,0,0,20],aspect=6)
    plt.figure()
    plt.plot(d,KH[:,50])
    plt.plot(d,KC[:,50])
    plt.legend(['gaze','rand'])
    plt.figure()
    plt.plot(KH[0,:])
    plt.plot(KC[0,:])
    plt.legend(['gaze','rand'])
#dirChanges(D,si)
#def agentDensity(D,si):
D,si=computeD()
plt.close('all')
np.random.seed(1234)
I=[]; I2=[]
for f in range(0,200,5):
    H=np.zeros((14*D.shape[0],2))
    C=np.zeros((14*D.shape[0],2))
    G=np.zeros((14*D.shape[0],2))
    r=np.random.permutation(D.shape[0])
    n=D.shape[0]
    #f=-1
    for a in range(14):
        H[a*n:(a+1)*n,0]= D[:,f,a,0]-si[:,1,2]
        H[a*n:(a+1)*n,1]= D[:,f,a,1]-si[:,1,3]
        C[a*n:(a+1)*n,0]= D[:,f,a,0]-si[r,1,2]
        C[a*n:(a+1)*n,1]= D[:,f,a,1]-si[r,1,3]
        G[a*n:(a+1)*n,0]= D[:,f,a,0]-D[:,f,2,0]
        G[a*n:(a+1)*n,1]= D[:,f,a,1]-D[:,f,2,1]
    H=H[~np.isnan(H[:,0]),:]
    C=C[~np.isnan(C[:,0]),:]
##    plt.figure()
##    bn=np.linspace(-20,20,41)
##    b1,c,d=np.histogram2d(H[:,0],H[:,1],bins=[bn,bn])
##    b2,c,d=np.histogram2d(C[:,0],C[:,1],bins=[bn,bn])
##    plt.imshow(b1-b2, extent=[-20,20,-20,20],origin='lower',
##               interpolation='nearest')
##    plt.set_cmap('hot')
##    plt.colorbar()
##    plt.figure()
    
    bn=np.arange(0,26,0.5)
    d= bn+(bn[1]-bn[0])/2.0
    d=d[:-1]
    a,b=np.histogram(np.sqrt((C*C).sum(axis=1)),bins=bn)
    c=np.diff(np.pi*bn**2)
    I2.append(a[0]/c[0]/n*(bn.size-1))
    #plt.plot(d,a/c/n*(bn.size-1))
    a,b=np.histogram(np.sqrt((H*H).sum(axis=1)),bins=bn)
    #plt.plot(d,a/c/n*(bn.size-1))
    I.append(a[0]/c[0]/n*(bn.size-1))
    a,b=np.histogram(np.sqrt((G*G).sum(axis=1)),bins=bn)
    c[0]=0
##    plt.plot(d,a/c/n*(bn.size-1))
##    plt.ylim([0,7])
##    plt.xlabel('Distance from Saccade Target Location')
##    plt.ylabel('Agent Density') #[# agents per deg^2]
##    plt.legend(['rand pos','gaze','rand ag'])
plt.figure()
plt.plot(np.linspace(-100,100,len(I)),I)
plt.plot(np.linspace(-100,100,len(I)),I2)
#plt.legend(['gaze','rand pos'])
    
#agentDensity(D,si)

##def radialHist(M,bn,nn):
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
##radialHist(C,bn,range(1,26))
##radialHist(H,bn,range(1,26))
##radialHist(G,bn,range(1,26))
##plt.legend(['gaze','rand pos','rand ag'])



