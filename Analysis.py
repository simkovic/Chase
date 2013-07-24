import numpy as np
import pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
#plt.ion()
from os import getcwd
path=getcwd()
path=path.rstrip('code')
path+='/evaluation/searchTargets/'
vp=1
bs=range(1,22)
def plotSearchInfo(plot=True):
    D=[]
    for b in bs:
        D.append(np.load(path+'SIvp%03db%d.npy'%(vp,b)))
    D=np.concatenate(D,axis=0)
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
    D=[]
    for b in bs:
        D.append(np.load(path+'vp%03db%d.npy'%(vp,b)))
    D=np.concatenate(D,axis=0)
    
    sel=np.logical_not(np.all(np.all(np.all(np.isnan(D[:,:14]),3),2),1))
    D=D[sel,:,:,:]
    #print D.shape, si.shape
    sel=si[:,10]==0
    D=D[sel,:,:,:]
    si=si[sel,:]

    E=D[:,:,:14,:]
    r=np.random.permutation(D.shape[0])
    for a in range(14):
        for f in range(D.shape[1]):
            E[:,f,a,0]-= si[:,6]
            E[:,f,a,1]-= si[:,7]
    #E=E[:,range(0,200,5),:,:]
    return E,D,si


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
    plt.close('all')
    J=np.zeros((14*D.shape[0],D.shape[1]-2))
    H=np.zeros((14*D.shape[0],2))
    C=np.zeros((14*D.shape[0],2))
    n=D.shape[0]
    f=90
    r=np.random.permutation(D.shape[0])
    for a in range(14):
        H[a*n:(a+1)*n,0]= D[:,f,a,0]-si[:,7]
        H[a*n:(a+1)*n,1]= D[:,f,a,1]-si[:,8]
        C[a*n:(a+1)*n,0]= D[:,f,a,0]-si[r,7]
        C[a*n:(a+1)*n,1]= D[:,f,a,1]-si[r,8]
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
        KH[i,:]=J[sel,:].mean(0)*500/14.0
        KC[i,:]=J[sel2,:].mean(0)*500/14.0
    from scipy.ndimage.filters import gaussian_filter
    sig=[0.5,2.5]
    KH=gaussian_filter(KH,sig)
    KC=gaussian_filter(KC,sig)
    #sel=H<5
    #ot=J[sel,:].sum(0)
    
    plt.imshow(KH,origin='lower',extent=[-200,100,0,20],aspect=12,cmap='hot')   
        
    plt.colorbar()
    plt.figure()
    plt.imshow(KC,origin='lower',extent=[-200,100,0,20],aspect=12,cmap='hot')
    plt.colorbar()
    plt.figure()
    plt.plot(d,KH[:,5])
    plt.plot(d,KC[:,5])
    plt.legend(['gaze','rand'])
    plt.figure()
    x=np.linspace(-200,100,KH.shape[1])
    plt.plot(x,KH.mean(0))
    plt.plot(x,KC.mean(0))
    plt.legend(['gaze','rand'])
##    ci=np.zeros((J.shape[1],2))
##    for f in range(J.shape[1]):
##        print f
##        R=np.zeros(1000)
##        for rep in range(1000):
##            rnd=np.random.randint(0,J.shape[0],J.shape[0])
##            R[rep]=J[rnd,f].mean()
##        R.sort()
##        ci[f,0]=R[50]
##        ci[f,1]=R[950]
##    ci=ci*500/14.0
    plt.figure()
    x=np.linspace(-200,100,J.shape[1])
    plt.plot(x,gaussian_filter(J.mean(0)*500/14.0,2.5),'-k')
    ci=np.load('ci.npy')
    plt.plot(x,gaussian_filter(ci[:,0],2.5),'--k')
    plt.plot(np.linspace(-200,100,J.shape[1]),gaussian_filter(ci[:,1],2.5),'--k')
    return J
def agentDensity(D,si):
    np.random.seed(1234)
    I=[]; I2=[]
    plot=True
    g=1
    flim=[-200,100]
    figs=range(1,100)
    fig1=figs.pop(0)
    fig2=figs.pop(0)
    for f in range(0,D.shape[1],5):
        H=np.zeros((14*D.shape[0],2))
        C=np.zeros((14*D.shape[0],2))
        G=np.zeros((14*D.shape[0],2))
        r=np.random.permutation(D.shape[0])
        n=D.shape[0]
        #f=-1
        for a in range(14):
            H[a*n:(a+1)*n,0]= D[:,f,a,0]-si[:,6]
            H[a*n:(a+1)*n,1]= D[:,f,a,1]-si[:,7]
            C[a*n:(a+1)*n,0]= D[:,f,a,0]-si[r,6]
            C[a*n:(a+1)*n,1]= D[:,f,a,1]-si[r,7]
            G[a*n:(a+1)*n,0]= D[:,f,a,0]-D[:,f,2,0]
            G[a*n:(a+1)*n,1]= D[:,f,a,1]-D[:,f,2,1]
        H=H[~np.isnan(H[:,0]),:]
        C=C[~np.isnan(C[:,0]),:]
        if plot:
            if g>12: fig1=figs.pop(0); fig2=figs.pop(0); g=1
            plt.figure(fig1)
            plt.subplot(3,4,g)
            plt.title('Frame '+str(f*2+flim[0]))
            bn=np.linspace(-20,20,41)
            b1,c,d=np.histogram2d(H[:,0],H[:,1],bins=[bn,bn])
            b2,c,d=np.histogram2d(C[:,0],C[:,1],bins=[bn,bn])
            plt.imshow(b1-b2, extent=[-20,20,-20,20],origin='lower',
                       interpolation='nearest',vmin=-140,vmax=40)
            plt.set_cmap('hot')
            plt.colorbar()
            plt.figure(fig2)
            plt.subplot(3,4,g)
            plt.title('Frame '+str(f*2+flim[0]))
            g+=1
        
        bn=np.arange(0,26,0.5)
        d= bn+(bn[1]-bn[0])/2.0
        d=d[:-1]
        a,b=np.histogram(np.sqrt((C*C).sum(axis=1)),bins=bn)
        c=np.diff(np.pi*bn**2)
        bbb=0
        I2.append(a[bbb]/c[bbb]/n*(bn.size-1))
        if plot: plt.plot(d,a/c/n*(bn.size-1))
        a,b=np.histogram(np.sqrt((H*H).sum(axis=1)),bins=bn)
        if plot: plt.plot(d,a/c/n*(bn.size-1))
        I.append(a[bbb]/c[bbb]/n*(bn.size-1))
        a,b=np.histogram(np.sqrt((G*G).sum(axis=1)),bins=bn)
        c[0]=0
        if plot:
            plt.plot(d,a/c/n*(bn.size-1))
            plt.ylim([0,2])
            plt.xlabel('Distance from Saccade Target Location')
            plt.ylabel('Agent Density') #[# agents per deg^2]
            if g==2:plt.legend(['rand pos','gaze','rand ag'])
    plt.figure()
    plt.plot(np.linspace(flim[0],flim[1],len(I)),I)
    plt.plot(np.linspace(flim[0],flim[1],len(I)),I2)
    plt.legend(['gaze','rand'])
    plt.show()
    #return H,C,G,bn
    
E,D,si=loadData()
##J=dirChanges(D,si)
agentDensity(D,si)

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

plt.show()

