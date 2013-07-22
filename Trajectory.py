from Constants import *
from Settings import Q
from psychopy.event import xydist
import numpy as np
import random, os, pickle
from Maze import *
class Diagnosis:
    def __init__(self,replications=100,nragents=[8,11,14,17,20],
                 dispSizes=[18,22,26,29,32], rejDists=[0.0,1.5,3.0]):
        def dist(a,b):
            return np.sqrt((a[:,0]-b[:,0])**2+(a[:,1]-b[:,1])**2)
        self.rejDists=rejDists
        self.dispSizes=dispSizes
        self.nragents=nragents
        self.ndirchange=np.zeros((len(dispSizes),len(nragents),len(rejDists),3))
        self.ncrashes= np.zeros((len(dispSizes),len(nragents),len(rejDists),3))
        self.nbacktracks=np.zeros((len(dispSizes),len(nragents),len(rejDists)))
        self.dists=np.zeros((len(dispSizes),len(nragents),len(rejDists),4))
        self.adens=np.zeros((len(dispSizes),len(nragents),len(rejDists),3,4))
        self.acrashes=np.zeros((len(dispSizes),len(nragents),len(rejDists),3))
        self.adirs=np.zeros((len(dispSizes),len(nragents),len(rejDists),3,36))
        self.nndists=np.zeros((len(dispSizes),len(nragents),len(rejDists),3))
        edges=range(0,370,10)
        for d in range(len(dispSizes)):
            maze=EmptyMaze((1,1),dispSize=(dispSizes[d],dispSizes[d]))
            print 'disp',d
            for na in range(len(nragents)):
                for rd in range(len(rejDists)):
                    print '\trejdist',rd
                    for r in range(replications):
                        print 'r', r
                        pos=None
                        while pos==None:
                            pos,phi,crashes,bts=generateTrial(nragents[na],maze,
                                    STATISTICS=True,rejectionDistance=rejDists[rd])
                        #np.save('phi.npy',phi)
                        self.ndirchange[d,na,rd,:]+=(phi!=np.roll(phi,1,axis=0)).sum(axis=0)
                        self.ncrashes[d,na,rd,:]+=crashes
                        self.nbacktracks[d,na,rd]+=bts
                        self.dists[d,na,rd,0]+=dist(pos[:,0,:],pos[:,1,:]).mean()
                        self.dists[d,na,rd,1]+=dist(pos[:,0,:],pos[:,2,:]).mean()
                        self.dists[d,na,rd,2]+=dist(pos[:,2,:],pos[:,1,:]).mean()
                        self.dists[d,na,rd,3]+=dist(pos[:,2,:],pos[:,3,:]).mean()
                        for a1 in range(3):
                            mindist=np.inf*np.ones((Q.nrframes))
                            for a2 in range(nragents[na]):
                                if a1!=a2:
                                    aadist=dist(pos[:,a1,:],pos[:,a2,:])
                                    mindist=np.concatenate((np.matrix(mindist),np.matrix(aadist)),axis=1).min(axis=1)
                                    self.adens[d,na,rd,a1,0]+=(aadist<2).sum()
                                    self.adens[d,na,rd,a1,1]+=(aadist<3).sum()
                                    self.adens[d,na,rd,a1,2]+=(aadist<4).sum()
                                    self.adens[d,na,rd,a1,3]+=(aadist<5).sum()
                                    overlap=(aadist<1)
                                    self.acrashes[d,na,rd,a1]+=(np.bitwise_and(overlap,
                                        np.bitwise_not(np.roll(overlap,1,axis=0)))).sum()
                            self.nndists[d,na,rd,a1]+=mindist.mean()        
                            df=np.mod(360+np.diff(phi[:,a1]),360)
                            df[df>180]=360-df[df>180]
                            self.adirs[d,na,rd,a1,:]+= np.histogram(df,bins=edges)[0]
        # nr dir changes per second
        self.ndirchange=self.ndirchange/float(replications)/float(Q.trialDur)
        # nr crashes per second
        self.ncrashes=self.ncrashes/float(replications)/float(Q.trialDur)
        # nr backtracks per trial
        self.nbacktracks=self.nbacktracks/float(replications)
        # average distance during a trial
        self.dists=self.dists/float(replications)
        self.adens=self.adens/float(replications)/float(Q.nrframes)
        self.acrashes= self.acrashes/float(replications)/float(Q.trialDur)
    def plot(self):
        plt.close('all')
        print self.rejDists
        plt.figure(figsize=(12,6))
        for rd in range(len(self.rejDists)):
            plt.subplot(1,3,rd+1)
            plt.plot(self.dispSizes,self.ndirchange[:,:,rd,:].mean(axis=1))
            plt.legend(['Chasee','Chaser','Distractor'],loc=1)
            plt.title('Rejection Dist = %.1f'%self.rejDists[rd])
            if rd==0:
                plt.ylabel('Number of Direction Changes')
            else:
                plt.gca().set_yticklabels([])
            plt.xlabel('Display Size')
            plt.ylim([4,7])
            plt.grid()
        plt.subplots_adjust(wspace=0.1)
        # crashes
        plt.figure(figsize=(12,6))
        for rd in range(len(self.rejDists)):
            plt.subplot(1,3,rd+1)
            plt.plot(self.dispSizes,self.ncrashes[:,:,rd,:].mean(axis=1))
            plt.legend(['Chasee','Chaser','Distractor'],loc=1)
            plt.title('Rejection Dist = %.1f'%self.rejDists[rd])
            if rd==0:
                plt.ylabel('Number of Crashes')
            else:
                plt.gca().set_yticklabels([])
            plt.xlabel('Display Size')
            plt.ylim([-0.1,1.4])
            plt.grid()
        plt.subplots_adjust(wspace=0.1)
        # backtracks
        plt.figure()
        plt.plot(self.dispSizes,self.nbacktracks.mean(axis=1))
        plt.legend(['RD=0.0','RD=1.5','RD=3.0'],loc=1)
        plt.ylabel('Number of Sampling Rejections per Trial')
        plt.xlabel('Display Size')
        plt.ylim([-10,250])
        plt.grid()
        
        # average distance
        tit=['Avg Chasee-Chaser Distance','Avg Chasee-Distractor Distance',
             'Avg Chaser-Distractor Distance','Avg Distractor-Distractor Distance']
        plt.figure(figsize=(10,10))
        for rd in range(4):
            plt.subplot(2,2,rd+1)
            plt.plot(self.dispSizes,self.dists[:,:,:,rd].mean(axis=1))
            if rd==2:
                plt.legend(['RD=0.0','RD=1.5','RD=3.0'],loc=4)
            plt.title(tit[rd])
            plt.ylabel('Distance in Deg')
            plt.xlabel('Display Size')
            #plt.ylim([2,15])
            plt.grid()
        #plt.subplots_adjust(wspace=0.1)

        # agent density
        for wsize in [2,3,4,5]:
            plt.figure(figsize=(12,12))
            ags=['Chasee','Chaser','Distractor']
            for rd in range(len(self.rejDists)):
                for a in range(3):
                    plt.subplot(3,3,3*rd+a+1)
                    plt.plot(self.dispSizes,self.adens[:,:,rd,a,wsize-2])
                    #plt.legend(['Chasee','Chaser','Distractor'],loc=1)
                    plt.title('WS=%d,RD = %.1f, %s'%(wsize,self.rejDists[rd],ags[a]))
                    if a==0: plt.ylabel('Agent Density')
                    else: plt.gca().set_yticklabels([])
                    if rd==2: plt.xlabel('Display Size')
                    else: plt.gca().set_xticklabels([])
                    if wsize==2: plt.ylim([0,1.4])
                    elif wsize==3: plt.ylim([0,3])
                    elif wsize==4: plt.ylim([0,4])
                    elif wsize==5: plt.ylim([0,6])
                    plt.grid()
            plt.subplots_adjust(wspace=0.1)

        plt.figure(figsize=(12,12))
        ags=['Chasee','Chaser','Distractor']
        for rd in range(len(self.rejDists)):
            for a in range(3):
                plt.subplot(3,3,3*rd+a+1)
                plt.plot(self.dispSizes,self.acrashes[:,:,rd,a])
                #plt.legend(['Chasee','Chaser','Distractor'],loc=1)
                plt.title('RD = %.1f, %s'%(self.rejDists[rd],ags[a]))
                if a==0: plt.ylabel('Nr Crashes')
                else: plt.gca().set_yticklabels([])
                if rd==2: plt.xlabel('Display Size')
                else: plt.gca().set_xticklabels([])
                plt.ylim([0,4])
                plt.grid()
        plt.subplots_adjust(wspace=0.1)

    def plotDirs(self):
        plt.close('all')
        ags=['Chasee','Chaser','Distr']
        plt.figure(figsize=(12,12))
        edges=np.array(range(0,360,10))+5
        print 'here'
        for ds in range(len(self.dispSizes)):
            for na in range(len(self.nragents)):
                plt.subplot(5,5,ds*5+na+1)
                #print self.adirs[ds,na,0,:,:].squeeze().transpose().shape
                plt.plot(edges,np.log(self.adirs[ds,na,0,:,:].squeeze().transpose()))
                #plt.ylim([0,8000])
                plt.xlim([0,360])
                plt.title('DS=%d, NA=%d'%(self.dispSizes[ds],self.nragents[na]))
                if na==0: plt.ylabel('Frequency')
                else: plt.gca().set_yticklabels([])
                if ds==4: plt.xlabel('Angle')
                else: plt.gca().set_xticklabels([])
                if na==0 and ds==0: plt.legend(ags,loc=0)
        plt.subplots_adjust(wspace=0.1)
        
class RandomAgent():
    def __init__(self,nrframes,dispSize,pdc,sd,moveRange):
        self.ds=dispSize
        self.nrframes=nrframes
        self.traj=np.zeros((nrframes,3))
        self.reset()
        self.pdc=pdc
        self.sd=sd
        self.nrcrashes=np.zeros((nrframes))
        self.moveRange=moveRange/2.0
    
    def reset(self):
        """ choose Random position within arena """
        self.f=0
        self.i=0
        self.traj[self.f,:]=np.array((random.random()*self.ds[X]-self.ds[X]/2.0,
            random.random()*self.ds[Y]-self.ds[Y]/2.0,random.random()*360))
        
    def backtrack(self):
        self.f-=51#31
        return self.f<0 or self.i>100000#10000
    def getPosition(self,dec=0):
        return self.traj[self.f+dec,[X,Y]]
    def getTrajectory(self):
        return self.traj
    def move(self):
        self.f+=1
        self.i+=1
        f=self.f
        self.nrcrashes[f]=0
        rnd = random.random()<self.pdc
        if rnd: # change direction chasee
            self.traj[f,PHI]=(self.traj[f-1,PHI]
                    +random.uniform(-self.moveRange,self.moveRange))%360
        else:
            self.traj[f,PHI]= self.traj[f-1,PHI]
        adjust =np.array((np.cos(self.traj[f,PHI]/180.0*np.pi) 
                *self.sd,np.sin(self.traj[f,PHI]/180.0*np.pi)*self.sd))
        self.traj[f,[X,Y]]=self.traj[f-1,[X,Y]]+adjust
        return (f+1)==self.nrframes
    def crashed(self,newD):
        self.nrcrashes[self.f]=1
        self.traj[self.f,PHI]=newD[1]
        self.traj[self.f,[X,Y]]=newD[0]

class HeatSeekingChaser(RandomAgent):
    def __init__(self,*args,**kwargs):
        isGreedy=args[-1]        
        RandomAgent.__init__(self,*args[:-1],**kwargs)
        self.isGreedy=isGreedy
    def move(self,targetPos,crash=False):
        if not crash:
            self.f+=1
            self.i+=1
        f=self.f
        rnd = random.random()<self.pdc
        if rnd or crash:
            self.traj[f,PHI]=np.arctan2(targetPos[Y],
                            targetPos[X])/np.pi * 180
            self.traj[f,PHI]=(360+self.traj[f,PHI]
                +random.uniform(-self.moveRange,self.moveRange))%360
        else:
            self.traj[f,PHI]= self.traj[f-1,PHI]   
        adjust =np.array((np.cos(self.traj[f,PHI]/180.0*np.pi) 
                *self.sd,np.sin(self.traj[f,PHI]/180.0*np.pi)*self.sd))
        self.traj[f,[X,Y]]=self.traj[f-1,[X,Y]]+adjust
    def crashed(self,newD=None,targetPos=(0,0)):
        #print 'crashed', self.f
        if not self.isGreedy:
            RandomAgent.crashed(self,newD)
        else:
            #print 'before', self.getPosition(), targetPos
            self.move(targetPos,crash=True)
            #print 'after', self.getPosition()

            
def generateTrial(nragents,maze,rejectionDistance=0.0,STATISTICS=False):
    if STATISTICS: nrbacktracks=0
    # init chaser chasee
    chasee=RandomAgent(Q.nrframes,maze.dispSize,
            Q.pDirChange[CHASEE],Q.aSpeed,Q.phiRange[0])
    chaser=HeatSeekingChaser(Q.nrframes,maze.dispSize,
            Q.pDirChange[CHASER],Q.aSpeed,Q.phiRange[CHASER],True)
    while (xydist(chaser.getPosition(),chasee.getPosition())<Q.initDistCC[MIN]
        and xydist(chaser.getPosition(),chasee.getPosition())>Q.initDistCC[MAX]):
        # resample until valid distance between chaser and chasee is obtained
        chasee.reset(); chaser.reset()
    agents=[chasee,chaser]
    # init distractors
    for d in range(nragents-2):
        distractor=RandomAgent(Q.nrframes,maze.dispSize,
            Q.pDirChange[DISTRACTOR],Q.aSpeed,Q.phiRange[CHASEE])
        agents.append(distractor)
    # check for wall collisions
    for a in range(nragents):
        d,edge=maze.shortestDistanceFromWall(agents[a].getPosition())
        while d<=Q.agentRadius:
            agents[a].reset()
            d,edge=maze.shortestDistanceFromWall(agents[a].getPosition())
            
    # generate the movement of chasee and chaser
    finished=False
    while not finished:
        # check the distance
        (dx,dy)=chasee.getPosition() - chaser.getPosition()
        if np.sqrt(dx**2+dy**2)<rejectionDistance:
            if STATISTICS: nrbacktracks+=1
            deadend=chaser.backtrack()
            chasee.backtrack()
            if deadend:
                print 'dead end', chasee.f
                if STATISTICS: return None, None, None, None
                else: return None
            (dx,dy)=chasee.getPosition() - chaser.getPosition()
        # move chaser and avoid walls
        chaser.move((dx,dy))
        d,edge=maze.shortestDistanceFromWall(chaser.getPosition())
        if d<=Q.agentRadius:
            newD=maze.bounceOff(chaser.getPosition(),
                chaser.getPosition(-1),edge,Q.agentRadius)
            chaser.crashed(newD=newD,targetPos=(dx,dy))
        # move chasee and avoid walls
        finished=chasee.move()
        d,edge=maze.shortestDistanceFromWall(chasee.getPosition())
        if d<=Q.agentRadius:
            newD=maze.bounceOff(chasee.getPosition(),
                chasee.getPosition(-1),edge,Q.agentRadius)
            chasee.crashed(newD)
        #if chaser.f>401:
        #    raise NameError('stop')
    # generate distractor movement
    finished=False
    while not finished and nragents>2:
        for a in range(2,nragents):
            finished=agents[a].move()
            d,edge=maze.shortestDistanceFromWall(agents[a].getPosition())
            if d<=Q.agentRadius:
                newD=maze.bounceOff(agents[a].getPosition(),
                    agents[a].getPosition(-1),edge,Q.agentRadius)
                agents[a].crashed(newD)
    trajectories=np.zeros((Q.nrframes,nragents,3))
    for a in range(nragents):
        tt=agents[a].getTrajectory()
        trajectories[:,a,X]=tt[:,X]
        trajectories[:,a,Y]=tt[:,Y]
        trajectories[:,a,PHI]=tt[:,PHI]
        
    if STATISTICS:
        #statistics=np.zeros((nrframes,nragents,3))
        statistics=[trajectories,np.zeros((Q.nrframes,3)),
                    np.zeros((3)),nrbacktracks]
        for a in range(3):
            statistics[1][:,a]=agents[a].getTrajectory()[:,PHI]
            statistics[2][a]=agents[a].nrcrashes.sum()
        return statistics
    else: return trajectories
    
def generateShortTrial(maze,nrdirch=4,rejectionDistance=0.0):
    Q.trialDur=5
    Q.nrframes=Q.trialDur*Q.refreshRate+1
    traj=generateTrial(2,maze,rejectionDistance)
    ende=np.nonzero(np.cumsum(np.diff(traj[:,1,PHI])>1)==nrdirch)[0]
    print len(ende)
    ende=np.array(ende).min()
    return traj[:ende,:,:]

def generateExperiment(vpn,nrtrials,conditions=None,dispSizes=None,maze=None,rejectionDistance=0):
    #os.chdir('..')
    os.chdir('input/')
    conditions=np.repeat(conditions,nrtrials)
    dispSizes=np.repeat(dispSizes,nrtrials)
    mazes=[]
    for d in dispSizes:
        mazes.append(EmptyMaze((1,1),dispSize=(d,d)))
    print 'Generating Trajectories'
    for vp in vpn:
        vpname='vp%03d' % vp
        os.mkdir(vpname)
        os.chdir(vpname)
        order=np.random.permutation(conditions.size)
        for trial in range(conditions.size):
            trajectories=None
            while trajctories ==None:
                trajectories=generateTrial(conditions[order[trial]],
                    maze=mazes[order[trial]],
                    rejectionDistance=rejectionDistance)
            #fn='%str%03dcond%02d'% (vpname,trial,conditions[order[trial]])
            fn = 'trial%03d' % trial
            print fn
            np.save(fn,trajectories)  
        os.chdir('..')
    os.chdir('..')
    
def generateMixedExperiment(vpn,trialstotal,blocks=4,condition=14,
        dispSize=26,maze=None,probeTrials=False):
    #os.chdir('..')
    os.chdir(Q.inputPath)
    mazes=[]
    if probeTrials: bs=range(0,blocks+1)
    else: bs=range(1,blocks+1)
    print 'Generating Trajectories'
    for vp in vpn:
        vpname='vp%03d' % vp
        os.mkdir(vpname)
        os.chdir(vpname)
        Q.save('SettingsTraj.pkl')
        
        for block in bs:
            if block ==0: nrtrials=10
            else: nrtrials=trialstotal
            for trial in range(nrtrials):
                if trial >= nrtrials*0.9: rd=0.0
                else: rd=3.0
                trajectories=None
                while trajctories ==None:
                    trajectories=generateTrial(condition, 
                        maze=EmptyMaze((1,1),dispSize=(dispSize,dispSize)),rejectionDistance=rd)
                #fn='%str%03dcond%02d'% (vpname,trial,conditions[order[trial]])
                #fn = 'trial%03d' % trial
                fn='%sb%dtrial%03d'% (vpname,block,trial)
                print fn
                np.save(fn,trajectories)
            while True:# check that more than 1 consecutive control trials do not occur
                r=np.random.permutation(nrtrials)
                r2=np.roll(np.random.permutation(nrtrials)>=nrtrials-0.1*nrtrials,1)
                #r3=np.roll(np.random.permutation(50)>=45,2)
                if not np.any(np.bitwise_and(r,r2)):
                    break
            np.save('order%sb%d'% (vpname,block),r)
        os.chdir('..')
    os.chdir('..')
    
def generateBabyExperiment(vpn,nrtrials=10,blocks=1,conditions=[6,8],
        dispSize=29,maze=None):
    #os.chdir('..')
    os.chdir(Q.inputPath)
    mazes=[]
    Q.nrframes+= Q.refreshRate *5
    print 'Generating Trajectories'
    for vp in vpn:
        vpname='vp%03d' % vp
        os.mkdir(vpname)
        os.chdir(vpname)
        r=[]
        phase=[0,1,1,2]
        for i in range((len(conditions)*nrtrials-len(phase))/2):
            if np.random.rand()>0.5: phase.extend([1,2])
            else: phase.extend([2,1])
        print 'phase', phase
        for block in range(blocks):
            i=0
            for condition in conditions:
                for trial in range(nrtrials):
                    if condition==conditions[0]: 
                        if np.random.rand()>0.5: r.extend([trial, trial+nrtrials])
                        else: r.extend([trial+nrtrials,trial])
                    trajectories=None
                    while trajectories ==None:
                        trajectories=generateTrial(condition, 
                            maze=EmptyMaze((1,1),dispSize=(dispSize,dispSize)),rejectionDistance=3.0)
                    #fn='%str%03dcond%02d'% (vpname,trial,conditions[order[trial]])
                    #fn = 'trial%03d' % trial
                    trajectories=trajectories[(Q.refreshRate*5):]
                    #print trajectories.shape
                    fn='%sb%dtrial%03d'% (vpname,block,i)
                    i+=1
                    print fn
                    np.save(fn,trajectories)
            #r=np.random.permutation(nrtrials*len(conditions))
            r=np.array(r)
            print r
            np.save('order%sb%d'% (vpname,block),r)
            np.save('phase%sb%d'% (vpname,block),phase)
            Q.save('SettingsTraj.pkl')
        os.chdir('..')
    os.chdir('..')

def generateTremouletTrial(phi=0, lam=1):
    refreshRate=60 # Hz
    speed=3.25/refreshRate
    angle=0
    duration=0.75 # in seconds
    N=np.int32(duration*refreshRate) # nr of frames
    traj = np.zeros((N,1,2))
    traj[0,0,X] = -speed*(N/2-0.5); traj[0,0,Y]=0
    for i in range(1,N):
        traj[i,0,X]=np.cos(angle/180.0*np.pi)*speed+traj[i-1,0,X]
        traj[i,0,Y]=np.sin(angle/180.0*np.pi)*speed+traj[i-1,0,Y]
        if i==((N-1)/2):
            speed=speed*lam
            angle=angle+phi
    return traj
def generateGao09e1(vpn):
    nrtrials=15
    maze=EmptyMaze((1,1),dispSize=(32,24))
    chs=[0,60,120,180,240,300]
    block=1
    #os.chdir('..')
    Q.trialDur=10
    
    os.chdir('input/')
    for vp in vpn:
        vpname='vp%03d' % vp
        os.mkdir(vpname)
        os.chdir(vpname)
        i=0
        r=np.zeros((2*6*nrtrials,2))
        r[:,0]=np.random.permutation(2*6*nrtrials)
        for cond in range(6):
            for trial in range(nrtrials):
                Q.phiRange=(chs[cond],120)
                trajectories=generateTrial(5,maze=maze,
                    rejectionDistance=5.0)
                #target present trial
                r[i,1]=cond 
                fn='gao09e1%sb%dtrial%03d'% (vpname,block,i); 
                np.save(fn,trajectories[:,:-1,:]);i+=1
                #target absent trial
                r[i,1]=cond+6
                fn='gao09e1%sb%dtrial%03d'% (vpname,block,i); 
                np.save(fn,trajectories[:,1:,:]);i+=1

        np.save('gao09e1order%sb%d'% (vpname,block),r)
        Q.save('SettingsTraj.pkl')
        os.chdir('..')
    os.chdir('..')
def exportSvmGao09(nrtrials=10000):
    def saveTraj(fout,traj,label):
        sample=5
        fout.write('%d '%label)
        i=1
        for f in range(traj.shape[0]/sample):
            for a in range(2):
                fout.write('%d:%f '%(i,traj[f*sample,a,0]));i+=1
                fout.write('%d:%f '%(i,traj[f*sample,a,1]));i+=1
        fout.write('\n')            
        
    maze=EmptyMaze((1,1),dispSize=(32,24))
    chs=[300,240,180,120,60,0]
    block=1
    os.chdir('input/')
##    fout=open('svmGao2.train','w')
##    for trial in range(nrtrials):
##        print trial
##        for cond in range(6):
##            trajectories=generateTrial(5,maze=maze,rejectionDistance=5.0,
##                moveSubtlety=(chs[cond],120),trialDur=10)
##            trajectories[:,:,0]/= 32.0
##            trajectories[:,:,1]/= 24.0
##            saveTraj(fout,trajectories[:,[0,1],:],1);
##            saveTraj(fout,trajectories[:,[4,1],:],-1);
##    fout.close()

    nrtrials=10000
        
    for cond in range(5):
        print cond
        fout1=open('svmGaoCond%03dT.train'%chs[cond],'w')
        fout2=open('svmGaoCond%03dF.train'%chs[cond],'w')
        for trial in range(nrtrials):
            trajectories=generateTrial(5,maze=maze,rejectionDistance=5.0,
                moveSubtlety=(chs[cond],120),trialDur=10)
            trajectories[:,:,0]/= 32.0
            trajectories[:,:,1]/= 24.0
            saveTraj(fout1,trajectories[:,[0,1],:],1);
            saveTraj(fout2,trajectories[:,[4,1],:],-1);
        fout1.close()
        fout2.close()
    os.chdir('..')
    

if __name__ == '__main__':
    
    #d=28
    #random.seed(3)
    #maze=EmptyMaze((1,1),dispSize=(32,24))
    
    #generateMixedExperiment([84],40,blocks=1,condition=14,dispSize=26,probeTrials=True)
    #t=generateTrial(5,maze,rejectionDistance=5.0,moveSubtlety=(0,120),trialDur=10)
    #print t.shape
    #t=np.load('input/vp023/gao09e1vp023b1trial003.npy')
    #TD=TrajectoryData(t,trajRefresh=60.0,highlightChase=True)
    #TD.replay(tlag=0)
    #TD.showFrame(t[333,:,:])
    #generateExperiment([105],1000,conditions=[20],dispSizes=[32],rejectionDistance=3.0)
    #generateExperiment([0],1,conditions=[8,11,14,17,20],
    #                   dispSizes=[18,22,26,29,32],rejectionDistance=3.0)
    #generateExperiment([92],2,[6],[22])
        
    #generateBabyExperiment([201])
    
    #t=generateShortTrial(maze)
    #print t.shape
    #np.save('t.npy',t)
    #exportSvmGao09()
    #generateGao09e1([23])
##    d=Diagnosis(replications=100,nragents=[6],dispSizes=[29], rejDists=[3])
##    f=open('diagBaby4.pkl','w')
##    pickle.dump(d,f)
##    f.close()
    f=open('diagBaby4.pkl')
    diag=pickle.load(f)
    f.close()
    
