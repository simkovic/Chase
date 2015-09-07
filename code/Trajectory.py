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

from psychopy.event import xydist
import numpy as np
import random, os, pickle
# NOTE: most of the motion settings are setup in Settings.py, not here
#   in particular check that the monitor frame rate parameter is set up
#   correctly in Settings.py, the actual monitor frame rate doesnt matter
#   for the trajectory generation
from Settings import Q 
from Constants import *
from Maze import *

class Diagnosis:
    ''' Due to the rejection sampling procedure that manipulates the
        chaser-chasee minimum distance the motion properties are do not have
        the average nominal values designated in Settings.py
        This class measures the average empirical values for a
        groups of nominal values. It can be used to choose nominal values
        such that the empirical values are matched across conditions
        or studies.
    '''
    def __init__(self,replications=100,nragents=[8,11,14,17,20],
                 dispSizes=[18,22,26,29,32], rejDists=[0.0,1.5,3.0]):
        '''
            The function samples batches of trials
            replications - number of trials that will be sampled
            nragents - tuple of ints, each element gives the number
                of agents in the batch
            dispSizes - tuple of ints, each element gives the size of
                the square movement area in degrees (defines NxN area)
            rejDists - tuple of floats, chaser-chasee minimum distance
                in degrees
            Function computes one sample batch for each combination
                parameters ie the total nr of batches is
                len(nragents)*len(dispSizes)*len(rejDists)
        '''
        def dist(a,b):
            return np.sqrt((a[:,0]-b[:,0])**2+(a[:,1]-b[:,1])**2)
        self.rejDists=rejDists
        self.dispSizes=dispSizes
        self.nragents=nragents
        self.dirchs=[]
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
                            pos,phi,crashes,bts,ndc=generateTrial(nragents[na],maze,
                                    STATISTICS=True,rejectionDistance=rejDists[rd])
                        #np.save('phi.npy',phi)
                        self.ndirchange[d,na,rd,:]+=np.array(ndc)
                        self.dirchs.append(ndc)
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
        '''
            Plots the results of the sampling, plots
            * average number of direction changes per second
            * number of crashes, ie average number of wall/boundary contacts
            * average number of rejectios of the rejection sampling algo
            * average agent distance in degrees
            * average agent density (agents per degree squared?)
        '''
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
        '''
            Plots the angle distribution during direction changes
        '''
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
######################################
# load/save routines
    @staticmethod
    def save(d,fname):
        f=open(fname,'w')
        pickle.dump(d,f)
        f.close()
    @staticmethod
    def load(fname):
        f=open(fname)
        d=pickle.load(f)
        f.close()
        return d
    @staticmethod
    def multiload(N,prefix='diagBaby'):
        D=[]
        for i in range(N):
            f=open(prefix+'%d.pkl'%(i+1))
            diag=pickle.load(f)
            f.close()
            D.extend(diag.dirchs)
        D=np.array(D)/float(Q.trialDur)
        print D.mean(0)
        print D.std(0)
        return D

######################################
# trajectory generation routines
        
class RandomAgent():
    '''generates the trajectory for a random agent '''
    def __init__(self,nrframes,dispSize,pos,pdc,sd,moveRange):
        ''' nrframes - number of frames (consecutive positions) to generate
            dispSize - size of the square movement area
            pos - initial position
            pdc - probability of a direction change
            sd - agent speed
            moveRange - size of the range from which new motion
                direction is selected after a direction change
        '''
        self.offset=pos
        self.ds=dispSize
        self.nrframes=nrframes
        self.traj=np.zeros((nrframes,3))
        self.reset()
        # some vars used for loging stats for the Diagnosis class
        self.pdc=pdc
        self.sd=sd
        self.nrcrashes=np.zeros((nrframes))
        self.ndc=np.zeros((self.nrframes))
        self.moveRange=moveRange/2.0
    
    def reset(self):
        """ reset the agent to initial position
            discards any previously generated trajectory
        """
        self.ndc=np.zeros((self.nrframes))
        self.nrcrashes=np.zeros((self.nrframes))
        self.f=0
        self.i=0
        self.traj[self.f,:]=np.array((random.random()*self.ds[X]-self.ds[X]/2.0+self.offset[X],
            random.random()*self.ds[Y]-self.ds[Y]/2.0+self.offset[Y],random.random()*360))
    def backtrack(self):
        ''' should the algorithm backtrack to previous position?
            returns boolean
        '''
        self.f-=51#31
        return self.f<0 or self.i>100000#10000
    def getPosition(self,dec=0):
        ''' return current/latest position'''
        return self.traj[self.f+dec,[X,Y]]
    def getTrajectory(self):
        return self.traj
    def move(self):
        ''' generate next position'''
        self.f+=1
        self.i+=1
        f=self.f
        self.nrcrashes[f]=0
        rnd = random.random()<self.pdc
        self.ndc[f]=float(rnd)
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
        ''' adjust direction and position after a contact with the boundary
            newD - new direction
        '''
        self.nrcrashes[self.f]=1
        self.traj[self.f,PHI]=newD[1]
        self.traj[self.f,[X,Y]]=newD[0]

class HeatSeekingChaser(RandomAgent):
    '''generates the trajectory for a heat-seeking chaser '''
    def __init__(self,*args,**kwargs):
        ''' the last argument (isGreedy) determines how boundary
            colisions are handle
            if True chaser makes random direction change after collision
            otherwise it moves towards chasee
        '''
        isGreedy=args[-1]        
        RandomAgent.__init__(self,*args[:-1],**kwargs)
        self.isGreedy=isGreedy
    def move(self,targetPos,crash=False):
        ''' generate next position
            targetPos - chasee's position
            crash - chaser made contact with boundary
        '''
        if not crash:
            self.f+=1
            self.i+=1
        f=self.f
        rnd = random.random()<self.pdc
        self.ndc[f]=int(rnd)
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
        ''' adjust direction and position after a contact with the boundary
            targetPos - chasee's position
            newD - new direction
        '''
        #print 'crashed', self.f
        if not self.isGreedy:
            RandomAgent.crashed(self,newD)
        else:
            #print 'before', self.getPosition(), targetPos
            self.move(targetPos,crash=True)
            self.nrcrashes[self.f]=1
            #print 'after', self.getPosition()

            
def generateTrial(nragents,maze,rejectionDistance=0.0,STATISTICS=False):
    ''' generates the agent trajectories for a single trial
        the generation may take considerably longer when rejectionDistance>0
        nragents - number of agents to generate
        maze - Maze class instance (see Maze.py)
        rejectionDistance - chaser-chasee minimum distance in degrees
        STATISTICS - if True will log stats for the Diagnosis class
        
        returns ndarray of size (nrframes x nragents x 3)
            first dim - number of frames is derived based on trial duration
                and frame rate (both in Settings.py)
            second dim - the first agents is chasee,
                         the second agents is chaser,
                         the rest are distractors
            third dim - X position in degrees, Y position in degrees,
                direction in radians  
        if STATISTICS is True returns statistics
        
    '''
    if STATISTICS: nrbacktracks=0
    # init chaser chasee
    chasee=RandomAgent(Q.nrframes,maze.dispSize,maze.pos,
            Q.pDirChange[CHASEE],Q.aSpeed,Q.phiRange[0])
    chaser=HeatSeekingChaser(Q.nrframes,maze.dispSize,maze.pos,
            Q.pDirChange[CHASER],Q.aSpeed,Q.phiRange[CHASER],True)
    while (xydist(chaser.getPosition(),chasee.getPosition())<Q.initDistCC[MIN]
        and xydist(chaser.getPosition(),chasee.getPosition())>Q.initDistCC[MAX]):
        # resample until valid distance between chaser and chasee is obtained
        chasee.reset(); chaser.reset()
    agents=[chasee,chaser]
    # init distractors
    for d in range(nragents-2):
        distractor=RandomAgent(Q.nrframes,maze.dispSize,maze.pos,
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
            if deadend: # reset the algorithm 
                print 'dead end', chasee.f
                if STATISTICS: return None, None, None, None,None
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
                    np.zeros((3)),nrbacktracks,[chasee.ndc.sum(),chaser.ndc.sum(),agents[2].ndc.sum()]]
        for a in range(3):
            statistics[1][:,a]=agents[a].getTrajectory()[:,PHI]
            statistics[2][a]=agents[a].nrcrashes.sum()
        return statistics
    else: return trajectories

###########################################################
# Routines for generating trajectories for some published experiments
# The experiments are based on information provided in the publication
# TODO use path prefix instead of os.chdir()

def generateExperiment(vpn,nrtrials,conditions=None,dispSizes=None,
                       maze=None,rejectionDistance=0):
    '''my work in progress, experiment without chatch trials'''
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
    '''my work in progress, experiment with chatch trials'''
    #os.chdir('..')
    os.chdir(Q.inputPath)
    mazes=[]
    if probeTrials: bs=range(0,blocks+1)
    else: bs=range(22,blocks+1)
    print 'Generating Trajectories'
    for vp in vpn:
        vpname='vp%03d' % vp
        #os.mkdir(vpname)
        os.chdir(vpname)
        Q.save('SettingsTraj.pkl')
        
        for block in bs:
            if block ==0: nrtrials=10
            else: nrtrials=trialstotal
            for trial in range(nrtrials):
                if vp>1 and vp<10: continue
                if trial >= nrtrials*0.9: rd=0.0
                else: rd=3.0
                trajectories=None
                while trajectories ==None:
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
    

if __name__ == '__main__':
    generateMixedExperiment([1],40,blocks=24,probeTrials=True)
    
        

    
