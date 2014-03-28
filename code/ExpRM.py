from Constants import *
from Settings import Q
from psychopy.misc import pix2deg,pol2cart
from psychopy.event import xydist
import numpy as np
import random, os, pickle
from Maze import *
from Trajectory import RandomAgent,HeatSeekingChaser


def generateRMTrial(maze,nrdirchange,pdch,speed,ch):
    Q.trialDur=10
    Q.nrframes=Q.trialDur*Q.refreshRate+1
    Q.pDirChange=[pdch,pdch]
    #Q.aSpeed=speed/Q.refreshRate
    #Q.refreshRate=Q.aSpeed*60/speed
    Q.phiRange[CHASER]=ch
    chasee=RandomAgent(Q.nrframes,maze.dispSize,
            Q.pDirChange[CHASEE]*2,Q.aSpeed,Q.phiRange[0])
    chaser=HeatSeekingChaser(Q.nrframes,maze.dispSize,
            Q.pDirChange[CHASER]*2,Q.aSpeed,Q.phiRange[CHASER],True)
    dist=5
    phi=random.uniform(0,360)
    chasee.traj[0,PHI]=(phi+random.uniform(-chasee.moveRange,chasee.moveRange))%360
    chaser.traj[0,PHI]=(phi+random.uniform(-chaser.moveRange,chaser.moveRange))%360
    chasee.traj[0,[X,Y]]=-np.array(pol2cart((phi-180),dist/2.0))
    chaser.traj[0,[X,Y]]= -chasee.traj[0,[X,Y]]
        
    agents=[chasee,chaser]
     
    # generate the movement of chasee and chaser
    for dc in range(2*nrdirchange):
        finished=False
        if dc%2==1:
            chasee.pdc=0; chaser.pdc=Q.pDirChange[CHASER]*2
            agent=chaser
        else:
            chaser.pdc=0; chasee.pdc=Q.pDirChange[CHASEE]*2
            agent=chasee
        while not finished:
            #print 'here'
            (dx,dy)=chasee.getPosition() - chaser.getPosition()
            #print dx,dy,agent.traj[agent.f,PHI]
            chaser.move((dx,dy))
            chasee.move()
            finished=agent.traj[agent.f,PHI]!=agent.traj[agent.f-1,PHI]
    trajectories=np.zeros((chasee.f,2,3))
    for a in range(2):
        tt=agents[a].getTrajectory()
        trajectories[:,a,X]=tt[:chasee.f,X]
        trajectories[:,a,Y]=tt[:chasee.f,Y]
        trajectories[:,a,PHI]=tt[:chasee.f,PHI]
    return trajectories

def generateExperiment(vpn,nrtrials,blocks=4,ndchs=[4,5,6],
        pdchs=[0.045,0.09,0.18],speeds=[4,9,14],chs=[0,30,60,120]):
    #os.chdir('..')
    os.chdir('input/')
    maze=EmptyMaze((1,1),(32,24))
    print 'Generating Trajectories'
    for vp in vpn:
        vpname='vp%03d' % vp
        os.mkdir(vpname)
        os.chdir(vpname)
        for block in range(1,blocks+1):
            for trial in range(nrtrials):
                ndch=ndchs[trial%len(ndchs)]
                pdch=pdchs[(trial/len(ndchs))%len(pdchs)]
                speed=speeds[(trial/len(ndchs)/len(pdchs))%len(speeds)]
                ch=chs[trial/len(ndchs)/len(pdchs)/len(speeds)]
                print trial, ndch,pdch,speed,ch
                trajectories=generateTrial(maze,ndch,pdch,speed,ch)
                
                fn='%sb%dtrial%03d'% (vpname,block,trial)
                print fn
                np.save(fn,trajectories)
            np.save('order%sb%d'% (vpname,block),r)
            f=open('Settings.pkl','w')
            pickle.dump(Q,f)
            f.close()
        os.chdir('..')
    os.chdir('..')


def showTrial(trajectories,maze,wind=None):
    """
        shows the trial as given by TRAJECTORIES
    """
    if type(wind)==type(None):
        wind=Q.initDisplay(1000,fullscr=False)
        
    nrframes=int(trajectories.shape[0])
    cond=trajectories.shape[1]
    clrs=np.ones((cond,3))
    elem=visual.ElementArrayStim(wind,fieldShape='sqr',
        nElements=cond,sizes=Q.agentSize,rgbs=clrs,
        elementMask=RING,elementTex=None)
    wind.flip()
    t0=core.getTime()
    for f in range(nrframes):
        pos=trajectories[f,:,[X,Y]].transpose()
        elem.setXYs(pos)
        elem.draw()
        wind.flip()
        #core.wait(0.1)
        for key in event.getKeys():
            if key in ['escape']:
                wind.close()
                return
                #core.quit()
    wind.flip()
    print core.getTime() - t0
    core.wait(1.0)
    wind.close()


maze=EmptyMaze((1,1),dispSize=(32,32))
t=generateRMTrial(maze,5,0.09,4,0)
showTrial(t,maze)
