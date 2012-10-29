from Settings import *
from Constants import *
from psychopy import visual, core, event,gui
from psychopy.misc import pix2deg
import numpy as np


def exportFrame(positions,fn,maze=None,wind=None):
    if type(wind)==type(None):
        wind=initDisplay()
    try:
        cond=positions.shape[0]
        clrs=np.ones((cond,3))
        clrs[CHASEE,:]=(0,1,0)
        clrs[CHASER,:]=(1,0,0)
##        if a1>1:
##            clrs[a1]=(0,0,1)
##        else:
##            clrs[a2]=(0,0,1)
        if type(maze)!=type(None):
            maze.draw(wind)
            maze.draw(wind)
        elem=visual.ElementArrayStim(wind,fieldShape='circle',
            nElements=cond,sizes=agentSize,rgbs=clrs,
            elementMask='circle',elementTex=None)
        elem.setXYs(positions)      
        elem.draw()    
        #wind.flip()
        wind.getMovieFrame(buffer='back')
        wind.saveMovieFrames(fn+'.png')
        wind.close()
    except: 
        wind.close()
        raise
    
def exportTrial(outname,trajectories,wind=None):
    if type(wind)==type(None):
        wind=initDisplay()
    try:
        nrframes=trajectories.shape[0]
        cond=trajectories.shape[1]
        elem=visual.ElementArrayStim(wind,fieldShape='sqr',nElements=cond,
            sizes=agentRadius*scale,elementMask='circle',elementTex=None)
        for f in range(nrframes):
            t0=core.getTime()
            elem.setXYs(trajectories[f,:,[X,Y]].transpose()*scale)
            elem.draw()
            #print core.getTime()-t0
            wind.update()
            wind.getMovieFrame(buffer='front')
            #wind.clearBuffer()
            for key in event.getKeys():
                if key in ['escape']:
                    wind.close()
                    break;
                    #core.quit()
        wind.saveMovieFrames('%s.mpeg'%outname,fps=60,mpgCodec='mpeg1video')
    except: 
        wind.close()
        raise
    
def exportExperiment(vp,wind):
    dn='vp%02d' % vp
    os.chdir('C:\Users\matus\Desktop\promotion\distractorVariation\code\%s' % dn)
    fnames=os.listdir('.')
    fnames.sort()
    for fname in fnames:
        exportTrial(np.load(fname),wind,fname[:-10])

    
def exportTremoulet(wind=None):
    from Trajectory import generateTremoulet
    if type(wind)==type(None):
        wind=initDisplay()
    try:
        agentRadius=0.23
        angles=[0,10,20,40,80]
        lambdas=[0.5,1,2,4]
        i=0
        os.chdir('tremoulet')
        for a in angles:
            for l in lambdas:
                t=generateTremouletTrial(a,l)
                exportTrial(t,wind,'trem%02d'%i)
                np.save('trem%02d.npy'%i,t)
                i+=1
    except: 
        wind.close()
        raise
    
    
def showFrame(positions,maze=None,wind=None, elem=None,highlightChase=False):
    if type(wind)==type(None):
        wind=initDisplay()
    if type(elem)==type(None):
        cond=positions.shape[0]
        clrs=np.ones((cond,3))
        if highlightChase:
            clrs[0,[0,2]]=0
            clrs[1,[1,2]]=0
        elem=visual.ElementArrayStim(wind,fieldShape='sqr',
            nElements=cond,sizes=agentSize,rgbs=clrs,
            elementMask='circle',elementTex=None)
    try:
        if type(maze)!=type(None):
            maze.draw(wind)
        elem.setXYs(positions)
        elem.draw()
        wind.flip()
    except: 
        wind.close()
        raise
    
def showTrial(trajectories,maze=None,wind=None,highlightChase=False,
        origRefresh=100.0,gazeData=None,gazeDataRefresh=250.0):
    """
        shows the trial as given by TRAJECTORIES
    """
    
    if type(wind)==type(None):
        wind=Q.initDisplay(1000)
    core.wait(2)
    try:
        nrframes=int(trajectories.shape[0]/origRefresh*Q.refreshRate)
        cond=trajectories.shape[1]
        if gazeData!=None:
            cond+=1
            gazeData=pix2deg(gazeData,wind.monitor)
        clrs=np.ones((cond,3))
        if gazeData!=None: clrs[-1,[0,1]]=0
        if highlightChase:
            clrs[0,[0,2]]=0
            clrs[1,[1,2]]=0
        elem=visual.ElementArrayStim(wind,fieldShape='sqr',
            nElements=cond,sizes=Q.agentSize,rgbs=clrs,
            elementMask='circle',elementTex=None)
        if type(maze)!=type(None):
            maze.draw(wind)
        wind.flip()
        t0=core.getTime()
        for f in range(nrframes):
            if origRefresh!=Q.refreshRate:
                fnew=f*origRefresh/Q.refreshRate
                if round(fnew)==fnew:
                    pos=trajectories[int(fnew),:,[X,Y]].transpose()
                else: # interpolate
                    pos0=trajectories[np.floor(fnew),:,[X,Y]].transpose()
                    pos1=trajectories[np.ceil(fnew),:,[X,Y]].transpose()
                    pos=pos0+(pos1-pos0)*(fnew-np.floor(fnew))
                    #print pos0[0],pos1[0],pos[0]
            else: pos=trajectories[f,:,[X,Y]].transpose()
            if gazeData!=None:
                fnew=f*gazeDataRefresh/Q.refreshRate
                if np.ceil(fnew)>=gazeData.shape[0]:
                    break
                if round(fnew)==fnew:
                    #print int(fnew)
                    gaze=gazeData[int(fnew),:]
                else: # interpolate
                    pos0=gazeData[np.floor(fnew),:]
                    pos1=gazeData[np.ceil(fnew),:]
                    gaze=pos0+(pos1-pos0)*(fnew-np.floor(fnew))
                
                pos=np.array(np.concatenate((pos,np.matrix(gaze)),axis=0))
            showFrame(pos, wind=wind,elem=elem, highlightChase=highlightChase)
            #core.wait(0.02)
            for key in event.getKeys():
                if key in ['escape']:
                    wind.close()
                    return
                    #core.quit()
        wind.flip()
        print core.getTime() - t0
        core.wait(2)
        wind.close()
    except: 
        wind.close()
        raise



if __name__ == '__main__':
    #t=np.load('test.npy')
    from Settings import *
    from Constants import *
    from Maze import *
    from evalETdata import *
    #data=readEdf(fname='VP001.ASC')
    #maze=TestMaze(10,dispSize=24)
    #traj=np.load('input/vp001/trial019.npy')
    #showTrial(traj,highlightChase=True,gazeData=data[18][:,[1,2]])
    #r=traj2image(t[113,:,:].squeeze())
    t=np.load('t.npy')
    showTrial(t)
