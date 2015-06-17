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

from Settings import *
from Constants import *
from psychopy import visual, core, event,gui
from psychopy.misc import pix2deg
import numpy as np
from subprocess import call

####################################################
# export functionality
def exportFrame(positions,fn,maze=None,wind=None):
    ''' exports a frame as an image in png format
        positions - Nx2 ndarray with coordinates of N agents
        fn - output file name (without suffix)
        maze & wind - provide window and maze for drawing 
    '''
    if type(wind)==type(None):
        wind=initDisplay()
    try:
        cond=positions.shape[0]
        clrs=np.ones((cond,3))
        clrs[CHASEE,:]=(0,1,0)
        clrs[CHASER,:]=(1,0,0)
        if type(maze)!=type(None):
            maze.draw(wind)
            maze.draw(wind)
        elem=visual.ElementArrayStim(wind,fieldShape='circle',
            nElements=cond,sizes=agentSize,rgbs=clrs,
            elementMask='circle',elementTex=None)
        elem.setXYs(positions)      
        elem.draw()    
        wind.getMovieFrame(buffer='back')
        wind.saveMovieFrames(fn+'.png')
        wind.close()
    except: 
        wind.close()
        raise
    
def exportTrial(outname,trajectories,wind=None,hz=60):
    ''' exports trajectories as mpeg movie
        trajectories - FxNx2 ndarray with coordinates of N agents
            for each of the F frames
        outname - output file name (without suffix)
        wind - provide window for drawing
        hz - frate rate of the output file
    '''
    #breaks=[250,500,750,1000,1250,1500,1750,2000,2250,2550]
    #breaks=[200,400,600]
    # you may wish to define breaks which lists values of frames on
    # which RAM will be flushed, helpful for long movies/low RAM machines
    if type(wind)==type(None):
        wind=Q.initDisplay()
    try:
        nrframes=trajectories.shape[0]
        cond=trajectories.shape[1]
        elem=visual.ElementArrayStim(wind,fieldShape='sqr',nElements=cond,
            sizes=Q.agentSize,elementMask=RING,elementTex=None)
        t0=core.getTime()
        for f in range(nrframes):
            
            elem.setXYs(trajectories[f,:,[X,Y]].transpose())
            elem.draw()
            
            wind.update()
            wind.getMovieFrame(buffer='front')
            #wind.clearBuffer()
            for key in event.getKeys():
                if key in ['escape']:
                    wind.close()
                    break;
                    #core.quit()
            #print f
            if f in breaks:
                wind.saveMovieFrames('%s%d.mpeg'%(outname,f/250-1),mpgCodec='mpeg1video')
        s='copy/b '
        for i in range(len(breaks)): s+= '%s%d.mpeg+'% (outname,i)
        s=s[:-1]
        s+= ' %s.mpeg'%outname
        #print s
        call(s,shell=True)
        for i in range(len(breaks)): call('DEL %s%d.mpeg'%(outname,i),shell=True)
        #print core.getTime()-t0
        wind.close()
    except: 
        wind.close()
        raise
    
def exportTremoulet(wind=None):
    ''' creates Tremoulet & Feldman (2000) experiment and
        exports each trial as mpeg'''
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
##################################################    
# functionality for viewing trials
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
            elementMask=RING,elementTex=None)
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
        shows the trial given by trajectories
        helpful for viewing trials
        trajectories - FxNx2 ndarray with coordinates of N agents
            for each of the F frames
        maze & wind - provide window and maze for drawing
        highlightChase - if True chaser and chasee are highlighted in color
        origRefresh - frame rate of the trajectory data
        gazeData & gazeDataRefresh - display gaze data (eyetracking)
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
            elementMask=RING,elementTex=None)
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
