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

import numpy as np
import pylab as plt
from psychopy import visual,core, event
from psychopy.misc import pix2deg, deg2pix
from psychopy.event import xydist
from Constants import *

###################################
# basic geometry routines
def circleLine(p0,lin1,lin2,r):
    """ compute circle line intersetion
        p0 - circle center
        lin1, lin2 - line edges
        r - circle radius
    """
    # translate first
    p2=lin2-p0; p1=lin1-p0
    dpos=p2-p1
    dr=dpos[X]**2+dpos[Y]**2
    det=p1[X]*p2[Y]-p1[Y]*p2[X]
    discriminant= dr*r**2-det**2
    if discriminant<0:
        raise NameError('No intersection')
    temp1=(2*bool(dpos[Y]>=0)-1)*dpos[X]*discriminant**0.5
    temp2=abs(dpos[Y])*discriminant**0.5
    inter=(((det*dpos[Y]+temp1)/dr,
        (-det*dpos[X]+temp2)/dr),
        ((det*dpos[Y]-temp1)/dr,
        (-det*dpos[X]-temp2)/dr))
    # output point which is nearer to p1
    i= (xydist(p1,inter[0])+xydist(p2,inter[0]) >
        xydist(p1,inter[1])+xydist(p2,inter[1]))
    # check whether inter lies between line edges
    cw=xydist(p1,p2)
    if xydist(p1,inter[i])>=cw or xydist(p2,inter[i])>=cw:
        raise NameError('Intersection out of bounds')
    return np.array(inter[i])+p0

def lineLine(x1,y1,x2,y2,x3,y3,x4,y4):
    """ compute intersection of two lines
        points (x1,y1) and (x2,y2) belong to line 1 and
        points (x3,y3) and (x4,y4) to line 2
    """
    d=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if d==0:
        raise NameError('Lines are parallel')
    x=((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/d
    y=((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/d
    return np.array((x,y))
###############################################

class Maze(np.ndarray):
    ''' handles environment and agent's collisions with boundaries
        in particular
    '''
    H=0; V=1;I=0;J=1 # map indices of horizontal and vertical 
    def __new__(cls,sz,dispSize,pos=(0,0),lw2cwRatio=0.1):
        inst= super(Maze, cls).__new__(cls,(sz[0],sz[1],2),np.bool)
        #inst.dispSize=dispSize
        return inst
    def __init__(self,sz=(0,0),dispSize=(18,18),lw2cwRatio=0.1,pos=(0,0)):
        """
            sz - size of the grid, tuple with two values
            dispSize - total size of the rectangular area,  tuple with two values in degrees
            lw2cwRatio - ratio of the line width to the cell width
            pos - position of the maze on the screen,  tuple with two values in degrees
        """
        self.pos=np.array(pos)
        #mon=visual.monitors.Monitor(Q.monname)
        self.dispSize=np.array(dispSize)
        self.cw=self.dispSize/(np.float32(self.shape)[[0,1]])
        self.lw=self.cw[0]*lw2cwRatio
        #self.edgeType=edgeType #Edge type may be 'flat' 0 or 'circle' 1
        offset=self.dispSize/2.0
        self.lineXY=[]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                posS = (i*self.cw[0]-offset[0],
                    j*self.cw[1]-offset[1])
                if self[i,j,Maze.V] and i>0:
                    posE= (i*self.cw[0]-offset[0],
                        (j+1)*self.cw[1]-offset[1])
                    self.lineXY.append((posS,posE))
                if self[i,j,Maze.H] and j>0:
                    posE= ((i+1)*self.cw[0]-offset[0],
                           j*self.cw[1]-offset[1])
                    self.lineXY.append((posS,posE))

    def draw(self, wind):
        ''' draws the maze to window wind'''
        try:
            self.maze.draw()
            return
        except AttributeError:
            'do nothing'
        elem=[]

        temp=np.array(self.lineXY)
        edges=visual.ElementArrayStim(wind,fieldShape='sqr',
            nElements=len(self.lineXY)*2,sizes=self.lw,units='deg',
            elementMask='circle',elementTex=None,
            xys=np.reshape(temp,(temp.shape[0]*2,2)))
        elem.append(edges)
        # todo try elem array instead of line
        for line in self.lineXY:
            #print line
            body=visual.Line(wind,line[0],line[1],
                lineWidth=deg2pix(self.lw,wind.monitor))
            elem.append(body)
        frame=visual.Rect(wind,width=self.dispSize[0],
            height=self.dispSize[1],lineColor=(1,1,1),pos=(0,0),
            lineWidth=deg2pix(self.lw,wind.monitor),units='deg')
        if len(self.lineXY)==0:
            self.maze=frame
        else:
            elem.append(frame)
            self.maze=visual.BufferImageStim(wind,stim=elem)
        #self.maze.draw()
        
    def shortestDistanceFromWall(self, point):
        '''returns the shortest distance of the point to
            a maze wall in degrees'''
        from psychopy.event import xydist
        point-=self.pos
        X=Maze.I; Y = Maze.J
        (gx,gy)=self.dispSize/2.0
        arena=(((-gx,-gy),(gx,-gy)), ((gx,-gy),(gx,gy)),
            ((-gx,gy),(gx,gy)),((-gx,gy),(-gx,-gy)))
        # check outside walls
        dists=(abs(point[Y]+gy), abs(-point[X]+gx),
            abs(-point[Y]+gy),abs(point[X]+gx))
        shd = min(dists)
        edge=arena[dists.index(shd)]
        # check the lines
        for line in self.lineXY:
            dim=int(line[1][X]==line[0][X])
            # 1 is vertical
            mx=int(line[0][dim]<line[1][dim])
            mn=1-mx
            if point[dim]>line[mx][dim]:
                d=xydist(point,line[mx])
                ol=line[mx]
            elif point[dim]<line[mn][dim]:
                d=xydist(point,line[mn])
                ol=line[mn]
            else:
                d=abs(point[1-dim]-line[0][1-dim])
                ol=line
            if d<shd:
                shd=d; edge=ol
        #print shd,point, edge
        return shd-self.lw/2.0, edge

    def bounceOff(self,posT1,posT0,edge,agentRadius):
        ''' returns new position and direction based on
            two positions (posT1 and posT0) that intersect a
            boundary (edge)
        '''
        posT1-= self.pos; posT0-=self.pos
        if type(edge[0])==type(0.1):# edge is a point
            #print 'bounce ',posT1, posT0,edge, agentRadius
            # first translate the agent center
            phi=np.arctan2(edge[Y]-posT0[Y],edge[X]-posT0[X])
            posT0[X]+= np.cos(phi)*agentRadius
            posT0[Y]+= np.sin(phi)*agentRadius
            phi=np.arctan2(edge[Y]-posT1[Y],edge[X]-posT1[X])
            posT1[X]+= np.cos(phi)*agentRadius
            posT1[Y]+= np.sin(phi)*agentRadius 
            oldPhi=np.arctan2(posT1[Y]-posT0[Y],posT1[X]-posT0[X])
            inters=circleLine(edge,posT0,posT1,self.lw/2.0)
            # compute new angle
            normPhi=np.arctan2(inters[Y]-edge[Y],
                            inters[X]-edge[X])
            newPhi=2*normPhi-np.pi-oldPhi
            # compute new position
            # translate
            newPos=posT1-inters
            # rotate
            R=np.array([[np.cos(newPhi-oldPhi),
                -np.sin(newPhi-oldPhi)],
                [np.sin(newPhi-oldPhi),
                 np.cos(newPhi-oldPhi)]])
            newPos = R.dot(newPos)
            # translate back
            newPos=newPos+inters
            phi=np.arctan2(edge[Y]-newPos[Y],edge[X]-newPos[X])
            newPos[X]+= np.cos(phi)*-agentRadius
            newPos[Y]+= np.sin(phi)*-agentRadius
        else: # edge is a line
            oldPhi=np.arctan2(posT1[Y]-posT0[Y],posT1[X]-posT0[X])
            if edge[0][X]==edge[1][X]: #vertical
                if edge[0][X]>posT0[X]:
                    offset=-self.lw/2.0
                    agoff=agentRadius
                else:
                    offset=self.lw/2.0
                    agoff=-agentRadius
                posT1[X]+=agoff; posT0[X]+=agoff
                inters=lineLine(posT1[X],posT1[Y],
                    posT0[X],posT0[Y],edge[0][X]+offset,
                    edge[0][Y],edge[1][X]+offset,edge[1][Y])
                newPos=np.copy(posT1)
                newPos[X]=2*inters[X]-posT1[X]-agoff
                newPhi=np.pi-oldPhi
            else: #horizontal
                if edge[0][Y]>posT0[Y]:
                    offset=-self.lw/2.0
                    agoff=agentRadius
                else:
                    offset=self.lw/2.0
                    agoff=-agentRadius
                posT1[Y]+=agoff; posT0[Y]+=agoff
                inters=lineLine(posT1[X],posT1[Y],posT0[X],
                    posT0[Y],edge[0][X],edge[0][Y]+offset,
                    edge[1][X],edge[1][Y]+offset)
                newPos=np.copy(posT1)
                newPos[Y]=2*inters[Y]-posT1[Y]-agoff
                newPhi=2*np.pi-oldPhi
        newPos+= self.pos
        return (newPos,(180*newPhi/np.pi)%360) 
##############################################################
# Library of some simple mazes
class GridMaze(Maze):
    def __new__(cls,sz=(10,10)):
        return super(GridMaze, cls).__new__(cls,sz)
    def __init__(self):
        self.sz=self.shape[0]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self[i,j,:]= np.array((True,True))

class EmptyMaze(Maze):
    def __new__(cls,*args,**kwargs):
        return super(EmptyMaze, cls).__new__(cls,*args,**kwargs)
    def __init__(self,*args,**kwargs):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self[i,j,:]= np.array((False,False))
        super(EmptyMaze, self).__init__(*args,**kwargs)
                
class TestMaze(Maze):
    def __new__(cls,*args,**kwargs):
        return super(TestMaze, cls).__new__(cls,*args,**kwargs)
    def __init__(self,*args,**kwargs):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i==3:
                    self[i,j,Maze.H]=True
                else:
                    self[i,j,Maze.H]=False
                self[i,j,Maze.V]=False
        super(TestMaze, self).__init__(*args,**kwargs)
class ZbrickMaze(Maze):
    def __new__(cls,*args,**kwargs):
        return super(ZbrickMaze, cls).__new__(cls,*args,**kwargs)
    def __init__(self,*args,**kwargs):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self[i,j,Maze.V]= (i%2==0 and j%2==0 or
                    i%4==3 and j%4==3 or i%4==1 and j%4==1)
                self[i,j,Maze.H]= (i%2==1 and j%2==1 or
                    i%4==0 and j%4==0 or i%4==2 and j%4==2)
        super(ZbrickMaze, self).__init__(*args,**kwargs)
                   
if __name__ == '__main__':
    # last test: PsychoPy 1.82.01-py2.7
    # use this script to display a maze
    from Settings import eizo
    wind=visual.Window(fullscr=False,size=(900,900),units='deg',
            color=[-1,-1,-1],winType='pyglet',monitor=eizo)
    try:
        np.random.seed()
        plt.close()       
        m = ZbrickMaze(sz=(10,10),dispSize=(18,12))
        #m=EmptyMaze((1,1),dispSize=(18,18))
        #bla
        #m.draw(wind)
        wind.flip()
        stop=False
        while not stop:
            for key in event.getKeys():
                if key in ['escape']:
                    stop=True
            m.draw(wind)
            wind.flip()
        wind.close()
    except:
        wind.close()
        raise
