# -*- coding: utf-8 -*-
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

from Settings import Q
from Constants import *
from psychopy import visual, core, event,gui,sound,parallel
from psychopy.misc import pix2deg, deg2pix
import time, sys,os
import numpy as np
from Trajectory import HeatSeekingChaser

import random
try: from Eyelink import TrackerEyeLink
except ImportError: print 'Warning >> Eyelink import failed'
try: from SMI import TrackerSMI
except ImportError: print 'Warning >> SMI import failed'
from eyetracking.Tobii import TobiiController,TobiiControllerFromOutput

class Experiment():
    ''' Base experiment class'''
    def __init__(self,vp=None):
        ''' inits variables and presents the intro dialog
            vp - subject id, useful for  replay functionality'''
        # ask infos
        myDlg = gui.Dlg(title="Experiment zur Bewegungswahrnehmung",pos=Q.guiPos)   
        myDlg.addText('VP Infos')   
        myDlg.addField('Subject ID:',201)# subject id
        myDlg.addField('Block:',0) # block id
        myDlg.addField('Alter:', 21) # age
        myDlg.addField('Geschlecht (m/w):',choices=(u'weiblich',u'maennlich')) #gender
        myDlg.addField(u'Händigkeit:',choices=('rechts','links'))# handedness
        myDlg.addField(u'Dominantes Auge:',choices=('rechts','links'))# dominant eye
        myDlg.addField(u'Sehschärfe: ',choices=('korrigiert','normal')) # visual acuity
        # weekly hours spent on computer screen 
        myDlg.addField(u'Wochenstunden vor dem Komputerbildschirm:', choices=('0','0-2','2-5','5-10','10-20','20-40','40+'))
        # weekly hours spent playing video games
        myDlg.addField(u'Wochenstunden Komputerspielen:', choices=('0','0-2','2-5','5-9','10-20','20+'))
        myDlg.addField('Starte bei Trial:', 0) # start trial id, for debug only
        if vp is None:
            myDlg.show()#show dialog and wait for OK or Cancel
            vpInfo = myDlg.data
        else: vpInfo=[vp,0,21,'','','','','','',0]
        self.id=vpInfo[0]
        self.block=vpInfo[1]
        self.initTrial=vpInfo[-1]
        self.scale=1#vpInfo[2]
        try:#then the user pressed OK
            subinf = open(Q.outputPath+'vpinfo.res','a')
            subinf.write('%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%d\n'% tuple(vpInfo))
            subinf.close()               
        except: print 'Experiment cancelled'
        # save settings, which we will use
        Q.save(Q.inputPath+'vp%03d'%self.id+Q.delim+'SettingsExp.pkl')
        #init stuff
        self.wind=Q.initDisplay()
        self.mouse = event.Mouse(False,None,self.wind)
        self.mouse.setVisible(False)
        fcw=0.1; fch=0.8 #fixcross width and height
        fclist=[ visual.ShapeStim(win=self.wind, pos=[0,0],fillColor='white',
            vertices=((fcw,fch),(-fcw,fch),(-fcw,-fch),(fcw,-fch)),interpolate=False),
            visual.ShapeStim(win=self.wind, pos=[0,0],fillColor='white',
            vertices=((fch,fcw),(-fch,fcw),(-fch,-fcw),(fch,-fcw)),interpolate=False),
            visual.Circle(win=self.wind, pos=[0,0],fillColor='black',radius=0.1,interpolate=False)]
        self.fixcross=visual.BufferImageStim(self.wind,stim=fclist)
        self.wind.flip(); self.wind.flip()
        self.score=0
        self.rt=0
        # init text
        fs=1 # font size
        self.text1=visual.TextStim(self.wind,text='Error',wrapWidth=30,pos=[0,2])
        self.text2=visual.TextStim(self.wind,text='Error',wrapWidth=30,pos=[0,0])
        self.text3=visual.TextStim(self.wind, text='Error',wrapWidth=30,pos=[0,-10])
        self.text1.setHeight(fs)
        self.text2.setHeight(fs)
        self.text3.setHeight(fs)
        self.f=0
        self.permut=np.load(Q.inputPath+'vp%03d'%self.id+Q.delim
            +'ordervp%03db%d.npy'%(self.id,self.block))
        if len(self.permut.shape)>1 and self.permut.shape[1]>1:
            self.data=self.permut[:,1:]
            self.permut=self.permut[:,0]
        self.nrtrials=self.permut.size
        
    def getWind(self):
        '''returns the window handle '''
        try: return self.wind
        except AttributeError: 
            self.wind=Q.initDisplay()
            return self.wind

    def getJudgment(self,giveFeedback=False):
        '''asks subject to select chaser chasee'''
        position=np.transpose(self.pos)
        cond=position.shape[1]
        self.mouse.clickReset();self.mouse.setVisible(1)
        elem=self.elem
        t0=core.getTime();selected=[]
        mkey=self.mouse.getPressed()
        lastPress=t0
        while sum(mkey)>0:
            elem.draw()
            self.wind.flip()
            mkey=self.mouse.getPressed()
        released=True
        clrs=np.ones((cond,1))*Q.agentCLR
        while len(selected) <2:
            elem.draw()
            self.wind.flip()
            mpos=self.mouse.getPos()
            mkey=self.mouse.getPressed()
            mtime=core.getTime()
            for a in range(cond):
                if (event.xydist(mpos,np.squeeze(position[:,a]))
                    < Q.agentRadius*self.scale):
                    if 0<sum(mkey) and released: # button pressed
                        if selected.count(a)==0: # avoid selecting twice
                            clrs[a]=Q.selectedCLR
                            elem.setColors(clrs,'rgb')
                            selected.append(a)
                            self.output.write('\t%d\t%2.4f' % (a,mtime-t0))
                        released=False
                    elif a in selected: # no button pressed but selected already
                        clrs[a]=Q.selectedCLR
                        elem.setColors(clrs,'rgb')
                    else: # no button pressed but mouse cursor over agent
                        clrs[a]=Q.mouseoverCLR
                        elem.setColors(clrs,'rgb')
                elif a in selected: # no button pressed, no cursor over agent, but already selected
                    clrs[a]=Q.selectedCLR
                    elem.setColors(clrs,'rgb')
                else: # no button press, no cursor over agent, not selected
                    clrs[a]=Q.agentCLR
                    elem.setColors(clrs,'rgb')
            if 0==sum(mkey) and not released:
                released=True       
        t0=core.getTime()
        while core.getTime()-t0<1:
            elem.draw()
            self.wind.flip()
        self.mouse.setVisible(0)
        if (selected[0]==0 and selected[1]==1
            or selected[0]==1 and selected[1]==0):
            return 1
        else: return 0
    def trialIsFinished(self): return False
    def omission(self): pass
    def getf(self): return self.f
    def flip(self): 
        self.elem.draw()
        self.wind.flip()
    def runTrial(self,trajectories,fixCross=True):
        ''' runs a trial
            trajectories - NxMx2, where N is number of frames,
                            M is number of agents
            fixCross - if True presents fixation cross before the trial
        '''
        self.nrframes=trajectories.shape[0]
        self.cond=trajectories.shape[1]
        self.elem=visual.ElementArrayStim(self.wind,fieldShape='sqr',
            nElements=self.cond, sizes=Q.agentSize*self.scale,interpolate=True,
            elementMask=MASK,elementTex=None,colors=Q.agentCLR)
        # display fixation cross
        if fixCross:
            self.fixcross.draw()
            self.wind.flip()
            core.wait(1+random.random()/2)
            #core.wait(0.5+random.random()/2)
            #self.eyeTracker.sendMessage('Movement Start')
        self.mouse.clickReset()
        event.clearEvents() #agents[a].setLineColor(agentCLR)
        self.noResponse=True
        self.t0=core.getTime()
        #t0=core.getTime()
        #times=[]
        self.f=0
        while self.f<self.nrframes:
            self.pos=trajectories[self.f,:,[X,Y]].transpose()*self.scale
            self.phi=trajectories[self.f,:,PHI].squeeze()
            self.elem.setXYs(self.pos)
            #t0=core.getTime()
            self.flip()
            #print core.getTime()-t0

            # check for termination signal
            for key in event.getKeys():
                if key in ['escape']:
                    self.wind.close()
                    core.quit()
                    sys.exit()
            if self.trialIsFinished(): break
            self.f+=1
            #times.append(core.getTime()-t0)
            #t0=core.getTime()
        #np.save('times',np.array(times))
        if self.noResponse:
            self.omission() 
        self.wind.flip()
        core.wait(1.0)
        self.output.write('\n')
    def destroy(self):
        try: self.wind.close()
        except: pass
        core.quit()
        
    def run(self,mouse=None,prefix='', initScreen=False):
        ''' mouse & prefix - used for replay functionality,
                for no replay keep at default
        '''
        self.output = open(Q.outputPath+prefix+'vp%03d.res'%self.id,'a')
        if initScreen: # show initial screen until key pressed
            self.text2.setText(u'Bereit ?')
            self.text2.draw()
            self.wind.flip()
            self.mouse.clickReset()
            mkey = self.mouse.getPressed(False)
            while not (mkey[0]>0 or mkey[1]>0 or mkey[2]>0):
                mkey = self.mouse.getPressed(False)
        # loop trials
        for trial in range(self.initTrial,self.nrtrials):   
            self.t=trial; 
            self.output.write('%d\t%d\t%d\t%s'% (self.id,self.block,trial,int(self.permut[trial])))
            # load trajectories
            if self.id>1 and self.id<10:
                fname=prefix+'vp001b%dtrial%03d.npy' % (self.block,self.permut[trial])
                self.trajectories= np.load(Q.inputPath+'vp001'+Q.delim+fname)
            elif self.id>300 and self.id<400:
                fname=prefix+'vp300trial%03d.npy' % self.permut[trial]
                self.trajectories= np.load(Q.inputPath+'vp300'+Q.delim+fname)
            elif self.id>400 and self.id<500:
                fname=prefix+'vp400b0trial%03d.npy' % self.permut[trial]
                self.trajectories= np.load(Q.inputPath+'vp400'+Q.delim+fname)
            else:
                fname=prefix+'vp%03db%dtrial%03d.npy' % (self.id,self.block,self.permut[trial])
                self.trajectories= np.load(Q.inputPath+'vp%03d'%self.id+Q.delim+fname)
            self.runTrial(self.trajectories)
            
class BehavioralExperiment(Experiment):
    ''' wrapper for Experiment class, logs behavioral response
        and display some additional info to the subject'''
    def omission(self):
        self.output.write('\t0\t%d\t30.0\t-1\t-1\t-1\t-1\t0' %self.cond)
        self.rt+=30.0
    def trialIsFinished(self):
        mkey = self.mouse.getPressed(False)
        if mkey[0]>0 or mkey[1]>0 or mkey[2]>0:
            mtime=core.getTime()-self.t0
            self.output.write('\t1\t%d\t%2.4f' %(self.cond,mtime))
            self.rt+=mtime
            resp=self.getJudgment()
            if resp==1: self.score+=1
            self.output.write('\t%d'%resp)
            self.noResponse=False
            return True
        else: return False
    def runTrial(self,trajectories,fixCross=True):
        Experiment.runTrial(self,trajectories,fixCross)
        # display progress info
        if self.t % 10==9:
            self.text1.setText(u'Sie hatten %d Prozent richtig.'% (self.score/10.0*100))
            self.text1.draw()
            self.score=0
            self.text2.setText('(Es geht gleich automatisch weiter.)')
            self.text2.draw()
            #text3.setText('Durchschnittliche Reaktionszeit: %d Sekunden.' % (self.rt/10.0))
            #text3.draw()
            self.rt=0
            self.wind.flip()
            core.wait(4)
        elif False:#permut.size==11+selt.t:
            self.text1.setText('Noch zehn Trials!')
            text1.draw()
            text2.setText('(Es geht gleich automatisch weiter.)')
            text2.draw()
            self.wind.flip()
            core.wait(5)
        if self.t==39: # say goodbye
            self.text1.setText(u'Der Block ist nun zu Ende.')
            self.text1.draw()
            self.text2.setText(u'Bitte, verständigen Sie den Versuchsleiter.')
            self.text2.draw()
            self.wind.flip()
            core.wait(10)
            self.output.close()
            self.wind.close()

    
class EyelinkExperiment(BehavioralExperiment):
    ''' wrapper that implements eyetracking functionality'''
    def __init__(self,doSetup=True):
        BehavioralExperiment.__init__(self)
        self.eyeTracker = TrackerEyeLink(self.getWind(), core.Clock(),sj=self.id,block=self.block,doSetup=doSetup,target=self.fixcross)
        self.eyeTracker.sendMessage('MONITORDISTANCE %f'% self.wind.monitor.getDistance())
    def run(self):
        BehavioralExperiment.run(self)
        self.eyeTracker.closeConnection()
    def runTrial(self,*args):
        self.eyeTracker.preTrial(self.t,False,self.getWind(),autoDrift=True)
        self.eyeTracker.sendMessage('START')
        BehavioralExperiment.runTrial(self,*args,fixCross=False)
        self.eyeTracker.postTrial()
    def getJudgment(self,*args):
        self.eyeTracker.sendMessage('DETECTION')
        resp=BehavioralExperiment.getJudgment(self,*args)
        return resp
    def omission(self):
        self.eyeTracker.sendMessage('OMISSION')
        BehavioralExperiment.omission(self)
    def flip(self):
        self.eyeTracker.sendMessage('FRAME %d %f'%(self.f, core.getTime()))
        return BehavioralExperiment.flip(self)
    
if __name__ == '__main__':
    # to run the experiment
    E=EyelinkExperiment()
    E.run()
