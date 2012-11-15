
# -*- coding: utf-8 -*-
from Settings import Q
from Constants import *
from psychopy import visual, core, event,gui,sound
from psychopy.misc import pix2deg, deg2pix
import time, sys
from pickle import dump
import numpy as np

import random
try: from Eyelink import TrackerEyeLink
except ImportError: print 'Warning >> Eyelink import failed'
from Tobii import TobiiController,TobiiControllerFromOutput,  TobiiControllerFromOutputPaced

class Experiment():
    def __init__(self):
        # ask infos
        myDlg = gui.Dlg(title="Experiment zur Bewegungswahrnehmung",pos=Q.guiPos)   
        myDlg.addText('VP Infos')   
        myDlg.addField('Subject ID:',0)
        myDlg.addField('Block:',0)
        #myDlg.addField('Scale (1 or 0.6):',1)
        myDlg.addField('Alter:', 21)
        myDlg.addField('Geschlecht (m/w):',choices=(u'weiblich',u'maennlich'))
        myDlg.addField(u'Händigkeit:',choices=('rechts','links'))
        myDlg.addField(u'Dominantes Auge:',choices=('rechts','links'))
        myDlg.addField(u'Sehschärfe: ',choices=('korrigiert','normal'))
        myDlg.addField(u'Wochenstunden vor dem Komputerbildschirm:', choices=('0','0-2','2-5','5-10','10-20','20-40','40+'))
        myDlg.addField(u'Wochenstunden Komputerspielen:', choices=('0','0-2','2-5','5-9','10-20','20+'))
        myDlg.addField('Starte bei Trial:', 0)
        myDlg.show()#show dialog and wait for OK or Cancel
        vpInfo = myDlg.data
        self.id=vpInfo[0]
        self.block=vpInfo[1]
        self.initTrial=vpInfo[-1]
        self.scale=1#vpInfo[2]
        if myDlg.OK:#then the user pressed OK
            subinf = open(Q.outputPath+'vpinfo.res','a')
            subinf.write('%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%d\n'% tuple(vpInfo))
            subinf.close()               
        else: print 'Experiment cancelled'
        # save settings, which we will use
        f=open('Settings.pkl','w')
        dump(Q,f)
        f.close()
        #init stuff
        self.wind=Q.initDisplay()
        self.mouse = event.Mouse(False,None,self.wind)
        self.mouse.setVisible(False)
        fcw=0.1; fch=0.8 #fixcross width and height
        fclist=[ visual.ShapeStim(win=self.wind, pos=[0,0],fillColor='white',
            vertices=((fcw,fch),(-fcw,fch),(-fcw,-fch),(fcw,-fch))),
            visual.ShapeStim(win=self.wind, pos=[0,0],fillColor='white',
            vertices=((fch,fcw),(-fch,fcw),(-fch,-fcw),(fch,-fcw))),
            visual.Circle(win=self.wind, pos=[0,0],fillColor='black',radius=0.1)]
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
        
    def getWind(self):
        try: return self.wind
        except AttributeError: 
            self.wind=Q.initDisplay()
            return self.wind

    def getJudgment(self,giveFeedback=False):
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
    def trialIsFinished(self):
        return False
    def omission(self):
        pass

    def runTrial(self,trajectories,fixCross=True):
        nrframes=trajectories.shape[0]
        self.cond=trajectories.shape[1]
        self.elem=visual.ElementArrayStim(self.wind,fieldShape='sqr',
            nElements=self.cond, sizes=Q.agentSize*self.scale,
            elementMask=RING,elementTex=None,colors=Q.agentCLR)
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
        while self.f<nrframes:
            self.pos=trajectories[self.f,:,[X,Y]].transpose()*self.scale
            self.elem.setXYs(self.pos)
            self.elem.draw()
            self.wind.flip()
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
        
    def run(self,mouse=None,prefix=''):        
        self.output = open(Q.outputPath+prefix+'vp%03d.res'%self.id,'a')
        
        permut=np.load(Q.inputPath+'vp%03d'%self.id+Q.delim
            +prefix+'ordervp%03db%d.npy'%(self.id,self.block))
        if len(permut.shape)>1 and permut.shape[1]>1:
            self.data=permut[:,1:]
            permut=permut[:,0]
        self.nrtrials=permut.size
        # show initial screen until key pressed
#        self.text2.setText(u'Bereit ?')
#        self.text2.draw()
#        self.wind.flip()
#        self.mouse.clickReset()
#        mkey = self.mouse.getPressed(False)
#        while not (mkey[0]>0 or mkey[1]>0 or mkey[2]>0):
#            mkey = self.mouse.getPressed(False)
        # loop trials
        for trial in range(self.initTrial,self.nrtrials):   
            self.t=trial; self.pt=permut[trial]
            #print self.t
            self.output.write('%d\t%d\t%d\t%s'% (self.id,self.block,trial,int(permut[trial])))
            fname=prefix+'vp%03db%dtrial%03d.npy' % (self.id,self.block,permut[trial])
            print 'showing',fname
            self.trajectories= np.load(Q.inputPath+'vp%03d'%self.id+Q.delim+fname)
            self.runTrial(self.trajectories)
            
        
class Gao09Experiment(Experiment):
    def trialIsFinished(self):
        return False
    def omission(self):
        self.text3.setText('Wurde Verfolgung gezeigt?')
        t0=core.getTime();selected=[]
        self.mouse.clickReset()
        mkey=self.mouse.getPressed()
        while sum(mkey)==0:
            self.elem.draw()
            self.text3.draw()
            self.wind.flip()
            mkey=self.mouse.getPressed()
        if mkey[0]>0:
            self.output.write('\t%d\t%2.4f\t1'%(self.data[self.pt,0],core.getTime()-t0))
            if self.data[self.pt,0]>=6: self.sFalse.play()
            resp=self.getJudgment()
            res= self.data[self.pt,0]<6 and resp==1
        else: 
            self.output.write('\t%d\t%2.4f\t0\t-1\t-1\t-1\t-1'%(self.data[self.pt,0],core.getTime()-t0))
            res= self.data[self.pt,0]>=6
            if self.data[self.pt,0]<6: self.sFalse.play()
        if not res: self.output.write('\t0')
        else: self.output.write('\t1')
        
    def run(self):
        sFalse=sound.SoundPygame()
        Experiment.run(self,prefix='gao09e1')
        
class BABY():
    NOTLOOKING=0
class BabyExperiment(Experiment):
    colored=False
    criterion=6 # abort trial after infant is continuously not looking for more than criterion sec. 
    fixRadius=2 # threshold in deg
    sacToReward=3 # reward is presented after enough saccades are made
    maxSacPerPursuit=10 # trial will be terminated if the pursuit is upheld
    blinkTolerance=5 # iterations
    maxRewards=10# todo
    rewardIterations=100
    maxDuration = 5 # minutes
    
    
    def __init__(self):
        Experiment.__init__(self)
        self.etController = TobiiControllerFromOutputPaced(self.getWind(),sid=self.id,block=self.block)
        self.nrRewards=0
        self.etController.doMain()
        self.clrOscil=0.05
        self.rewardColor1=np.array((0,-1,1))
        self.rewardColor2=np.array((1,1,1))
        
    def run(self):
        if BabyExperiment.colored:
            c=[(-1,-1,0),(-1,0,-1),(0,-1,-1),(1,1,0),(1,0,1),(0,1,1)] #c=['blue','red','green','black','white']
            c=np.array(2*c); self.colors= c[np.random.permutation(len(c))]
        Experiment.run(self)
        self.etController.closeConnection()
        
    def runTrial(self,*args):
        self.babyStatus=0 # -1 no signal, 0 saccade, 1 fixation,
        self.sacPerPursuit=0
        self.pursuedAgents=None
        self.rewardIter=0
        self.blinkCount=0
        if BabyExperiment.colored:
            #print self.t
            #print self.colors[self.t,:]
            Q.agentCLR=self.colors[self.t,:]
            self.etController.sendMessage('Color\t%.2f %.2f %.2f'%(Q.agentCLR[0],Q.agentCLR[1],Q.agentCLR[2]))
        #colors=((1,0,-1),(1,-1,0),(0,-1,1),(0,1,-1),(-1,0,1),(-1,1,0))
        #self.rewardColor=np.array(colors[np.random.randint(len(colors))])
        self.timeNotLooking=0
        self.etController.preTrial(driftCorrection=True)
        self.etController.sendMessage('Trial\t%d'%self.t)
        Experiment.runTrial(self,*args,fixCross=False)
        self.etController.postTrial()
        
    def trialIsFinished(self):
        gc,fc,f,incf=self.etController.getCurrentFixation(units='deg')
        self.f+=incf
        if np.isnan(gc[0]): self.babyStatus=-1; self.blinkCount+=1
        elif self.babyStatus==-1: 
            self.babyStatus=0
            self.blinkCount=0
        
        temp=self.trajectories[max(self.f-8,0),:,:]
        #print np.array([fc.tolist()]*self.cond).shape,temp[:,:2].shape
        dxy=np.array([fc.tolist()]*self.cond)-temp[:,:2]#self.elem.xys
        distance= np.sqrt(np.power(dxy[:,0],2)+np.power(dxy[:,1],2))
        agentsInView=BabyExperiment.fixRadius>distance
        #print 'show ',f, self.babyStatus
        if not f and self.babyStatus == 1: # saccade initiated
            self.babyStatus=0
        elif f and self.babyStatus == 0: # fixation started
            self.babyStatus=1
            if self.pursuedAgents is None or not self.pursuedAgents: # first fixation
                self.pursuedAgents=agentsInView[0] or agentsInView[1] 
                if self.pursuedAgents:  self.etController.sendMessage('1 st saccade \t'+str(agentsInView[0])+'\t'+str(agentsInView[1]))
            else: # consecutive fixation
                self.pursuedAgents= (agentsInView[0] or agentsInView[1]) and self.pursuedAgents
                if self.pursuedAgents:  self.etController.sendMessage('%d th saccade \t' % (self.sacPerPursuit+1)+str(agentsInView[0])+'\t'+str(agentsInView[1]))
                #print 'second fixation to',agentsInView
            if self.pursuedAgents: #is chaser-chasee being pursued?
                self.sacPerPursuit+=1
                #print 'stage ',self.sacPerPursuit,' pursued:' ,self.pursuedAgents.nonzero()[0]
            else: self.sacPerPursuit=0
                
        if self.sacPerPursuit>= BabyExperiment.sacToReward and self.blinkCount<BabyExperiment.blinkTolerance: 
            if self.rewardIter!=2:
                clrs=np.ones((self.cond,3))
                clrs[0,:]=self.rewardColor1
                clrs[1,:]=self.rewardColor1
                self.elem.setColors(clrs)
                self.etController.sendMessage('Reward On')
            self.rewardIter=1
            
        #print self.rewardIter
        if self.rewardIter>0 and self.rewardIter<BabyExperiment.rewardIterations:
            self.reward()#self.pursuedAgents.nonzero()[0])
            self.rewardIter+=1
        elif self.rewardIter>=BabyExperiment.rewardIterations:
            self.etController.sendMessage('Reward Off')
            self.rewardIter=0
            self.elem.setColors(np.ones((self.cond,3)))
        
        if self.babyStatus is -1: self.timeNotLooking+=1
        else: self.timeNotLooking=0
        out=(self.timeNotLooking>BabyExperiment.criterion*Q.refreshRate 
            or self.sacPerPursuit>=BabyExperiment.maxSacPerPursuit)
        #print gc,f,self.babyStatus,self.timeNotLooking,out
        return out
        
    def omission(self):
        self.etController.sendMessage('Omission')
    def reward(self):
        ''' Present reward '''
        clrs=np.ones((self.cond,3))
        if self.elem.colors.shape[0]!=self.cond:
            self.elem.setColors(clrs)

        nc=self.elem.colors[0,:]-self.clrOscil*(self.rewardColor1-self.rewardColor2)
        #print '1 ',nc
        if (np.linalg.norm(self.rewardColor1-nc)>np.linalg.norm(self.rewardColor1-self.rewardColor2)
            or np.linalg.norm(self.rewardColor2-nc)>np.linalg.norm(self.rewardColor2-self.rewardColor1)): 
                self.clrOscil *= -1
                nc=self.elem.colors[0,:]-self.clrOscil*(self.rewardColor1-self.rewardColor2)
        #print '2 ',nc
        clrs[0,:]=nc
        clrs[1,:]=nc
        self.elem.setColors(clrs)
        #print nc,pa, self.elem.colors[pa,:]
    
class BehavioralExperiment(Experiment):
            
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
#        # display query
#        self.mouse.clickReset()
#        self.fixcross.draw()
#        #self.text1.setText(u'Bereit ?')
#        #self.text1.draw()
#        self.wind.flip()
#        while True:
#            mkey=self.mouse.getPressed()
#            if sum(mkey)>0: break
#            core.wait(0.01)
        #self.eyeTracker.sendMessage('Drift Correction')
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
        
class TobiiExperiment(BehavioralExperiment):
    def __init__(self,doSetup=True):
        BehavioralExperiment.__init__(self)
        self.eyeTracker = TobiiControllerFromOutputPaced(self.getWind(),sj=self.id,block=self.block,doSetup=doSetup)
        self.eyeTracker.sendMessage('Monitor Distance\t%f'% self.wind.monitor.getDistance())
    def run(self):
        BehavioralExperiment.run(self)
        self.eyeTracker.closeConnection()
    def runTrial(self,*args):
        self.eyeTracker.preTrial(False)#self.t,False,self.getWind(),autoDrift=True)
        self.eyeTracker.sendMessage('Trial\t%d'%self.t)
        BehavioralExperiment.runTrial(self,*args,fixCross=True)
        self.eyeTracker.postTrial()
    def getJudgment(self,*args):
        self.eyeTracker.sendMessage('Detection')
        resp=BehavioralExperiment.getJudgment(self,*args)
        return resp 
    def omission(self):
        self.eyeTracker.sendMessage('Omission')
        pass
    

if __name__ == '__main__':
    from Settings import Q
    #E=BehavioralExperiment()
    #E=TobiiExperiment()
    #E=Gao09Experiment()
    E=BabyExperiment()
    E.run()


