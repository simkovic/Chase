import os, struct,time
import numpy as np
import pylab as plt
from Tkinter import *
import tkFileDialog

X=0;Y=1
def pfm2npy(fpath):
    f=open(fpath)
    f.readline()
    rows, cols = f.readline()[:-1].rsplit(' ')
    rows=int(rows); cols=int(cols)
    f.readline()

    m = [ list(struct.unpack('<%df' % rows, f.read(4*rows)))
        for c in range(cols)]
    #m = np.array(zip(*m))
    m=np.array(m)
    f.close()
    return m

def rawvideo2npy(fpath):
    f=open(fpath)
    fname = fpath.rsplit('/')[-1]
    info= fname.rsplit('.')
    wh=info[-2]
    rows, cols = np.int32(wh.rsplit('x'))
    if info[-1]=='grey':
        m = [ list(struct.unpack('<%dB' % rows, f.read(rows)))
            for c in range(cols)]
        m = np.array(zip(*m))
    f.close()
    return m
def mraw2npy(fpath):
    vid=Mraw(fpath)
    out=np.zeros((5000,vid.rows,vid.cols),dtype=np.uint8)
    m = vid.nextFrame()
    i=0
    while len(m.shape)==2:
        i+=1
        m = vid.nextFrame()
    return out[:i,:,:]

class InputSeries():
    """ A superclass for Movie input files
        contains the computeSaliency method
    """
    def computeSaliency(self,traj,method='fast',radius=0.05):
        """ vid - Mraw object, make sure the frame buffer is at the start
            traj - FramesXAgentsX2 matrix with the agents' coordinates in pixels
            radius - radius of the circle within which the saliency at each
                frame is computed, the value is in percent of the frame height
        """
        radius =(radius*min(self.rows,self.cols))**2
        res=np.zeros((traj.shape[0],traj.shape[1]))
        if method=='fast':      
            # pad the edges so that mask doesn't spill across when shifted
            mask = np.zeros((2*self.rows,2*self.cols))
            for i in range(2*self.rows):
                for j in range(2*self.cols):
                    if ((i-self.rows)**2+(j-self.cols)**2)<radius:
                        mask[i,j]=1
            surface = np.sum(mask)
        else:
            surface = np.pi*radius

        for f in range(traj.shape[0]):
            frame=self.nextFrame()
            if len(frame.shape)!=2:
                print 'Error: no more frames at f= %d\n'% f
                break
            for a in range (traj.shape[1]):      
                if method=='fast':
                    maskshifted=np.roll(mask,
                        int(round(traj[f,a,X]-self.cols/2)),1)
                    maskshifted=np.roll(maskshifted,
                        int(round(traj[f,a,Y]-self.rows/2)),0)
                    maskshifted=maskshifted[(self.rows/2):(3*self.rows/2),
                        (self.cols/2):(3*self.cols/2)]
                    r=np.sum(frame*maskshifted)
                else:
                    r=0
                    for i in range(self.rows):
                        for j in range(self.cols):
                            if ((j-traj[f,a,X])**2+(i-traj[f,a,Y])**2)<radius:
                                r+= frame[i,j]
                res[f,a]=float(r)/surface/self.norm
        return res
    

class PfmSeries(InputSeries):
    def __init__(self,fpath):
        self.fid=open(fpath,'r')
        self.fpath=fpath
        #fname = fpath.rsplit('/')[-1]
        info = fpath.rsplit('.')
        self.name =info[0]
        self.channel=info[1][1:-6]
        self.frame=0 # frame pointer index
        self.data=pfm2npy(self.name+'.-'+
            self.channel+'%06d' % self.frame+'.pfm')
        self.rows, self.cols = self.data.shape
        # we normalize nA with 0.001, that is the unit will be in micA
        self.norm = 1#0.001
        

    def nextFrame(self):   
        m=self.data
        self.frame+=1
        try:
            self.data=pfm2npy(self.name+'.-'+
                self.channel+'%06d' % self.frame+'.pfm')
            return m  
        except IOError:
            #print 'The sequence %s ends at frame nr %d' % (self.name+'-'+self.channel,self.frame)
            self.data=np.array([])
            return m
    
    

class Mraw(InputSeries):
    def __init__(self,fpath):
        self.fid=open(fpath,'r')
        self.fpath=fpath
        fname = fpath.rsplit('/')[-1]
        info = fname.rsplit('.')
        self.form =info[-1]
        self.rows, self.cols = np.int32(info[-2].rsplit('x'))
        self.data=self.fid.read(self.rows)
        # we normalize uint with 255, that is the unit will be in [0,1]
        self.norm = 255.0

    def nextFrame(self):   
        if self.data!='':
            if self.form=='mgrey':
                m=[]
                for c in range(self.cols):
                    m += [list(struct.unpack('<%dB' % self.rows,
                            self.data))]
                    self.data=self.fid.read(self.rows)
                #m = np.array(zip(*m))
                m=np.array(m,dtype=np.uint8)
                return m  
        else:
            self.fid.close()
            return np.array([])
    def close(self):
        self.fid.close()

def showEzvisionOutput():
    X=0;Y=1
    master = Tk()
    master.withdraw() #hiding tkinter window
     
    filePath = tkFileDialog.askopenfilename(title="Open file",
        filetypes=[("pfm file",".pfm"),
                   ("mgrey file",".mgrey"),("All files",".*"),])
    plt.close('all')
    plt.ion()
    showMovie = False
    if filePath != "":
        if filePath[-3:]=='pfm':
            try: # check whether the pfm is part of
                # sequence by checking for its integer suffix
                int(filePath[-10:-4])
                vid= PfmSeries(filePath)
                showMovie=True
            except ValueError: 
                m=pfm2npy(filePath)
                plt.figure()
                plt.imshow(m)        
        elif filePath[-5:]=='mgrey':
            vid= Mraw(filePath)
            showMovie=True
        elif filePath[-4:]=='grey':
            m=rawvideo2npy(filePath)
            plt.figure()
            plt.imshow(m)  
        else:
            print "Unsupported file type" 
    else:
       print "Cancelled"
    if showMovie:
        m = vid.nextFrame()
        # unpack the "tremXX" name from filePath
        fn=filePath.rsplit('/')
        fn=fn[-1].rsplit('.')
        i=0
        tt=0
        while len(m.shape)==2:
            print i
            i+=1
            plt.cla()
            plt.imshow(m,cmap='gray',aspect='equal',vmax=255,vmin=0)
            if i==1: plt.colorbar()
            m = vid.nextFrame()
        


if __name__ == '__main__':
    showEzvisionOutput()
    fpath='saliency/output/vp001b1trial000COmotion-.64x64.mgrey'
    



    


