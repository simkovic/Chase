import os, struct,time
import numpy as np
import pylab as plt
from Tkinter import *
import tkFileDialog
from scipy.interpolate import interp2d

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



    

class PfmSeries():
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
    
    

class Mraw():
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
        self.vals=self.tonpy()

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

    def tonpy(self):
        out=np.zeros((5000,self.rows,self.cols),dtype=np.uint8)
        i=0
        m = self.nextFrame()
        while True:
            if len(m.shape)!=2: break
            else: out[i,:,:]=m
            i+=1
            m = self.nextFrame()
        return out[:i,:,:]
    def computeSaliency(self,pos,twind,rdb,bins=range(-20,21),inpRes=1024):
        """ self - Mraw object, make sure the frame buffer is at the start
            pos - in (deg,deg)
            radius - radius of the circle within which the saliency at each
                frame is computed, in deg
            twindow - time window in ms
        """
        from Settings import eyelinklab, Settings
        Q=Settings(**eyelinklab)
        assert self.rows==self.cols

        if False:# this is precise but too slow
            oldbns=np.linspace(inpRes/self.rows/2,inpRes-inpRes/self.rows/2,self.rows)
            oldbns= Q.pix2deg(oldbns-inpRes/2)
            newbns=np.array(bins)
            grid = np.zeros((twind[1]-twind[0]+1,newbns.size,newbns.size))
            for f in range(twind[0],twind[1]+1):
                func= interp2d(oldbns,oldbns,self.vals[f,:,:],fill_value=0)
                grid[f-twind[0],:,:]=func(newbns+pos[X],newbns+pos[Y])
            return grid
        # less precise but fast, discards the requested BINS
        grid=np.zeros((twind[1]-twind[0],self.rows,self.cols))
        pos=np.array((Q.deg2pix(pos[X]),Q.deg2pix(pos[Y])),dtype=np.float32)
        pos+=inpRes/2
        pos/= (inpRes/self.rows)
        posi=np.int32(np.round(pos))
        xs=posi[X]-self.cols/2; xe=posi[X]+self.cols/2;
        ys=posi[Y]-self.rows/2; ye=posi[Y]+self.rows/2;
        #print xs,xe,ys,ye
        xxs=max(xs,0)-xs; xxe=self.cols-(xe-min(xe,self.cols))
        yys=max(ys,0)-ys; yye=self.rows-(ye-min(ye,self.rows))
        #print xxs,xxe,yys,yye
        grid[:,xxs:xxe,yys:yye]=self.vals[twind[0]:twind[1],max(xs,0):xe,max(ys,0):ye]

        # compute radial saliency
        masks=[]
        temp=[]
        for r in rdb: temp.append(Q.deg2pix(r)/(inpRes/self.rows))
        rdb=temp
        for k in range(len(rdb)):
            masks.append(np.zeros((1,self.rows,self.cols),dtype=np.int32))
            for x in range(self.rows):
                for y in range(self.cols):
                    masks[k][0,x,y]= (k==0 and (x-pos[X])**2+(y-pos[Y])**2<rdb[k]**2 or
                        k>0 and (x-pos[X])**2+(y-pos[Y])**2<rdb[k]**2 and
                        (x-pos[X])**2+(y-pos[Y])**2>=rdb[k-1]**2)  
        rad=[]
        for m in masks:
            if m.sum()==0: 'warning: mask.sum()==0'
            rad.append((self.vals[twind[0]:twind[1],:,:]*
                np.repeat(m,grid.shape[0],axis=0)).sum(2).sum(1)/float(m.sum())/self.norm)
        return grid, np.array(rad)
        

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
            plt.show()
        


if __name__ == '__main__':
    #showEzvisionOutput()
    fpath='saliency/output/vp001/vp001b10trial002COmotion-.64x64.mgrey'
    vid=Mraw(fpath)
    res, rad=vid.computeSaliency([ 5.79374473,  4.79934528],[8,76],rdb=np.arange(1,15))
    



    


