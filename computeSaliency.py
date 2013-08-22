import os,commands
import numpy as np
import pylab as plt

""" put all data into a single folder. The format of the data name
    should be <name>XX.mpeg.

"""
# Call ezvision and get saliency maps

##path='//home//matus//Desktop//promotion//saliencyControl//tremoulet//'
##files = os.listdir(path)
##files.sort()
##os.chdir(path)
##os.chdir('..')
##for f in files:
##    folder=path.rsplit('//')[-2]+'//'
##    status,output=commands.getstatusoutput('ezvision '+
##        '--in='+folder+f+' --input-frames=0-MAX@60Hz --rescale-input=1600x1600 '+
##        '--out=pfm:output//'+f[:-4]+' --output-frames=@60Hz --rescale-output=400x400 ' +
##        #'--save-channel-outputs --vc-chans=s --sm-type=None '+
##        '--save-channel-outputs --vc-chans=IM --sm-type=None '+
##        '--nodisplay-foa --nodisplay-patch --nodisplay-traj '+
##        '--nodisplay-additive --wta-type=None --nouse-random '+
##        '--vc-type=Thread:Std ' +
##        '-j 2 --nouse-fpe --logverb=Error')
##    if status!=0:
##        print output

###compute saliency at each frame for each channel

from ezvisiontools import *
files = os.listdir('saliency/input')
files.sort()
print files
X=0;Y=1
f=files[0]
channel2id = {'SOintensity-':0,'COmotion-':1,'SOdir_0-':2,
              'SOdir_1-':3,'SOdir_2-':4,'SOdir_3-':5}
sal=[]
for f in files:
    vid = Mraw('saliency/input/'+f)
    name,channel=f.rsplit('.')[0:2]
    traj = np.load('saliency/input/'+name+'.npy')
    # transform from deg to pix
    offset=201
    traj=offset+traj*400.0 /13.5#11.951193368719736
    # in addition I have to rotate Y axis
    traj[:,:,Y]=offset-(traj[:,:,Y]-offset)
    if len(sal)==0:
        sal=np.ones((len(files)/len(channel2id),len(channel2id),traj.shape[0]))*-1
    
    if True:#channel2id.get(channel)<2:
        print f
        u=vid.computeSaliency(traj,radius=0.2)
        sal[int(name[-2:]),channel2id.get(channel),:]=u.squeeze()
    vid.close()
np.save('sal1600stdFloat.npy',sal)
plt.ion()
if False:      
    #sal=np.load('sal1600.npy')
    from mpl_toolkits.axes_grid1 import AxesGrid
    CHANNEL=1
    speeds=('0.5','1','2','4')
    dirs=('0','10','20','40','80')
    sm=np.mean(sal[:,CHANNEL,:],axis=1).squeeze()
    sm=np.reshape(sm,(4,5))
    plt.ion()
    plt.close('all')
    plt.plot([0,10,20,40,80],sm.transpose())
    plt.legend(speeds)
    #plt.close()
    plt.figure()

    ss=np.reshape(range(20),(5,4))

    plt.subplots_adjust(wspace=0,hspace=0)
    for j in range(4):
        s=ss[:,j]
        plt.subplot(2,2,j+1)
        for i in s:
            plt.plot(sal[i,CHANNEL,:])
            #grid[j].add_line(plt.Line2D(range(45),sal[i,CHANNEL,:]))
            #grid[j].set_xlim((10,45))
            #grid[j].set_ylim((0.01,0.036))
        if j !=2:
            plt.gca().set_xticklabels('')
            plt.gca().set_yticklabels('')
        plt.xlim((10,45))
        plt.ylim((0,1))
        #plt.title('Lambda = '+speeds[j])
    plt.legend(('0','10','20','40','80'),loc=4)
    plt.figure()

    plt.subplots_adjust(wspace=0,hspace=0)
    for i in range(5):
        for j in range(4):
            plt.subplot(5,4,i*4+j+1)
            plt.plot(sal[i*4+j,CHANNEL,:])
            plt.xlim((10,45))
            plt.ylim((0,0.036))
            if j!=0:
                plt.gca().set_yticklabels('')
            else:
                plt.ylabel('Surprise')
                plt.gca().set_yticks(np.arange(0.015,0.035,0.005))
            if i!=4:
                plt.gca().set_xticklabels('')
            else:
                plt.gca().set_xticks(range(15,45,5))
                plt.xlabel('Frame')
            if i==0:
                plt.title('Lambda='+speeds[j])
            if j==3:
                plt.text(plt.xlim()[0]+1.05*(plt.xlim()[1]-plt.xlim()[0]),
                         plt.ylim()[0]+0.5*(plt.ylim()[1]-plt.ylim()[0]),
                         'Phi='+dirs[i],fontsize='large',
                         rotation=270,
                         verticalalignment='center',
                         horizontalalignment='left')
            plt.grid()
                

        #plt.title('Lambda = '+speeds[j])
    #plt.legend(('0','10','20','40','80'),loc=4)



    #plt.legend(('0.5','1','2','4'),loc=4)




