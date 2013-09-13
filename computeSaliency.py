import os,commands
import numpy as np
import pylab as plt

""" put all data into a single folder INPATH. The format of the data name
    should be <name>.mpeg. Create folder OUTPATH and run

"""
# Call ezvision and get saliency maps
vp=1
inpath='saliency/input/vp%03d/'%vp
outpath='saliency/output/vp%03d/'%vp
files = os.listdir(inpath)
files.sort()

for f in files:
    print f
    status,output=commands.getstatusoutput('ezvision '+
        '--in='+inpath+f+' --input-frames=0-MAX@85Hz --rescale-input=1024x1024 '+
        ' --crop-input=128,0,1152,1028 --rescale-output=64x64 '+
        '--out=mraw:'+outpath+f[:-5]+' --output-frames=@85Hz ' +
        '--save-channel-outputs --vc-chans=IM --sm-type=None '+
        '--nodisplay-foa --nodisplay-patch --nodisplay-traj '+
        '--nodisplay-additive --wta-type=None --nouse-random '+
        '--direction-sqrt --num-directions=8 --vc-type=Thread:Std ' +
        '-j 2 --nouse-fpe --logverb=Error')
    commands.getstatusoutput('rm '+outpath+f[:-5]+'SOdir_*')
    if status!=0:
        print output
    


