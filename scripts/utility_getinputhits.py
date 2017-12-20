from root_pandas import read_root
from pylab import *
import pandas as pd
from utility_rechitcalibration import *
from IPython.display import clear_output


DatasetFolder = sys.argv[1]
DatasetFile   = sys.argv[2] #'CMSSW9304_partGun_PDGid22_x100_E30.0To30.0_NTUP'
ECUT          = float(sys.argv[3]) #3 or 5 are reasonable numbers
NEVENTS       = int(sys.argv[4]) 

df      = read_root(DatasetFolder+DatasetFile+".root",'ana/hgc')
collist = [ 'rechit_x', 'rechit_y','rechit_z','rechit_energy','rechit_layer','rechit_thickness']
for col in df.columns:
    if not col in collist:
        df.drop(col, axis=1, inplace=True) 



# start to calibrate rechits, based on sigma noise
RecHitCalib = RecHitCalibration()
collist     = ["id","layer","energy", "ox","oy","oz","x","y","z"]
dfrech      = pd.DataFrame(columns=collist)
for index, row in df.iterrows():
    if (index < NEVENTS) and (index>0):
        print("Calibrate Event: {}/{}".format(index+1,NEVENTS))
        clear_output(wait=True)
        
        thicknessindex  = (row["rechit_thickness"]/100 - 1).astype(int)
        layer,energy    = row['rechit_layer'],row['rechit_energy']
        ox, oy, oz      = row['rechit_x'],row['rechit_y'],row['rechit_z']
        sigmaNoise      = 0.001 * RecHitCalib.sigmaNoiseMeV(layer, thicknessindex) 
        aboveTreshold   = (energy >= ECUT*sigmaNoise)
        
        ## 1. zside>0 ## 
        sel         = aboveTreshold & (oz>0) #sel = (energy>0.1)&(oz>0)
        layer_      = layer[sel]
        energy_     = energy[sel]
        ox_,oy_,oz_ = ox[sel],oy[sel],oz[sel]
        eventid_    = 1 * index * np.ones(layer_.size)
        x_,y_,z_    = ox_*320/oz_, oy_*320/oz_, layer_
        temp        = np.c_[eventid_,layer_,energy_,ox_,oy_,oz_,x_,y_,z_]
        temp        = pd.DataFrame(temp,columns=collist)
        dfrech      = dfrech.append(temp,ignore_index=True)
        
        ## 2. zside<0 ## 
        sel         = aboveTreshold & (oz<0) #sel = (energy>0.1)&(oz>0)
        layer_      = layer[sel]
        energy_     = energy[sel]
        ox_,oy_,oz_ = ox[sel],oy[sel],oz[sel]
        eventid_    = -1 * index * np.ones(layer_.size)
        x_,y_,z_    = ox_*320/oz_, oy_*320/oz_, layer_
        temp        = np.c_[eventid_,layer_,energy_,ox_,oy_,oz_,x_,y_,z_]
        temp        = pd.DataFrame(temp,columns=collist)
        dfrech      = dfrech.append(temp,ignore_index=True)
        
dfrech.to_hdf(DatasetFolder+"input/"+DatasetFile+"_rechit.h5",key="DUMMY",mode='w')