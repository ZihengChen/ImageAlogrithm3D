from root_pandas import read_root
from pylab import *
import pandas as pd
from utility_rechitcalibration import *


ECUT = 6
DatasetFolder = '/Users/zihengchen/Documents/HGCal/clustering/data/'
DatasetFile   = 'SinglePi_PU200'

df = read_root(DatasetFolder+DatasetFile+".root",'ana/hgc')
collist = [ 'rechit_x', 'rechit_y','rechit_z','rechit_energy','rechit_layer','rechit_thickness',
            'genpart_dvx','genpart_dvy','genpart_dvz','genpart_energy']
for col in df.columns:
    if not col in collist:
        df.drop(col, axis=1, inplace=True) 

RecHitCalib = RecHitCalibration()

collist    = ["id","layer","energy", "ox","oy","oz","x","y","z"]
collistgen = ["id","gx","gy","gz","ge"]
dfrech     = pd.DataFrame(columns=collist)
dfgen      = pd.DataFrame(columns=collistgen)

for index, row in df.iterrows():
    if index <100:
        thicknessindex  = (row["rechit_thickness"]/100 - 1).astype(int)
        
        layer,energy = row['rechit_layer'],row['rechit_energy']
        ox, oy, oz   = row['rechit_x'],row['rechit_y'],row['rechit_z']
        
        sigmaNoise   = 0.001 * RecHitCalib.sigmaNoiseMeV(layer, thicknessindex) 
        aboveTreshold = (energy >= ECUT*sigmaNoise)
        sel = aboveTreshold & (oz>0)

        layer,energy,ox,oy,oz = layer[sel],energy[sel],ox[sel],oy[sel],oz[sel]
        eventid  = index * np.ones(layer.size)
        x,y,z    = ox*320/oz, oy*320/oz, layer
        
        temp = np.c_[eventid,layer,energy,ox,oy,oz,x,y,z]
        temp = pd.DataFrame(temp,columns=collist)
        dfrech = dfrech.append(temp,ignore_index=True)
        
        
        temp = np.array([[index,
                          row["genpart_dvx"][0]/row["genpart_dvz"][0]*320,
                          row["genpart_dvy"][0]/row["genpart_dvz"][0]*320,
                          320.0,
                          row["genpart_energy"][0]
                         ]])
        temp = pd.DataFrame(temp,columns=collistgen)
        dfgen = dfgen.append(temp,ignore_index=True)

dfrech.to_pickle(DatasetFolder+DatasetFile+"_rechit.pkl")
dfgen.to_pickle(DatasetFolder+DatasetFile+"_gen.pkl")