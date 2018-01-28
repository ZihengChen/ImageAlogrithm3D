from pylab import *
import pandas as pd
from root_pandas import read_root
from HGCal_Calibration import *
from IPython.display import clear_output

DatasetFolder = sys.argv[1]
DatasetFile   = sys.argv[2]
ECUT          = float(sys.argv[3]) #3 or 5 are reasonable numbers
NEVENTS       = int(sys.argv[4]) 
RecHitCalib   = RecHitCalibration()

df = read_root(DatasetFolder+DatasetFile+".root",'ana/hgc', 
               columns=['rechit_x', 'rechit_y','rechit_z','rechit_energy','rechit_layer','rechit_thickness',
                        'genpart_dvx','genpart_dvy','genpart_dvz','genpart_energy','genpart_gen'])

collist    = ["id","layer","energy", "ox","oy","oz","x","y","z"]
collistgen = ["id","gx","gy","gz","ge"]
dfrech     = pd.DataFrame(columns=collist)
dfgen      = pd.DataFrame(columns=collistgen)


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
        sel = aboveTreshold & (oz>0)
        layer_      = layer[sel]
        energy_     = energy[sel]
        ox_,oy_,oz_ = ox[sel],oy[sel],oz[sel]
        eventid_    = 1 * index * np.ones(layer_.size)
        x_,y_,z_    = ox_*320/oz_, oy_*320/oz_, layer_
        temp        = np.c_[eventid_,layer_,energy_,ox_,oy_,oz_,x_,y_,z_]
        temp        = pd.DataFrame(temp,columns=collist)
        dfrech      = dfrech.append(temp,ignore_index=True)
        
        
        ## 2. zside<0 ## 
        sel = aboveTreshold & (oz<0)
        layer_      = layer[sel]
        energy_     = energy[sel]
        ox_,oy_,oz_ = ox[sel],oy[sel],oz[sel]
        eventid_    = -1 * index * np.ones(layer_.size)
        x_,y_,z_    = ox_*320/oz_, oy_*320/oz_, layer_
        temp        = np.c_[eventid_,layer_,energy_,ox_,oy_,oz_,x_,y_,z_]
        temp        = pd.DataFrame(temp,columns=collist)
        dfrech      = dfrech.append(temp,ignore_index=True)
        
        
        
        isgen = np.array(row["genpart_gen"])
        gendvz= np.array(row["genpart_dvz"])
        
        slt = (isgen>=0) &(gendvz>0)
        thisgen = int(np.arange(isgen.size)[slt][0])
        genz =  320
        genx =  row["genpart_dvx"][thisgen]/row["genpart_dvz"][thisgen] *genz
        geny =  row["genpart_dvy"][thisgen]/row["genpart_dvz"][thisgen] *genz
        gene =  row["genpart_energy"][thisgen]
        temp = np.array([[index,genx,geny,genz,gene]])
        temp = pd.DataFrame(temp,columns=collistgen)
        dfgen = dfgen.append(temp,ignore_index=True)
        
        slt = (isgen>=0) &(gendvz<0)
        thisgen = int(np.arange(isgen.size)[slt][0])
        genz =  -320.0
        genx =  row["genpart_dvx"][thisgen]/row["genpart_dvz"][thisgen]*genz
        geny =  row["genpart_dvy"][thisgen]/row["genpart_dvz"][thisgen]*genz
        gene =  row["genpart_energy"][thisgen]
        temp = np.array([[-index,genx,geny,genz,gene]])
        temp = pd.DataFrame(temp,columns=collistgen)
        dfgen = dfgen.append(temp,ignore_index=True)
        

#dfrech.to_hdf(DatasetFolder+"input/"+DatasetFile+"_rechit.h5",key="table",mode='w',complevel=3)
dfrech.to_pickle(DatasetFolder+"input/"+DatasetFile+"_rechit.pkl")
dfgen.to_pickle(DatasetFolder+"input/"+DatasetFile+"_gen.pkl")
