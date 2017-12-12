from pylab import *
import pandas as pd

def energyeff(DatasetDir,DatasetFile,deltarho=5):
    dfresultclus = pd.read_pickle(DatasetDir+"output/"+DatasetFile+"_OutputClus.pkl")
    dfgen        = pd.read_pickle(DatasetDir+"input/"+DatasetFile+"_gen.pkl")
    energyeff = []
    for ievt, temp in dfresultclus.iterrows():
        genx,geny,gene = dfgen["gx"][ievt],dfgen["gy"][ievt],dfgen["ge"][ievt]
        tempe = temp.clust_energy
        tempx = temp.clust_x/temp.clust_z*320
        tempy = temp.clust_y/temp.clust_z*320
    
        slt   = (((tempx-genx)**2+(tempy-geny)**2)**0.5 < deltarho)
        energy = sum(tempe[slt])
        energyeff.append(energy/gene)
    return np.array(energyeff)

def effsigma(arr):
    ntotal = len(arr)
    npeak  = int(0.683*ntotal)
    ewidth = int(1e5)

    e = sort(arr)[::-1]
    for i in range(ntotal-npeak-1):
        temp = e[i]-e[i+npeak]
        if temp<ewidth:
            ewidth = temp
            mean = e[i:i+npeak].mean()
    return ewidth/2,mean