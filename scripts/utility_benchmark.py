from pylab import *
import pandas as pd
from root_pandas import read_root

def energyeff(DatasetDir,DatasetFile,N,deltarho=5,test=None):
    dfgen = pd.read_pickle(DatasetDir+"input/"+DatasetFile+"_gen.pkl")
    if test is not None:
        dfresultclus = pd.read_pickle(test)
    else:
        dfresultclus = pd.read_pickle(DatasetDir+"output/"+DatasetFile+"_OutputClus.pkl")
    
    energyeff = []
    for i, tempclus in dfresultclus.iterrows():
        if abs(tempclus.id) < N:
            evtid   = tempclus.id
            tempgen = dfgen[dfgen.id==evtid]
            genx    = float(tempgen['gx'])
            geny    = float(tempgen['gy'])
            genz    = float(tempgen['gz'])
            gene    = float(tempgen['ge'])
        
            clusz = genz
            clusx = tempclus.clust_x/tempclus.clust_z*clusz
            clusy = tempclus.clust_y/tempclus.clust_z*clusz
            cluse = tempclus.clust_energy
    
            slt   = (((clusx-genx)**2+(clusy-geny)**2)**0.5 < deltarho)
            energy = sum(cluse[slt])
            #print("{},{}".format(energy,gene))
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

def getcmsswmultclust(DatasetDir,DatasetFile,N):
    dfc = read_root(DatasetDir+DatasetFile+".root",'ana/hgc')
    energy3d,count3d = [],[]
    for i in np.arange(len(dfc)):
        if i < N:
            slt = dfc.multiclus_z[i]>0
            energyi = dfc.multiclus_energy[i][slt]
            energy3d.append(np.sum(energyi))
            count3d.append(energyi.size)
    
            slt = dfc.multiclus_z[i]<0
            energyi = dfc.multiclus_energy[i][slt]
            energy3d.append(np.sum(energyi))
            count3d.append(energyi.size)
    
    energy3d = np.array(energy3d)
    count3d = np.array(count3d)
    return energy3d, count3d

def getoutputclust(DatasetDir,DatasetFile,N,deltarho=5):
    dfresultclus = pd.read_pickle(DatasetDir+"output/"+DatasetFile+"_OutputClus.pkl")
    dfgen        = pd.read_pickle(DatasetDir+"input/"+DatasetFile+"_gen.pkl")
    genparticle = []
    energy3d,count3d = [],[]
    for i, tempclus in dfresultclus.iterrows():
        if abs(tempclus.id) < N:
            evtid   = tempclus.id
            tempgen = dfgen[dfgen.id==evtid]
            genx    = float(tempgen['gx'])
            geny    = float(tempgen['gy'])
            genz    = float(tempgen['gz'])
            gene    = float(tempgen['ge'])
        
            clusz = genz
            clusx = tempclus.clust_x/tempclus.clust_z*clusz
            clusy = tempclus.clust_y/tempclus.clust_z*clusz
            cluse = tempclus.clust_energy
    
            slt   = (((clusx-genx)**2+(clusy-geny)**2)**0.5 < deltarho)
            energy = sum(cluse[slt])
            #print("{},{}".format(energy,gene))
            genparticle.append(gene)
            energy3d.append(energy)
            count3d.append(cluse[slt].size)
    return np.array(genparticle), np.array(energy3d),np.array(count3d)