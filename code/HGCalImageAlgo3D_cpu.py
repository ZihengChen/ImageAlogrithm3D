from pylab import *
import pandas as pd


KERNAL_R,KERNAL_Z= 2,2 #cm
MAXDISTANCE = 200 #cm
DECISION_RHO_KAPPA = 10
DECISION_NHD = 5 #cm
AFFINITY_Z = 0.5
CONTINUITY_NHD = 4 #cm

def ImageAlgorithm_cpu(dfevt_input,ievent):
    dfevt = dfevt_input
    dfevt = dfevt.reset_index(drop=True)
    x,y,z,energy = np.array(dfevt.x),np.array(dfevt.y),np.array(dfevt.z),np.array(dfevt.energy)
    z =  AFFINITY_Z*z
    
    nrech = energy.size
    # 1.find rho and rhorank
    rho = []
    for i in range(nrech):
        dr = ((x-x[i])**2 + (y-y[i])**2)**0.5
        dz = np.abs(z-z[i])
        local = (dr<KERNAL_R) & (dz<KERNAL_Z)
        irho = np.sum(energy[local] 
                      *np.exp(-dr[local]/1.0) # fix 0.5*KERNAL_R as decay rate
                      *np.exp(-dz[local]/4.0))  # fix 2*KERNAL_Z as decay rate
        rho.append(irho)
    rho = np.array(rho)
    argsortrho = rho.argsort()[::-1]
    rhorank = np.empty(len(rho), int)
    rhorank[argsortrho] = np.arange(len(rho))
    dfevt['rho'] = pd.Series(rho, index=dfevt.index)
    dfevt['rhorank'] = pd.Series(rhorank, index=dfevt.index)

    
    # 2.find NearstHiger and distance to NearestHigher
    nh,nhd = [],[]
    for i in range(nrech):
        irho = rho[i]
        irank= rhorank[i]
        if irank==0:
            nh. append(i)
            nhd.append(MAXDISTANCE)
        else:
            higher = rho>irho
            drr  = ((x[higher]-x[i])**2 + (y[higher]-y[i])**2 + (z[higher]-z[i])**2)**0.5
            temp = np.arange(len(rho))[higher]
            nh. append(temp[np.argmin(drr)])
            nhd.append(np.min(drr))
    nh = np.array(nh)
    nhd= np.array(nhd)
    dfevt['nh'] = pd.Series(nh, index=dfevt.index)
    dfevt['nhd'] = pd.Series(nhd, index=dfevt.index)
    
    
    DECISION_RHO = rho.max()/DECISION_RHO_KAPPA
    cluster = -np.ones(nrech,int)
    # argsortrho has been done in part1.
    # 3.find seeds
    selectseed = (rho>DECISION_RHO) & (nhd>DECISION_NHD)
    seedrho = rho[selectseed]
    temp = seedrho.argsort()[::-1]
    seedid = np.empty(len(seedrho), int)
    seedid[temp] = np.arange(len(seedrho))
    cluster[selectseed] = seedid
    dfevt['isseed'] = pd.Series(selectseed.astype(int), index=dfevt.index)
    

    # 4.asign clusters to seeds
    for ith in range(nrech):
        i = argsortrho[ith]
        if (cluster[i]<0) & (nhd[i]<CONTINUITY_NHD):
            cluster[i] = cluster[nh[i]]
    dfevt['cluster'] = pd.Series(cluster, index=dfevt.index)
    
    
    ########################################
    ##        END of image algorithm      ##
    ##             output result          ##
    ########################################
    clustx,clusty,clustz,clustenergy = [],[],[],[]
    for seed in seedid:
        sel = (dfevt.cluster == seed)
        seedenergy = np.sum(dfevt.energy[sel])
        seedx = np.sum(dfevt.energy[sel]*dfevt.ox[sel])/seedenergy
        seedy = np.sum(dfevt.energy[sel]*dfevt.oy[sel])/seedenergy
        seedz = np.sum(dfevt.energy[sel]*dfevt.oz[sel])/seedenergy
        
        clustenergy.append(seedenergy)
        clustx.append(seedx)
        clusty.append(seedy)
        clustz.append(seedz)
    clustenergy = np.array(clustenergy)
    clustx = np.array(clustx)
    clusty = np.array(clusty)
    clustz = np.array(clustz)
    clust_inputenergy    = np.sum(dfevt.energy)
    clust_includedenergy = np.sum(clustenergy)

    dfclus = pd.DataFrame({"id"     :[ievent],
                           "clust_n":[len(clustx)],
                           "clust_x":[clustx],
                           "clust_y":[clusty],
                           "clust_z":[clustz],
                           "clust_energy":[clustenergy],
                           "clust_inputenergy":[clust_inputenergy],
                           "clust_includedenergy":[clust_includedenergy]})
    return dfevt,dfclus

