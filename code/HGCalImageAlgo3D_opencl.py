from pylab import *
import pandas as pd
from HGCalImageAlgo3D_opencl_kernel import *

#KERNAL_R,KERNAL_Z= 2,2 #cm
#MAXDISTANCE = 200 #cm
DECISION_RHO_KAPPA = 10
DECISION_NHD = 4 #cm
AFFINITY_Z = 0.5
CONTINUITY_NHD = 4 #cm

def ImageAlgorithm_opencl(dfevt_input,ievent,device):
    lsz,context,prg = openclkernel(DeviceID=device)
    queue = cl.CommandQueue(context)
    
    dfevt = dfevt_input
    dfevt = dfevt.reset_index(drop=True)
    x,y,z,e = np.array(dfevt.x),np.array(dfevt.y),np.array(dfevt.z),np.array(dfevt.energy)
    z =  AFFINITY_Z*z
    
    ##########################################
    # STARTING CUDA
    # 1. copy rechits x,y,z,e from CPU to GPU
    nrech = np.int32(e.size)
    LOCALSIZE = int(lsz)
    GLOBALSIZE= (int(nrech/LOCALSIZE)+1)*LOCALSIZE

    x = x.astype(np.float32)
    y = y.astype(np.float32)
    z = z.astype(np.float32)
    e = e.astype(np.float32)
    rho     = np.zeros_like(e)
    rhorank = np.zeros_like(e).astype(np.int32)
    nh      = np.zeros_like(e).astype(np.int32)
    nhd     = np.zeros_like(e)

    mem_flags = cl.mem_flags
    d_x = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=x)
    d_y = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=y)
    d_z = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=z)
    d_e = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=e)
    d_rho     = cl.Buffer(context, mem_flags.READ_WRITE, rho.nbytes)
    d_rhorank = cl.Buffer(context, mem_flags.READ_WRITE, rhorank.nbytes)
    d_nh      = cl.Buffer(context, mem_flags.READ_WRITE, nh.nbytes)
    d_nhd     = cl.Buffer(context, mem_flags.READ_WRITE, nhd.nbytes)

    prg.rho_opencl(queue, (GLOBALSIZE,), (LOCALSIZE,),
                   d_rho,
                   nrech,d_x,d_y,d_z,d_e
                  )

    prg.rhorank_opencl(queue, (GLOBALSIZE,), (LOCALSIZE,),
                       d_rhorank,
                       nrech,d_rho
                       )
    prg.nh_opencl(queue, (GLOBALSIZE,), (LOCALSIZE,),
                  d_nh,d_nhd,
                  nrech,d_x,d_y,d_z,d_rho
                 )

    cl.enqueue_copy(queue, rho, d_rho)
    cl.enqueue_copy(queue, rhorank, d_rhorank)
    cl.enqueue_copy(queue, nh, d_nh)
    cl.enqueue_copy(queue, nhd, d_nhd)

    dfevt['rho'] = pd.Series(rho, index=dfevt.index)
    dfevt['rhorank'] = pd.Series(rhorank, index=dfevt.index)
    dfevt['nh'] = pd.Series(nh, index=dfevt.index)
    dfevt['nhd'] = pd.Series(nhd, index=dfevt.index)
    # ENDING CUDA
    ##########################################
    
    
    
    # 3. now decide seeds and asign rechits to seeds
    # rho,rhorank,nh,nhd = dfevt.rho,dfevt.rhorank,dfevt.nh,dfevt.nhd
    
    cluster = -np.ones(nrech,int)
    DECISION_RHO = rho.max()/DECISION_RHO_KAPPA

    # 2.1 convert rhorank to argsortrho 0(N)
    argsortrho = np.zeros(nrech,int)
    argsortrho[rhorank] = np.arange(nrech)

    # 2.2 find seeds
    selectseed = (rho>DECISION_RHO) & (nhd>DECISION_NHD)
    seedrho = rho[selectseed]
    temp = seedrho.argsort()[::-1]
    seedid = np.empty(len(seedrho), int)
    seedid[temp] = np.arange(len(seedrho))
    cluster[selectseed] = seedid
    dfevt['isseed'] = pd.Series(selectseed.astype(int), index=dfevt.index)

    # 2.3 asign clusters to seeds
    for ith in range(nrech):
        i = argsortrho[ith]
        if  (cluster[i]<0) & (nhd[i]<CONTINUITY_NHD):
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
