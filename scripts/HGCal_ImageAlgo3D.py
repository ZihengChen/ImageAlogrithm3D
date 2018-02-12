from pylab import *
import pandas as pd
import tqdm
from HGCal_ImageAlgo3D_kernel_cuda import *


class ImagingAlgo3D():

    def __init__(self, 
                 MAXDISTANCE        = 200,  #cm
                 LAYER_DISTANCE     = 1.2,  #cm
                 KERNAL_R           = 4.0,  #cm
                 KERNAL_R_NORM      = 2.0,  #cm
                 KERNAL_R_POWER     = 2.0,  
                 KERNAL_LAYER       = 2.0,  #number of layer
                 DECISION_RHO_KAPPA = 10.0, #fractio of max rho
                 DECISION_NHD       = 4.0,  #cm
                 CONTINUITY_NHD     = 6.0   #cm 
                 ):

        self.MAXDISTANCE    = np.float32(MAXDISTANCE)
        self.LAYER_DISTANCE = np.float32(LAYER_DISTANCE)

        self.KERNAL_R       = np.float32(KERNAL_R)
        self.KERNAL_R_NORM  = np.float32(KERNAL_R_NORM)
        self.KERNAL_R_POWER = np.float32(KERNAL_R_POWER)
        self.KERNAL_LAYER   = np.float32(KERNAL_LAYER)
        self.KERNAL_Z       = np.float32(KERNAL_LAYER*LAYER_DISTANCE)

        self.DECISION_RHO_KAPPA = np.float32(DECISION_RHO_KAPPA)
        self.DECISION_NHD   = np.float32(DECISION_NHD)
        self.CONTINUITY_NHD = np.float32(CONTINUITY_NHD)



    def RunImagingAlgo(self, df, Nevent=100, verb=True):
        dfresultclus     = pd.DataFrame()
        if verb:
            looplist = tqdm.tqdm(np.unique(np.abs(df.id)))
        else:
            looplist = np.unique(np.abs(df.id))

        for ievt in looplist:
            if ievt < Nevent:
                dfevtclus   = self.ImageAlgorithm_cuda(df[df.id==ievt])
                dfresultclus  = dfresultclus.append(dfevtclus, ignore_index=True)
                dfevtclus   = self.ImageAlgorithm_cuda(df[df.id==-ievt])
                dfresultclus  = dfresultclus.append(dfevtclus, ignore_index=True)
        return dfresultclus



    def ImageAlgorithm_cuda(self, dfevt_input, ReturnDecision = False):

        dfevt = dfevt_input
        dfevt = dfevt.reset_index(drop=True)
        
        x = np.array(dfevt.x).astype(np.float32)
        y = np.array(dfevt.y).astype(np.float32)
        z = (np.array(dfevt.z) * self.LAYER_DISTANCE).astype(np.float32)
        e = np.array(dfevt.energy).astype(np.float32)

        N = np.int32(e.size)
        
        ##########################################
        # STARTING CUDA
        # 1. copy rechits x,y,z,e from CPU to GPU

        rho     = np.zeros(N).astype(np.float32)
        rhorank = np.zeros(N).astype(np.int32)
        nh      = np.zeros(N).astype(np.int32)
        nhd     = np.zeros(N).astype(np.float32)

        d_x = cuda.mem_alloc(x.nbytes)
        d_y = cuda.mem_alloc(y.nbytes)
        d_z = cuda.mem_alloc(z.nbytes)
        d_e = cuda.mem_alloc(e.nbytes)
        d_rho     = cuda.mem_alloc(rho.nbytes)
        d_rhorank = cuda.mem_alloc(rhorank.nbytes)
        d_nh      = cuda.mem_alloc(nh.nbytes)
        d_nhd     = cuda.mem_alloc(nhd.nbytes)
        cuda.memcpy_htod(d_x,x)
        cuda.memcpy_htod(d_y,y)
        cuda.memcpy_htod(d_z,z)
        cuda.memcpy_htod(d_e,e)

        # 2. Calculate rho, rhorank, nh, nhd on GPU
        rho_cuda(d_rho,
                 d_x, d_y, d_z, d_e,
                 N, self.KERNAL_R, self.KERNAL_R_NORM, self.KERNAL_R_POWER, self.KERNAL_Z,
                grid=( int(N/1024)+1,1,1), block=(int(1024),1,1) )

        rhoranknh_cuda(d_rhorank,d_nh,d_nhd,
                       d_x,d_y,d_z,d_rho,
                       N, self.MAXDISTANCE,
                       grid=( int(N/1024)+1,1,1), block=(int(1024),1,1) )


        # 2. Copy out the result from GPU to CPU
        cuda.memcpy_dtoh(rho,d_rho)
        cuda.memcpy_dtoh(rhorank,d_rhorank)
        cuda.memcpy_dtoh(nh,d_nh)   
        cuda.memcpy_dtoh(nhd,d_nhd)

        d_x.free()
        d_y.free()
        d_z.free()
        d_e.free()
        d_rho.free()
        d_rhorank.free()
        d_nh.free()
        d_nhd.free()


        # ENDING CUDA
        ##########################################
        
        
        # 3. now decide seeds and asign rechits to seeds
        # rho,rhorank,nh,nhd = dfevt.rho,dfevt.rhorank,dfevt.nh,dfevt.nhd
        argsortrho          = np.zeros(N,int)
        argsortrho[rhorank] = np.arange(N)
        cluster             = -np.ones(N,int)
        DECISION_RHO        = rho.max()/self.DECISION_RHO_KAPPA
        # find seeds
        selectseed = (rho>DECISION_RHO) & (nhd>self.DECISION_NHD)
        seedrho = rho[selectseed]
        temp = seedrho.argsort()[::-1]
        seedid = np.empty(len(seedrho), int)
        seedid[temp] = np.arange(len(seedrho))
        cluster[selectseed] = seedid
        
        # asign clusters to seeds
        for ith in range(N):
            i = argsortrho[ith]
            if  (cluster[i]<0) & (nhd[i]<self.CONTINUITY_NHD):
                cluster[i] = cluster[nh[i]]
        

        if ReturnDecision:  
            dfevt['rho']     = pd.Series(rho,        index=dfevt.index)
            dfevt['rhorank'] = pd.Series(rhorank,    index=dfevt.index)
            dfevt['nh']      = pd.Series(nh,         index=dfevt.index)
            dfevt['nhd']     = pd.Series(nhd,        index=dfevt.index)
            dfevt['isseed']  = pd.Series(selectseed, index=dfevt.index)
            dfevt['cluster'] = pd.Series(cluster,    index=dfevt.index)
   
        
        ########################################
        ##        END of image algorithm      ##
        ##             output result          ##
        ########################################
        ievent = dfevt.id[0]


        clustx,clusty,clustz,clustenergy = [],[],[],[]
        for seed in seedid:
            sel = (cluster == seed)
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
        
        if ReturnDecision:
            return dfevt,dfclus
        else:
            return dfclus



