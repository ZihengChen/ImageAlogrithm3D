from pylab import *
import pandas as pd
import tqdm
from HGCal_ImageAlgo3D_kernel import *


class ImagingAlgo():
    def __init__(self, 
                 AFFINITY_Z     = 1.2,
                 KERNAL_R       = 4.0, 
                 KERNAL_Z       = 2, # layer
                 KERNAL_EXPC    = 0.5,
                 MAXDISTANCE    = 200,
                 DECISION_RHO_KAPPA = 10,
                 DECISION_NHD   = 4.0,
                 CONTINUITY_NHD = 6.0
                 ):
        self.AFFINITY_Z     = AFFINITY_Z
        self.KERNAL_R       = KERNAL_R
        self.KERNAL_Z       = AFFINITY_Z * KERNAL_Z
        self.KERNAL_EXPC    = KERNAL_EXPC
        self.MAXDISTANCE    = MAXDISTANCE
        self.DECISION_RHO_KAPPA = DECISION_RHO_KAPPA
        self.DECISION_NHD   = DECISION_NHD
        self.CONTINUITY_NHD = CONTINUITY_NHD



    def RunImagingAlgo(self, df, N=100, verb=True):
        dfresultclus     = pd.DataFrame()
        if verb:
            looplist = tqdm.tqdm(np.unique(np.abs(df.id)))
        else:
            looplist = np.unique(np.abs(df.id))
        for ievt in looplist:
            if ievt < N:
                _,dfevtclus   = self.ImageAlgorithm_cuda(df[df.id==ievt],ievt)
                dfresultclus  = dfresultclus.append(dfevtclus, ignore_index=True)
                _,dfevtclus   = self.ImageAlgorithm_cuda(df[df.id==-ievt],-ievt)
                dfresultclus  = dfresultclus.append(dfevtclus, ignore_index=True)
        return dfresultclus



    def ImageAlgorithm_cuda(self, dfevt_input,ievent):
        dfevt = dfevt_input
        dfevt = dfevt.reset_index(drop=True)
        x,y,z,e = np.array(dfevt.x),np.array(dfevt.y),np.array(dfevt.z),np.array(dfevt.energy)
        z =  self.AFFINITY_Z*z
        
        ##########################################
        # STARTING CUDA
        # 1. copy rechits x,y,z,e from CPU to GPU
        nrech = np.int32(e.size)
        x = x.astype(np.float32)
        y = y.astype(np.float32) 
        z = z.astype(np.float32)
        e = e.astype(np.float32)
        rho     = np.zeros_like(e)
        rhorank = np.zeros_like(e).astype(np.int32)
        nh      = np.zeros_like(e).astype(np.int32)
        nhd     = np.zeros_like(e)

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
                nrech,np.float32( self.KERNAL_R),np.float32( self.KERNAL_Z),np.float32( self.KERNAL_EXPC),
                d_x,d_y,d_z,d_e,
                grid=(int(nrech/1024)+1,1,1),block=(int(1024),1,1))
        rhoranknh_cuda(d_rhorank,d_nh,d_nhd,
                    nrech,d_x,d_y,d_z,d_rho,
                    grid=(int(nrech/1024)+1,1,1),block=(int(1024),1,1))

        #rhorank_cuda(d_rhorank,nrech,d_rho,
        #             grid=(int(nrech/1024)+1,1,1),block=(int(1024),1,1))

        #nh_cuda(d_nh,d_nhd,
        #        nrech,d_x,d_y,d_z,d_rho,
        #        grid=(int(nrech/1024)+1,1,1),block=(int(1024),1,1))

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

        dfevt['rho'] = pd.Series(rho, index=dfevt.index)
        dfevt['rhorank'] = pd.Series(rhorank, index=dfevt.index)
        dfevt['nh'] = pd.Series(nh, index=dfevt.index)
        dfevt['nhd'] = pd.Series(nhd, index=dfevt.index)
        # ENDING CUDA
        ##########################################
        
        
        
        # 3. now decide seeds and asign rechits to seeds
        # rho,rhorank,nh,nhd = dfevt.rho,dfevt.rhorank,dfevt.nh,dfevt.nhd
        argsortrho          = np.zeros(nrech,int)
        argsortrho[rhorank] = np.arange(nrech)
        cluster             = -np.ones(nrech,int)
        DECISION_RHO        = rho.max()/self.DECISION_RHO_KAPPA
        # find seeds
        selectseed = (rho>DECISION_RHO) & (nhd>self.DECISION_NHD)
        seedrho = rho[selectseed]
        temp = seedrho.argsort()[::-1]
        seedid = np.empty(len(seedrho), int)
        seedid[temp] = np.arange(len(seedrho))
        cluster[selectseed] = seedid
        dfevt['isseed'] = pd.Series(selectseed.astype(int), index=dfevt.index)
        # asign clusters to seeds
        for ith in range(nrech):
            i = argsortrho[ith]
            if  (cluster[i]<0) & (nhd[i]<self.CONTINUITY_NHD):
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



