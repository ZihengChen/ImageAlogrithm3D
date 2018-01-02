from pylab import *
import pandas as pd
import multiprocessing as mp
import tqdm
from root_pandas import read_root

from HGCalImageAlgo3D_cuda_kernel import *
from HGCalImageAlgo3D_opencl_kernel import *


class ImagingAlgo():
    def __init__(self, 
                 AFFINITY_Z_     = 1.2,
                 KERNAL_R_       = 4.0, 
                 #KERNAL_Z_       = 2.4,
                 KERNAL_EXPC_    = 0.5,
                 MAXDISTANCE_    = 200,
                 DECISION_RHO_KAPPA_ = 10,
                 DECISION_NHD_   = 4.0,
                 CONTINUITY_NHD_ = 6.0
                 ):
        self.AFFINITY_Z     = AFFINITY_Z_
        self.KERNAL_R       = KERNAL_R_
        self.KERNAL_Z       = AFFINITY_Z_*2
        self.KERNAL_EXPC    = KERNAL_EXPC_
        self.MAXDISTANCE    = MAXDISTANCE_
        self.DECISION_RHO_KAPPA = DECISION_RHO_KAPPA_
        self.DECISION_NHD   = DECISION_NHD_
        self.CONTINUITY_NHD = CONTINUITY_NHD_



    def RunImagingAlgo(self, df, N=100, verb=True):
        dfresultclus     = pd.DataFrame()
        if verb:
            for ievt in tqdm.tqdm(np.unique(np.abs(df.id))):
                if ievt < N:
                    _,dfevtclus   = self.ImageAlgorithm_opencl(df[df.id==ievt],ievt,1)
                    dfresultclus  = dfresultclus.append(dfevtclus, ignore_index=True)
                    _,dfevtclus   = self.ImageAlgorithm_opencl(df[df.id==-ievt],-ievt,1)
                    dfresultclus  = dfresultclus.append(dfevtclus, ignore_index=True)
        else:
            for ievt in np.unique(np.abs(df.id)):
                if ievt < N:
                    _,dfevtclus   = self.ImageAlgorithm_opencl(df[df.id==ievt],ievt,1)
                    dfresultclus  = dfresultclus.append(dfevtclus, ignore_index=True)
                    _,dfevtclus   = self.ImageAlgorithm_opencl(df[df.id==-ievt],-ievt,1)
                    dfresultclus  = dfresultclus.append(dfevtclus, ignore_index=True)
        return dfresultclus



    def ImageAlgorithm_cpu(self, dfevt_input, ievent):
        dfevt = dfevt_input
        dfevt = dfevt.reset_index(drop=True)
        x,y,z,energy = np.array(dfevt.x),np.array(dfevt.y),np.array(dfevt.z),np.array(dfevt.energy)
        z =  self.AFFINITY_Z*z
        
        nrech = energy.size
        # 1.find rho and rhorank
        rho = []
        for i in range(nrech):
            dr = ((x-x[i])**2 + (y-y[i])**2)**0.5
            dz = np.abs(z-z[i])
            local = (dr<self.KERNAL_R) & (dz<self.KERNAL_Z)
            irho = np.sum(energy[local] 
                        *np.exp(-dr[local]/1.0) # fix 0.5*self.KERNAL_R as decay rate
                        *np.exp(-dz[local]/4.0))  # fix 2*self.KERNAL_Z as decay rate
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
                nhd.append(self.MAXDISTANCE)
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
        
        
        DECISION_RHO = rho.max()/self.DECISION_RHO_KAPPA
        cluster = -np.ones(nrech,int)
        # argsortrho has been done in part1.
        # 3.find seeds
        selectseed = (rho>DECISION_RHO) & (nhd>self.DECISION_NHD)
        seedrho = rho[selectseed]
        temp = seedrho.argsort()[::-1]
        seedid = np.empty(len(seedrho), int)
        seedid[temp] = np.arange(len(seedrho))
        cluster[selectseed] = seedid
        dfevt['isseed'] = pd.Series(selectseed.astype(int), index=dfevt.index)
        

        # 4.asign clusters to seeds
        for ith in range(nrech):
            i = argsortrho[ith]
            if (cluster[i]<0) & (nhd[i]<self.CONTINUITY_NHD):
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


    def ImageAlgorithm_opencl(self, dfevt_input,ievent,device):
        lsz,context,prg = openclkernel(DeviceID=device)
        queue = cl.CommandQueue(context)
        
        dfevt = dfevt_input
        dfevt = dfevt.reset_index(drop=True)
        x,y,z,e = np.array(dfevt.x),np.array(dfevt.y),np.array(dfevt.z),np.array(dfevt.energy)
        z =  self.AFFINITY_Z*z
        
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
                    nrech,np.float32( self.KERNAL_R),np.float32( self.KERNAL_Z),np.float32( self.KERNAL_EXPC),
                    d_x,d_y,d_z,d_e
                    )
        prg.rhoranknh_opencl(queue, (GLOBALSIZE,), (LOCALSIZE,),
                            d_rhorank,d_nh,d_nhd,
                            nrech,d_x,d_y,d_z,d_rho
                            )

        #prg.rhorank_opencl(queue, (GLOBALSIZE,), (LOCALSIZE,),
        #                   d_rhorank,
        #                   nrech,d_rho
        #                   )
        #prg.nh_opencl(queue, (GLOBALSIZE,), (LOCALSIZE,),
        #              d_nh,d_nhd,
        #              nrech,d_x,d_y,d_z,d_rho
        #             )

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
        DECISION_RHO = rho.max()/self.DECISION_RHO_KAPPA

        # 2.1 convert rhorank to argsortrho 0(N)
        argsortrho = np.zeros(nrech,int)
        argsortrho[rhorank] = np.arange(nrech)

        # 2.2 find seeds
        selectseed = (rho>DECISION_RHO) & (nhd>self.DECISION_NHD)
        seedrho = rho[selectseed]
        temp = seedrho.argsort()[::-1]
        seedid = np.empty(len(seedrho), int)
        seedid[temp] = np.arange(len(seedrho))
        cluster[selectseed] = seedid
        dfevt['isseed'] = pd.Series(selectseed.astype(int), index=dfevt.index)

        # 2.3 asign clusters to seeds
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


