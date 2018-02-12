from pylab import *
import pandas as pd
import multiprocessing as mp
import tqdm
from root_pandas import read_root

from HGCal_ImageAlgo3D_kernel_cuda import *
from HGCal_ImageAlgo3D_kernel_opencl import *


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



    def RunImagingAlgo(self, df, Nevent=100, verb=True, framework='opencl'):
        # switches of verboros
        if verb:
            looplist = tqdm.tqdm(np.unique(np.abs(df.id)))
        else:
            looplist = np.unique(np.abs(df.id))

        # switches of framework
        if framework == 'opencl':
            clustering = self.ImageAlgorithm_opencl
        if framework == 'cuda':
            clustering = self.ImageAlgorithm_cuda
        if framework == 'numpy':
            clustering = self.ImageAlgorithm_numpy

        # start to run
        dfresultclus = pd.DataFrame()
        for ievt in looplist:
            if ievt < Nevent:
                dfevtclus   = clustering( df[df.id==ievt] )
                dfresultclus  = dfresultclus.append(dfevtclus, ignore_index=True)
                dfevtclus   = clustering( df[df.id==-ievt] )
                dfresultclus  = dfresultclus.append(dfevtclus, ignore_index=True)
        return dfresultclus


    def ImageAlgorithm_numpy(self, dfevt_input, ievent):
        dfevt = df.query(demoevent)
        dfevt = dfevt.reset_index(drop=True)

        x = np.array(dfevt.x).astype(np.float32)
        y = np.array(dfevt.y).astype(np.float32)
        z = (np.array(dfevt.z) * self.LAYER_DISTANCE).astype(np.float32)
        e = np.array(dfevt.energy).astype(np.float32)

        N = np.int32(e.size)
        
        # 1.find rho and rhorank
        rho = []
        for i in range(N):
            dr = ((x-x[i])**2 + (y-y[i])**2)**0.5
            dz = np.abs(z-z[i])
            local = (dr<self.KERNAL_R) & (dz<=self.KERNAL_Z)
            irho  = np.sum( e[local] * np.exp( -(dr[local]/self.KERNAL_R_NORM)**self.KERNAL_R_POWER  ))
            rho.append(irho)
        rho = np.array(rho)
        argsortrho = rho.argsort()[::-1]
        rhorank = np.empty(len(rho), int)
        rhorank[argsortrho] = np.arange(len(rho))

        
        # 2.find NearstHiger and distance to NearestHigher
        nh,nhd = [],[]
        for i in range(N):
            irho = rho[i]
            irank= rhorank[i]
            
            higher = rho > irho
            # if no points is higher
            if not (True in higher): 
                nh. append(i)
                nhd.append(self.MAXDISTANCE)
            else:
                drr  = ((x[higher]-x[i])**2 + (y[higher]-y[i])**2 + (z[higher]-z[i])**2)**0.5
                temp = np.arange(len(rho))[higher]
                nh. append(temp[np.argmin(drr)])
                nhd.append(np.min(drr))
        nh = np.array(nh)
        nhd= np.array(nhd)
        
        
        
        # 3.find seeds
        cluster = -np.ones(N,int)
        DECISION_RHO = rho.max()/self.DECISION_RHO_KAPPA

        # 2.1 convert rhorank to argsortrho 0(N)
        argsortrho = np.zeros(N,int)
        argsortrho[rhorank] = np.arange(N)

        # 2.2 find seeds
        selectseed = (rho>DECISION_RHO) & (nhd>self.DECISION_NHD)
        seedrho = rho[selectseed]
        temp = seedrho.argsort()[::-1]
        seedid = np.empty(len(seedrho), int)
        seedid[temp] = np.arange(len(seedrho))
        cluster[selectseed] = seedid


        # 2.3 asign clusters to seeds
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

        dfclus = pd.DataFrame({"id"  :[ievent],
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


    def ImageAlgorithm_opencl(self, dfevt_input, device=1,  ReturnDecision = False):
        lsz,context,prg = openclkernel(DeviceID=device)
        queue = cl.CommandQueue(context)
        
        dfevt = dfevt_input
        dfevt = dfevt.reset_index(drop=True)

        x = np.array(dfevt.x).astype(np.float32)
        y = np.array(dfevt.y).astype(np.float32)
        z = (np.array(dfevt.z) * self.LAYER_DISTANCE).astype(np.float32)
        e = np.array(dfevt.energy).astype(np.float32)

        N = np.int32(e.size)
        
        ##########################################
        # STARTING opencl
        # 1. copy rechits x,y,z,e from CPU to GPU

        LOCALSIZE = int(lsz)
        GLOBALSIZE= (int(N/LOCALSIZE)+1)*LOCALSIZE

        rho     = np.zeros(N).astype(np.float32)
        rhorank = np.zeros(N).astype(np.int32)
        nh      = np.zeros(N).astype(np.int32)
        nhd     = np.zeros(N).astype(np.float32)

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
                    d_x, d_y, d_z, d_e,
                    N, self.KERNAL_R, self.KERNAL_R_NORM, self.KERNAL_R_POWER, self.KERNAL_Z
                    )

        prg.rhoranknh_opencl(queue, (GLOBALSIZE,), (LOCALSIZE,),
                    d_rhorank,d_nh,d_nhd,
                    d_x,d_y,d_z,d_rho,
                    N, self.MAXDISTANCE
                    )


        cl.enqueue_copy(queue, rho, d_rho)
        cl.enqueue_copy(queue, rhorank, d_rhorank)
        cl.enqueue_copy(queue, nh, d_nh)
        cl.enqueue_copy(queue, nhd, d_nhd)
        # ENDING opencl
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

        dfclus = pd.DataFrame({"id"  :[ievent],
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
        DECISION_RHO   = rho.max()/self.DECISION_RHO_KAPPA
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

        dfclus = pd.DataFrame({"id"  :[ievent],
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

