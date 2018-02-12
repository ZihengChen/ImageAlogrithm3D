from pylab import *
import pandas as pd
from ImageAlgoKD_kernel_cuda import *

class ImageAlgoKD():
    def __init__(self,
                 MAXDISTANCE        = 1.5,
                 KERNAL_R           = 0.6,
                 KERNAL_R_NORM      = 0.4,
                 KERNAL_R_POWER     = 2.0,
                 DECISION_RHO_KAPPA = 4.0,
                 DECISION_NHD       = 0.6,
                 CONTINUITY_NHD     = 0.7
                ):
        
        self.MAXDISTANCE    = np.float32(MAXDISTANCE)
        self.KERNAL_R       = np.float32(KERNAL_R)
        self.KERNAL_R_NORM  = np.float32(KERNAL_R_NORM)
        self.KERNAL_R_POWER = np.float32(KERNAL_R_POWER)
        self.DECISION_RHO_KAPPA = np.float32(DECISION_RHO_KAPPA)
        self.DECISION_NHD   = np.float32(DECISION_NHD)
        self.CONTINUITY_NHD = np.float32(CONTINUITY_NHD)
    
    def run_ImageAlgoKD_cuda(self, Points, wPoints, ReturnDF=False):
        nPoints,kPoints = Points.shape
        
        Points  = Points .astype(np.float32)
        wPoints = wPoints.astype(np.float32)
        nPoints = np.int32(nPoints)
        kPoints = np.int32(kPoints)
    
        rho     = np.zeros(nPoints).astype(np.float32)
        rhorank = np.zeros(nPoints).astype(np.int32)
        nh      = np.zeros(nPoints).astype(np.int32)
        nhd     = np.zeros(nPoints).astype(np.float32)
        
        d_Points  = cuda.mem_alloc(Points.nbytes)
        d_wPoints = cuda.mem_alloc(wPoints.nbytes)
        d_rho     = cuda.mem_alloc(rho.nbytes)
        d_rhorank = cuda.mem_alloc(rhorank.nbytes)
        d_nh      = cuda.mem_alloc(nh.nbytes)
        d_nhd     = cuda.mem_alloc(nhd.nbytes)

        ### Run Cuda ###
        cuda.memcpy_htod( d_Points , Points )
        cuda.memcpy_htod( d_wPoints, wPoints )

        rho_cuda(d_rho, d_Points, d_wPoints,
                 nPoints, kPoints, 
                 self.KERNAL_R, self.KERNAL_R_NORM, self.KERNAL_R_POWER,
                 grid=(int(nPoints/1024)+1,1,1),block=(int(1024),1,1))

        rhoranknh_cuda(d_rhorank, d_nh,d_nhd, d_Points, d_rho,
                       nPoints, kPoints, 
                       self.MAXDISTANCE,
                       grid=(int(nPoints/1024)+1,1,1),block=(int(1024),1,1))


        cuda.memcpy_dtoh(rho,d_rho)
        cuda.memcpy_dtoh(rhorank,d_rhorank)
        cuda.memcpy_dtoh(nh,d_nh)
        cuda.memcpy_dtoh(nhd,d_nhd)
        ### Finish Cuda ###


        d_Points.free()
        d_wPoints.free()

        d_rho.free()
        d_rhorank.free()
        d_nh.free()
        d_nhd.free()
        
        
        cluster = -np.ones(nPoints,int)
        self.DECISION_RHO = rho.max()/self.DECISION_RHO_KAPPA

        # 2.1 convert rhorank to argsortrho 0(N)
        argsortrho = np.zeros(nPoints,int)
        argsortrho[rhorank] = np.arange(nPoints)

        # 2.2 find seeds
        selectseed = (rho>self.DECISION_RHO) & (nhd>self.DECISION_NHD)
        seedrho = rho[selectseed]
        temp = seedrho.argsort()[::-1]
        seedid = np.empty(len(seedrho), int)
        seedid[temp] = np.arange(len(seedrho))
        cluster[selectseed] = seedid

        # 2.3 asign clusters to seeds
        for ith in range(nPoints):
            i = argsortrho[ith]
            if  (cluster[i]<0) & (nhd[i]<self.CONTINUITY_NHD):
                cluster[i] = cluster[nh[i]]
                
        
        if ReturnDF:
            result = pd.DataFrame()
            result['rho'] = pd.Series(rho)
            result['rhorank'] = pd.Series(rhorank)
            result['nh'] = pd.Series(nh)
            result['nhd'] = pd.Series(nhd)
            result['isseed'] = pd.Series(selectseed)
            result['cluster'] = pd.Series(cluster)
            return result
        else:
            return cluster
    

    def run_ImageAlgoKD_numpy(self, Points, wPoints, ReturnDF=False):
        nPoints,kPoints = Points.shape

        # 1. find rho 
        rho = []
        for i in range(nPoints):
            dr = self.dis_numpy(Points, Points[i])
            local = (dr<self.KERNAL_R)
            irho = np.sum( wPoints[local] * np.exp( - (dr[local]/self.KERNAL_R_NORM)**self.KERNAL_R_POWER ))
            rho.append(irho)

        rho = np.array(rho)


        # 2. find rhorank
        argsortrho = rho.argsort()[::-1]
        rhorank = np.empty(len(rho), int)
        rhorank[argsortrho] = np.arange(len(rho))



        # 3. find NearstHiger and distance to NearestHigher
        nh,nhd = [],[]
        for i in range(nPoints):
            irho  = rho[i]
            irank = rhorank[i]
            
            higher = rho>irho
            # if no points is higher
            if not (True in higher): 
                nh. append(i)
                nhd.append(self.MAXDISTANCE)
            else:
                drr  = self.dis_numpy(Points[higher], Points[i])
                temp = np.arange(len(rho))[higher]
                nh. append(temp[np.argmin(drr)])
                nhd.append(np.min(drr))
                    
        nh = np.array(nh)
        nhd= np.array(nhd)
        
        # 4 Assign cluster
        cluster = -np.ones(nPoints,int)
        self.DECISION_RHO = rho.max()/self.DECISION_RHO_KAPPA

        # 4.1 make decision and find seeds
        selectseed = (rho>self.DECISION_RHO) & (nhd>self.DECISION_NHD)
        seedrho = rho[selectseed]
        temp = seedrho.argsort()[::-1]
        seedid = np.empty(len(seedrho), int)
        seedid[temp] = np.arange(len(seedrho))
        cluster[selectseed] = seedid

        # 4.2 asign clusters to seeds
        for ith in range(nPoints):
            i = argsortrho[ith]
            if  (cluster[i]<0) & (nhd[i]<self.CONTINUITY_NHD):
                cluster[i] = cluster[nh[i]]
        
        # 5. output
        if ReturnDF:
            result = pd.DataFrame()
            result['rho'] = pd.Series(rho)
            result['rhorank'] = pd.Series(rhorank)
            result['nh'] = pd.Series(nh)
            result['nhd'] = pd.Series(nhd)
            result['isseed'] = pd.Series(selectseed)
            result['cluster'] = pd.Series(cluster)
            return result
        else:
            return cluster

    def dis_numpy(self, p1,p2):
        return np.sqrt( np.sum((p1-p2)**2, axis=-1) )