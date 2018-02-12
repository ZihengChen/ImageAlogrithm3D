## cuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
mod = SourceModule("""

    ////////////////////////////
    // 1. get rho
    ////////////////////////////

    __global__ void rho_cuda( float *d_rho, float *d_Points, 
                              float *d_wPoints,
                              int nPoints, int kPoints, float KERNAL_R, float KERNAL_R_NORM, float KERNAL_R_POWER){

        int i = blockDim.x*blockIdx.x + threadIdx.x;
        int inx_point = i * kPoints;

        //int NP = nPoints;
        //int KP = kPoints;

        if( i < nPoints ) {

            
            float pointi[784];
            //float * pointi;
            for (int k = 0; k<kPoints; k++){
                pointi[k] = d_Points[inx_point + k];
            }
        
            
            // loop over other points to calculate rho
            float rhoi = 0.0;
            
            for (int j=0; j<nPoints; j++){

                float dr = 0;

                for (int k = 0; k < kPoints; k++){
                    dr += pow( d_Points[ j*kPoints + k] - pointi[k] , 2) ;
                }

                dr = sqrt(dr);

                if (dr<KERNAL_R){
                    rhoi += d_wPoints[j] * exp(- pow(dr/KERNAL_R_NORM, KERNAL_R_POWER) );
                }
            }

            d_rho[i] = rhoi;
        }
    }

    __global__ void rho_cuda_test( float *d_rho, float *d_Points, 
                              float *d_wPoints,
                              int nPoints, int kPoints, float KERNAL_R, float KERNAL_R_NORM){

        int i = blockDim.x*blockIdx.x + threadIdx.x;

        int N = nPoints;

        if( i < N ) {
            d_rho[i] = 1;
        }
    }


    

    ////////////////////////////
    // 2. get rhorank and nh+nhd 2in1
    ////////////////////////////
    __global__ void rhoranknh_cuda( int *d_rhorank, int *d_nh, float *d_nhd,
                                    float *d_Points, float *d_rho,
                                    int nPoints, int kPoints, float MAXDISTANCE){
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        int inx_point = i * kPoints;

        //int NP = nPoints;
        //int KP = kPoints;

        if( i < nPoints ) {

            
            float pointi[784];
            //float *pointi;
            for (int k = 0; k<kPoints; k++){
                pointi[k] = d_Points[inx_point + k];
            }

            float rhoi = d_rho[i];


            // loop over other points to calculate rhorank, nh,nhd
            int rhoranki = 0;
            int nhi      = i;
            float nhdi   = MAXDISTANCE;
            
            for (int j=0; j<nPoints; j++){
                // calc rhorank
                if( d_rho[j] >  rhoi ) rhoranki++;
                if( d_rho[j] == rhoi ){
                    if (j<i) rhoranki++;
                }
                
                
                // find nh
                float dr = 0;
                for (int k = 0; k < kPoints; k++){
                    dr += pow( d_Points[ j*kPoints + k] - pointi[k] , 2) ;
                }
                dr = sqrt(dr);

                // if nearer AND higher rho
                if ( (dr<nhdi) and (d_rho[j]>rhoi)){ 
                    nhdi = dr;
                    nhi  = j;
                }
            }
                
            d_rhorank[i] = rhoranki;
            d_nh[i]      = nhi;
            d_nhd[i]     = nhdi;        
        }
    }
""")
rho_cuda       = mod.get_function("rho_cuda")
rho_cuda_test  = mod.get_function("rho_cuda_test")
rhoranknh_cuda = mod.get_function("rhoranknh_cuda")
