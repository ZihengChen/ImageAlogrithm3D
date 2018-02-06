## cuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
mod = SourceModule("""

    __global__ void rho_cuda(float *d_rho,
                         int nrech, float KERNAL_R, float KERNAL_Z, float KERNAL_EXPC,
                         float *d_x, float *d_y,float *d_z, float *d_e )
    {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if(i<nrech){
            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];
    
            float rhoi = 0.0;
            for (int j=0;j<nrech; j++){
                float dr = sqrt((d_x[j]-xi)*(d_x[j]-xi) + (d_y[j]-yi)*(d_y[j]-yi));
                float dz = abs(d_z[j]-zi);
            
                if ( dz<=KERNAL_Z && dr<KERNAL_R ){ 
                    rhoi = rhoi + d_e[j] * exp(- KERNAL_EXPC*dr);
                }
            }
            d_rho[i] = rhoi;
        }
    }
    
    __global__ void rhoranknh_cuda(int *d_rhorank, int *d_nh, float *d_nhd, 
                                   int nrech, float *d_x, float *d_y,float *d_z,
                                   float *d_rho )
    {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if( i<nrech ){
            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];
            float rhoi = d_rho[i];
        
            int rhoranki = 0;
            int nhi = i;
            float nhdi = 200.0; // MAXDISTANCE = 200
        
            for (int j=0; j<nrech; j++){
                if(d_rho[j]>rhoi) rhoranki++;
                if(d_rho[j]==rhoi){
                   if (j<i) rhoranki++;
                }
               
                float drr = sqrt((d_x[j]-xi)*(d_x[j]-xi) + (d_y[j]-yi)*(d_y[j]-yi)+ (d_z[j]-zi)*(d_z[j]-zi));
                if ( (drr<nhdi) and (d_rho[j]>rhoi)){ //if nearer AND higher rho
                    nhdi = drr;
                    nhi = j;
                }
            }
            
            d_rhorank[i] = rhoranki;
            d_nh[i] = nhi;
            d_nhd[i]= nhdi;
        }
    }
    
    
    // split function for rhorank and nh
    __global__ void rhorank_cuda(int *d_rhorank, int nrech, float *d_rho )
    {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if( i<nrech ){
            float rhoi = d_rho[i];
        
            int rhoranki = 0;
            for (int j=0; j<nrech; j++){
                if(d_rho[j]>rhoi) rhoranki++;
                if(d_rho[j]==rhoi){
                    if (j<i) rhoranki++;
                }
            }
            d_rhorank[i] = rhoranki;
        }
    }
    
    __global__ void nh_cuda(int *d_nh, float *d_nhd,
                            int nrech, float *d_x, float *d_y,float *d_z,
                            float *d_rho ) 
    {

        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if( i<nrech ){
            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];
            float rhoi = d_rho[i];
        
            int nhi = i;
            float nhdi = 200.0; // MAXDISTANCE = 200
            for (int j=0; j<nrech; j++){
                float drr = sqrt((d_x[j]-xi)*(d_x[j]-xi) + (d_y[j]-yi)*(d_y[j]-yi)+ (d_z[j]-zi)*(d_z[j]-zi));
                if ( (drr<nhdi) and (d_rho[j]>rhoi)){ //if nearer
                    nhdi = drr;
                    nhi = j;
                }
            }
            d_nh[i] = nhi;
            d_nhd[i]= nhdi;
        }
    }


    



    __global__ void rho_cuda_smem( float *d_rho,
                              int nrech, float KERNAL_R, float KERNAL_Z, float KERNAL_EXPC,
                              float *d_x, float *d_y,float *d_z, float *d_e )
    {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        const int PARTWIDTH = 1024;
        __shared__ float x_shared[PARTWIDTH];
        __shared__ float y_shared[PARTWIDTH];
        __shared__ float z_shared[PARTWIDTH];
        __shared__ float e_shared[PARTWIDTH];
        if(i<nrech){

            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];

            float rhoi = 0.0;
        
            int npart = int (nrech/PARTWIDTH) ;
            if (nrech%PARTWIDTH>0) npart++;

            for(int ipart=0;ipart<npart;ipart++){

                if (ipart*PARTWIDTH + threadIdx.x < nrech ){
                    x_shared[threadIdx.x] = d_x[ipart*PARTWIDTH + threadIdx.x];
                    y_shared[threadIdx.x] = d_y[ipart*PARTWIDTH + threadIdx.x];
                    z_shared[threadIdx.x] = d_z[ipart*PARTWIDTH + threadIdx.x];
                    e_shared[threadIdx.x] = d_e[ipart*PARTWIDTH + threadIdx.x];
                }
                else{
                    x_shared[threadIdx.x] = 0;
                    y_shared[threadIdx.x] = 0;
                    z_shared[threadIdx.x] = 0;
                    e_shared[threadIdx.x] = 0;     
                }

                __syncthreads();

                for (int j=0;(j<PARTWIDTH) && (ipart * PARTWIDTH + j < nrech); j++){

                    float dr = sqrt((x_shared[j]-xi)*(x_shared[j]-xi) + 
                                    (y_shared[j]-yi)*(y_shared[j]-yi));

                    float dz = abs(z_shared[j]-zi);

                    if ( dz<=KERNAL_Z && dr<KERNAL_R ){ 
                        rhoi += e_shared[j] * exp(- KERNAL_EXPC*dr);
                    }
                }
                __syncthreads(); 
            }
        
            d_rho[i] = rhoi;
        }
    }

__global__ void rhoranknh_cuda_smem(int *d_rhorank, int *d_nh, float *d_nhd, 
                                   int nrech, float *d_x, float *d_y,float *d_z,
                                   float *d_rho )
    {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        const int PARTWIDTH = 1024;
        __shared__ float x_shared[PARTWIDTH];
        __shared__ float y_shared[PARTWIDTH];
        __shared__ float z_shared[PARTWIDTH];
        __shared__ float rho_shared[PARTWIDTH];

        if( i<nrech ){
            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];
            float rhoi = d_rho[i];
        
            int rhoranki = 0;
            int nhi = i;
            float nhdi = 200.0; // MAXDISTANCE = 200


            int npart = int (nrech/PARTWIDTH) ;
            if (nrech%PARTWIDTH>0) npart++;

            for(int ipart=0;ipart<npart;ipart++){

                if (ipart*PARTWIDTH + threadIdx.x < nrech ){
                    x_shared[threadIdx.x] = d_x[ipart*PARTWIDTH + threadIdx.x];
                    y_shared[threadIdx.x] = d_y[ipart*PARTWIDTH + threadIdx.x];
                    z_shared[threadIdx.x] = d_z[ipart*PARTWIDTH + threadIdx.x];

                    rho_shared[threadIdx.x] = d_rho[ipart*PARTWIDTH + threadIdx.x];
                }
                else{
                    x_shared[threadIdx.x] = 0;
                    y_shared[threadIdx.x] = 0;
                    z_shared[threadIdx.x] = 0;

                    rho_shared[threadIdx.x] = 0;     
                }
                __syncthreads();

        
                for (int j=0;(j<PARTWIDTH) && (ipart * PARTWIDTH + j< nrech); j++){
                    int j_global = ipart * PARTWIDTH + j ;

                    if(rho_shared[j]>rhoi) rhoranki++;
                    if(rho_shared[j]==rhoi){
                        if (j_global < i) rhoranki++;
                    }
               
                    float drr = sqrt((x_shared[j]-xi)*(x_shared[j]-xi) + 
                                     (y_shared[j]-yi)*(y_shared[j]-yi) + 
                                     (z_shared[j]-zi)*(z_shared[j]-zi));
                    // nearer AND higher rho
                    if ( (drr<nhdi) and (d_rho[j]>rhoi)){ 
                        nhdi = drr;
                        nhi = j_global;
                    }
                }
                __syncthreads(); 
            }
            
            d_rhorank[i] = rhoranki;
            d_nh[i] = nhi;
            d_nhd[i]= nhdi;
        }
    }
    
    
    
""")
rho_cuda       = mod.get_function("rho_cuda")
rhoranknh_cuda = mod.get_function("rhoranknh_cuda")
rhorank_cuda   = mod.get_function("rhorank_cuda")
nh_cuda        = mod.get_function("nh_cuda")

rho_cuda_smem        = mod.get_function("rho_cuda_smem")
rhoranknh_cuda_smem  = mod.get_function("rhoranknh_cuda_smem")




'''     
__global__ void rho_cuda_smem2(float *d_rho,
                              int nrech, float KERNAL_R, float KERNAL_Z, float KERNAL_EXPC,
                              float *d_x, float *d_y,float *d_z, float *d_e ){

        const int PARTWIDTH = 1024;
        const int NHITS = 8;

        int idx_starting = NHITS * (blockDim.x*blockIdx.x + threadIdx.x);
        
        __shared__ float x_shared[PARTWIDTH];
        __shared__ float y_shared[PARTWIDTH];
        __shared__ float z_shared[PARTWIDTH];
        __shared__ float e_shared[PARTWIDTH];

        if(idx_starting < nrech){
            float xi[NHITS];
            float yi[NHITS];
            float zi[NHITS];
            float rhoi[NHITS];

            for( int i = 0; (i<NHITS) ; ++i ){
                xi[i] = 0;
                yi[i] = 0;
                zi[i] = 0;
                rhoi[i] = 0;
            } 

            for( int i = 0; (i<NHITS) && (idx_starting+i < nrech); ++i ){
                xi[i] = d_x[idx_starting+i];
                yi[i] = d_y[idx_starting+i];
                zi[i] = d_z[idx_starting+i];
            }

            
            

            int npart = int (nrech/PARTWIDTH) ;
            if (nrech%PARTWIDTH>0) npart++;

            for(int ipart=0;ipart<npart;ipart++){
                // 1. copy partial data
                if (ipart*PARTWIDTH + threadIdx.x < nrech ){
                    x_shared[threadIdx.x] = d_x[ipart*PARTWIDTH + threadIdx.x];
                    y_shared[threadIdx.x] = d_y[ipart*PARTWIDTH + threadIdx.x];
                    z_shared[threadIdx.x] = d_z[ipart*PARTWIDTH + threadIdx.x];
                    e_shared[threadIdx.x] = d_e[ipart*PARTWIDTH + threadIdx.x];
                }
                else{
                    x_shared[threadIdx.x] = 0;
                    y_shared[threadIdx.x] = 0;
                    z_shared[threadIdx.x] = 0;
                    e_shared[threadIdx.x] = 0;     
                }   
                __syncthreads();

                // 2. compute partial result
                for( int i = 0; (i<NHITS) && (idx_starting+i < nrech); ++i ){                    
                    
                    for (int j=0; (j<PARTWIDTH) && (ipart * PARTWIDTH + j < nrech); j++){

                        float dr = sqrt(( x_shared[j] - xi[i] ) * ( x_shared[j] - xi[i] ) + 
                                        ( y_shared[j] - yi[i] ) * ( y_shared[j] - yi[i] )
                                        );

                        float dz = abs( z_shared[j] - zi[i] );

                        if ( dz<=KERNAL_Z && dr<KERNAL_R ){ 
                            rhoi[i] += e_shared[j] * exp(- KERNAL_EXPC*dr);
                        }
                    }
                }
                __syncthreads(); 
            }

            for( int i = 0; (i<NHITS) && (idx_starting+i < nrech); ++i ){
                d_rho[ idx_starting+i ] = rhoi[i];
            }
        }
    }
    '''