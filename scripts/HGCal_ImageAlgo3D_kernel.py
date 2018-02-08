## cuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
mod = SourceModule("""

    __global__ void rho_cuda(float *d_rho,
                             float *d_x, float *d_y,float *d_z, float *d_e,
                             int N, float KERNAL_R, float KERNAL_R_NORM, float KERNAL_R_POWER, float KERNAL_Z
                          )
    {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if(i<N){
            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];
    
            float rhoi = 0.0;
            for (int j=0;j<N; j++){
                float dr = sqrt( pow( (d_x[j]-xi), 2) + pow( (d_y[j]-yi), 2) );
                float dz = abs(d_z[j]-zi);
            
                if ( dz<=KERNAL_Z && dr<KERNAL_R ){ 
                    rhoi = rhoi + d_e[j] * exp(- pow(dr/KERNAL_R_NORM, KERNAL_R_POWER)  );
                }
            }
            d_rho[i] = rhoi;
        }
    }
    
    __global__ void rhoranknh_cuda(int *d_rhorank, int *d_nh, float *d_nhd, 
                                   float *d_x, float *d_y,float *d_z, float *d_rho,
                                   int N, float MAXDISTANCE)
    {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if( i<N ){
            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];
            float rhoi = d_rho[i];
        
            int rhoranki = 0;
            int nhi = i;
            float nhdi = MAXDISTANCE;
        
            for (int j=0; j<N; j++){
                if(d_rho[j]>rhoi) rhoranki++;
                if(d_rho[j]==rhoi){
                   if (j<i) rhoranki++;
                }
               
                float drr = sqrt( pow(d_x[j]-xi,2) + pow(d_y[j]-yi,2) + pow(d_z[j]-zi,2) );
                //if nearer AND higher rho
                if ( (drr<nhdi) and (d_rho[j]>rhoi)){ 
                    nhdi = drr;
                    nhi = j;
                }
            }
            
            d_rhorank[i] = rhoranki;
            d_nh[i]      = nhi;
            d_nhd[i]     = nhdi;
        }
    }
    

""")
rho_cuda       = mod.get_function("rho_cuda")
rhoranknh_cuda = mod.get_function("rhoranknh_cuda")






# other code using shared memeory
# do not spend time on them at this stage.
'''   



    __global__ void rho_cuda_smem( float *d_rho,
                              int N, float KERNAL_R, float KERNAL_Z, float KERNAL_EXPC,
                              float *d_x, float *d_y,float *d_z, float *d_e )
    {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        const int PARTWIDTH = 1024;
        __shared__ float x_shared[PARTWIDTH];
        __shared__ float y_shared[PARTWIDTH];
        __shared__ float z_shared[PARTWIDTH];
        __shared__ float e_shared[PARTWIDTH];
        if(i<N){

            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];

            float rhoi = 0.0;
        
            int npart = int (N/PARTWIDTH) ;
            if (N%PARTWIDTH>0) npart++;

            for(int ipart=0;ipart<npart;ipart++){

                if (ipart*PARTWIDTH + threadIdx.x < N ){
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

                for (int j=0;(j<PARTWIDTH) && (ipart * PARTWIDTH + j < N); j++){

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
                                   int N, float *d_x, float *d_y,float *d_z,
                                   float *d_rho )
    {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        const int PARTWIDTH = 1024;
        __shared__ float x_shared[PARTWIDTH];
        __shared__ float y_shared[PARTWIDTH];
        __shared__ float z_shared[PARTWIDTH];
        __shared__ float rho_shared[PARTWIDTH];

        if( i<N ){
            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];
            float rhoi = d_rho[i];
        
            int rhoranki = 0;
            int nhi = i;
            float nhdi = 200.0; // MAXDISTANCE = 200


            int npart = int (N/PARTWIDTH) ;
            if (N%PARTWIDTH>0) npart++;

            for(int ipart=0;ipart<npart;ipart++){

                if (ipart*PARTWIDTH + threadIdx.x < N ){
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

        
                for (int j=0;(j<PARTWIDTH) && (ipart * PARTWIDTH + j< N); j++){
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



    
    // split function for rhorank and nh
    __global__ void rhorank_cuda(int *d_rhorank, int N, float *d_rho )
    {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if( i<N ){
            float rhoi = d_rho[i];
        
            int rhoranki = 0;
            for (int j=0; j<N; j++){
                if(d_rho[j]>rhoi) rhoranki++;
                if(d_rho[j]==rhoi){
                    if (j<i) rhoranki++;
                }
            }
            d_rhorank[i] = rhoranki;
        }
    }
    
    __global__ void nh_cuda(int *d_nh, float *d_nhd,
                            int N, float *d_x, float *d_y,float *d_z,
                            float *d_rho ) 
    {

        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if( i<N ){
            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];
            float rhoi = d_rho[i];
        
            int nhi = i;
            float nhdi = 200.0; // MAXDISTANCE = 200
            for (int j=0; j<N; j++){
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


__global__ void rho_cuda_smem2(float *d_rho,
                              int N, float KERNAL_R, float KERNAL_Z, float KERNAL_EXPC,
                              float *d_x, float *d_y,float *d_z, float *d_e ){

        const int PARTWIDTH = 1024;
        const int NHITS = 8;

        int idx_starting = NHITS * (blockDim.x*blockIdx.x + threadIdx.x);
        
        __shared__ float x_shared[PARTWIDTH];
        __shared__ float y_shared[PARTWIDTH];
        __shared__ float z_shared[PARTWIDTH];
        __shared__ float e_shared[PARTWIDTH];

        if(idx_starting < N){
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

            for( int i = 0; (i<NHITS) && (idx_starting+i < N); ++i ){
                xi[i] = d_x[idx_starting+i];
                yi[i] = d_y[idx_starting+i];
                zi[i] = d_z[idx_starting+i];
            }

            
            

            int npart = int (N/PARTWIDTH) ;
            if (N%PARTWIDTH>0) npart++;

            for(int ipart=0;ipart<npart;ipart++){
                // 1. copy partial data
                if (ipart*PARTWIDTH + threadIdx.x < N ){
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
                for( int i = 0; (i<NHITS) && (idx_starting+i < N); ++i ){                    
                    
                    for (int j=0; (j<PARTWIDTH) && (ipart * PARTWIDTH + j < N); j++){

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

            for( int i = 0; (i<NHITS) && (idx_starting+i < N); ++i ){
                d_rho[ idx_starting+i ] = rhoi[i];
            }
        }
    }
    '''