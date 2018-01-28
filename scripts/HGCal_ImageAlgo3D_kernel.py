## cuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
mod = SourceModule("""
    __global__ void rho_cuda( float *d_rho,
                              int nrech, float KERNAL_R, float KERNAL_Z, float KERNAL_EXPC,
                              float *d_x, float *d_y,float *d_z, float *d_e ){
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
                // on some device e.g. Intel Iris Pro, function exp() is not defined
                // use Tylor expansion for exp() instead
                // rhoi = rhoi + d_e[j] * exp(- dr/1.0);
                float d   = KERNAL_EXPC*dr;
                float exp = 1/(1+d+d*d/2+d*d*d/6+d*d*d*d/24);
                rhoi = rhoi + d_e[j] * exp;
                }
            }
        d_rho[i] = rhoi;
        }
    }
    
    __global__ void rhoranknh_cuda(int *d_rhorank, int *d_nh, float *d_nhd, 
                                   int nrech, float *d_x, float *d_y,float *d_z,
                                   float *d_rho ){
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
    __global__ void rhorank_cuda(int *d_rhorank, int nrech, float *d_rho ){
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
                            float *d_rho ){
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
    
    
""")
rho_cuda       = mod.get_function("rho_cuda")
rhoranknh_cuda = mod.get_function("rhoranknh_cuda")
rhorank_cuda   = mod.get_function("rhorank_cuda")
nh_cuda        = mod.get_function("nh_cuda")