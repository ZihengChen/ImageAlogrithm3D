import pyopencl as cl

def openclkernel(DeviceID=0):

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[DeviceID]
    
    context = cl.Context([device])
    
    # For deviceid:
    # 0 ...... Intel-Core-i7 CPU: 0001 WI/WG
    # 1 ...... Intel-IrisPro GPU: 0512 WI/WG
    # 2 ...... NVidia-GT750M GPU: 1024 WI/WG
    if DeviceID==0:
        lsz = 1
    elif DeviceID==1:
        lsz = 512
    elif DeviceID==2:
        lsz = 1024
    else:
        lsz = 1
    print("Device: {}".format(device.name))
    print("Device MaxWorkGroupSize: {}".format(device.max_work_group_size))
    
    
    prg = cl.Program(context,"""
     ////////////////////////////
     // 1. get rho
     ////////////////////////////
     
     __kernel void rho_opencl(__global float *d_rho,
                              // input parameters
                              int nrech, 
                              __global float *d_x,
                              __global float *d_y,
                              __global float *d_z,
                              __global float *d_e
                              ){
        int i = get_group_id(0)*get_local_size(0)+get_local_id(0);
        if(i<nrech){
            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];
    
            float rhoi = 0.0;
           
            for (int j=0;j<nrech; j++){
                float dr = sqrt((d_x[j]-xi)*(d_x[j]-xi) + (d_y[j]-yi)*(d_y[j]-yi));
                float dz = (d_z[j]-zi);
                // cannot get abs(dz) work
                if (dz<0) dz=-dz; 
                //KERNAL_R,KERNAL_Z= 2,2
                //force the exp rate of kernals are 1,4 in r,z direction
                if ( dz<2.0 && dr<2.0 ){ 
                    rhoi = rhoi + d_e[j] * exp( (- dr/1.0) ) * exp( (- dz/4.0));
                    //rhoi = rhoi + d_e[j] * 1.0/(dr+1.0) * 1.0/(0.25*dz+1.0);
                    }
                }
            d_rho[i] = rhoi;
            }
        }
    
    
    
    ////////////////////////////
    // 2. get rhorank
    ////////////////////////////
    __kernel void rhorank_opencl(__global int *d_rhorank,
                                 // input parameters
                                 int nrech, 
                                 __global float *d_rho
                                 ){
        int i = get_group_id(0)*get_local_size(0)+get_local_id(0);
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
    
    
    
    ////////////////////////////
    // 3. get nh and nhd
    ////////////////////////////
    __kernel void nh_opencl(__global int *d_nh, 
                            __global float *d_nhd,
                            // input parameters
                            int nrech,
                            __global float *d_x, 
                            __global float *d_y,
                            __global float *d_z,
                            __global float *d_rho
                            ){
        int i = get_group_id(0)*get_local_size(0)+get_local_id(0);
        if(i<nrech){
            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];
            float rhoi = d_rho[i];
            
            int nhi = i;
            float nhdi = 200.0; // MAXDISTANCE = 1000
            
            for (int j=0; j<nrech; j++){
                float drr = sqrt((d_x[j]-xi)*(d_x[j]-xi) + (d_y[j]-yi)*(d_y[j]-yi)+ (d_z[j]-zi)*(d_z[j]-zi));
                //if nearer and higher
                if ( (drr<nhdi) && (d_rho[j]>rhoi)){ 
                    nhdi = drr;
                    nhi = j;
                    }
                }
            d_nh[i] = nhi;
            d_nhd[i]= nhdi;
            }
        }
    """
    ).build()
    return lsz,context,prg