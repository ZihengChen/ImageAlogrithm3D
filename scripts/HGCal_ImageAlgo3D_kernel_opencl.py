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
    #print("Device: {}".format(device.name))
    #print("Device MaxWorkGroupSize: {}".format(device.max_work_group_size))
    
    
    prg = cl.Program(context,"""
    ////////////////////////////
    // 1. get rho
    ////////////////////////////
     
    __kernel void rho_opencl(__global float *d_rho,
                              // input parameters
                              __global float *d_x,
                              __global float *d_y,
                              __global float *d_z,
                              __global float *d_e,
                              int N, float KERNAL_R, float KERNAL_R_NORM, float KERNAL_R_POWER, float KERNAL_Z
                              )
    {
        int i = get_group_id(0)*get_local_size(0)+get_local_id(0);
        if(i<N){
            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];
    
            float rhoi = 0.0;
           
            for (int j=0;j<N; j++){
                float dr = sqrt((d_x[j]-xi)*(d_x[j]-xi) + (d_y[j]-yi)*(d_y[j]-yi));
                float dz = (d_z[j]-zi);
                // cannot get abs(dz) work
                if (dz<0) dz=-dz; 


                if ( dz<=KERNAL_Z && dr<KERNAL_R ){ 
                    // on some device e.g. Intel Iris Pro, function exp() is not defined
                    // use Tylor expansion for exp() instead

                    float d = pow(dr/KERNAL_R_NORM, KERNAL_R_POWER);
                    

                    float exp = 1 / (1 + d + d*d/2 + d*d*d/6 + d*d*d*d/24);
                    rhoi = rhoi + d_e[j] * exp;
                }
            }
            d_rho[i] = rhoi;
        }
    }
    
    
    
        
    ////////////////////////////
    // 4. get rhorank and nh+nhd 2in1
    ////////////////////////////
    __kernel void rhoranknh_opencl(__global int *d_rhorank,
                                   __global int *d_nh, 
                                   __global float *d_nhd,
                                   // input parameters
                                   __global float *d_x, 
                                   __global float *d_y,
                                   __global float *d_z,
                                   __global float *d_rho,
                                   int N, float MAXDISTANCE
                                   )
    {
        int i = get_group_id(0)*get_local_size(0)+get_local_id(0);
        if(i<N){
            float xi = d_x[i];
            float yi = d_y[i];
            float zi = d_z[i];
            float rhoi = d_rho[i];
            
            int rhoranki = 0;
            int nhi = i;
            float nhdi = MAXDISTANCE; // MAXDISTANCE = 200
            
            for (int j=0; j<N; j++){
                // rhorank
                if(d_rho[j]>rhoi) rhoranki++;
                if(d_rho[j]==rhoi){
                    if (j<i) rhoranki++;
                }
                
           
                // nh and nhd
                float drr = sqrt((d_x[j]-xi)*(d_x[j]-xi) + (d_y[j]-yi)*(d_y[j]-yi)+ (d_z[j]-zi)*(d_z[j]-zi));
                //if nearer and higher
                if ( (drr<nhdi) && (d_rho[j]>rhoi)){ 
                    nhdi = drr;
                    nhi = j;
                }
            }
            
            d_rhorank[i] = rhoranki;
            d_nh[i]      = nhi;
            d_nhd[i]     = nhdi;
        }
    }

    """
    ).build()
    return lsz,context,prg


