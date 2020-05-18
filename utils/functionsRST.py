import numpy as np
from numba import jit



#@jit
def calcRST(X, kv, density):

    tmp = int(np.size(kv, 0))
    H = np.zeros((tmp, 6))

    for i in range(np.size(kv, 0)):
        H[i, :] = [kv[i, 0] ** 2, kv[i, 1] ** 2, kv[i, 2] ** 2,
                   2 * kv[i, 0] * kv[i, 1], 2 * kv[i, 0] * kv[i, 2], 2 * kv[i, 1] * kv[i, 2]]

    # H dims: nEnc x 6
    kv_magsq = np.reshape(np.sum(kv ** 2, 1), (6, 1))
    Hdir = H / kv_magsq;
    Hinv = np.linalg.pinv(Hdir);

    ivsd = 2 * np.log(np.expand_dims(np.abs(X[:, :, :, :, 0]), 4) / np.abs(X[:, :, :, :, 1:]))
    ivsd = ivsd / np.reshape(kv_magsq, (1, 1, 1, 1, np.size(kv, 0)));

    # ivsd shape: Nvoxels x N_encodings
    ivsd = np.reshape(ivsd, (np.prod(ivsd.shape[:4]), np.size(ivsd, 4)), order='F');
    RST = ivsd.copy()

    # todo: replace for loop with matmul
    for i in range(np.size(ivsd, 0)):
        for j in range(np.size(ivsd, 1)):
            RST[i, j] = np.sum(ivsd[i, :] * (Hinv[j, :]))  # np.transpose(Hinv[:,j]))

    sz = np.shape(X);

    #    RST = np.matmul(ivsd,np.linalg.pinv(Hdir) );

    szn = np.array(sz);
    szn[4] = np.size(H, 1);
    RST = density* np.reshape(RST, szn, order='F');

    return RST


#@jit
def calcRST_ivsdNB(ivsd, density):
    # input:
    #   ivsd: intra voxel standard deviations (i.e. sigma, not sigma^2)
    #   density: fluid density (for blood or whatever fluid was used)
    # returns Reynols Stress Tensor, i.e. densitity * [u'u', v'v', w'w', u'v',u'w',v'w']


    kv = np.array([[1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [1., 1., 0.],
                       [1., 0., 1.],
                       [0., 1., 1.]])

    var = ivsd.copy()**2

    var[0, 0, 0, 0, :] = np.array([10, 0, 1, 3, 2, 0])  # for testing only!!

    RST2 = np.zeros_like(var)
    RST2[:, :, :, :, 0] = var[:, :, :, :, 0]
    RST2[:, :, :, :, 1] = var[:, :, :, :, 1]
    RST2[:, :, :, :, 2] = var[:, :, :, :, 2]
    RST2[:, :, :, :, 3] = var[:, :, :, :, 3] - 0.5 * (var[:, :, :, :, 0] + var[:, :, :, :, 1])
    RST2[:, :, :, :, 4] = var[:, :, :, :, 4] - 0.5 * (var[:, :, :, :, 0] + var[:, :, :, :, 2])
    RST2[:, :, :, :, 5] = var[:, :, :, :, 5] - 0.5 * (var[:, :, :, :, 1] + var[:, :, :, :, 2])


#####
    #     sz0 = np.shape(var)
    #
    #     H = np.zeros((np.size(kv, 0), 6))
    #
    #     for i in range(np.size(kv, 0)):
    #         H[i, :] = [kv[i, 0] ** 2, kv[i, 1] ** 2, kv[i, 2] ** 2,
    #                    2 * kv[i, 0] * kv[i, 1], 2 * kv[i, 0] * kv[i, 2], 2 * kv[i, 1] * kv[i, 2]]
    #
    #     #   print('H:' , H)
    #     # H dims: nEnc x 6
    #     kv_magsq = np.reshape(np.sum(kv ** 2, 1), (6, 1))
    #     Hdir = H / kv_magsq;
    #
    #
    #
    #     Hinv = np.linalg.pinv(Hdir);
    #
    #     Hinv = np.around(Hinv, decimals=2)
    #
    #     sz = np.array(np.shape(var)).astype(int)
    #
    #
    #
    #     var_prev = var.copy()
    #
    #
    #
    # #    var = np.reshape(np.transpose(var,[4,0,1,2,3]), (np.size(var, 4),np.prod(var.shape[:4])), order='F');
    #
    #     # var shape: Nvoxels x N_encodings
    #     var = np.reshape(var, (np.prod(var.shape[:4]), np.size(var, 4)), order='F');
    # #    print('var_prev', np.around(var_prev[0, 0, 0, 0, :] , decimals=2))
    # #    print('var', np.around(var[0,:] , decimals=2))
    #
    #
    #
    #     assert ((var_prev[0, 0, 0, 0, :] == var[0,:]).all())
    #
    #     RST = np.zeros_like(var)
    #
    #
    #     print(np.around(np.transpose(var)[:,0], decimals=2))
    #
    #     var[100, :] = np.array([10,0,1,3,2,0])
    #
    #     RST = np.matmul(var, np.transpose(Hinv))
    #
    #     #print(RST.shape)
    #
    #     print('Hinv', np.around(  Hinv[0, :] , decimals=2))
    #     print('var', np.around( var[100,:] , decimals=2))
    #     print('RST', np.around( RST[0,:] , decimals=2))
    #
    #     print('RST2', np.around( RST2[0, 0, 0, 0, :] , decimals=2))
    #
    # #    [:, :, :, :, 0]
    #
    # #    print(RST[:, 0])
    #
    #     '''
    #         #check if reshape worked the right way
    #         print(var_prev[0,0,0,0,:],var[:,0])
    #
    #         assert((var_prev[0,0,0,0,:] == var[:,0]).all() )
    #
    #         RST = np.matmul(Hinv, var );
    #
    #         print(np.matmul(Hinv[0,None,:], var))
    #
    #
    #         print('Hinv[5,:]',Hinv[5,:])
    #         RST[0,:] = np.sum(Hinv[0,None,:]* np.transpose(var),1) #np.matmul(Hinv[0,None,:], var);
    #         RST[1,:] = np.sum(Hinv[1,None,:]* np.transpose(var),1) #np.matmul(Hinv[1,None,:], var);
    #         RST[2,:] = np.sum(Hinv[2,None,:]* np.transpose(var),1) #np.matmul(Hinv[2,None,:], var);
    #         RST[3,:] = np.sum(Hinv[3,None,:]* np.transpose(var),1) #np.matmul(Hinv[3,None,:], var);
    #         RST[4,:] = np.sum(Hinv[4,None,:]* np.transpose(var),1)  #np.matmul(Hinv[4,None,:], var);
    #         RST[5:,] = np.sum(Hinv[5,None,:]* np.transpose(var),1) #np.matmul(Hinv[5,None,:], var);
    #
    #         RST_dbg = RST.copy()
    #         RST = np.reshape(RST, np.append(sz[4], sz[:4]))
    #         print('RST shape', RST.shape)
    #         RST = np.transpose(RST, [1, 2, 3,4, 0])
    #
    #         assert((RST[0,0,0,0,:] == RST_dbg[:,0]).all() )
    #     #    print(RST[0,0,0,0,:], RST_dbg[:,0])
    #
    #     '''
    #
    #
    #     szn = np.array(sz);
    #     szn[4] = np.size(H, 1);
    #     RST = density * np.reshape(RST, szn, order='F');
    #
    #     print('mean diff: ',np.mean(np.abs(RST.flatten() - RST2.flatten())))

       # print(RST[0, 0, 0, 0, :], RST2[0, 0, 0, 0, :])
        # check if we did all the reshapes etc. right
    #    assert ((RST.flatten() == RST2.flatten()).all())
        #RST = density* RST#np.reshape(RST, szn, order='F');
########################


    RST2 = density*RST2

    return RST2






def calcPrincipalStresses(RST):
    # execute only for loop with numba, as it has some issues with np.shape command
    @jit(nopython=True)
    def calcPS(ps, RST):
        for i in range(len(RST[0, :])):
            RST_curr = np.array([[RST[0, i], RST[3, i], RST[4, i]],
                                 [RST[3, i], RST[1, i], RST[5, i]],
                                 [RST[4, i], RST[5, i], RST[2, i]]])

            w = np.linalg.eigvalsh(RST_curr)
            ps[i, :] = w
        return ps

    sz = np.array(np.shape(RST))
    sz[4] = 3
    RST = np.transpose(RST, (4, 0, 1, 2, 3))
    RST = np.reshape(RST, (6, np.size(RST[0, :, :, :, :])))

    ps = np.zeros((np.size(RST, 1), 3))

    ps = calcPS(ps, RST)

    sz[4] = 3;
    ps = np.reshape(ps, sz);

    return ps

def calcLaminarStress(vel,res,mu = 8.89E-4):
    #laminar viscous stress: tau_ij =  mu * ( d u_i / d x_j + d u_j/ d x_i)
    #mu = dynamic viscosity
    tau_xy = np.gradient(vel[:,:,:,:,0],res[0],axis=1) + np.gradient(vel[:,:,:,:,1],res[1],axis=0)
    tau_xz = np.gradient(vel[:,:,:,:,2],res[2],axis=0) + np.gradient(vel[:,:,:,:,0],res[0],axis=2)
    tau_yz = np.gradient(vel[:,:,:,:,1],res[1],axis=2) + np.gradient(vel[:,:,:,:,2],res[2],axis=1)

    return mu*tau_xy,mu*tau_xz,mu*tau_yz


def calcTVSSsq(vel,RST,res,density,mu):
    #square of Turbulent Viscous Shear Stress: tau_TVSS**2 = 0.5 * density * viscosity * RST_ij * S_ij; with the strain rate tesnor S_ij


    dvxdx = np.gradient(vel[:, :, :, :, 0], res[0],axis=0)
    dvxdy = np.gradient(vel[:, :, :, :, 0], res[1], axis=1)
    dvxdz = np.gradient(vel[:, :, :, :, 0], res[2], axis=2)

    dvydx = np.gradient(vel[:, :, :, :, 1], res[0], axis=0)
    dvydy = np.gradient(vel[:, :, :, :, 1], res[1], axis=1)
    dvydz = np.gradient(vel[:, :, :, :, 1], res[2], axis=2)

    dvzdx = np.gradient(vel[:, :, :, :, 2], res[0], axis=0)
    dvzdy = np.gradient(vel[:, :, :, :, 2], res[1], axis=1)
    dvzdz = np.gradient(vel[:, :, :, :, 2], res[2], axis=2)

    # RST was already scaled with density, therefore we do not need to multiply by it here.
    TVSS_xx =  dvxdx * RST[:,:,:,:,0]
    TVSS_yy =  dvydy * RST[:,:,:,:,1]
    TVSS_zz =  dvzdz * RST[:,:,:,:,2]

    TVSS_xy =  (dvxdy + dvydx) * RST[:,:,:,:,3] #omit factor 0.5 because the term occurs twice in final sum
    TVSS_xz =  (dvxdz + dvzdx) * RST[:,:,:,:,4]
    TVSS_yz =  (dvydz + dvzdy) * RST[:,:,:,:,5]


    return mu * (TVSS_xx+TVSS_yy+TVSS_zz+TVSS_xy+TVSS_xz)
