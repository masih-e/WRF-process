import numpy as np 
import numba as nb


@nb.jit(nopython=True)
def calc_rdz(ph, phb, G):      #o.k. less than 1 percent
    IEND = ph.shape[2]
    JEND = ph.shape[1]
    KEND = ph.shape[0] - 1
    z_at_w = np.zeros((KEND+1, JEND, IEND))
    rdz    = np.zeros((KEND+1, JEND, IEND))
    rdzw   = np.zeros((KEND+1, JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
            for k in range(KEND+1):
                z_at_w[k, j, i] = ( ph[k, j, i] + phb[k, j, i]) / G
        for j in range(JEND):
            for k in range(KEND):
                rdzw[k, j, i] = 1.0 / ( z_at_w[k+1, j, i] - z_at_w[k, j, i] )

        for j in range(JEND):
            for k in range(1, KEND):
                rdz[k, j, i] = 2.0 / ( z_at_w[k+1, j, i] - z_at_w[k-1, j, i] )

        for j in range(ph.shape[1]):
            rdz[0, j, i] = 1./(z_at_w[1, j, i]-z_at_w[0, j, i])
    return rdz, rdzw
    
@nb.jit(nopython=True)
def calc_zx(RDX, phb, ph, G):
    IEND = ph.shape[2]
    JEND = ph.shape[1]
    KEND = ph.shape[0] - 1
    
    zx = np.zeros((KEND+1, JEND, IEND+1))
    for i in range(1, IEND):
        for j in range(JEND):
            for k in range(KEND+1): 
                zx[k, j, i] = RDX * ( ph[k, j, i] - ph[k, j, i-1] + phb[k, j, i] - phb[k, j, i-1] ) / G
                
    # if the domain is not periodic then zx is zero on the boundary
    for j in range(JEND):
        for k in range(KEND+1): 
            zx[k, j, 0] = .0
            zx[k, j, -1] = .0
                
    # if periodic
#    for j in range(JEND):
#        for k in range(KEND+1): 
#            zx[k,j, -1] = RDX * ( ph[k, j, 0] - ph[k, j, -1] + phb[k, j, 0] - phb[k, j, -1] ) / G # is this correct?
#            zx[k,j, 0]  = RDX * ( ph[k, j, 0] - ph[k, j, -1] + phb[k, j, 0] - phb[k, j, -1] ) / G
                
    return zx


@nb.jit(nopython=True)
def calc_zy(RDY, phb, ph, G):
    IEND = ph.shape[2]
    JEND = ph.shape[1]
    KEND = ph.shape[0] - 1
    
    zy = np.zeros((KEND+1, JEND+1, IEND))
    for i in range(IEND):
        for j in range(1, JEND):
            for k in range(KEND+1): 
                zy[k, j, i] = RDY * ( ph[k, j, i] - ph[k, j-1, i] +  phb[k, j, i] - phb[k, j-1, i] ) / G
                
    # if the domain is not periodic then zx is zero on the boundary
    for i in range(IEND):
        for k in range(KEND+1):
            zy[k, 0, i] = .0
            zy[k, -1, i] = .0
            
    # if periodic
#    for i in range(IEND):
#        for k in range(KEND+1):
#            zy[k,-1, i] = RDY * ( ph[k, 0, i] - ph[k, -1, i] + phb[k, 0, i] - phb[k, -1, i] ) / G # is this correct?
#            zy[k,0, i]  = RDY * ( ph[k, 0, i] - ph[k, -1, i] + phb[k, 0, i] - phb[k, -1, i] ) / G
    return zy

@nb.jit(nopython=True)
def calc_du_dx(U, msftx, msfty, msfuy, fnm, fnp, zx, rdzw, cf1, cf2, cf3, cft1, cft2, RDX):  # o.k.!
    IEND = U.shape[2] - 1
    JEND = U.shape[1]
    KEND = U.shape[0]
    
    #        ! Square the map scale factor at
    #        ! mass points
    mm = np.zeros((JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
              mm[j, i] = msftx[j, i] * msfty[j, i]
                
    #        ! hat = u/m_uy
    hat = np.zeros((KEND, JEND, IEND+1))
    for i in range(IEND+1):
        for j in range(JEND):
            for k in range(KEND):
                hat[k, j, i] = U[k, j, i] / msfuy[j, i]
                
    hatavg = np.zeros((KEND+1, JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
            for k in range(1, KEND):
                hatavg[k, j, i] = 0.5 * (fnm[k] * (hat[k,   j, i] + hat[k,   j, i+1]) + \
                                         fnp[k] * (hat[k-1, j, i] + hat[k-1, j, i+1]))

    for i in range(IEND):
        for j in range(JEND):
            # ! Surface
            hatavg[1, j, i] = 0.5 * (cf1 * hat[1, j, i] + \
                                     cf2 * hat[2, j, i] + \
                                     cf3 * hat[3, j, i] + \
                                     cf1 * hat[1, j, i+1] + \
                                     cf2 * hat[2, j, i+1] + \
                                     cf3 * hat[3, j, i+1])
            # ! Top face
            hatavg[-1, j, i] =  0.5 * (cft1 * (hat[-2, j, i] + hat[-2, j, i+1]) + \
                                       cft2 * (hat[-3, j, i] + hat[-3, j, i+1])) 
                                       
    tmp1 = np.zeros((KEND,JEND,IEND))
    for i in range(IEND): # is the index correct?
        for j in range(JEND):
            for k in range(KEND):
                tmpzx = 0.25 * (zx[k, j, i] + zx[k, j, i+1] + \
                              zx[k+1, j, i] + zx[k+1, j, i+1]) 
                tmp1[k, j, i] = (hatavg[k+1, j, i] - hatavg[k, j, i]) * tmpzx * rdzw[k, j, i]

    du_dx = np.zeros((KEND, JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
            for k in range(KEND):
                du_dx[k, j ,i] = mm[j, i] * (RDX * (hat[k, j, i+1] - hat[k, j, i]) -  \
                    tmp1[k, j, i])

    return du_dx

@nb.jit(nopython=True)
def calc_du_dy(U, msfux, msfvy, fnm, fnp, zy, rdzw, cf1, cf2, cf3, cft1, cft2, RDY):
    IEND = U.shape[2] - 1
    JEND = U.shape[1]
    KEND = U.shape[0]
    
    #        ! Square the map scale factor at
    #        ! mass points
    mm = np.ones((JEND, IEND))
    for i in range(1,IEND):
        for j in range(1,JEND):
              mm[j, i] = 0.25 * (msfux[j-1, i] + msfux[j, i]) * \
                (msfvy[j, i-1] + msfvy[j, i])

    #        ! hat = u/m_uy
    hat = np.zeros((KEND, JEND, IEND+1))
    for i in range(IEND+1):
        for j in range(JEND):
            for k in range(KEND):
                hat[k, j, i] = U[k, j, i] / msfux[j, i]
                
    hatavg = np.zeros((KEND, JEND, IEND+1))
    for i in range(IEND+1):
        for j in range(1, JEND):
            for k in range(1, KEND):
                hatavg[k, j, i] = 0.5 * (fnm[k] * (hat[k, j-1, i] + hat[k, j, i]) + \
                                         fnp[k] * (hat[k-1, j-1, i]+hat[k-1, j, i]))

    for i in range(IEND + 1):
        for j in range(1, JEND):
            # ! Surface
            hatavg[1, j, i] = 0.5 * (cf1 * hat[1, j-1, i] + \
                                     cf2 * hat[2, j-1, i] + \
                                     cf3 * hat[3, j-1, i] + \
                                     cf1 * hat[1, j, i] + \
                                     cf2 * hat[2, j, i] + \
                                     cf3 * hat[3, j, i])
            # ! Top face
            hatavg[-1, j, i] =  0.5 * (cft1 * (hat[-2, j-1, i] + hat[-2, j, i]) + \
                                       cft2 * (hat[-3, j-1, i] + hat[-3, j, i])) 
                                       
    tmp1 = np.zeros((KEND,JEND,IEND))
    for i in range(1,IEND): # is the index correct?
        for j in range(1,JEND):
            for k in range(KEND-1):
                tmpzy = 0.25 * (zy[k,   j, i-1] + zy[k,   j, i] + \
                               zy[k+1, j, i-1] + zy[k+1, j, i]) 
                tmp1[k, j, i] = (hatavg[k+1, j, i] - hatavg[k, j, i]) * \
                0.25 * tmpzy * (rdzw[k, j, i] + rdzw[k, j, i-1] + \
                                rdzw[k, j-1, i] + rdzw[k, j-1, i-1])

    du_dy = np.zeros((KEND, JEND, IEND))
    for i in range(IEND):
        for j in range(1, JEND):
            for k in range(KEND):
                du_dy[k, j ,i] = mm[j, i] * (RDY * (hat[k, j, i] - hat[k, j-1, i]) -  \
                    tmp1[k, j, i])
    return du_dy

@nb.jit(nopython=True)
def calc_dv_dx(V, msfux, msfvy, fnm, fnp, zx, rdzw, cf1, cf2, cf3, cft1, cft2, RDX):
    IEND = V.shape[2]
    JEND = V.shape[1] - 1
    KEND = V.shape[0]
    
    #        ! Square the map scale factor at
    #        ! mass points
    mm = np.ones((JEND, IEND))
    for i in range(1,IEND):
        for j in range(1,JEND):
              mm[j, i] = 0.25 * (msfux[j-1, i] + msfux[j, i]) * \
                (msfvy[j, i-1] + msfvy[j, i])
                
    hat = np.zeros((KEND, JEND + 1, IEND))
    for i in range(IEND):
        for j in range(JEND + 1):
            for k in range(KEND):
                hat[k, j, i] = V[k, j, i] / msfvy[j, i]
                
    hatavg = np.zeros((KEND, JEND+1, IEND))
    for i in range(1,IEND):
        for j in range(JEND + 1):
            for k in range(1, KEND):
                hatavg[k, j, i] = 0.5 * (fnm[k] * (hat[k, j, i-1] + hat[k, j, i]) + \
                                         fnp[k] * (hat[k-1, j, i-1]+hat[k-1, j, i]))


    for i in range(1,IEND):
        for j in range(JEND + 1):
            # ! Surface
            hatavg[1, j, i] = 0.5 * (cf1 * hat[1, j, i-1] + \
                                     cf2 * hat[2, j, i-1] + \
                                     cf3 * hat[3, j, i-1] + \
                                     cf1 * hat[1, j, i] + \
                                     cf2 * hat[2, j, i] + \
                                     cf3 * hat[3, j, i])
            # ! Top face
            hatavg[-1, j, i] =  0.5 * (cft1 * (hat[-2, j, i] + hat[-2, j, i-1]) + \
                                       cft2 * (hat[-3, j, i] + hat[-3, j, i-1])) 
            
    tmp1 = np.zeros((KEND,JEND,IEND))
    for i in range(1, IEND): 
        for j in range(1, JEND):
            for k in range(KEND-1):
                tmpzx = 0.25 * (zx[k, j-1, i] + zx[k, j, i] + \
                              zx[k+1, j-1, i] + zx[k+1, j, i]) 
                tmp1[k, j, i] = (hatavg[k+1, j, i] - hatavg[k, j, i]) * \
                0.25 * tmpzx * (rdzw[k, j, i] + rdzw[k, j-1, i] + \
                                rdzw[k, j, i-1]+ rdzw[k, j-1, i-1])

    dv_dx = np.zeros((KEND,JEND,IEND))
    for i in range(1, IEND):
        for j in range(JEND):
            for k in range(1, KEND):
                dv_dx[k, j ,i] = mm[j, i] * (RDX * (hat[k, j, i] - hat[k, j, i-1]) -  \
                    tmp1[k, j, i])
    
    return dv_dx

@nb.jit(nopython=True)
def calc_dv_dy(V, msftx, msfty, msfvx, fnm, fnp, zy, rdzw, cf1, cf2, cf3, cft1, cft2, RDY): # ok
    IEND = V.shape[2]
    JEND = V.shape[1] - 1
    KEND = V.shape[0]
    
    #        ! Square the map scale factor at
    #        ! mass points
    mm = np.zeros((JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
              mm[j, i] = msftx[j, i] * msfty[j, i]
                
    #        ! hat = u/m_uy
    hat = np.zeros((KEND, JEND+1, IEND))
    for i in range(IEND):
        for j in range(JEND+1):
            for k in range(KEND):
                hat[k, j, i] = V[k, j, i] / msfvx[j, i]
                
    hatavg = np.zeros((KEND+1, JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
            for k in range(1, KEND):
                hatavg[k, j, i] = 0.5 * (fnm[k] * (hat[k, j+1, i] + hat[k, j, i]) + \
                                         fnp[k] * (hat[k-1, j+1, i]+hat[k-1, j, i]))

    for i in range(IEND):
        for j in range(JEND):
            # ! Surface
            hatavg[1, j, i] = 0.5 * (cf1 * hat[1, j, i] + \
                                     cf2 * hat[2, j, i] + \
                                     cf3 * hat[3, j, i] + \
                                     cf1 * hat[1, j+1, i] + \
                                     cf2 * hat[2, j+1, i] + \
                                     cf3 * hat[3, j+1, i])
            # ! Top face
            hatavg[-1, j, i] =  0.5 * (cft1 * (hat[-2, j, i] + hat[-2, j+1, i]) + \
                                       cft2 * (hat[-3, j, i] + hat[-3, j+1, i])) 
                                              
    tmp1 = np.zeros((KEND,JEND,IEND))
    for i in range(IEND): 
        for j in range(JEND):
            for k in range(KEND-1):
                tmpzy = 0.25 * (zy[k,   j, i] + zy[k,   j+1, i] + \
                                zy[k+1, j, i] + zy[k+1, j+1, i]) 
                tmp1[k, j, i] = (hatavg[k+1, j, i] - hatavg[k, j, i]) * tmpzy * rdzw[k, j, i]

    dv_dy = np.zeros((KEND,JEND,IEND))
    for i in range(IEND):
        for j in range(JEND):
            for k in range(KEND):
                dv_dy[k, j ,i] = mm[j, i] * (RDY * (hat[k, j+1, i] - hat[k, j, i]) -  \
                    tmp1[k, j, i])

    return dv_dy

# 13 & 23
@nb.jit(nopython=True)
def calc_du_dz(U, rdz):
    IEND = U.shape[2] - 1
    JEND = U.shape[1]
    KEND = U.shape[0]
    
    du_dz = np.zeros((KEND+1,JEND,IEND))
    for i in range(1,IEND): # is the index correct?
        for j in range(JEND):
            for k in range(1, KEND):
                du_dz[k, j, i] = (U[k, j, i] - U[k-1, j, i]) * \
                0.5 * (rdz[k, j, i] + rdz[k, j, i-1])
    return du_dz

@nb.jit(nopython=True)
def calc_dv_dz(V, rdz):
    IEND = V.shape[2]
    JEND = V.shape[1] - 1
    KEND = V.shape[0]
    
    dv_dz = np.zeros((KEND+1,JEND,IEND))
    for i in range(IEND): # is the index correct?
        for j in range(1,JEND):
            for k in range(1, KEND):
                dv_dz[k, j, i] = (V[k, j, i] - V[k-1, j, i] ) *  \
                              0.5 * ( rdz[k, j, i] + rdz[k, j-1, i])
    return dv_dz

@nb.jit(nopython=True)
def calc_dw_dx(W, msfux, msfuy, msfty, zx, rdz, RDX):
    IEND = W.shape[2]
    JEND = W.shape[1] 
    KEND = W.shape[0] - 1
    
    #        ! Square the map scale factor at
    #        ! mass points
    mm = np.zeros((JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
              mm[j, i] = msfux[j, i] * msfuy[j, i]

    #        ! hat = u/m_uy
    hat = np.zeros((KEND+1, JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
            for k in range(KEND+1):
                hat[k, j, i] = W[k, j, i] / msfty[j, i]
                
    hatavg = np.zeros((KEND+1, JEND, IEND))
    for i in range(1,IEND):
        for j in range(JEND):
            for k in range(KEND):
                hatavg[k, j, i] = 0.25 * (  \
                              hat[k, j, i] +  \
                              hat[k+1, j, i] +  \
                              hat[k, j, i-1] +  \
                              hat[k+1, j, i-1])
                                       
    tmp1 = np.zeros((KEND+1,JEND,IEND))
    for i in range(1,IEND): # is the index correct?
        for j in range(JEND):
            for k in range(1, KEND+1):
                tmp1[k, j, i] = (hatavg[k, j, i] - hatavg[k-1, j, i]) * \
                zx[k, j, i] * 0.5 * (rdz[k, j, i] + rdz[k, j, i-1])

    dw_dx = np.zeros((KEND+1,JEND,IEND))
    for i in range(1,IEND): # is the index correct?
        for j in range(JEND):
            for k in range(1, KEND+1):
                dw_dx[k, j, i] = mm[j, i] * ( \
                    RDX * (hat[k, j, i] - hat[k, j, i-1]) - tmp1[k, j, i])

    for i in range(IEND):
        for j in range(JEND):
            dw_dx[0, j, i] = 0.0
            dw_dx[-1, j, i] = 0.0            
            
                
    return dw_dx

@nb.jit(nopython=True)
def calc_dw_dy(W, msfvx, msfvy, msftx, zy, rdz, RDY):
    IEND = W.shape[2]
    JEND = W.shape[1] 
    KEND = W.shape[0] - 1
    
    #        ! Square the map scale factor at
    #        ! mass points
    mm = np.zeros((JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
              mm[j, i] = msfvx[j, i] * msfvy[j, i]

    #        ! hat = u/m_uy
    hat = np.zeros((KEND+1, JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
            for k in range(KEND+1):
                hat[k, j, i] = W[k, j, i] / msftx[j, i]
                
    hatavg = np.zeros((KEND+1, JEND, IEND))
    for i in range(IEND):
        for j in range(1,JEND):
            for k in range(KEND):
                hatavg[k, j, i] = 0.25 * (  \
                              hat[k, j, i] +  \
                              hat[k+1, j, i] +  \
                              hat[k, j-1, i] +  \
                              hat[k+1, j-1, i])
                                       
    tmp1 = np.zeros((KEND+1,JEND,IEND))
    for i in range(IEND): # is the index correct?
        for j in range(1,JEND):
            for k in range(1, KEND+1):
                tmp1[k, j, i] = (hatavg[k, j, i] - hatavg[k-1, j, i]) * \
                zy[k, j, i] * 0.5 * (rdz[k, j-1, i] + rdz[k, j, i])

    dw_dy = np.zeros((KEND+1,JEND,IEND))
    for i in range(IEND): # is the index correct?
        for j in range(1,JEND):
            for k in range(1, KEND+1):
                dw_dy[k, j, i] = mm[j, i] * ( \
                    RDY * (hat[k, j, i] - hat[k, j-1, i]) - tmp1[k, j, i])
                
    for i in range(IEND):
        for j in range(JEND):
            dw_dy[0, j, i] = 0.0
            dw_dy[-1, j, i] = 0.0            
                                
    return dw_dy

@nb.jit(nopython=True)
def calc_dw_dz(W, rdzw):  # ok
    IEND = W.shape[2]
    JEND = W.shape[1] 
    KEND = W.shape[0] - 1
    
    dw_dz = np.zeros((KEND+1, JEND, IEND)) # I don't know why D33 is staggered? the top row is zero
#    for i in range(IEND):
#        for j in range(JEND):
#            for k in range(KEND):
#                tmp1[k, j, i] = 0.5 * (W[k + 1, j, i] + W[k, j, i])
                
#        ! Calc partial w / partial z
    for i in range(IEND):
        for j in range(JEND):
            for k in range(KEND):
                dw_dz[k, j, i] = (W[k+1, j, i] - W[k , j, i]) * rdzw[k, j, i]
    
    return dw_dz

# Theta derivatives
@nb.jit(nopython=True)
def calc_dt_dx(T, msfux, msfuy, msfty, fnm, fnp, zx, rdzw, RDX):
    IEND = T.shape[2]
    JEND = T.shape[1] 
    KEND = T.shape[0]
    
    #        ! Square the map scale factor at
    #        ! mass points
    mm = np.zeros((JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
              mm[j, i] = msfux[j, i] * msfuy[j, i]

    #        ! hat = u/m_uy
    hat = np.zeros((KEND, JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
            for k in range(KEND):
                hat[k, j, i] = T[k, j, i] / msfty[j, i]
                
    hatavg = np.zeros((KEND, JEND, IEND))
    for i in range(1,IEND):
        for j in range(JEND):
            for k in range(1, KEND):
                hatavg[k, j, i] = 0.5 * (fnm[k] * (hat[k, j, i-1] + hat[k, j, i]) + \
                                         fnp[k] * (hat[k-1, j, i-1] + hat[k-1, j, i]))
                                       
    tmp1 = np.zeros((KEND,JEND,IEND))
    for i in range(1,IEND): # is the index correct?
        for j in range(JEND):
            for k in range(1, KEND-1):
                tmpzx = 0.5 * (zx[k, j, i] + zx[k+1, j, i])
                rdzu = 2.0/(1.0/rdzw[k, j, i]+1.0/rdzw[k, j, i-1])
                tmp1[k, j, i] = tmpzx * (hatavg[k+1, j, i] - hatavg[k, j, i]) * rdzu

    tmp2 = np.zeros((KEND,JEND,IEND))
    for i in range(1,IEND): # is the index correct?
        for j in range(JEND):
            for k in range(1,KEND):
                tmp2[k, j, i] = mm[j, i] * ( \
                    RDX * (hat[k, j, i] - hat[k, j, i-1]) - tmp1[k, j, i])

    dt_dx = np.zeros((KEND,JEND,IEND))       
    for i in range(IEND-1): # is the index correct?
        for j in range(JEND):
            for k in range(1, KEND):                
                dt_dx[k, j, i]=0.5*(fnm[k]*(tmp2[k,  j, i+1] + tmp2[k  ,j, i])+  \
                                   fnp[k]*(tmp2[k-1,j, i+1] + tmp2[k-1,j, i]))
                

    for i in range(IEND):
        for j in range(JEND):
            dt_dx[0, j, i] = 0.0
            dt_dx[-1, j, i] = 0.0            
            
                
    return dt_dx

@nb.jit(nopython=True)
def calc_dt_dy(T, msfvx, msfvy, msftx, fnm, fnp, zy, rdzw, RDY):
    IEND = T.shape[2]
    JEND = T.shape[1] 
    KEND = T.shape[0]
    
    #        ! Square the map scale factor at
    #        ! mass points
    mm = np.zeros((JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
              mm[j, i] = msfvx[j, i] * msfvy[j, i]

    #        ! hat = u/m_uy
    hat = np.zeros((KEND, JEND, IEND))
    for i in range(IEND):
        for j in range(JEND):
            for k in range(KEND):
                hat[k, j, i] = T[k, j, i] / msftx[j, i]
                
    hatavg = np.zeros((KEND, JEND, IEND))
    for i in range(IEND):
        for j in range(1,JEND):
            for k in range(1, KEND):
                hatavg[k, j, i] = 0.5 * (fnm[k]*(T[k, j-1, i]+T[k,  j, i]) + \
                                         fnp[k]*(T[k-1,j-1,i]+T[k-1,j, i]))
                                       
    tmp1 = np.zeros((KEND,JEND,IEND))
    for i in range(IEND): # is the index correct?
        for j in range(1,JEND):
            for k in range(1, KEND-1):
                tmpzy = 0.5*( zy[k,j,i]+ zy[k+1,j,i])
                rdzv = 2./(1./rdzw[k,j,i] + 1./rdzw[k,j-1,i])
                tmp1[k, j, i] = tmpzy * (hatavg[k+1, j, i] - hatavg[k, j, i]) * rdzv

    tmp2 = np.zeros((KEND,JEND,IEND))
    for i in range(IEND): # is the index correct?
        for j in range(1,JEND):
            for k in range(1, KEND):
                tmp2[k, j, i] = mm[j, i] * ( \
                    RDY * (hat[k, j, i] - hat[k, j-1, i]) - tmp1[k, j, i])

    dt_dy = np.zeros((KEND,JEND,IEND))
    for i in range(IEND): # is the index correct?
        for j in range(JEND-1):
            for k in range(1, KEND):                
                dt_dy[k, j, i]=0.5*(fnm[k]*(tmp2[k,   j+1, i]+tmp2[k,   j, i])+  \
                                    fnp[k]*(tmp2[k-1, j+1, i]+tmp2[k-1, j, i]))
                
    for i in range(IEND):
        for j in range(JEND):
            dt_dy[0, j, i] = 0.0
            dt_dy[-1, j, i] = 0.0            
                                
    return dt_dy

@nb.jit(nopython=True)
def calc_dt_dz(T, rdz):
    IEND = T.shape[2]
    JEND = T.shape[1] 
    KEND = T.shape[0]
    
    dt_dz = np.zeros((KEND,JEND,IEND))
    for i in range(IEND): # is the index correct?
        for j in range(JEND):
            for k in range(1, KEND):
                dt_dz[k, j, i] = (T[k, j, i] - T[k-1, j, i]) * \
                 rdz[k, j, i]
    return dt_dz

@nb.jit(nopython=True)
def calc_l(z, q, rdz, KARMAN):
    IEND = z.shape[2]
    JEND = z.shape[1] 
    KEND = z.shape[0] - 1
    
    #l_master = np.zeros((KEND, JEND, IEND))
    lt = np.zeros((JEND, IEND))
    for j in range(JEND):
        for i in range(IEND):

            l0_num = np.zeros(KEND)
            l0_den = np.zeros(KEND)
            z_profile = z[:,j,i]-z[0,j,i]
            for k in range(KEND):


                q_dz = q[k, j, i] * rdz[k, j, i]
                l0_num[k] = q_dz * z_profile[k]
                l0_den[k] = q_dz
                


            l0 = 0.1 * l0_num.sum() / max(l0_den.sum(), 1e-6)
            lt[j, i] = l0 / .1
            # Calculates master length scale
            #for k in range(KEND):
            #    l_master[k, j, i] = l0 * KARMAN * z_profile[k] / (KARMAN * z_profile[k] + l0)
    return lt
