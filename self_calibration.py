import os
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import sys, os, glob, time
import gc
from astropy.io import ascii
import random
import copy

# draw random realization
def get_cp_array(cp_, covij_, N_sample_, N_theta_, N_MC_):    
    cp_array = np.zeros( (N_sample_, N_sample_, N_theta_, N_MC_) )
    for MC in range(N_MC_): 
        cp_MC = np.zeros_like(cp_)
        # 生成一组观测
        for i in range(N_sample_):
            # 随机抽样
            cp_MC[i,i,:] = np.random.multivariate_normal(cp_[i,i,:], covij_[i,i,:,:])     
            for j in range(i+1,N_sample_):
                cp_MC[i,j,:] = np.random.multivariate_normal(cp_[i,j,:], covij_[i,j,:,:])
                cp_MC[j,i,:] = copy.deepcopy(cp_MC[i,j,:]) 
        cp_MC[cp_MC<0] = 1e-5 #非负
        # 存入大矩阵
        cp_array[:,:,:,MC] = cp_MC
        
    return cp_array

# generate a random diagnoal dominate matrix
def diag_domin_pij_init2(N_sample):    
    fij = np.random.rand(N_sample, N_sample)
    for i in range(N_sample):
        fij[i,i] = 0.6+0.3*np.random.rand()
        a = np.random.rand(N_sample-1)
        a = a/np.sum(a)*(1-fij[i,i])
        a_sort = np.sort(a)[::-1] # big to small
        if i == 0:
            fij[i+1:,i] = a_sort
        elif i == N_sample-1:
            fij[:i, i] = a_sort[::-1]
        else:
            a_sort_ = np.array([a_sort[0],a_sort[1]])
            np.random.shuffle(a_sort_)
            fij[i-1, i] = a_sort_[0]
            fij[i+1, i] = a_sort_[1]
            indices = list(range(i-1))+list(range(i+2,N_sample))
            a_sort_ = a_sort[2:]
            np.random.shuffle(a_sort_)
            fij[indices, i] = a_sort_
    return fij


@jit(nopython=True)
def update_EM(Yk,W,Ck,nbin,nsample):
    W1=np.copy(W)
    Rys = np.zeros_like(Yk)
    for iq in range(0,nbin):
        Ainv= np.linalg.pinv(W1)
        Ck[:,:,iq] = np.dot(np.dot(Ainv,Yk[:,:,iq]),Ainv.T)
        Rys[:,:,iq] = np.dot(Yk[:,:,iq],Ainv.T) #
        Rsst = np.sum(Ck,axis=2)  # 
        Ryst = np.sum(Rys,axis=2)
        W1= np.copy(np.abs(np.dot(Ryst,np.linalg.pinv(Rsst)))) # 

        norm= np.sum(W1,axis=1) # don't change it
        for i in range(0, nsample):# don't change it
            W1[i,:] =  (W1[i,:]+1e-10)/(norm[i]+1e-10) # don't change it
#             W1[i,:] =  (W1[i,:]+1e-10)/(norm[i]+1e-10) # don't change it
    for iq in range(0,nbin):
        Ck[:,:,iq] =np.copy(np.diag(np.diag(np.abs(Ck[:,:,iq]))))
    return W1,Ck

@jit(nopython=True)
def fixed_point_based_J(fij, cp, cr, N_max_interation, N_theta, N_sample,covij):
    J_dummy = []
    chi2_dummy = []
    fij_dummy = []
    for i in range(N_max_interation):
        fij_input = fij.copy() 
        w, ck = update_EM(cp, fij_input.transpose(),cr,N_theta,N_sample)
        fij = w.transpose()
        cr = ck.copy()
        
        fij_dummy.append(fij)

        # 计算 J
        
        J = 0.
        cp_model = np.zeros_like(cp)
        for j in range(N_theta):
            cp_model[:,:,j] = np.dot(np.dot(fij.transpose(),np.ascontiguousarray(cr[:,:,j])), fij)
            J = J + 0.5*np.linalg.norm( cp[:,:,j] - cp_model[:,:,j] )**2
        J_dummy.append(J)
                
        # 计算\chi^2
        chi2 = 0.
        for ii in range(N_sample):
            for j in range(ii,N_sample):
                cp_diff = cp_model[ii,j,:]-cp[ii,j,:]
                chi2 = chi2 + np.dot(cp_diff, np.linalg.solve(covij[ii,j,:,:],cp_diff))
        
        chi2_dummy.append(chi2)
        
        
        if i > 3: 
            # stop criterion 
            if J > 1.5*(J_dummy[i-3]+J_dummy[i-2]+J_dummy[i-1])/3: # 1.5 times larger than last three J
                print('==> Algorithm 1 breaks at iteration', i)
                break
    
    return fij_dummy, chi2_dummy, J_dummy


#------------------------Algorithm2----------------
# update C^{gg,R}_l, Eq.A13 Zhang+2017
@jit(nopython=True)
def updatecr(cp, w, N_theta, N_sample):
    cr_ = np.zeros_like(cp)
    for i in range(N_theta):
        u = np.zeros( ( N_sample*N_sample, N_sample )  )
        vec_vl = np.ravel(cp[:,:,i].transpose())
        for j in range(N_sample):
            u[:,j] = np.kron(w[:,j], w[:,j])
        cl = np.dot(np.linalg.solve(np.dot(u.transpose(), u), u.transpose()), vec_vl)
        np.fill_diagonal(cr_[:,:,i],np.abs(cl))

    return cr_

# update w, Eq.A11
# vl = cp
@jit(nopython=True)
def updatew(cp, cr, w, N_theta, N_sample):
    hl = np.zeros_like(cr)
    Nabla_minus = np.zeros( ( N_sample, N_sample )  )
    Nabla_plus = np.zeros( ( N_sample, N_sample )  )
    new_w = np.zeros_like(w)
    for i in range(N_theta):
        hl[:,:,i] = np.dot(np.ascontiguousarray(cr[:,:,i]),w.transpose()) # Pr = W.T
        Nabla_minus = Nabla_minus + np.dot(np.ascontiguousarray(cp[:,:,i]), np.ascontiguousarray(hl[:,:,i].transpose()))
        Nabla_plus = Nabla_plus + np.dot(np.dot(w,np.ascontiguousarray(hl[:,:,i])),np.ascontiguousarray(hl[:,:,i].transpose()))
    
    A = np.zeros( ( N_sample, N_sample )  )
    B = np.zeros( ( N_sample, N_sample )  )
    for i in range(N_sample):
        A[i,:] = np.sum(w[i,:]/Nabla_plus[i,:])
        B[i,:] = np.sum(w[i,:]*Nabla_minus[i,:]/Nabla_plus[i,:])
    
    for i in range(N_sample):
        for j in range(N_sample):
            numerator = Nabla_minus[i,j]*A[i,j] + 1 - B[i,j]
            denominator = Nabla_plus[i,j]*A[i,j]
            new_w[i,j] = w[i,j]*numerator/denominator
    
    if np.any(new_w < 0):
        for i in range(N_sample):
            for j in range(N_sample):
                numerator = Nabla_minus[i,j]*A[i,j] + 1 
                denominator = Nabla_plus[i,j]*A[i,j] + B[i,j]
                new_w[i,j] = w[i,j]*numerator/denominator        
    
    return new_w

@jit(nopython=True)
def updatehl(cr,w,N_theta):
    hl = np.zeros_like(cr)
    for i in range(N_theta):
        hl[:,:,i] = np.dot(np.ascontiguousarray(cr[:,:,i]), w.transpose())
    return hl

@jit(nopython=True)
def NMF_w_theta_J(fij,cp,cr,N_max_interation, N_theta, N_sample, covij):
    J_dummy = []
    chi2_dummy = []
    fij_dummy = []
    cp_model_dummy = []
    cr_dummy = []
    
    for i in range(N_max_interation):
        fij_input = fij.copy()
        w = updatew(cp, cr, fij_input.transpose(), N_theta, N_sample)
        hl = updatehl(cr, w, N_theta)
        cr = updatecr(cp, w, N_theta, N_sample)
        fij = w.transpose()
        
        fij_dummy.append(fij)
        
        # 计算 J
        J = 0.
        cp_model = np.zeros_like(cp)
        for j in range(N_theta):
            cp_model[:,:,j] = np.dot(np.dot(fij.transpose(),np.ascontiguousarray(cr[:,:,j])), fij)
            J = J + 0.5*np.linalg.norm( cp[:,:,j] - cp_model[:,:,j] )**2        
        
        J_dummy.append(J)
        cp_model_dummy.append(cp_model)
        cr_dummy.append(cr)
        
        # 计算\chi^2
        chi2 = 0.
        for ii in range(N_sample):
            for j in range(ii,N_sample):
                cp_diff = cp_model[ii,j,:]-cp[ii,j,:]
                chi2 = chi2 + np.dot(cp_diff, np.linalg.solve(covij[ii,j,:,:],cp_diff))
        
        chi2_dummy.append(chi2)
        
        if i > 3: 
            # stop criterion 
            if J > 1.5*(J_dummy[i-3]+J_dummy[i-2]+J_dummy[i-1])/3: # 1.5 times larger than last three J
                print('==> Algorithm 1 breaks at iteration', i)
                break   

    return fij_dummy, chi2_dummy, J_dummy, cp_model_dummy,cr_dummy

def run_self_cali(cp, cov, N_first_theta, N_last_theta, N_sample, N_MC = 1):
    cp = cp[:,:, N_first_theta:N_last_theta]
    covij = cov[:,:, N_first_theta:N_last_theta,N_first_theta:N_last_theta]
    _, _, N_theta = cp.shape
    if N_MC > 1:
        # 产生100组data realization
        cp_array = get_cp_array(cp, covij, N_sample, N_theta, N_MC)
    else:
        cp_array = cp[:,:,:, np.newaxis]

    cr = np.zeros_like(cp)
    # MC 
    fij_array_minJ = np.zeros((N_sample, N_sample, N_MC))
    fij_array_minchi2 = np.zeros((N_sample, N_sample, N_MC))
    cr_array_minJ = np.zeros_like(cp_array)
    cr_array_minchi2 = np.zeros_like(cp_array)
    cp_model_array_minJ = np.zeros_like(cp_array)
    cp_model_array_minchi2 = np.zeros_like(cp_array)
    J_array = np.zeros(N_MC)
    chi2_array = np.zeros(N_MC)

    for MC in range(N_MC):
        print('===> MC #', MC)
        cp_MC = cp_array[:,:,:,MC]    

        # algorithm1, fixed-point-based
        # 完全随机产生一些P作为初始条件，跑算法1，直到J增加为止, 选个最小J对应的P作为输入算法2
        pij = diag_domin_pij_init2(N_sample) # 随机赋值,对角占优矩阵
        for i in range(N_theta):
            cr[:,:,i] = np.abs(np.matmul(np.linalg.solve(pij.transpose(), cp_MC[:,:,i]), np.linalg.pinv(pij) ))  # 初始化cr
        fij_a1, chi2_a1, J_a1 = fixed_point_based_J(pij, cp_MC, cr, N_max_interation1, N_theta, N_sample, covij)
    #   print('===> fixed-point-based min J=',np.min(J_a1))

        #--------------algorithm2, NMF
        fij_a1_min = fij_a1[np.argmin(chi2_a1)] # initial guess for algorithm2
        fij_a1_min = np.asarray(fij_a1_min, order='C') # 改变在内存的存储方式，方便jit运算
        cr = updatecr(cp_MC, fij_a1_min.transpose(), N_theta, N_sample) # 初始化cr
        fij_a2, chi2_a2, J_a2, cp_model_a2, cr_a2 = NMF_w_theta_J(fij_a1_min, cp_MC, cr, N_max_interation2, N_theta, N_sample, covij)
    #    print('NMF min J=',np.min(J_a2))
    #    print('===> NMF fij=',fij)

        # 存入大矩阵
        fij_array_minJ[:,:,MC] = fij_a2[np.argmin(J_a2)]
        fij_array_minchi2[:,:,MC] = fij_a2[np.argmin(chi2_a2)]
        J_array[MC] = J_a2[np.argmin(J_a2)]
        chi2_array[MC] = chi2_a2[np.argmin(chi2_a2)]
        cp_model_array_minJ[:,:,:,MC] = cp_model_a2[np.argmin(J_a2)]
        cp_model_array_minchi2[:,:,:,MC] = cp_model_a2[np.argmin(chi2_a2)]
        cr_array_minJ[:,:,:, MC] = cr_a2[np.argmin(J_a2)]
        cr_array_minchi2[:,:,:,MC] = cr_a2[np.argmin(chi2_a2)]

    filename='Test_Ntheta'+str(N_theta).zfill(2)+'_arrays'
    np.savez(filename, cp = cp, covij = covij, cp_array = cp_array,
            fij_array_minJ=fij_array_minJ,fij_array_minchi2=fij_array_minchi2,
            J_array = J_array, chi2_array=chi2_array, cp_model_array_minJ= cp_model_array_minJ,
            cp_model_array_minchi2 = cp_model_array_minchi2, 
             cr_array_minJ = cr_array_minJ, cr_array_minchi2 = cr_array_minchi2)

if __name__ == '__main__':
    t0 = time.time()
    # measurement array, (N_photoz_bins, N_photoz_bins, N_theta)
    cp_input = np.load('zmag21_NGC_cp_5bins.npy') 
    # covariance array, (N_photoz_bins, N_photoz_bins, N_theta, N_theta)
    covij_input = np.load('zmag21_NGC_cov_5bins.npy')

    N_sample, N_sample, N_all_theta = cp_input.shape

    # theta range to fit
    N_first_theta, N_last_theta = 0, 5 
    N_MC = 1

    # algorithm1 收敛条件
    N_max_interation1 = 1000
    # algorithm2 收敛条件
    N_max_interation2 = 10000

    run_self_cali(cp_input, covij_input, N_first_theta, N_last_theta, N_sample, N_MC = N_MC)

    if N_last_theta == None:
        N_last_theta = N_all_theta
    print(f'N_photoz_bins={N_sample}\nN_theta = {N_last_theta-N_first_theta}\nTime used:{time.time()-t0} s')