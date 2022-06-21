''' Functions in deblur flow '''

from pickletools import uint8
import numpy as np
import cv2 as cv
from scipy import ndimage, signal
from scipy.signal import convolve2d

import sys
DBL_MIN = sys.float_info.min





########################################################
def Wiener_deconv(img_in, k_in, SNR_F):
    """ Wiener deconvolution
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                SNR_F (float): Wiener deconvolution parameter
            Returns:
                Wiener_result (uint8 ndarray, shape(height, width, ch)): Wiener-deconv image
                
            Todo:
                Wiener deconvolution
    """
    # Convert to floating-point type and normalize the intensity of blurred image to [0, 1]
    img_in = img_in.astype(float) / 255.
    
    # Normalize the blur kernel such that its sum equals to 1
    k_in = k_in.astype(float)
    k_in = k_in / k_in.sum()

    # Expand the kernel size to the image resolution by padding zeros
    K = np.pad(k_in, ((0, img_in.shape[0] - k_in.shape[0]), (0, img_in.shape[1] - k_in.shape[1])), 'constant')
    # Roll for zero convention of DFT to avoid the phase problem
    K = np.roll(K, shift = (-(k_in.shape[0] // 2), -(k_in.shape[1] // 2)), axis = (0, 1))
    # Apply DFT to the blur kernel
    K_f = np.fft.rfft2(K)

    # For each color channel (R, G, B)
    Wiener_result = np.zeros(shape = img_in.shape)
    for ch in range(img_in.shape[2]):
        # Apply DFT to the blurred image
        B_f = np.fft.rfft2(img_in[:, :, ch])

        # Perform Wiener deconvolution
        I_hat_f = B_f / K_f * (np.abs(K_f)**2 / (np.abs(K_f)**2 + 1 / SNR_F))

        # Apply inverse DFT to the deconvolved result
        Wiener_result[:, :, ch] = np.fft.irfft2(I_hat_f, s = img_in[:, :, ch].shape)
    
    # Convert the range of Wiener-deconvolved image from [0, 1] to [0, 255]
    Wiener_result = np.floor(np.clip(Wiener_result, 0, 1) * 255 + 0.5).astype('uint8')

    return Wiener_result









########################################################
def RL(img_in, k_in, max_iter):
    """ RL deconvolution
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): blur kernel
                max_iter (int): total iteration count
                
            Returns:
                RL_result (uint8 ndarray, shape(height, width, ch)): RL-deblurred image
                
            Todo:
                RL deconvolution
    """
    # Convert to floating-point type and normalize the intensity of blurred image to [0, 1]
    img_in = img_in.astype(float) / 255.
    
    # Normalize the blur kernel such that its sum equals to 1
    k_in = k_in.astype(float)
    K = k_in / k_in.sum()

    # Calculate the adjoint kernel
    K_star = np.flip(K, axis = (0, 1))
    
    I = np.array(img_in)

    # For each iteration
    for iter in range(max_iter):
        # For each color channel (R, G, B)
        for ch in range(img_in.shape[2]):
            I_conv_K = convolve2d(I[:, :, ch], K, boundary = 'symm', mode = 'same')
            I[:, :, ch] = I[:, :, ch] * convolve2d(img_in[:, :, ch] / (I_conv_K + DBL_MIN), K_star, boundary = 'symm', mode = 'same')
    
    # Convert the range of RL-deblurred image from [0, 1] to [0, 255]
    RL_result = np.floor(np.clip(I, 0, 1) * 255 + 0.5).astype('uint8')
    
    return RL_result



########################################################
def BRL(img_in, k_in, max_iter, lamb_da, sigma_r, rk):
    """ BRL deconvolution
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                max_iter (int): Total iteration count
                lamb_da (float): BRL parameter
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                
            Returns:
                BRL_result (uint8 ndarray, shape(height, width, ch)): BRL-deblurred image
                
            Todo:
                BRL deconvolution
    """
    # Convert to floating-point type and normalize the intensity of blurred image to [0, 1]
    img_in = img_in.astype(float) / 255.
    
    # Normalize the blur kernel such that its sum equals to 1
    k_in = k_in.astype(float)
    K = k_in / k_in.sum()

    # Calculate the adjoint kernel
    K_star = np.flip(K, axis = (0, 1))

    # Define parameters
    r_omega = int(0.5 * rk)
    sigma_s = (r_omega / 3)**2

    # Construct the kernel of spatial penalty
    axis = np.linspace(-r_omega, r_omega, 2 * r_omega + 1)
    k, l = np.meshgrid(axis, axis, indexing = 'ij')
    spatial_kernel = np.exp(-(k**2 + l**2) / (2 * sigma_s))
    
    I = np.array(img_in)

    # For each iteration
    for iter in range(max_iter):
        # For each color channel (R, G, B)
        for ch in range(img_in.shape[2]):
            I_conv_K = convolve2d(I[:, :, ch], K, boundary = 'symm', mode = 'same')

            I_padded = np.pad(I[:, :, ch], (r_omega, r_omega), 'symmetric')

            # Calculate the intensity gradient of E_B(I)
            grad_E_B = np.zeros(shape = I.shape[0:2])
            for i in range(I.shape[0]):
                for j in range(I.shape[1]):
                    neighborhood = I_padded[i:(i + 2 * r_omega + 1), j:(j + 2 * r_omega + 1)]
                    # Construct the kernel of range penalty
                    range_kernel = np.exp(-(I[i, j, ch] - neighborhood)**2 / (2 * sigma_r))
                    # Combine
                    grad_E_B[i, j] = 2 * (spatial_kernel * range_kernel * (I[i, j, ch] - neighborhood) / sigma_r).sum()

            I[:, :, ch] = I[:, :, ch] / (1 + lamb_da * grad_E_B) * convolve2d(img_in[:, :, ch] / (I_conv_K + DBL_MIN), K_star, boundary = 'symm', mode = 'same')
    
    # Convert the range of BRL-deblurred image from [0, 1] to [0, 255]
    BRL_result = np.floor(np.clip(I, 0, 1) * 255 + 0.5).astype('uint8')

    return BRL_result
    





########################################################
def RL_energy(img_in, k_in, I_in):
    """ RL Energy
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                I_in (uint8 ndarray, shape(height, width, ch)): Deblurred image
                
            Returns:
                RL_energy (float): RL_energy
                
            Todo:
                Calculate RL energy
    """
    # Convert to floating-point type and normalize the intensity of images to [0, 1]
    img_in = img_in.astype(float) / 255.
    I_in = I_in.astype(float) / 255.
    
    # Normalize the blur kernel such that its sum equals to 1
    k_in = k_in.astype(float)
    K = k_in / k_in.sum()

    # For each color channel (R, G, B)
    RL_energy = 0
    for ch in range(img_in.shape[2]):
        I_conv_K = convolve2d(I_in[:, :, ch], K, boundary = 'symm', mode = 'same')
        RL_energy += (I_conv_K - img_in[:, :, ch] * np.log(I_conv_K)).sum()
    
    return RL_energy




########################################################
def BRL_energy(img_in, k_in, I_in, lamb_da, sigma_r, rk):
    """ BRL Energy
            Args:
                img_in (uint8 ndarray, shape(height, width, ch)): Blurred image
                k_in (uint8 ndarray, shape(height, width)): Blur kernel
                I_in (uint8 ndarray, shape(height, width, ch)): Deblurred image
                lamb_da (float): BRL parameter
                sigma_r (float): BRL parameter
                rk (int): BRL parameter
                
            Returns:
                BRL_energy (float): BRL_energy
                
            Todo:
                Calculate BRL energy
    """
    # Convert to floating-point type and normalize the intensity of blurred image to [0, 1]
    img_in = img_in.astype(float) / 255.
    I_in = I_in.astype(float) / 255.
    
    # Normalize the blur kernel such that its sum equals to 1
    k_in = k_in.astype(float)
    K = k_in / k_in.sum()

    # Define parameters
    r_omega = int(0.5 * rk)
    sigma_s = (r_omega / 3)**2

    # Construct the kernel of spatial penalty
    axis = np.linspace(-r_omega, r_omega, 2 * r_omega + 1)
    k, l = np.meshgrid(axis, axis, indexing = 'ij')
    spatial_kernel = np.exp(-(k**2 + l**2) / (2 * sigma_s))

    # For each color channel (R, G, B)
    BRL_energy = 0
    for ch in range(img_in.shape[2]):
        I_padded = np.pad(I_in[:, :, ch], (r_omega, r_omega), 'symmetric')

        # Calculate the edge-preserving regularization term E_B(I)
        E_B = 0
        for i in range(I_in.shape[0]):
            for j in range(I_in.shape[1]):
                neighborhood = I_padded[i:(i + 2 * r_omega + 1), j:(j + 2 * r_omega + 1)]
                # Construct the kerenl of range penalty
                range_kernel = 1 - np.exp(-(I_in[i, j, ch] - neighborhood)**2 / (2 * sigma_r))
                # Combine
                E_B += (spatial_kernel * range_kernel).sum()
        
        # Calculate E(I)
        I_conv_K = convolve2d(I_in[:, :, ch], K, boundary = 'symm', mode = 'same')
        BRL_energy += ((I_conv_K - img_in[:, :, ch] * np.log(I_conv_K)).sum() + lamb_da * E_B)
    
    return BRL_energy