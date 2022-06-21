''' Functions in HDR flow '''

import os
from turtle import shape
from unittest import result
import cv2 as cv
import numpy as np

Z = 256  # intensity levels
Z_max = 255
Z_min = 0
gamma = 2.2


def ReadImg(path, flag=1):
    img = cv.imread(path, flag)  # flag = 1 means to load a color image
    img = np.transpose(img, (2,0,1))
    return img


def SaveImg(img, path):
    img = np.transpose(img, (1,2,0))
    cv.imwrite(path, img)
    
    
def LoadExposures(source_dir):
    """ load bracketing images folder

    Args:
        source_dir (string): folder path containing bracketing images and a image_list.txt file
                             image_list.txt contains lines of image_file_name, exposure time, ... 
    Returns:
        img_list (uint8 ndarray, shape (N, ch, height, width)): N bracketing images (3 channel)
        exposure_times (list of float, size N): N exposure times
    """
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *_) = line.split()
        filenames += [filename]
        exposure_times += [float(exposure)]
    img_list = [ReadImg(os.path.join(source_dir, f)) for f in filenames]
    img_list = np.array(img_list)
    
    return img_list, exposure_times


def PixelSample(img_list):
    """ Sampling

    Args:
        img_list (uint8 ndarray, shape (N, ch, height, width))
        
    Returns:
        sample (uint8 ndarray, shape (N, ch, height_sample_size, width_sample_size))
    """
    # trivial periodic sample
    sample = img_list[:, :, ::64, ::64]
    
    return sample


def EstimateResponse(img_samples, etime_list, lambda_=50):
    """ Estimate camera response for bracketing images

    Args:
        img_samples (uint8 ndarray, shape (N, height_sample_size, width_sample_size)): N bracketing sampled images (1 channel)
        etime_list (list of float, size N): N exposure times
        lambda_ (float): Lagrange multiplier (Defaults to 50)
    
    Returns:
        response (float ndarray, shape (256)): response map
    """
    
    ''' TODO '''
    I = img_samples.shape[1] * img_samples.shape[2]     # num of sampled pixels
    J = img_samples.shape[0]                            # num of images

    # Calculate the weighting (hat) function
    w = np.linspace(0, 255, num = 256, dtype = int)
    w = np.piecewise(w, [w <= (Z_min + Z_max) / 2, w > (Z_min + Z_max) / 2], [lambda x: x - Z_min, lambda x: Z_max - x])

    # Construct the objective function in matrix form: Ax = b
    A = np.zeros(shape = (I * J + 1 + (Z - 2), Z + I))
    b = np.zeros(shape = (A.shape[0], 1))

    img_samples_flatten = img_samples.reshape((J, -1))     # shape: (J, I)

    k = 0
    for i in range(I):
        for j in range(J):
            wij = w[img_samples_flatten[j, i]]
            A[k, img_samples_flatten[j, i]] = wij
            A[k, Z + i] = -wij
            b[k, 0] = wij * np.log(etime_list[j])
            k += 1
    
    # Apply a unit exposure assumption by letting g(127) = 0
    A[k, 127] = 1
    k += 1

    # Include the smoothness term
    for i in range(1, Z - 1):
        A[k, i - 1] = lambda_ * w[i]
        A[k, i] = -2 * lambda_ * w[i]
        A[k, i + 1] = lambda_ * w[i]
        k += 1
    
    # Solve the least square problem using SVD
    x = np.linalg.lstsq(A, b, rcond = None)[0]

    response = x[:Z, 0]
    
    return response


def ConstructRadiance(img_list, response, etime_list):
    """ Construct radiance map from brackting images

    Args:
        img_list (uint8 ndarray, shape (N, height, width)): N bracketing images (1 channel)
        response (float ndarray, shape (256)): response map
        etime_list (list of float, size N): N exposure times
    
    Returns:
        radiance (float ndarray, shape (height, width)): radiance map
    """

    ''' TODO '''
    # Calculate the weighting (hat) function
    w = np.linspace(0, 255, num = 256, dtype = int)
    w = np.piecewise(w, [w <= (Z_min + Z_max) / 2, w > (Z_min + Z_max) / 2], [lambda x: x - Z_min, lambda x: Z_max - x])

    # Estimate the radiance map
    I = img_list.shape[1] * img_list.shape[2]        # num of sampled pixels
    J = img_list.shape[0]                            # num of images

    img_list_flatten = img_list.reshape((J, -1))     # shape: (J, I)

    radiance = np.zeros(shape = (I,))

    for i in range(I):
        numerator = (w[img_list_flatten[:, i]] * (response[img_list_flatten[:, i]] - np.log(etime_list))).sum()
        denominator = (w[img_list_flatten[:, i]]).sum()
        if denominator == 0:
            denominator = 1
        ln_Ei = numerator / denominator
        radiance[i] = np.exp(ln_Ei)
    
    radiance = radiance.reshape((img_list.shape[1], img_list.shape[2]))

    return radiance


def CameraResponseCalibration(src_path, lambda_):
    img_list, exposure_times = LoadExposures(src_path)
    radiance = np.zeros_like(img_list[0], dtype=np.float32)
    pixel_samples = PixelSample(img_list)
    for ch in range(3):
        response = EstimateResponse(pixel_samples[:,ch,:,:], exposure_times, lambda_)
        radiance[ch,:,:] = ConstructRadiance(img_list[:,ch,:,:], response, exposure_times)
        
    return radiance


def WhiteBalance(src, y_range, x_range):
    """ White balance based on Known to be White(KTBW) region

    Args:
        src (float ndarray, shape (ch, height, width)): source radiance
        y_range (tuple of 2 int): location range in y-dimension
        x_range (tuple of 2 int): location range in x-dimension
        
    Returns:
        result (float ndarray, shape (ch, height, width))
    """
   
    ''' TODO '''
    # Calculate average values in the KTBW region
    B_avg = np.mean(src[0, y_range[0]:y_range[1], x_range[0]:x_range[1]])
    G_avg = np.mean(src[1, y_range[0]:y_range[1], x_range[0]:x_range[1]])
    R_avg = np.mean(src[2, y_range[0]:y_range[1], x_range[0]:x_range[1]])

    # Scaling (fix R)
    result = np.zeros(shape = src.shape)
    result[0] = src[0, :, :] / B_avg * R_avg
    result[1] = src[1, :, :] / G_avg * R_avg
    result[2] = src[2, :, :]
    
    return result


def GlobalTM(src, scale=1.0):
    """ Global tone mapping

    Args:
        src (float ndarray, shape (ch, height, width)): source radiance image
        scale (float): scaling factor (Defaults to 1.0)
    
    Returns:
        result(uint8 ndarray, shape (ch, height, width)): result HDR image
    """
    
    ''' TODO '''
    result = np.zeros(shape = src.shape)

    # Apply global tone mapping for each channel
    for ch in range(3):
        src_max = src[ch].max()
        log2_X_hat = scale * (np.log2(src[ch]) - np.log2(src_max)) + np.log2(src_max)
        X_hat = np.exp2(log2_X_hat)
        result[ch] = X_hat ** (1 / gamma)
        # Map the range to [0, 255], round to the nearest integer, and cast to uint8 type
        # result[ch] = np.floor(np.clip(result[ch] * 255, 0, 255) + 0.5).astype('uint8')
        result[ch] = np.floor(np.clip(result[ch], 0, 1) * 255 + 0.5).astype('uint8')
    
    return result


def LocalTM(src, imgFilter, scale=3.0):
    """ Local tone mapping

    Args:
        src (float ndarray, shape (ch, height, width)): source radiance image
        imgFilter (function): filter function with preset parameters
        scale (float): scaling factor (Defaults to 3.0)
    
    Returns:
        result(uint8 ndarray, shape (ch, height, width)): result HDR image
    """
    
    ''' TODO '''
    result = np.zeros(shape = src.shape)

    # Seperate intensity map (I) and color ratio (C_r, C_g, C_b) for radiance (R, G, B)
    I = np.mean(src, axis = 0)
    color_ratio = src / I

    # Take log of intensity
    L = np.log2(I)

    # Seperate the detail layer (L_D) and base layer (L_B) of L
    L_B = imgFilter(L)
    L_D = L - L_B
    # Compress the contrast of base layer
    L_min = np.min(L_B)
    L_max = np.max(L_B)
    L_B_prime = (L_B - L_max) * scale / (L_max - L_min)
    # Reconstruct intensity map with detail layer and adjusted base layer
    I_prime = np.exp2(L_B_prime + L_D)

    # For each channel
    for ch in range(3):
        # Reconstruct color map with color ratio and adjusted intensity
        result[ch] = color_ratio[ch] * I_prime

        # Gamma correction
        result[ch] = result[ch] ** (1 / gamma)
        # Map the range to [0, 255], round to the nearest integer, and cast to uint8 type
        result[ch] = np.floor(np.clip(result[ch], 0, 1) * 255 + 0.5).astype('uint8')
    
    return result


def GaussianFilter(src, N=35, sigma_s=100):
    """ Gaussian filter

    Args:
        src (float ndarray, shape (height, width)): source intensity
        N (int): window size of the filter (Defaults to 35)
                 filter indices span [-N/2, N/2]
        sigma_s (float): standard deviation of Gaussian filter (Defaults to 100)
    
    Returns:
        result (float ndarray, shape (height, width))
    """
    
    ''' TODO '''
    # Construct Gaussain kernel
    axis = np.linspace(-(N - 1) / 2, (N - 1) / 2, N)
    k, l = np.meshgrid(axis, axis, indexing = 'ij')
    gaussian_kernel = np.exp(-(k**2 + l**2) / (2 * sigma_s**2))

    # Pad the input image
    pad_size = np.floor(N / 2).astype(int)
    src_padded = np.pad(src, (pad_size, pad_size), 'symmetric')

    # Convolution with normalized kernel
    result = np.zeros(shape = src.shape)

    kernel_sum = gaussian_kernel.sum()

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            result[i, j] = (src_padded[i:(i + N), j:(j + N)] * gaussian_kernel).sum() / kernel_sum
    
    return result


def BilateralFilter(src, N=35, sigma_s=100, sigma_r=0.8):
    """ Bilateral filter

    Args:
        src (float ndarray, shape (height, width)): source intensity
        N (int): window size of the filter (Defaults to 35)
                 filter indices span [-N/2, N/2]
        sigma_s (float): spatial standard deviation of bilateral filter (Defaults to 100)
        sigma_r (float): range standard deviation of bilateral filter (Defaults to 0.8)
    
    Returns:
        result (float ndarray, shape (height, width))
    """
    
    ''' TODO '''    
    # Construct domain kernel
    axis = np.linspace(-(N - 1) / 2, (N - 1) / 2, N)
    k, l = np.meshgrid(axis, axis, indexing = 'ij')
    domain_kernel = np.exp(-(k**2 + l**2) / (2 * sigma_s**2))

    # Pad the input image
    pad_size = np.floor(N / 2).astype(int)
    src_padded = np.pad(src, (pad_size, pad_size), 'symmetric')

    # Convolution with normalized kernel
    result = np.zeros(shape = src.shape)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            conv_region = src_padded[i:(i + N), j:(j + N)]
            # Construct range kerenl
            range_kerenl = np.exp(-(src[i, j] - conv_region)**2 / (2 * sigma_r**2))
            # Combine domain & range kernels to get bilateral filter
            bilateral_kernel = domain_kernel * range_kerenl

            result[i, j] = (conv_region * bilateral_kernel).sum() / bilateral_kernel.sum()
    
    return result
