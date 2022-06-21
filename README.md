# NTHU EE6620 Computational Photography (Spring 2022)

This repository contains my code for the 3 assignments in EE6620 Computational Photography, \
taught by Prof. Chao-Tsung Huang at National Tsing Hua University.

## HW1 High Dynamic Range Imaging
Modern cameras are unable to capture the full dynamic range of commonly encountered natural scenes. High-dynamic-range (HDR) photographs are generally achieved by capturing multiple standard-exposure images, often using exposure bracketing, and then merging them into a single HDR image. Also, to view the HDR image on an ordinary low-dynamic-range (LDR) display, tone mapping operation from HDR to LDR on images is required. In this assignment, we implement the whole HDR photography flow, including **image bracketing**, **camera response calibration** [1], **white balance**, and finally, **tone mapping**, to visualize our results. More details can be found in the file `hw1/EE6620-hw1.pdf`.

## HW2 Non-Blind Deblurring
A slow shutter speed will introduce blurred images due to camera shake. The objective of this assignment is to implement several non-blind deblurring algorithms and analyze the effects on blurred images. In part 1-3, we implement **Wiener deconvolution**, **Richarson-Lucy (RL) deconvolution** [2, 3] and its bilateral variant (**BRL**) [4]. And in part 4, we solve the deblurring problem by **total variation regularization** using ProxImaL [5]. More details can be found in the file `hw2/EE6620-hw2.pdf`.

## HW3 Super-Resolution
Super-resolution is a class of techniques which can enhance the resolution of images. In this assignment, we work on two types of super-resolution methods, **optimization-based** and **convnet-based**. For the optimization-based part, we solve the SR problem with a **single-image** method and **multi-image** method using ProxImaL [5]. For the convnet-based part, we build and train a SR model based on **WDSR** [6], and then try to increase its size. More details can be found in the file `hw3/EE6620-hw3.pdf`.

## References
1. P. E. Debevec and J. Malik. Recovering High Dynamic Range Radiance Maps from Photographs. In *ACM SIGGRAPH*, 2008.
2. W. H. Richardson. Bayesian-based Iterative Method of Image Restoration. In *Journal of the Optical Society of America*, 1972.
3. L. B. Lucy. An Iterative Technique for the Rectification of Observed Distributions. In *Astronomical Journal*, 1974.
4. L. Yuan *et al*. Progressive Inter-scale and Intra-scale Non-blind Image Deconvolution. In *ACM Transactions on Graphics*, 2008.
5. F. Heide *et al*. ProxImaL: Efficient Image Optimization Using Proximal Algorithms. In *ACM Transactions on Graphics*, 2016.
6. J. Yu *et al*. Wide Activation for Efficient and Accurate Image Super-Resolution. In *arXiv preprint arXiv:1808.08718*, 2018.
