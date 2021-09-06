"""This module contains simple helper functions """
import torch
import numpy as np

def focal_loss(fake, real):
    epsilon = 1e-8

    alpha = 1

    ## 2D DFT with orthonomalization
    fake_fft = torch.fft.fft2(fake, norm = 'ortho')

    real_fft = torch.fft.fft2(real, norm = 'ortho')

    x_dist = (real_fft.real - fake_fft.real) ** 2

    y_dist = (real_fft.imag - fake_fft.imag) ** 2

    distance_matrix = torch.sqrt(x_dist + y_dist + epsilon) 

    ## squared Eucliedean distance
    squared_distance = distance_matrix ** 2

    ## weight for spatial frequency
    weight_matrix = distance_matrix ** alpha

    # normalization weight_matrix to [0,1]
    norm_weight_matrix = (weight_matrix - torch.min(weight_matrix)) /  (torch.max(weight_matrix) - torch.min(weight_matrix))


    prod = torch.mul(squared_distance, norm_weight_matrix)

    FFL = torch.sum(prod) / (256 * 256 * 3)

    return FFL
    
def focal_loss_np(fake, real):
    epsilon = 1e-8
    
    alpha = 1
    fake = fake.clone().detach().cpu().numpy()
    real = real.clone().detach().cpu().numpy()
    
    ## 2D DFT with orthonomalization
    fake_fft = np.fft.fft2(fake, norm = 'ortho')

    real_fft = np.fft.fft2(real, norm = 'ortho') 

    x_dist = (real_fft.real - fake_fft.real) ** 2

    y_dist = (real_fft.imag - fake_fft.imag) ** 2

    distance_matrix = np.sqrt(x_dist + y_dist + epsilon) 

    ## squared Eucliedean distance
    squared_distance = distance_matrix ** 2

    ## weight for spatial frequency
    weight_matrix = distance_matrix ** alpha

    # normalization weight_matrix to [0,1]
    norm_weight_matrix = (weight_matrix - np.min(weight_matrix)) /  (np.max(weight_matrix) - np.min(weight_matrix))


    prod = np.multiply(squared_distance, norm_weight_matrix)

    FFL = np.sum(prod) / (256 * 256 * 3)

    return FFL
