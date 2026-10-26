import numpy as np
import matplotlib.pyplot as plt 
import astropy.units as u
from astropy.io import fits
import cfpack as cfp

from scipy import fftpack
import matplotlib.patches as patches
import math
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter
import multiprocessing as mp
import os

#kernel functions, defining a gaussian kernel, a gaussian-weighted mean function, a nan-gaussian filter function, 
def gaussian_2d(size, sigma, nanmask=None):
    '''Generates a 2D Gaussian kernel
    Inputs: 
    - size : the size of the kernel in pixels (format: [x_size, y_size])
    - sigma : the standard deviation of the Gaussian kernel

    Output:
    - 2D Gaussian kernel
    '''
    x_size = size[0]
    y_size = size[1]
    x, y = np.meshgrid(np.linspace(-x_size//2, x_size//2, x_size), np.linspace(-y_size//2, y_size//2, y_size))
    if nanmask is not None: 
        gauss = (np.exp(-(x**2 + y**2) / (2.0 * sigma**2)))*nanmask
    elif nanmask == None: 
        gauss = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return gauss / np.nansum(gauss)

def weighted_meanmaker(map, FWHM, nanmask=None):
    '''returns the weighted mean of whatever input map assuming a gaussian filter with a full-width-half-max of FWHM pixels.
    Inputs:
    - map : the map to be analysed
    - FWHM : the full-width-half-max of the gaussian filter in pixels

    Outputs: 
    - the weighted mean of the map
    '''
    # Convert the FWHM to sigma value;
    kernel_sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    map_size = np.shape(map)
    gauss_weighter = gaussian_2d(map_size, kernel_sigma, nanmask)

    # Normalize the Gaussian weighting function
    gauss_weighter /= np.nansum(gauss_weighter)

    # Calculate the weighted mean, ignoring NaN values in the map
    weighted_mean = np.nansum(gauss_weighter * map)
    return weighted_mean

def nan_gaussian_filter(values, sigma):
    ''' Applies a Gaussian filter to a 2D array while ignoring NaN values, and they have no contribution to the weighted mean.
    
    Inputs:
    - values: 1D array of values to be filtered (flattened from a 2D array)
    - sigma: standard deviation of the Gaussian kernel
    Outputs:
    - The filtered value at the central pixel of the original window, or NaN if the central pixel is NaN
    
    '''

    # Reshape the flat array back to the filter window size
    window_size = int(np.sqrt(len(values)))  # Assuming a square window
    values_2d = values.reshape((window_size, window_size))

    # Check if the central pixel of the original window is NaN
    center_pixel_index = window_size // 2
    if np.isnan(values_2d[center_pixel_index, center_pixel_index]):
        return np.nan

    # Create a mesh grid for the Gaussian kernel
    y, x = np.indices((window_size, window_size))
    center = window_size // 2

    # Compute the 2D Gaussian weights
    gaussian_kernel = np.exp(-0.5 * (((x - center) ** 2 + (y - center) ** 2) / sigma ** 2))

    # Normalize the Gaussian kernel
    gaussian_kernel /= np.sum(gaussian_kernel)

    # Apply the Gaussian kernel to the valid values (non-NaN)
    weighted_values = np.where(np.isnan(values_2d), 0, values_2d * gaussian_kernel)
    weights = np.where(np.isnan(values_2d), 0, gaussian_kernel)

    # Normalize by the sum of weights
    result = np.sum(weighted_values) / np.sum(weights)

    return result if not np.isnan(result) else np.nan


def NaNMasker(map, mode, returnnanmask = False, FWHM=None):
    '''Replaces NaNs in a map with either the mean, median or zeros. 
    Inputs:
    - map : the map to be analysed
    - mode : the mode of replacement (Mean, Median or Zeros)
    - returnnanmask : if True, returns the mask of the NaNs

    Outputs:
    - the map with the NaNs replaced
    - (if returnnanmask is True) the mask of the NaNs (1 for non-NaN, np.nan for NaN values)
    '''
    if np.isnan(np.sum(map))==False:
        return(map)
    nonan_map = map.copy()
    nanmask = np.ones_like(nonan_map)
    nanmask[np.isnan(nonan_map)] = np.nan
    #replace the NaNs with the mean, median or zeros;
    if mode=='Mean':
        nonan_map[np.isnan(nonan_map)] = np.nanmean(map)
    elif mode=='Median':
        nonan_map[np.isnan(nonan_map)] = np.nanmedian(map)
    elif mode=='Zeros':
        nonan_map[np.isnan(nonan_map)] = 0.
    elif mode=='Noise':
        nonan_map[np.isnan(nonan_map)] = 0.2
    elif mode == 'linear' or mode=='Interp':
        y, x = np.indices(nonan_map.shape)
        valid_mask = ~np.isnan(nonan_map)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        values_valid = nonan_map[valid_mask]

        nan_mask = np.isnan(nonan_map)
        x_nan = x[nan_mask]
        y_nan = y[nan_mask]

        # Interpolate using the valid points
        interpolated_values = griddata(points=(x_valid, y_valid), values=values_valid, xi=(x_nan, y_nan), method='linear')

        # Replace the NaN values with the interpolated values
        nonan_map[nan_mask] = interpolated_values
    elif mode == 'cubic':
        y, x = np.indices(nonan_map.shape)
        valid_mask = ~np.isnan(nonan_map)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        values_valid = nonan_map[valid_mask]

        nan_mask = np.isnan(nonan_map)
        x_nan = x[nan_mask]
        y_nan = y[nan_mask]

        # Interpolate using the valid points
        interpolated_values = griddata(points=(x_valid, y_valid), values=values_valid, xi=(x_nan, y_nan), method='cubic')

        # Replace the NaN values with the interpolated values
        nonan_map[nan_mask] = interpolated_values
    elif mode == 'WeightedMean':
        weighted_mean = weighted_meanmaker(map, FWHM, nanmask = nanmask)
        nonan_map[np.isnan(nonan_map)] = weighted_mean
    else:
        print('Invalid mode. Only Mean, Median, Zeros and Interp supported. Returning Nan-filled map')
    #return the mask if requested;
    if returnnanmask ==True: 
        return(nonan_map, nanmask)
    else:
        return(nonan_map)


def get_power_spectrum(mom0, zero_pad=False, weights=None, mirror=False, normalise=False, show=False):
    ''' Computes the power spectrum of the moment 0 map using Fourier transform.
    This code is taken from Gerrard et al, 2023. 

    Inputs: 
    - mom0: 2D map of the moment 0 for which the power spectrum is to be computed. This should not be a - - full galaxy or a large area, it should be a small subsection. 
    - zero_pad: If True, zero-pad the map before computing the power spectrum.
    - weights: Array of weights to apply to the moment 0 map.
    - mirror: If True, mirror the map before computing the power spectrum.
    - normalise: If True, normalise the map before computing the power spectrum.
    - show: If True, display the power spectrum.

    Outputs: 
    - power_spectrum_map: 2D array of the power spectrum. 
    - k: 2D array of the wave numbers corresponding to the power spectrum.

    '''
    if weights is None: # if weight is not provided, we make an array of ones; i.e., no change to the data
        weights = np.ones(mom0.shape)
    # construct map to do power spectrum analysis on
    scaled_map = np.copy(mom0)
    x_inds = np.argwhere(~np.isnan(scaled_map))[:,0]
    y_inds = np.argwhere(~np.isnan(scaled_map))[:,1]
    xmin = np.nanmin(x_inds)
    xmax = np.nanmax(x_inds)
    ymin = np.nanmin(y_inds)
    ymax = np.nanmax(y_inds)
    # select minimum bounding box that does not contain NaNs and is scaled by the SNR
    scaled_map = (scaled_map * weights)[xmin:xmax, ymin:ymax]
    scaledmapmean = np.nanmean(scaled_map)
    scaled_map[np.where(np.isnan(scaled_map))] = scaledmapmean # set NaNs to zero
    # apply mirroring to ensure periodicity of the moment-0 map before Fourier transform
    if mirror:
        mirrored_upperhalf = np.hstack((np.flipud(scaled_map),np.fliplr(np.flipud(scaled_map))))
        mirrored_lowerhalf = np.hstack((scaled_map,np.fliplr(scaled_map)))
        mirrored_map = np.vstack((mirrored_lowerhalf,mirrored_upperhalf))
        scaled_map = mirrored_map
    # zero padding (this is now default)
    if zero_pad:
        dims = scaled_map.shape
        scaled_map = np.pad(scaled_map, ((dims[0],dims[0]),(dims[1],dims[1])), 'constant')
    # make sure the data is float type
    data = scaled_map.astype(float)
    # pre-processing following Federrath et al. (2016); i.e., define fluctuations map
    if normalise: data = data/data.mean() - 1.0
    # 2D Fourier transformation
    ft = fftpack.fft2(data) / (data.shape[0]*data.shape[1])
    # center the transform so k = (0,0) is in the center
    ft_c = fftpack.fftshift(ft)
    # compute the power spectrum map
    power_spectrum_map = np.abs(ft_c*np.conjugate(ft_c))
    # undoing mirroring after Fourier transform
    if mirror:
        nx = power_spectrum_map.shape[0]
        ny = power_spectrum_map.shape[1]
        power_spectrum_map = power_spectrum_map[nx%2::2, ny%2::2]
    # define k (wave number) grid
    nx = power_spectrum_map.shape[0]
    ny = power_spectrum_map.shape[1]
    kx = np.linspace(-(nx//2), nx//2+(nx%2-1), nx)
    ky = np.linspace(-(ny//2), ny//2+(ny%2-1), ny)
    kx_grid, ky_grid = np.meshgrid(kx,ky,indexing="ij")
    k = np.hypot(kx_grid,ky_grid)
    # trim LHS (in case of even input data dimension; i.e., k=0 is in the center)
    power_spectrum_map = power_spectrum_map[(-(nx%2)+1):,(-(ny%2)+1):]
    k = k[(-(nx%2)+1):,(-(ny%2)+1):]
    if show:
        if show != "Fourier image": cfp.plot_map(data, show=True)
        else: cfp.plot_map(power_spectrum_map, log=True, cmap_label="Fourier power", show=True)
    return power_spectrum_map, k

def get_Brunt_sqrt_R(power_spectrum_map, k, exclude_k0=True):
    ''' From Gerrard et al, 2023. Computes the Brunt et al. (2010) sqrt(R) from the power spectrum of the moment 0 map.
    
    Inputs:
    - power_spectrum_map : 2D array of the power spectrum of the moment 0 map
    - k : 2D array of the wave numbers corresponding to the power spectrum
    - exclude_k0 : If True, exclude the k=0 component from the calculation

    Outputs:
    - Brunt_sqrt_R : the Brunt et al. (2010) sqrt(R) value
    '''
    # deal with wave number (k) input shape to get spacing of wave number map
    if k.shape[0] > 1 and k.shape[1] > 1:
        dkx = k[0,0]-k[0,1]
        dky = k[0,0]-k[1,0]
    else:
        dkx = 1
        dky = 1
    dk  = np.hypot(dkx,dky)
    # copy power spectrum
    P = np.copy(power_spectrum_map)
    # compute total of 2D and 3D power, excluding k=0 (excluding the mean)
    if exclude_k0: P[P.shape[0]//2, P.shape[1]//2] = 0.0
    
    P2D = 2 * np.pi * np.sum(P  ) * dk
    P3D = 4 * np.pi * np.sum(P*k) * dk
    Brunt_sqrt_R  = np.sqrt(P2D / P3D) # sqrt of ratio of 2D-to-3D power
    return Brunt_sqrt_R



def single_segment_turbulence_mom1(kernel_FWHM_in_n_pixels, column_density_map, mom1, nanmask, subsect_coords, soundspeed = 0.6, number_sigma_FFT = 3.0, show_kernelsubtr = True, show_kernelsize=True, PrintValues = False, show_beamsize=False, return_corrmaps=False, return_Mach_sigma_rho = False, full_step_returner = False, Gauss_Smoothing_FWHM_factor = 2, beam_FWHM_in_n_pixels=6, kernel_fwhm_in_n_beams = 5):
    '''Evaluates the turbulence parameter b of one section of data.
    Inputs;
    - kernel_fwhm_in_n_beams : the number of beams for the kernel's FWHM
    - mom0 : moment 0 map in K km/s
    - mom1 : moment 1 map 
    - peak_temp : Peak temperature mapping
    - subsect_coords : the coordinates of the subsection which we are analysing in the form [[x_start, x_end], [y_start, y_end]]
    - header : the header of the original datacube
    - soundspeed : the soundspeed of the gas in km/s 

    Options:
    - show_kernelsubtr : if True, shows the moment 0 and moment 1 maps before and after the kernel subtraction
    - show_kernelsize : if True, shows the size of the kernel when(/if) plotting the kernel subtraction as a green dashed circle. Requires show_kernelsubtr=True. 
    - show_Xco_map : if True, shows the Xco map
    - show_powerspect : if True, shows the power spectrum map
    - PrintValues : if True, prints the values of the Brunt sqrt(R), sigma_rho_3D, sigma_vel_3D, Mach number and b-value

    Output: 
    - turbulence parameter 'b' for the individual kernel
    '''
    #First, extract the subsections of the moment maps and peak temperature map;
    column_density_map = (column_density_map[subsect_coords[0][0]:subsect_coords[0][1], subsect_coords[1][0]:subsect_coords[1][1]]).value
    mom_1_Tmap = (mom1[subsect_coords[0][0]:subsect_coords[0][1], subsect_coords[1][0]:subsect_coords[1][1]]).value
    nanmask = nanmask[subsect_coords[0][0]:subsect_coords[0][1], subsect_coords[1][0]:subsect_coords[1][1]]
    #consider fwhm of the pixel in the number of pixels;
    if kernel_FWHM_in_n_pixels is None:
        kernel_fwhm_pixels = beam_FWHM_in_n_pixels*kernel_fwhm_in_n_beams
    else:
        kernel_fwhm_pixels = kernel_FWHM_in_n_pixels
    #set the FWHM of the gaussian smoothing function to half the FWHM of the kernel;
    FWHM_gaussfit = kernel_fwhm_pixels/Gauss_Smoothing_FWHM_factor
    sigma_gaussfit = FWHM_gaussfit / (2*np.sqrt(2*np.log(2)))
    # Correcting for larger-scale motions; 
    #for mom-0 need to do on log scales. We also want to normalise with gaussian-weighted mean density;
    columndensity_mean = weighted_meanmaker(column_density_map, kernel_fwhm_pixels, nanmask=nanmask) #the mean column density
    column_density_map_normalised = column_density_map/columndensity_mean
    #consider the nan values as the mean of the non-log column density, then the log distribution;
    column_density_map_normalised_nonan = column_density_map_normalised #NaNMasker(column_density_map_normalised, mode='Noise', returnnanmask=True)
    log_n = np.log10(column_density_map_normalised_nonan) #the log of the column density
    #replace the nan values with the mean values (need mean values as log scale)
    log_n_smoothed = generic_filter(log_n, nan_gaussian_filter, size=(50,50), extra_arguments=(sigma_gaussfit,)) # nanmask
    log_corr_n = log_n - log_n_smoothed
    corr_n= (10**log_corr_n) #(return the NaNs back so that we can divide it up however we like later)
    #remove the NaNs for later use (both for the FT and the st deviation calculations);
    corr_n_nonan_for_fft = corr_n.copy() 
    #Determine the Brunt R factor;
    #find the power spectrum of the moment 0 map; 
    center_x = int(np.shape(corr_n_nonan_for_fft)[0]/2)
    center_y = int(np.shape(corr_n_nonan_for_fft)[1]/2)
    sigma_pixels = kernel_fwhm_pixels/ (2*np.sqrt(2*np.log(2)))
    if number_sigma_FFT == 3.0:
        corr_n_nonan_for_fft_reducedregion = corr_n_nonan_for_fft
    elif number_sigma_FFT<3.0:
        extract_radius = math.ceil(number_sigma_FFT * sigma_pixels)
        corr_n_nonan_for_fft_reducedregion = corr_n_nonan_for_fft[center_x - extract_radius : center_x + extract_radius, center_y - extract_radius : center_y + extract_radius]
    elif number_sigma_FFT>3.0:
        print('number_sigma_FFT greater than 3subsect region size, defaulting to 3')
        extract_radius = math.ceil(3. * sigma_pixels)
        corr_n_nonan_for_fft_reducedregion = corr_n_nonan_for_fft[center_x - extract_radius : center_x + extract_radius, center_y - extract_radius : center_y + extract_radius]
    else: 
        print('number_sigma_FFT not recognised, defaulting to 3')
        extract_radius = math.ceil(3. * sigma_pixels)
        corr_n_nonan_for_fft_reducedregion = corr_n_nonan_for_fft[center_x - extract_radius : center_x + extract_radius, center_y - extract_radius : center_y + extract_radius]
    powerspect, k_mom0 = get_power_spectrum(corr_n_nonan_for_fft_reducedregion, mirror=True)
    #Use the previously defined function;
    brunt_sqrt_r = get_Brunt_sqrt_R(powerspect, k_mom0)
    #determine the st dev of the moment 0 map for the sigma_\rho;
    avg_0 = weighted_meanmaker(corr_n_nonan_for_fft*nanmask, kernel_fwhm_pixels, nanmask=nanmask)
    msq_0 = weighted_meanmaker(corr_n_nonan_for_fft**2*nanmask, kernel_fwhm_pixels, nanmask=nanmask)
    sigma_rho_2D = np.sqrt(np.abs(msq_0 - avg_0**2))
    #and from 2D st dev to 3D st dev; 
    sigma_rho_3D = sigma_rho_2D/brunt_sqrt_r
    #determine the st dev of the moment 1 map for the sigma_\vel;
    #remove the Nans in the moment 1;
    mom1_nonan = NaNMasker(mom_1_Tmap, mode='WeightedMean', FWHM=kernel_fwhm_pixels)
    #for moment-1 just need subtraction; 
    mom1_smoothed = generic_filter(mom_1_Tmap, nan_gaussian_filter, size=(50,50), extra_arguments=(sigma_gaussfit,))
    corr_mom1 = (mom1_nonan - mom1_smoothed)
    #Account for the NaNs;
    corr_mom1_nonan = corr_mom1 # NaNMasker(corr_mom1, mode='Mean')
    #and now the standard dev of the moment 1 map for \sigma_vel in 1D; 
    avg_1 = weighted_meanmaker(corr_mom1_nonan*nanmask, kernel_fwhm_pixels, nanmask=nanmask)
    msq_1 = weighted_meanmaker(corr_mom1_nonan**2*nanmask, kernel_fwhm_pixels, nanmask=nanmask)
    sigma_vel_1D = np.sqrt(np.abs(msq_1 - avg_1**2))
    #using the Federrath and Stewart 2022 conversion from 1D to 3D. 
    # Averaging parameters across the three energy ratios;
    p_0 = ((3.66+3.22+3.09)/3)
    p_1 = ((6.01+3.45+3.48)/3)
    p_2 = ((20.2+13.6+10.1)/3)
    f_R_ratio = (2/kernel_fwhm_in_n_beams)
    
    C_los_stat = p_2*f_R_ratio**2 + p_1*f_R_ratio + p_0
    sigma_vel_3D = C_los_stat*sigma_vel_1D

    Mach = (sigma_vel_3D/soundspeed)
    b = sigma_rho_3D / Mach
    #print the values if requested;
    if PrintValues == True:
        print('The Brunt sqrt(R) is '+str(brunt_sqrt_r))
        print('Sigma rho 3D is '+str(sigma_rho_3D))
        print('And the sigma vel 3D is '+ str(sigma_vel_3D))
        print('Mach number is '+str(Mach))
        print('so the b-value is '+str(b))
    # show the kernel subtraction if requested;
    if show_kernelsubtr==True: 
        #moment 0 maps;
        #make subplots of the three moment 0 maps; 
        fig, axs = plt.subplots(1, 4, figsize=(24, 5))
        # Plot for column_density_map_normalised
        im_0 = axs[0].imshow(column_density_map_normalised, cmap='magma', origin='lower', interpolation='none')
        axs[0].set_title('Raw Column Density Raw Map')
        axs[0].set_xlabel('Pixel Count')
        axs[0].set_ylabel('Pixel Count')
        cbar = fig.colorbar(im_0, ax=axs[0])
        cbar.set_label('Normalised Column Density (unitless)')
        axs[0].axis('on')
        centre_coords = [np.shape(column_density_map_normalised)[0] / 2, np.shape(column_density_map_normalised)[1] / 2]
        if show_kernelsize:
            kernel_demo = patches.Circle((math.ceil(centre_coords[0]), math.ceil(centre_coords[1])), radius=kernel_fwhm_pixels / 2, edgecolor='green', facecolor='none', linestyle='dashed')
            axs[0].add_patch(kernel_demo)
        if show_beamsize:
            beam_demo = patches.Circle((46, 3), radius=3, edgecolor='black', facecolor='orange')
            axs[0].add_patch(beam_demo)
        # Plot for 10**log_n_smoothed
        im_1 = axs[1].imshow(10**log_n_smoothed, cmap='magma', origin='lower', interpolation='none')
        axs[1].set_title('Smoothed Column Density Map')
        axs[1].set_xlabel('Pixel Count')
        axs[1].set_ylabel('Pixel Count')
        cbar = fig.colorbar(im_1, ax=axs[1])
        cbar.set_label('Normalised Column Density (unitless)')
        axs[1].axis('on')
        if show_kernelsize:
            kernel_demo = patches.Circle((math.ceil(centre_coords[0]), math.ceil(centre_coords[1])), radius=kernel_fwhm_pixels / 2, edgecolor='green', facecolor='none', linestyle='dashed')
            axs[1].add_patch(kernel_demo)
        if show_beamsize:
            beam_demo = patches.Circle((46, 3), radius=3, edgecolor='black', facecolor='orange')
            axs[1].add_patch(beam_demo)
        # Plot for corr_n_nonan_for_fft
        im_2 = axs[2].imshow(corr_n_nonan_for_fft, cmap='magma', origin='lower', interpolation='none')
        axs[2].set_title('Column Density Map - NaNs filled')
        axs[2].set_xlabel('Pixel Count')
        axs[2].set_ylabel('Pixel Count')
        cbar = fig.colorbar(im_2, ax=axs[2])
        cbar.set_label('Normalised Column Density (unitless)')
        axs[2].axis('on')
        if show_kernelsize:
            kernel_demo = patches.Circle((math.ceil(centre_coords[0]), math.ceil(centre_coords[1])), radius=kernel_fwhm_pixels / 2, edgecolor='green', facecolor='none', linestyle='dashed')
            axs[2].add_patch(kernel_demo)
        # Plot for corr_n
        im_3 = axs[3].imshow(corr_n, cmap='magma', origin='lower', interpolation='none')
        axs[3].set_title('Corrected Column Density Map')
        axs[3].set_xlabel('Pixel Count')
        axs[3].set_ylabel('Pixel Count')
        cbar = fig.colorbar(im_3, ax=axs[3])
        cbar.set_label('Normalised Column Density (unitless)')
        axs[3].axis('on')
        if show_kernelsize:
            kernel_demo = patches.Circle((math.ceil(centre_coords[0]), math.ceil(centre_coords[1])), radius=kernel_fwhm_pixels / 2, edgecolor='green', facecolor='none', linestyle='dashed')
            axs[3].add_patch(kernel_demo)
        if show_beamsize:
            beam_demo = patches.Circle((46, 3), radius=3, edgecolor='black', facecolor='orange')
            axs[3].add_patch(beam_demo)
        plt.show()
        #moment 1 maps;
        fig, axs = plt.subplots(1,3, figsize=(15,5))
        im_0 = axs[0].imshow(mom_1_Tmap, cmap='magma', origin='lower', interpolation='none')
        axs[0].set_title('Moment 1 Raw Map')
        axs[0].set_xlabel('Pixel Count')
        axs[0].set_ylabel('Pixel Count')
        cbar = fig.colorbar(im_0, ax=axs[0])
        cbar.set_label('km/s')
        axs[0].axis('on')
        if show_kernelsize==True:
            kernel_demo = patches.Circle((math.ceil(centre_coords[0]),math.ceil(centre_coords[1])), radius=kernel_fwhm_pixels/2, edgecolor='green', facecolor='none', linestyle='dashed')
            axs[0].add_patch(kernel_demo)
        if show_beamsize == True:
            beam_demo = patches.Circle((46,3), radius=3, edgecolor='black', facecolor='orange')
            axs[0].add_patch(beam_demo)
        im_1 = axs[1].imshow(mom1_smoothed, cmap='magma', origin='lower', interpolation='none')
        axs[1].set_title('Moment 1 Smoothed Map')
        axs[1].set_xlabel('Pixel Count')
        axs[1].set_ylabel('Pixel Count')
        cbar = fig.colorbar(im_1, ax=axs[1])
        cbar.set_label('km/s')
        axs[1].axis('on')
        if show_kernelsize==True:
            kernel_demo = patches.Circle((math.ceil(centre_coords[0]),math.ceil(centre_coords[1])), radius=kernel_fwhm_pixels/2, edgecolor='green', facecolor='none', linestyle='dashed')
            axs[1].add_patch(kernel_demo)
        if show_beamsize == True:
            beam_demo = patches.Circle((46,3), radius=3, edgecolor='black', facecolor='orange')
            axs[1].add_patch(beam_demo)
        im_2 = axs[2].imshow(corr_mom1, cmap='magma', origin='lower', interpolation='none')
        axs[2].set_title('Corrected Moment 1 Map')
        axs[2].set_xlabel('Pixel Count')
        axs[2].set_ylabel('Pixel Count')
        cbar = fig.colorbar(im_2, ax=axs[2])
        cbar.set_label('km/s')
        axs[2].axis('on')
        if show_kernelsize==True:
            kernel_demo = patches.Circle((math.ceil(centre_coords[0]),math.ceil(centre_coords[1])), radius=kernel_fwhm_pixels/2, edgecolor='green', facecolor='none', linestyle='dashed')
            axs[2].add_patch(kernel_demo)
        if show_beamsize == True:
            beam_demo = patches.Circle((46,3), radius=3, edgecolor='black', facecolor='orange')
            axs[2].add_patch(beam_demo)
        plt.show()
    #show the PDFs if requested;
    if return_corrmaps==True: 
        if return_Mach_sigma_rho == True:
            return(b, Mach, sigma_rho_3D, corr_n, corr_mom1)
        else:
            return(b, corr_n, corr_mom1)
    if return_Mach_sigma_rho == True: 
        return(b, Mach, sigma_rho_3D, sigma_rho_2D, brunt_sqrt_r, sigma_vel_1D, sigma_vel_3D)
    if full_step_returner ==True: 
        return(b, 10**log_n_smoothed, corr_n_nonan_for_fft, corr_n_nonan_for_fft*nanmask, mom1_nonan, mom1_smoothed, corr_mom1, corr_mom1*nanmask)
    return(b)


def single_segment_turbulence_mom2(kernel_FWHM_in_n_pixels, column_density_map, linewidth_map, nanmask, subsect_coords,  soundspeed = 0.6, number_sigma_FFT = 3., PrintValues = False, show_beamsize=False, return_Mach_sigma_rho = False, Gauss_Smoothing_FWHM_factor = 2, returnallmaps = False, return_powerspect = False, show_kernelsubtr=False, show_kernelsize=True, kernel_fwhm_in_n_beams=None, beam_FWHM_in_n_pixels=None):
    '''Evaluates the turbulence parameter b of one section of data.
    Inputs;
    - kernel_FWHM_in_n_pixels : the number of pixels in the kernel's FWHM
    - column_density_map : column density map in units of cm^-2
    - linewidth_map : the linewidth map - the square root of the moment 2 map in units of km/s
    - nanmask : the nanmask of the original datacube
    - subsect_coords : the coordinates of the subsection which we are analysing in the form [[x_start, x_end], [y_start, y_end]]
    - soundspeed : the soundspeed of the gas in km/s. Currently, a constant, we may want to change this to a map in the future.

    Options:
    - show_kernelsubtr : if True, shows the moment 0 map before and after the kernel subtraction/lowpass filtering. 
    - show_Xco_map : if True, shows the Xco map
    - show_powerspect : if True, shows the power spectrum map
    - PrintValues : if True, prints the values of the Brunt sqrt(R), sigma_rho_3D, sigma_vel_3D, Mach number and b-value

    Output: 
    - turbulence parameter 'b' for the individual kernel
    or;
    - b, Mach, sigma_rho_3D,sigma_rho_2D, brunt_sqrt_r,linewidth_value,sigma_vel_3D if return_Mach_sigma_rho is True and returnallmaps is False. 
    '''
    #extract subsect data;
    column_density_map = (column_density_map[subsect_coords[0][0]:subsect_coords[0][1], subsect_coords[1][0]:subsect_coords[1][1]]).value
    nanmask = nanmask[subsect_coords[0][0]:subsect_coords[0][1], subsect_coords[1][0]:subsect_coords[1][1]]
    linewidth_map = linewidth_map[subsect_coords[0][0]:subsect_coords[0][1], subsect_coords[1][0]:subsect_coords[1][1]].value
    #consider fwhm of the pixel in the number of pixels;
    kernel_fwhm_pixels = kernel_FWHM_in_n_pixels
    #set the FWHM of the gaussian smoothing function to half the FWHM of the kernel;
    FWHM_gaussfit = kernel_fwhm_pixels/Gauss_Smoothing_FWHM_factor
    sigma_gaussfit = FWHM_gaussfit / (2*np.sqrt(2*np.log(2)))
    # Correcting for larger-scale motions; 
    #for mom-0 need to do on log scales. We also want to normalise with mean density;
    columndensity_mean = weighted_meanmaker(column_density_map, kernel_fwhm_pixels, nanmask=nanmask) #the mean column density
    column_density_map_normalised = column_density_map/columndensity_mean
    #consider the nan values as the mean of the non-log column density, then the log distribution;
    column_density_map_normalised_nonan = column_density_map_normalised #NaNMasker(column_density_map_normalised, mode='Noise', returnnanmask=True)
    log_n = np.log10(column_density_map_normalised_nonan) #the log of the column density
    #replace the nan values with the mean values (need mean values as log scale)
    log_n_smoothed = generic_filter(log_n, nan_gaussian_filter, size=(50,50), extra_arguments=(sigma_gaussfit,)) # nanmask
    log_corr_n = log_n - log_n_smoothed
    corr_n= (10**log_corr_n) #(return the NaNs back so that we can divide it up however we like later)
    #remove the NaNs for later use (both for the FT and the st deviation calculations);
    corr_n_nonan_for_fft = corr_n.copy() 
    #Determine the Brunt R factor;
    #find the power spectrum of the moment 0 map; 
    #first, extract the central "number_sigma_FFT" sigma pixels to avoid edge effects in the FFT;
    center_x = int(np.shape(corr_n_nonan_for_fft)[0]/2)
    center_y = int(np.shape(corr_n_nonan_for_fft)[1]/2)
    sigma_pixels = kernel_fwhm_pixels/ (2*np.sqrt(2*np.log(2)))
    if number_sigma_FFT == 3.0:
        corr_n_nonan_for_fft_reducedregion = corr_n_nonan_for_fft
    elif number_sigma_FFT<3.0:
        extract_radius = math.ceil(number_sigma_FFT * sigma_pixels)
        corr_n_nonan_for_fft_reducedregion = corr_n_nonan_for_fft[center_x - extract_radius : center_x + extract_radius, center_y - extract_radius : center_y + extract_radius]
    elif number_sigma_FFT>3.0:
        print('number_sigma_FFT greater than 3subsect region size, defaulting to 3')
        extract_radius = math.ceil(3. * sigma_pixels)
        corr_n_nonan_for_fft_reducedregion = corr_n_nonan_for_fft[center_x - extract_radius : center_x + extract_radius, center_y - extract_radius : center_y + extract_radius]
    else: 
        print('number_sigma_FFT not recognised, defaulting to 3')
        extract_radius = math.ceil(3. * sigma_pixels)
        corr_n_nonan_for_fft_reducedregion = corr_n_nonan_for_fft[center_x - extract_radius : center_x + extract_radius, center_y - extract_radius : center_y + extract_radius]
    #now get the power spectrum;
    powerspect, k_mom0 = get_power_spectrum(corr_n_nonan_for_fft_reducedregion, mirror=True)
    #Use the previously defined function;
    brunt_sqrt_r = get_Brunt_sqrt_R(powerspect, k_mom0)
    #show the power spectrum if requested;

    #determine the st dev of the moment 0 map for the sigma_\rho;
    avg_0 = weighted_meanmaker(corr_n_nonan_for_fft*nanmask, kernel_fwhm_pixels, nanmask=nanmask)
    msq_0 = weighted_meanmaker(corr_n_nonan_for_fft**2*nanmask, kernel_fwhm_pixels, nanmask=nanmask)
    sigma_rho_2D = np.sqrt(np.abs(msq_0 - avg_0**2))
    #and from 2D st dev to 3D st dev; 
    sigma_rho_3D = sigma_rho_2D/brunt_sqrt_r

    linewidth_value = weighted_meanmaker(linewidth_map, kernel_fwhm_pixels, nanmask=nanmask)

    if kernel_fwhm_in_n_beams == None:
        if beam_FWHM_in_n_pixels is None:
            raise ValueError("beam_FWHM_in_n_pixels must be provided if kernel_fwhm_in_n_beams is None")
        else:
            kernel_fwhm_in_n_beams = kernel_fwhm_pixels/beam_FWHM_in_n_pixels
    #moment 2 doesn't require smoothing. Just take the central value, and then convert from 1D to 3D;
    sigma_vel_3D = linewidth_value * (2.34 + (2/kernel_fwhm_in_n_beams)*-0.97) # relationship from Stewart and Federrath 2021 table E1
    Mach = (sigma_vel_3D/soundspeed)
    b = sigma_rho_3D / Mach

    if show_kernelsubtr==True: 
        #moment 0 maps;
        #make subplots of the three moment 0 maps; 
        fig, axs = plt.subplots(1, 4, figsize=(24, 5))
        # Plot for column_density_map_normalised
        im_0 = axs[0].imshow(column_density_map_normalised, cmap='magma', origin='lower', interpolation='none')
        axs[0].set_title('Raw Column Density Raw Map')
        axs[0].set_xlabel('Pixel Count')
        axs[0].set_ylabel('Pixel Count')
        cbar = fig.colorbar(im_0, ax=axs[0])
        cbar.set_label('Normalised Column Density (unitless)')
        axs[0].axis('on')
        centre_coords = [np.shape(column_density_map_normalised)[0] / 2, np.shape(column_density_map_normalised)[1] / 2]
        if show_kernelsize:
            kernel_demo = patches.Circle((math.ceil(centre_coords[0]), math.ceil(centre_coords[1])), radius=kernel_fwhm_pixels / 2, edgecolor='green', facecolor='none', linestyle='dashed')
            axs[0].add_patch(kernel_demo)
        if show_beamsize:
            beam_demo = patches.Circle((46, 3), radius=3, edgecolor='black', facecolor='orange')
            axs[0].add_patch(beam_demo)
        # Plot for 10**log_n_smoothed
        im_1 = axs[1].imshow(10**log_n_smoothed, cmap='magma', origin='lower', interpolation='none')
        axs[1].set_title('Smoothed Column Density Map')
        axs[1].set_xlabel('Pixel Count')
        axs[1].set_ylabel('Pixel Count')
        cbar = fig.colorbar(im_1, ax=axs[1])
        cbar.set_label('Normalised Column Density (unitless)')
        axs[1].axis('on')
        if show_kernelsize:
            kernel_demo = patches.Circle((math.ceil(centre_coords[0]), math.ceil(centre_coords[1])), radius=kernel_fwhm_pixels / 2, edgecolor='green', facecolor='none', linestyle='dashed')
            axs[1].add_patch(kernel_demo)
        if show_beamsize:
            beam_demo = patches.Circle((46, 3), radius=3, edgecolor='black', facecolor='orange')
            axs[1].add_patch(beam_demo)
        # Plot for corr_n_nonan_for_fft
        im_2 = axs[2].imshow(corr_n_nonan_for_fft, cmap='magma', origin='lower', interpolation='none')
        axs[2].set_title('Column Density Map - NaNs filled')
        axs[2].set_xlabel('Pixel Count')
        axs[2].set_ylabel('Pixel Count')
        cbar = fig.colorbar(im_2, ax=axs[2])
        cbar.set_label('Normalised Column Density (unitless)')
        axs[2].axis('on')
        if show_kernelsize:
            kernel_demo = patches.Circle((math.ceil(centre_coords[0]), math.ceil(centre_coords[1])), radius=kernel_fwhm_pixels / 2, edgecolor='green', facecolor='none', linestyle='dashed')
            axs[2].add_patch(kernel_demo)
        # Plot for corr_n
        im_3 = axs[3].imshow(corr_n, cmap='magma', origin='lower', interpolation='none')
        axs[3].set_title('Corrected Column Density Map')
        axs[3].set_xlabel('Pixel Count')
        axs[3].set_ylabel('Pixel Count')
        cbar = fig.colorbar(im_3, ax=axs[3])
        cbar.set_label('Normalised Column Density (unitless)')
        axs[3].axis('on')
        if show_kernelsize:
            kernel_demo = patches.Circle((math.ceil(centre_coords[0]), math.ceil(centre_coords[1])), radius=kernel_fwhm_pixels / 2, edgecolor='green', facecolor='none', linestyle='dashed')
            axs[3].add_patch(kernel_demo)
        if show_beamsize:
            beam_demo = patches.Circle((46, 3), radius=3, edgecolor='black', facecolor='orange')
            axs[3].add_patch(beam_demo)
        plt.show()

    if PrintValues == True:
        print('The Brunt sqrt(R) is '+str(brunt_sqrt_r))
        print('Sigma rho 3D is '+str(sigma_rho_3D))
        print('And the sigma vel is '+ str(sigma_vel_3D))
        print('Mach number is '+str(sigma_vel_3D/soundspeed))
        print('so the b-value is '+str(b))
    
    if return_Mach_sigma_rho == True: 
        if returnallmaps == True:
            if return_powerspect == False:
                return(b, Mach, sigma_rho_3D, sigma_rho_2D, brunt_sqrt_r, linewidth_value, sigma_vel_3D, corr_n, log_corr_n, log_n_smoothed)
            else:
                return(b, Mach, sigma_rho_3D, sigma_rho_2D, brunt_sqrt_r, linewidth_value, sigma_vel_3D, corr_n, log_corr_n, log_n_smoothed, powerspect, k_mom0)
        else: 
            if return_powerspect == True:
                return(b, Mach, sigma_rho_3D,sigma_rho_2D,brunt_sqrt_r,linewidth_value,sigma_vel_3D, powerspect, k_mom0)
            else:
                return(b, Mach, sigma_rho_3D,sigma_rho_2D,brunt_sqrt_r,linewidth_value,sigma_vel_3D)
    else:
        return b


#now, the kernel requirements testing functions. One for an individual kernek, and one to make a mask of the whole map.
def kernel_req_tester_indiv(nanmap_subsection, gaussian_weighter): 
    '''Tests if there is at least a weighted 50% of not-nan pixels in the kernel (not the whole subsection)'''
    # extract just the circle of pixels from the centre of the subsection corresponding to the kernel size;
    subsect_size = np.shape(nanmap_subsection)
    ones_map = np.ones_like(nanmap_subsection)

    central_pixel = nanmap_subsection[int(subsect_size[0]/2), int(subsect_size[1]/2)]

    #multiply the nanmap by the gaussian weighter;
    weighted_nanmap = nanmap_subsection*gaussian_weighter
    weighted_onesmap = ones_map*gaussian_weighter

    weighted_sum = np.nansum(weighted_nanmap) / np.nansum(weighted_onesmap)
    if weighted_sum < 0.5:
        return False
    elif weighted_sum >= 0.5:
        if np.isnan(central_pixel) == True:
            return False
        
        elif np.isnan(central_pixel) == False:
            return True


# def kernelreq_maskmaker(nanmap, kernel_FWHM_in_pixels, subsectregion_size):
#     '''Returns a mask of the map where the kernel-weighted sum is greater than 0.5 of the maximum kernel-weighted sum, plus NaNs for all the boundary regions (where the subsection is too close to the edges of the map).'''

#     sigma_pixels = kernel_FWHM_in_pixels/(2*np.sqrt(2*np.log(2)))
#     gaussian_weighter = gaussian_2d([subsectregion_size+1, subsectregion_size+1], sigma = sigma_pixels)

#     subsection_rad = math.ceil(subsectregion_size/2)

#     map_height, map_width = nanmap.shape
#     kernel_req_mask = np.zeros(nanmap.shape, dtype=bool)
#     valid_init_coords = np.argwhere(~np.isnan(nanmap))

#     for y, x in valid_init_coords: #there's 
#         if x-subsection_rad < 0 or x+subsection_rad > map_width or y-subsection_rad < 0 or y+subsection_rad > map_height:
#             kernel_req_mask[y,x] = False
#         else: 
#             subsect_coords = [[y-subsection_rad, y+subsection_rad], [x-subsection_rad, x+subsection_rad]]
#             nanmask_slice = nanmap[subsect_coords[0][0]:subsect_coords[0][1], subsect_coords[1][0]:subsect_coords[1][1]]
#             kernel_req_mask[y,x] = kernel_req_tester_indiv(nanmask_slice, gaussian_weighter=gaussian_weighter)

#     return kernel_req_mask


def kernelreq_maskmaker(nanmap, kernel_FWHM_in_pixels, subsectregion_size):
    '''Returns a boolean mask identifying pixels whose Gaussian-weighted subsection
    meets the kernel requirement. Pixels too close to map boundaries are False.
    '''

    sigma_pixels = kernel_FWHM_in_pixels / (2 * np.sqrt(2 * np.log(2)))
    gaussian_weighter = gaussian_2d([subsectregion_size + 1, subsectregion_size + 1], sigma=sigma_pixels)

    subsection_height, subsection_width = gaussian_weighter.shape
    y_before = subsection_height // 2
    x_before = subsection_width // 2
    y_after = subsection_height - y_before
    x_after = subsection_width - x_before

    map_height, map_width = nanmap.shape
    kernel_req_mask = np.zeros(nanmap.shape, dtype=bool)
    valid_init_coords = np.argwhere(~np.isnan(nanmap))

    for y, x in valid_init_coords:
        y0, y1 = y - y_before, y + y_after
        x0, x1 = x - x_before, x + x_after

        if x0 < 0 or x1 > map_width or y0 < 0 or y1 > map_height:
            kernel_req_mask[y, x] = False
            continue

        nanmask_slice = nanmap[y0:y1, x0:x1]

        if nanmask_slice.shape != gaussian_weighter.shape:
            raise ValueError(f"Subsection shape {nanmask_slice.shape} does not match Gaussian shape {gaussian_weighter.shape}")

        kernel_req_mask[y, x] = kernel_req_tester_indiv(nanmask_slice, gaussian_weighter=gaussian_weighter)

    return kernel_req_mask



# ----------------------------------------------------------------------------

# Finally, the full roving kernel function. This makes the full maps of the turbulent parameters. 

# ----------------------------------------------------------------------------



def RovingKernel(Mom0, XCO_map, kernel_FWHM_in_pixels, mommapmethod="Linewidth", Mom1=None, Linewidth_map=None, Gauss_Smoothing_FWHM_factor=2, soundspeed=0.6, subsectregionsize_sigma = 3., kernel_FWHM_in_n_beams=5, overwrite_kernelmask=False, kernreqmask_filename=None): 
    ''' The full roving kernel function, generalised to be applied to anny CO(2-1) emission lines. Measure various turbulence parameters of the molecular gas in a galaxy, using the roving kernel method prescribed by Gerrard et al., 2023. 
    
    It has been adapted and extended by L.J.Miller et al, 2026 - please cite where used.
    
    Inputs: 
    - Mom0: moment 0 map, in units of K km/s
    - Mom1: moment 1 map, in units of km/s
    - Linewidth_map: line width map, in units of km/s (derived from square root of moment 2). The LSF must be removed prior to this step. 
    - kernel_FWHM_in_pixels: FWHM of the kernel in pixels
    - kernel_FWHM_in_n_beams: FWHM of the kernel in terms of the number of beams
    - XCO_map: map or value of the CO-to-H2 conversion factor, in units of cm^-2 (K km/s)^-1, used to convert Mom0 into column density. If a map is provided, it must be the same shape as Mom0. If a single value is provided, it will be applied to the entire map.
    - mommapmethod: string, either "Mom1", "Linewidth" (or "Mom2" accepted too), or "Parent", and it defines whether the moment-1, the linewidth (from moment-2) or the parent velocity dispersion (combination of both) is used to calculate the gas velocity dispersion.  

    Optional inputs:
    - Gauss_Smoothing_FWHM_factor: 2 by default. The fraction of the kernel FWHM to use to smooth the Mom0 and/or the Mom1 maps. So 2 means smoothing FWHM= 1/2 kernel FWHM.
    - soundspeed: the soundspeed in km/s. Assumed as a constant - may be updated in future iterations. Default is 0.6 km/s.
    - subsectregionsize_sigma: the number of sigma of the kernel to use for the subsection size. Default is 3, allowing a significant proportion to be included in the Fourier Transform process. 
    - overwrite_kernelmask: if True, the function will overwrite the existing kernel mask file if it exists. If False, it will import the existing file if it exists. See below in Notes for more information.
    - kernreqmask_filename: the filename of the kernel requirement mask. See Notes for more details.

    Outputs:
    b_map: map of the turbulence driving parameter, b. 
    Mach_map: map of the sonic Mach number.
    sigma_rho_2D_map: map of the 2D density (column density) variance.
    sigma_rho_3D_map: map of the 3D density variance.
    brunt_sqrt_r_map: map of the Brunt et al 2010 sqrt(R) factor, which converts the 2D density variance into the 3D density variance.
    sigma_v_1d_map: map of the 1D velocity dispersion.
    sigma_v_3d_map: map of the 3D velocity dispersion.

    Notes: 
    This function generates and applies a kernel-requirement mask. This follows the conditions listed in LJ Miller et al, 2026. This map is saved into a file caled '../Data/KernelReqMask_'+galaxyname+kernel_FWHM_in_pixels+'pixelFWHM.fits' IF such a file does not exist. If it does, it simply imports it, unless "overwrite_kernelmask" is set to True. 
    ''' 
    subsection_size = math.ceil(kernel_FWHM_in_pixels * (6/ (2*np.sqrt(2*np.log(2)))))
    subsection_rad = math.ceil(subsection_size/2)
    print('The subsection size is '+str(subsection_size))
    print('The subsection radius is '+str(subsection_rad))

    #convert mom0 into column density map using the XCO factor;
    raw_column_density_map =(Mom0 * XCO_map).to(u.cm**(-2))
    #covering the NaN values by interpolating to the low S/N data;
    column_density_map, nanmask_init = NaNMasker(raw_column_density_map.value,returnnanmask=True, mode='Interp')
    column_density_map = column_density_map * u.cm**(-2)

    if kernreqmask_filename is None:
        kernreqmask_filename = '../Data/KernelReqMask_'+str(kernel_FWHM_in_pixels)+'pixelFWHM.fits'

    kernelmask_exists = os.path.exists(kernreqmask_filename)
    if kernelmask_exists == True and overwrite_kernelmask == False:
        print('Kernel requirement mask already exists, importing it.')
        kernreqmask_hdul = fits.open(kernreqmask_filename)
        kernreqmask = kernreqmask_hdul[0].data
        kernreqmask_hdul.close()
    elif kernelmask_exists == True and overwrite_kernelmask == True:
        print('Kernel requirement mask could already exist, but overwrite_kernelmask is set to True. Overwriting it.')
        kernreqmask = kernelreq_maskmaker(nanmap=nanmask_init, kernel_FWHM_in_pixels=kernel_FWHM_in_pixels, subsectregion_size=int(subsectregionsize_sigma*kernel_FWHM_in_pixels))
        hdu = fits.PrimaryHDU(kernreqmask.astype(np.float32))
        hdu.writeto(kernreqmask_filename, overwrite=True)
    elif kernelmask_exists == False:
        print('Kernel requirement mask does not exist, generating it.')
        kernreqmask = kernelreq_maskmaker(nanmap=nanmask_init, kernel_FWHM_in_pixels=kernel_FWHM_in_pixels, subsectregion_size=int(subsectregionsize_sigma*kernel_FWHM_in_pixels))
        hdu = fits.PrimaryHDU(kernreqmask.astype(np.float32))
        hdu.writeto(kernreqmask_filename, overwrite=True)
    else: 
        print('Error in kernel requirement mask generation. Please check the inputs.')
        return None

    

    print('Column density map made, now generating empty maps;')
    b_map = np.full(np.shape(column_density_map), np.nan)
    Mach_map = np.full(np.shape(column_density_map), np.nan)
    sigma_rho_3D_map = np.full(np.shape(column_density_map), np.nan)
    sigma_rho_2D_map = np.full(np.shape(column_density_map), np.nan)
    brunt_sqrt_r_map = np.full(np.shape(column_density_map), np.nan)    
    sigma_v_3d_map = np.full(np.shape(column_density_map), np.nan)

    if mommapmethod == "Mom1":
        sigma_v_1d_map = np.full(np.shape(column_density_map), np.nan)
    elif mommapmethod == "Mom2" or mommapmethod=="Linewidth":
        linewidth_map_final = np.full(np.shape(column_density_map), np.nan)
    elif mommapmethod == "Parent":
        raise ValueError('This has not been implemented yet. Please use either Mom1 or Linewidth as the mommapmethod until this has been properly constructed.')

    
    print('Empty maps generated.')

    kernel_valid_coords = np.argwhere(kernreqmask==True)

    # make a counter/percentage tracker for the progress of the loop, and every 10% it sends out a message.
    num_spaxels_total = len(kernel_valid_coords)
    counter = 0
    checkpoints = {
        int(num_spaxels_total * 0.1):  r"$10\%$",
        int(num_spaxels_total * 0.2):  r"$20\%$",
        int(num_spaxels_total * 0.3):  r"$30\%$",
        int(num_spaxels_total * 0.4):  r"$40\%$",
        int(num_spaxels_total * 0.5):  r"$50\%$",
        int(num_spaxels_total * 0.6):  r"$60\%$",
        int(num_spaxels_total * 0.7):  r"$70\%$",
        int(num_spaxels_total * 0.8):  r"$80\%$",
        int(num_spaxels_total * 0.9):  r"$90\%$",
    }

    print('Great! Now running the roving kernel. ')
    if mommapmethod == "Mom1":
        if Mom1 is None:
            raise ValueError('Mom1 map is required for mommapmethod "Mom1". Please provide a valid Mom1 map.')
        print('Using the moment-1 map to derive the velocity dispersion. Smoothing the moment-1, subtracting, then taking the standard deviation of the residuals. Usual process for the column density. See LJ Miller et al, 2026 for details.')

        for y, x in kernel_valid_coords: 
            counter += 1
            if counter in checkpoints:
                print(f"Progress: {checkpoints[counter]} complete.")

            subsect_coords = [[y-subsection_rad, y+subsection_rad], [x-subsection_rad, x+subsection_rad]]
            nanmask_slice = nanmask_init[subsect_coords[0][0]:subsect_coords[0][1], subsect_coords[1][0]:subsect_coords[1][1]]

            b_curr, Mach_curr, sigma_rho_3D_curr, sigma_rho_2D_curr, brunt_sqrt_r_curr, sigma_vel_1D_curr, sigma_vel_3D_curr = single_segment_turbulence_mom1(kernel_fwhm_in_n_beams=None, column_density_map=column_density_map, mom1=Mom1, nanmask=nanmask_init, subsect_coords=subsect_coords, soundspeed=soundspeed, number_sigma_FFT=3., show_kernelsubtr=False, show_kernelsize=False, show_powerspect=False, PrintValues=False, show_beamsize=False, return_corrmaps=False, return_Mach_sigma_rho = True, full_step_returner = False, Gauss_Smoothing_FWHM_factor=Gauss_Smoothing_FWHM_factor, beam_FWHM_in_n_pixels=None, kernel_FWHM_in_n_pixels=kernel_FWHM_in_pixels)

            b_map[y,x] = b_curr
            Mach_map[y,x] = Mach_curr
            sigma_rho_3D_map[y,x] = sigma_rho_3D_curr
            sigma_rho_2D_map[y,x] = sigma_rho_2D_curr
            brunt_sqrt_r_map[y,x] = brunt_sqrt_r_curr
            sigma_v_1d_map[y,x] = sigma_vel_1D_curr
            sigma_v_3d_map[y,x] = sigma_vel_3D_curr

    elif mommapmethod == "Mom2" or mommapmethod=="Linewidth":
        if Linewidth_map is None:
            raise ValueError('Linewidth map is required for mommapmethod "Mom2" or "Linewidth". Please provide a valid Linewidth map.')

        print('Using the linewidth map to derive the velocity dispersion. Taking the central value of the linewidth map, then converting from 1D to 3D. Usual process for the column density. See LJ Miller et al, 2026 for details.')

        for y, x in kernel_valid_coords:
            counter += 1
            if counter in checkpoints:
                print(f"Progress: {checkpoints[counter]} complete.")

            subsect_coords = [[y-subsection_rad, y+subsection_rad], [x-subsection_rad, x+subsection_rad]]

            b_curr, Mach_curr, sigma_rho_3D_curr, sigma_rho_2D_curr, brunt_sqrt_r_curr, linewidth_curr, sigma_vel_3D_curr = single_segment_turbulence_mom2(kernel_FWHM_in_n_pixels=kernel_FWHM_in_pixels, column_density_map=column_density_map, linewidth_map=Linewidth_map, nanmask=nanmask_init, subsect_coords=subsect_coords,  soundspeed = 0.6, number_sigma_FFT = 3., PrintValues = False, show_beamsize=False, return_Mach_sigma_rho = True, Gauss_Smoothing_FWHM_factor = Gauss_Smoothing_FWHM_factor, returnallmaps = False, return_powerspect = False, show_kernelsubtr=False, show_kernelsize=True, kernel_fwhm_in_n_beams=kernel_FWHM_in_n_beams)

            b_map[y,x] = b_curr
            Mach_map[y,x] = Mach_curr
            sigma_rho_3D_map[y,x] = sigma_rho_3D_curr
            sigma_rho_2D_map[y,x] = sigma_rho_2D_curr
            brunt_sqrt_r_map[y,x] = brunt_sqrt_r_curr
            linewidth_map_final[y,x] = linewidth_curr
            sigma_v_3d_map[y,x] = sigma_vel_3D_curr

    else: 
        raise ValueError('mommapmethod not recognised. Please use either "Mom1" or "Linewidth" (or "Mom2") as the mommapmethod.')

    print('Roving kernel complete!')
    print('Now returning the maps of the turbulence measurements.')

    if mommapmethod == "Mom1":
        return b_map, Mach_map, sigma_rho_3D_map, sigma_rho_2D_map, brunt_sqrt_r_map, sigma_v_1d_map, sigma_v_3d_map
    elif mommapmethod == "Mom2" or mommapmethod=="Linewidth":
        return b_map, Mach_map, sigma_rho_3D_map, sigma_rho_2D_map, brunt_sqrt_r_map, linewidth_map_final, sigma_v_3d_map

    

# ---------------------------------------------------------------------------

# A multi-thread/multi-processing version of the roving kernel function. 

# ---------------------------------------------------------------------------


# Global variables used by the multiprocessing workers.
_ROVING_KERNEL_WORKER_ARGS = None


def _initialise_roving_kernel_worker(column_density_map, Mom1, Linewidth_map, nanmask_init, mommapmethod, subsection_rad, soundspeed, kernel_FWHM_in_pixels, Gauss_Smoothing_FWHM_factor, kernel_FWHM_in_n_beams):
    global _ROVING_KERNEL_WORKER_ARGS
    _ROVING_KERNEL_WORKER_ARGS = (column_density_map, Mom1, Linewidth_map, nanmask_init, mommapmethod, subsection_rad, soundspeed, kernel_FWHM_in_pixels, Gauss_Smoothing_FWHM_factor, kernel_FWHM_in_n_beams)


def _roving_kernel_worker(coord):
    y, x = coord
    column_density_map, Mom1, Linewidth_map, nanmask_init, mommapmethod, subsection_rad, soundspeed, kernel_FWHM_in_pixels, Gauss_Smoothing_FWHM_factor, kernel_FWHM_in_n_beams = _ROVING_KERNEL_WORKER_ARGS
    subsect_coords = [[y-subsection_rad, y+subsection_rad], [x-subsection_rad, x+subsection_rad]]

    if mommapmethod == "Mom1":
        b_curr, Mach_curr, sigma_rho_3D_curr, sigma_rho_2D_curr, brunt_sqrt_r_curr, sigma_vel_1D_curr, sigma_vel_3D_curr = single_segment_turbulence_mom1(kernel_fwhm_in_n_beams=None, column_density_map=column_density_map, mom1=Mom1, nanmask=nanmask_init, subsect_coords=subsect_coords, soundspeed=soundspeed, number_sigma_FFT=3., show_kernelsubtr=False, show_kernelsize=False, show_powerspect=False, PrintValues=False, show_beamsize=False, return_corrmaps=False, return_Mach_sigma_rho=True, full_step_returner=False, Gauss_Smoothing_FWHM_factor=Gauss_Smoothing_FWHM_factor, beam_FWHM_in_n_pixels=None, kernel_FWHM_in_n_pixels=kernel_FWHM_in_pixels)
        return y, x, b_curr, Mach_curr, sigma_rho_3D_curr, sigma_rho_2D_curr, brunt_sqrt_r_curr, sigma_vel_1D_curr, sigma_vel_3D_curr

    elif mommapmethod == "Mom2" or mommapmethod == "Linewidth":
        b_curr, Mach_curr, sigma_rho_3D_curr, sigma_rho_2D_curr, brunt_sqrt_r_curr, linewidth_curr, sigma_vel_3D_curr = single_segment_turbulence_mom2(kernel_FWHM_in_n_pixels=kernel_FWHM_in_pixels, column_density_map=column_density_map, linewidth_map=Linewidth_map, nanmask=nanmask_init, subsect_coords=subsect_coords, soundspeed=0.6, number_sigma_FFT=3., PrintValues=False, show_beamsize=False, return_Mach_sigma_rho=True, Gauss_Smoothing_FWHM_factor=Gauss_Smoothing_FWHM_factor, returnallmaps=False, return_powerspect=False, show_kernelsubtr=False, show_kernelsize=False, kernel_fwhm_in_n_beams=kernel_FWHM_in_n_beams)
        return y, x, b_curr, Mach_curr, sigma_rho_3D_curr, sigma_rho_2D_curr, brunt_sqrt_r_curr, linewidth_curr, sigma_vel_3D_curr

    else:
        raise ValueError('mommapmethod not recognised. Please use either "Mom1" or "Linewidth" (or "Mom2") as the mommapmethod.')


def RovingKernel_MP(Mom0, XCO_map, kernel_FWHM_in_pixels, mommapmethod="Linewidth", Mom1=None, Linewidth_map=None, Gauss_Smoothing_FWHM_factor=2, soundspeed=0.6, subsectregionsize_sigma = 3., kernel_FWHM_in_n_beams=5., overwrite_kernelmask=False, kernreqmask_filename=None, n_processes=32): 
    ''' A multi-processing version of the roving kernel function. The full roving kernel function, generalised to be applied to anny CO(2-1) emission lines. Measure various turbulence parameters of the molecular gas in a galaxy, using the roving kernel method prescribed by Gerrard et al., 2023. 
    
    It has been adapted and extended by L.J.Miller et al, 2026 - please cite where used.
    
    Inputs: 
    - Mom0: moment 0 map, in units of K km/s
    - Mom1: moment 1 map, in units of km/s
    - Linewidth_map: line width map, in units of km/s (derived from square root of moment 2). The LSF must be removed prior to this step. 
    - kernel_FWHM_in_pixels: FWHM of the kernel in pixels
    - kernel_FWHM_in_n_beams: FWHM of the kernel in terms of the number of beams. 5 is default (enough independent measurements within a kernel to be statistically significant). 
    - XCO_map: map or value of the CO-to-H2 conversion factor, in units of cm^-2 (K km/s)^-1, used to convert Mom0 into column density. If a map is provided, it must be the same shape as Mom0. If a single value is provided, it will be applied to the entire map.
    - mommapmethod: string, either "Mom1", "Linewidth" (or "Mom2" accepted too), or "Parent", and it defines whether the moment-1, the linewidth (from moment-2) or the parent velocity dispersion (combination of both) is used to calculate the gas velocity dispersion. Default is the moment-2 measure.

    Optional inputs:
    - Gauss_Smoothing_FWHM_factor: 2 by default. The fraction of the kernel FWHM to use to smooth the Mom0 and/or the Mom1 maps. So 2 means smoothing FWHM= 1/2 kernel FWHM.
    - soundspeed: the soundspeed in km/s. Assumed as a constant - may be updated in future iterations. Default is 0.6 km/s.
    - subsectregionsize_sigma: the number of sigma of the kernel to use for the subsection size. Default is 3, allowing a significant proportion to be included in the Fourier Transform process. 
    - overwrite_kernelmask: if True, the function will overwrite the existing kernel mask file if it exists. If False, it will import the existing file if it exists. See below in Notes for more information.
    - kernreqmask_filename: the filename of the kernel requirement mask. See Notes for more details.
    - n_processes: number of processes to use for multiprocessing. Default is 32, but should be adjusted for computational availability.

    Outputs:
    b_map: map of the turbulence driving parameter, b. 
    Mach_map: map of the sonic Mach number.
    sigma_rho_2D_map: map of the 2D density (column density) variance.
    sigma_rho_3D_map: map of the 3D density variance.
    brunt_sqrt_r_map: map of the Brunt et al 2010 sqrt(R) factor, which converts the 2D density variance into the 3D density variance.
    sigma_v_1d_map: map of the 1D velocity dispersion.
    sigma_v_3d_map: map of the 3D velocity dispersion.

    Notes: 
    This function generates and applies a kernel-requirement mask. This follows the conditions listed in LJ Miller et al, 2026. This map is saved into a file caled '../Data/KernelReqMask_'+galaxyname+kernel_FWHM_in_pixels+'pixelFWHM.fits' IF such a file does not exist. If it does, it simply imports it, unless "overwrite_kernelmask" is set to True. 
    ''' 
    subsection_size = math.ceil(kernel_FWHM_in_pixels * (6/ (2*np.sqrt(2*np.log(2)))))
    subsection_rad = math.ceil(subsection_size/2)
    print('The subsection size is '+str(subsection_size))
    print('The subsection radius is '+str(subsection_rad))

    #convert mom0 into column density map using the XCO factor;
    raw_column_density_map =(Mom0 * XCO_map).to(u.cm**(-2))
    #covering the NaN values by interpolating to the low S/N data;
    column_density_map, nanmask_init = NaNMasker(raw_column_density_map.value,returnnanmask=True, mode='Interp')
    column_density_map = column_density_map * u.cm**(-2)

    if kernreqmask_filename is None:
        kernreqmask_filename = '../Data/KernelReqMask_'+str(kernel_FWHM_in_pixels)+'pixelFWHM.fits'

    kernelmask_exists = os.path.exists(kernreqmask_filename)
    if kernelmask_exists == True and overwrite_kernelmask == False:
        print('Kernel requirement mask already exists, importing it.')
        kernreqmask_hdul = fits.open(kernreqmask_filename)
        kernreqmask = kernreqmask_hdul[0].data
        kernreqmask_hdul.close()
    elif kernelmask_exists == True and overwrite_kernelmask == True:
        print('Kernel requirement mask could already exist, but overwrite_kernelmask is set to True. Overwriting it.')
        kernreqmask = kernelreq_maskmaker(nanmap=nanmask_init, kernel_FWHM_in_pixels=kernel_FWHM_in_pixels, subsectregion_size=int(subsectregionsize_sigma*kernel_FWHM_in_pixels))
        hdu = fits.PrimaryHDU(kernreqmask.astype(np.float32))
        hdu.writeto(kernreqmask_filename, overwrite=True)
    elif kernelmask_exists == False:
        print('Kernel requirement mask does not exist, generating it.')
        kernreqmask = kernelreq_maskmaker(nanmap=nanmask_init, kernel_FWHM_in_pixels=kernel_FWHM_in_pixels, subsectregion_size=int(subsectregionsize_sigma*kernel_FWHM_in_pixels))
        hdu = fits.PrimaryHDU(kernreqmask.astype(np.float32))
        hdu.writeto(kernreqmask_filename, overwrite=True)
    else: 
        print('Error in kernel requirement mask generation. Please check the inputs.')
        return None

    

    print('Column density map made, now generating empty maps;')
    b_map = np.full(np.shape(column_density_map), np.nan)
    Mach_map = np.full(np.shape(column_density_map), np.nan)
    sigma_rho_3D_map = np.full(np.shape(column_density_map), np.nan)
    sigma_rho_2D_map = np.full(np.shape(column_density_map), np.nan)
    brunt_sqrt_r_map = np.full(np.shape(column_density_map), np.nan)    
    sigma_v_3d_map = np.full(np.shape(column_density_map), np.nan)

    if mommapmethod == "Mom1":
        sigma_v_1d_map = np.full(np.shape(column_density_map), np.nan)
    elif mommapmethod == "Mom2" or mommapmethod=="Linewidth":
        linewidth_map_final = np.full(np.shape(column_density_map), np.nan)
    elif mommapmethod == "Parent":
        raise ValueError('This has not been implemented yet. Please use either Mom1 or Linewidth as the mommapmethod until this has been properly constructed.')

    
    print('Empty maps generated.')

    kernel_valid_coords = np.argwhere(kernreqmask==True)

    # make a counter/percentage tracker for the progress of the loop, and every 10% it sends out a message.
    num_spaxels_total = len(kernel_valid_coords)
    counter = 0
    checkpoints = {
        int(num_spaxels_total * 0.1):  r"$10\%$",
        int(num_spaxels_total * 0.2):  r"$20\%$",
        int(num_spaxels_total * 0.3):  r"$30\%$",
        int(num_spaxels_total * 0.4):  r"$40\%$",
        int(num_spaxels_total * 0.5):  r"$50\%$",
        int(num_spaxels_total * 0.6):  r"$60\%$",
        int(num_spaxels_total * 0.7):  r"$70\%$",
        int(num_spaxels_total * 0.8):  r"$80\%$",
        int(num_spaxels_total * 0.9):  r"$90\%$",
    }

    print('Great! Now running the multiprocessing roving kernel.')

    if mommapmethod == "Mom1" and Mom1 is None:
        raise ValueError('Mom1 map is required for mommapmethod "Mom1". Please provide a valid Mom1 map.')

    if (mommapmethod == "Mom2" or mommapmethod == "Linewidth") and Linewidth_map is None:
        raise ValueError('Linewidth map is required for mommapmethod "Mom2" or "Linewidth". Please provide a valid Linewidth map.')

    if mommapmethod not in ("Mom1", "Mom2", "Linewidth"):
        raise ValueError('mommapmethod not recognised. Please use either "Mom1" or "Linewidth" (or "Mom2") as the mommapmethod.')

    print(f'Using {mommapmethod} to derive the velocity dispersion.')

    num_spaxels_total = len(kernel_valid_coords)

    if num_spaxels_total == 0:
        print('No valid kernel coordinates found. Returning empty maps.')

    else:
        #run the multiprocessing pool with the specified number of processes. 2 by default, but should be set by user. 
        num_processes = min(n_processes, num_spaxels_total)
        if num_processes < 1:
            raise ValueError('n_processes must be at least 1.')
        chunksize = max(1, num_spaxels_total // (num_processes * 8))
        print(f'Running {num_spaxels_total} kernel positions across {num_processes} worker processes, with chunksize {chunksize}.')

        worker_args = (column_density_map, Mom1, Linewidth_map, nanmask_init, mommapmethod, subsection_rad, soundspeed, kernel_FWHM_in_pixels, Gauss_Smoothing_FWHM_factor, kernel_FWHM_in_n_beams)

        with mp.Pool(processes=num_processes, initializer=_initialise_roving_kernel_worker, initargs=worker_args) as pool:
            for counter, result in enumerate(pool.imap_unordered(_roving_kernel_worker, kernel_valid_coords, chunksize=chunksize), start=1):
                y, x, b_curr, Mach_curr, sigma_rho_3D_curr, sigma_rho_2D_curr, brunt_sqrt_r_curr, velocity_curr, sigma_vel_3D_curr = result

                b_map[y, x] = b_curr
                Mach_map[y, x] = Mach_curr
                sigma_rho_3D_map[y, x] = sigma_rho_3D_curr
                sigma_rho_2D_map[y, x] = sigma_rho_2D_curr
                brunt_sqrt_r_map[y, x] = brunt_sqrt_r_curr
                sigma_v_3d_map[y, x] = sigma_vel_3D_curr

                if mommapmethod == "Mom1":
                    sigma_v_1d_map[y, x] = velocity_curr
                else:
                    linewidth_map_final[y, x] = velocity_curr

                if counter % max(1, num_spaxels_total // 10) == 0 or counter == num_spaxels_total:
                    print(f'Progress: {100 * counter / num_spaxels_total:.1f}% complete ({counter}/{num_spaxels_total} kernel positions).')

    print('Multiprocessing roving kernel complete!')
    print('Now returning the maps of the turbulence measurements.')

    if mommapmethod == "Mom1":
        return b_map, Mach_map, sigma_rho_3D_map, sigma_rho_2D_map, brunt_sqrt_r_map, sigma_v_1d_map, sigma_v_3d_map
    elif mommapmethod == "Mom2" or mommapmethod == "Linewidth":
        return b_map, Mach_map, sigma_rho_3D_map, sigma_rho_2D_map, brunt_sqrt_r_map, linewidth_map_final, sigma_v_3d_map



# ----------------------------------------------------------------------------

# Some functions specifically helpful for the NGC1313, NGC7793 and NGC628 data;

# ----------------------------------------------------------------------------

def Xco_maker(PeakTempMap, header, galaxyname='NGC7793', beamwidth=13, sol_metallicity=8.69): 
    ''' A function to return the CO-to-H2 conversion factor map, Xco, in units of cm^-2 (K km/s)^-1, for a given galaxy. This function is targetted for NGC7793, NGC1313 and NGC628, but can be adapted for other galaxies. It uses the metallicity gradient of the galaxy to determine the Xco factor, following Gong et al 2020.
    Inputs:
    - PeakTempMap: the peak temperature map of the CO(2-1) emission (K)
    - header: the header of the FITS file containing the PeakTempMap, used to determine the RA and DEC of each pixel
    - galaxyname: the name of the galaxy, either 'NGC7793', 'NGC1313' or 'NGC628' (for now)
    - beamwidth: the beamwidth of the observation in parsec (default is 13) 
    - sol_metallicity: the metallicity of the Sun (default is 8.69)

    Outputs: 
    - Xco_map: the CO-to-H2 conversion factor map in units of cm^-2 (K km/s)^-1
    '''

    #first isolate the galaxy, as NGC7793 has a metallicity gradient, but NGC1313 does not;
    if galaxyname=='NGC7793':
        #NGC7793 has a metallicity gradient from the centre, so the center coords; 
        galcent_RA = 359.457625
        galcent_DEC = -32.591028
        distance = np.zeros_like(PeakTempMap)
        for i in range(np.shape(PeakTempMap)[0]):
            for j in range(np.shape(PeakTempMap)[1]):
                curr_ra = (header['CRVAL1'] - (header['CRPIX1'] - j)*header['CDELT1']) 
                curr_dec = (header['CRVAL2'] + (header['CRPIX2'] - i)*header['CDELT2'])
                distance[i,j] = (np.sqrt((curr_ra - galcent_RA)**2 + (curr_dec - galcent_DEC)**2)) 
        distance_kpc = 3.7*10**3 * np.tan(np.radians(distance*u.deg)) * u.kpc
        #Metallicity decreasing as used in Finn et al 2024;
        logOH = 8.572 - 10**(0.054/u.kpc * distance_kpc).si
    elif galaxyname=='NGC1313':
        #NGC1313 does not have a metallicity gradient;
        logOH = 8.4       
    OH_curr = 10**(logOH-12)
    OH_sol = 10**(sol_metallicity-12)
    metallicity = OH_sol/OH_curr
    #Determine X_Co from Gong et al 2020, equation 3b:
    #if PeakTempMap is in u.K, converit it to dimensionless;
    if isinstance(PeakTempMap, u.Quantity):
        PeakTempMap = PeakTempMap.to(u.K).value
    Xco_map = (2.7*10**20)*(PeakTempMap**(-1.07+0.37*np.log10(beamwidth)) )* (metallicity**(-0.5))* (beamwidth**-0.13)*u.cm**-2/(u.K*u.km*u.s**-1)
    return(Xco_map)
