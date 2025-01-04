"""
Preprocessing Script for Astronomical Data Reduction
=====================================================

Description:
------------
This script handles the preprocessing stage of the data reduction pipeline.  
It processes flat and dark frames to generate master flats and masks needed  
for subsequent image correction steps.

Workflow:
---------
1. **Flat Frame Processing:**
   - Reads and sorts flat frames by exposure time.
   - Computes mean values for each frame and organizes data.

2. **Dark Frame Processing:**
   - Reads and sorts dark frames by exposure time.
   - Computes mean values and saves dark frames along with exposure times.

3. **Master Flat Creation:**
   - Generates a master flat by combining and normalizing flat frames.
   - Applies optional dark subtraction based on exposure time.

4. **Mask Creation:**
   - Computes pixel masks based on thresholds and filtering.
   - Flags bad pixels and artifacts for masking during later steps.

5. **Output:**
   - Saves processed data (master flats, masks, and darks) in both FITS and NPY formats.

"""


import os
import sys
import glob
import numpy as np
from astropy.io import fits
import logging
from scipy.ndimage import gaussian_filter, median_filter

# Add config directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
from parameters import config

# Setup logging
log_file = os.path.join(config['paths']['log_file'])
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO)


# Create output directory if not exists
os.makedirs(config['paths']['output_dir'], exist_ok=True)


def sort_flats(rep_flat):
    """
    Load and sort flat frames, extracting exposure times and computing mean values.

    Parameters
    ----------
    rep_flat : str
        Path to the directory containing flat FITS files.

    Returns
    -------
    times : numpy.ndarray
        Array of exposure times extracted from the FITS headers.
    flats : numpy.ndarray
        Array of mean values computed for each flat frame.

    Workflow
    --------
    1. Reads all FITS files from the specified directory.
    2. Extracts exposure times from the FITS headers.
    3. Computes the mean value for each flat frame (averaging along the spatial dimensions).
    4. Logs the number of flat frames processed and their exposure times.
    5. Returns arrays containing exposure times and mean values of the flats.
    """
    
    files_flat = sorted(glob.glob(rep_flat + '/*.fits'))
    times = []
    flats = []
    for f in files_flat:
        hdu = fits.open(f)
        header = hdu[0].header
        time_flat = header['ESO DET SEQ1 DIT']
        times.append(time_flat)
        flats.append(np.mean(hdu[0].data, 0))
    flats = np.array(flats)
    times = np.array(times)
    logging.info(f"Loaded {len(times)} flat frames with times: {times}")
    return times, flats


def sort_darks(rep_dark):
    """
    Load and sort dark frames, extracting exposure times and computing mean values.

    Parameters
    ----------
    rep_dark : str
        Path to the directory containing dark FITS files.

    Returns
    -------
    times : numpy.ndarray
        Array of exposure times extracted from the FITS headers.
    darks : numpy.ndarray
        Array of mean values computed for each dark frame.

    Workflow
    --------
    1. Reads all FITS files from the specified directory.
    2. Extracts exposure times from the FITS headers.
    3. Computes the mean value for each dark frame (averaging along the spatial dimensions).
    4. Logs the number of dark frames processed and their exposure times.
    5. Returns arrays containing exposure times and mean values of the darks.

    """

    """Sort and load dark frames, computing their mean values and exposure times."""
    files_dark = sorted(glob.glob(rep_dark + '/*.fits'))
    times = []
    darks = []
    for f in files_dark:
        hdu = fits.open(f)
        header = hdu[0].header
        time_dark = header['ESO DET SEQ1 DIT']
        times.append(time_dark)
        darks.append(np.mean(hdu[0].data, 0))
    darks = np.array(darks)
    times = np.array(times)
    logging.info(f"Loaded {len(times)} dark frames with times: {times}")
    return times, darks

def save_darks(times_darks, darks):
    """
    Save dark frames and their associated exposure times in FITS and NPY formats.

    Parameters
    ----------
    times_darks : numpy.ndarray
        Array of exposure times for the dark frames.
    darks : numpy.ndarray
        Array of mean values computed for each dark frame.

    Workflow
    --------
    1. Saves the dark frames as a FITS file in the output directory.
    2. Saves the dark frames and their exposure times as NPY files for faster loading in later stages.
    3. Logs a confirmation message once the files are saved successfully.

    Output
    ------
    - 'darks.fits': Dark frames stored in FITS format.
    - 'darks.npy': Dark frames stored in NPY format.
    - 'times_darks.npy': Exposure times stored in NPY format.

    """

    output_dir = config['paths']['output_dir']
    fits.writeto(os.path.join(output_dir, 'darks.fits'), darks, overwrite=config['output']['overwrite'])
    np.save(os.path.join(output_dir, 'darks.npy'), darks)
    np.save(os.path.join(output_dir, 'times_darks.npy'), times_darks)
    logging.info("Saved dark frames and associated times.")


def make_master_flat(times_flats, flats, times_darks, darks):
    """
    Create a master flat by combining minimum and maximum flats with optional dark subtraction.

    Parameters
    ----------
    times_flats : numpy.ndarray
        Array of exposure times for the flat frames.
    flats : numpy.ndarray
        Array of mean values computed for each flat frame.
    times_darks : numpy.ndarray
        Array of exposure times for the dark frames.
    darks : numpy.ndarray
        Array of mean values computed for each dark frame.

    Returns
    -------
    master_flat : numpy.ndarray
        The final master flat frame after dark subtraction (if applicable).

    Workflow
    --------
    1. Selects the flat frames with the minimum and maximum exposure times.
    2. Matches and subtracts corresponding dark frames if available:
       - Uses darks with matching exposure times for subtraction.
       - Logs a warning and skips dark subtraction if matching darks are missing.
    3. Computes the master flat by subtracting the minimum flat from the maximum flat.
    4. Saves the master flat in both FITS and NPY formats for further processing.

    Output
    ------
    - 'master_flat.fits': Master flat saved in FITS format.
    - 'master_flat.npy': Master flat saved in NPY format.

    """

    flat_max = flats[-1]
    flat_min = flats[0]
    time_max = times_flats[-1]
    time_min = times_flats[0]

    dark_min = np.zeros(np.shape(flat_min))
    dark_max = np.zeros(np.shape(flat_max))

    t_min_good, t_max_good = False, False
    for k in range(len(times_darks)):
        t = times_darks[k]
        if t == time_min:
            dark_min = darks[k]
            t_min_good = True
        if t == time_max:
            dark_max = darks[k]
            t_max_good = True

    # Handle missing darks by skipping subtraction
    if not t_min_good or not t_max_good:
        logging.warning('Missing dark for flats, using no dark subtraction')
        master_flat = flat_max - flat_min
    else:
        master_flat = (flat_max - dark_max) - (flat_min - dark_min)

    # Save master flat
    fits.writeto(os.path.join(config['paths']['output_dir'], 'master_flat.fits'), master_flat, overwrite=config['output']['overwrite'])
    np.save(os.path.join(config['paths']['output_dir'], 'master_flat.npy'), master_flat)
    logging.info("Master flat created and saved successfully.")
    return master_flat

def create_masks(master_flat):
    """
    Create pixel masks to filter out extreme values and artifacts based on thresholds and filtering.

    Parameters
    ----------
    master_flat : numpy.ndarray
        The master flat frame used for mask creation.

    Returns
    -------
    mask : numpy.ndarray
        Boolean array marking bad pixels (True) and good pixels (False).

    Workflow
    --------
    1. Percentile Thresholding:
       - Computes upper and lower percentile thresholds from the master flat.
       - Flags pixels outside these thresholds as bad pixels.

    2. Gaussian Filtering:
       - Applies a Gaussian filter to smooth the master flat.
       - Flags pixels with residuals (difference between flat and filtered data) above 1 standard deviation.

    3. Median Filtering:
       - Applies a median filter to remove noise.
       - Flags pixels with residuals above 1 standard deviation.

    4. Mask Combination:
       - Combines masks from percentile, Gaussian, and median filtering into a single binary mask.

    5. Output:
       - Saves the mask in both FITS and NPY formats for future use.

    Output
    ------
    - 'mask.fits': Mask saved in FITS format.
    - 'mask.npy': Mask saved in NPY format.

    """

    logging.info("Creating masks")

    # Compute percentiles for thresholds
    lower_thresh = np.percentile(master_flat, config['mask']['threshold_lower'] * 100)
    upper_thresh = np.percentile(master_flat, config['mask']['threshold_upper'] * 100)
    logging.info(f"Lower threshold: {lower_thresh}, Upper threshold: {upper_thresh}")

    # Create mask based on percentile thresholds
    mask = (master_flat < lower_thresh) | (master_flat > upper_thresh)
    logging.info(f"Number of masked pixels: {np.sum(mask)}, Unmasked pixels: {np.size(mask) - np.sum(mask)}")

    # # Ensure kernel size is a valid 2D tuple
    kernel_size = config['processing']['gaussian_filter_size']
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
        
    diff = master_flat-gaussian_filter(master_flat, kernel_size)
    std_diff = np.std(diff)
    mask_b = abs(diff)>std_diff
    
    # Ensure kernel size is a valid 2D tuple
    kernel_size = config['processing']['median_filter_size']
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
        
    diff = master_flat-median_filter(master_flat, kernel_size)
    std_diff = np.std(diff)
    mask_c = abs(diff)>(std_diff)

    # Mix all three masks
    mask = np.array(mask+mask_b+mask_c, dtype=bool)
    
    # Save mask
    fits.writeto(os.path.join(config['paths']['output_dir'], 'mask.fits'), mask.astype(int), overwrite=config['output']['overwrite'])
    np.save(os.path.join(config['paths']['output_dir'], 'mask.npy'), mask)
    logging.info("Mask created and saved successfully.")
    return mask


if __name__ == '__main__':
    # Preprocessing workflow
    times_flats, flats = sort_flats(config['paths']['flat_dir'])
    times_darks, darks = sort_darks(config['paths']['dark_dir'])
    save_darks(times_darks, darks)
    master_flat = make_master_flat(times_flats, flats, times_darks, darks)
    mask = create_masks(master_flat)
    logging.info("Preprocessing completed successfully.")
