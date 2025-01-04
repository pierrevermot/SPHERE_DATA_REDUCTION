"""
Reduction Script for Astronomical Data Processing
==================================================

Description:
------------
This script performs the data reduction step of the pipeline.  
It applies flat-field correction and dark subtraction to raw astronomical images,  
preparing them for alignment and calibration.

Workflow:
---------
1. **Load Object Files:**
   - Reads and organizes raw object frames for processing.

2. **Dark Frame Handling:**
   - Loads preprocessed dark frames and associated exposure times.
   - Matches dark frames to object frames based on exposure time.
   - Uses the closest match when an exact exposure time is unavailable.

3. **Flat-Field Correction:**
   - Applies master flat and pixel masks to normalize pixel intensities.
   - Masks bad pixels and artifacts.

4. **Output:**
   - Saves reduced images in both FITS and NPY formats for further processing.

"""


import os
import sys
import glob
import numpy as np
from astropy.io import fits
import logging

# Add config directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
from parameters import config

# Setup logging
log_file = os.path.join(config['paths']['log_file'])
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO)


def load_object_files(rep_object):
    """
    Load object FITS files and extract data and headers.

    Parameters
    ----------
    rep_object : str
        Path to the directory containing object FITS files.

    Returns
    -------
    files : list of str
        List of file paths for the loaded FITS files.
    data : list of numpy.ndarray
        List of image data arrays extracted from each FITS file.
    headers : list of astropy.io.fits.Header
        List of FITS headers associated with each file.

    Workflow
    --------
    1. Reads all FITS files from the specified directory, sorted alphabetically or numerically.
    2. Opens each FITS file and extracts:
       - Image data from the primary HDU.
       - Header metadata for reference.
    3. Logs the number of files processed and loaded.
    4. Returns the list of file paths, image data arrays, and headers.
    
    """
    
    files = sorted(glob.glob(rep_object + '/*.fits'))
    data = []
    headers = []
    for file in files:
        hdu = fits.open(file)
        data.append(hdu[0].data)
        headers.append(hdu[0].header)
        hdu.close()
    logging.info(f"Loaded {len(files)} object files.")
    return files, data, headers

def load_darks():
    """
    Load preprocessed dark frames and their associated exposure times.

    Returns
    -------
    times_darks : numpy.ndarray
        Array of exposure times for the dark frames.
    darks : numpy.ndarray
        Array of dark frame data loaded from preprocessed files.

    Workflow
    --------
    1. Loads dark frame data and exposure times from pre-saved NPY files.
    2. Logs a confirmation message indicating successful loading.

    Output
    ------
    - 'darks.npy': Contains the preprocessed dark frame data.
    - 'times_darks.npy': Contains the associated exposure times for each dark frame.

    """

    darks = np.load(os.path.join(config['paths']['output_dir'], 'darks.npy'))
    times_darks = np.load(os.path.join(config['paths']['output_dir'], 'times_darks.npy'))
    logging.info("Loaded dark frames and associated times.")
    return times_darks, darks

def subtract_dark(data, headers, times_darks, darks):
    """
    Subtract the corresponding dark frame based on exposure time.

    Parameters
    ----------
    data : list of numpy.ndarray
        List of image data arrays to be corrected.
    headers : list of astropy.io.fits.Header
        List of FITS headers associated with each data array.
    times_darks : numpy.ndarray
        Array of exposure times for the dark frames.
    darks : numpy.ndarray
        Array of preprocessed dark frame data.

    Returns
    -------
    corrected_data : list of numpy.ndarray
        List of dark-subtracted image data arrays.

    Workflow
    --------
    1. Extracts the exposure time for each image from its header.
    2. Matches the exposure time with the closest available dark frame:
       - Uses an exact match if available.
       - Falls back to the dark frame with the lowest exposure time if no exact match is found.
    3. Subtracts the matched dark frame from the image data.
    4. Logs the process, including warnings for mismatched exposure times.

    """

    corrected_data = []
    for i, img in enumerate(data):
        # Get exposure time from header
        exp_time = headers[i]['ESO DET SEQ1 DIT']
        logging.info(f"Image {i + 1} exposure time: {exp_time}")

        # Find the closest matching dark exposure time
        if exp_time in times_darks:
            dark = darks[np.where(times_darks == exp_time)[0][0]]
        else:
            logging.warning(f"No exact match for exposure time {exp_time}. Using offset dark with lowest exposure time.")
            dark = darks[np.argmin(times_darks)]

        # Subtract the dark
        corrected_img = img - dark
        corrected_data.append(corrected_img)
    return corrected_data

def apply_flat_field_correction(data, master_flat, mask):
    """
    Apply flat-field correction and masking to image data.

    Parameters
    ----------
    data : list of numpy.ndarray
        List of image data arrays (can be 2D or 3D).
    master_flat : numpy.ndarray
        Master flat field for normalizing pixel intensities.
    mask : numpy.ndarray
        Boolean mask marking bad pixels (True) and good pixels (False).

    Returns
    -------
    reduced_data : list of numpy.ma.MaskedArray
        List of flat-field corrected and masked image data arrays.

    Workflow
    --------
    1. Processes both 2D and 3D image data:
       - For 3D data, processes each slice independently.
       - For 2D data, applies correction directly.
    2. Normalizes each image using the master flat, avoiding division by zero.
    3. Applies the provided mask to flag bad pixels using NumPy masked arrays.
    4. Logs the status of flat-field correction and masking for each image.

    """

    reduced_data = []
    for i, img in enumerate(data):
        # Normalize and mask data (handle 3D data slices)
        if img.ndim == 3:
            corrected = []
            for j in range(img.shape[0]):
                temp = np.divide(img[j], master_flat, out=np.zeros_like(img[j]), where=(master_flat != 0))
                temp = np.ma.masked_array(temp, mask=mask)  # Apply mask using masked array
                corrected.append(temp)
            corrected = np.ma.array(corrected)
        else:
            corrected = np.divide(img, master_flat, out=np.zeros_like(img), where=(master_flat != 0))
            corrected = np.ma.masked_array(corrected, mask=mask)  # Apply mask using masked array

        reduced_data.append(corrected)
        logging.info(f"Image {i + 1} corrected with flat-field and mask applied.")
    return reduced_data

def save_reduced_images(output_dir, files, reduced_data, headers):
    """
    Save reduced images in both FITS and NPY formats.

    Parameters
    ----------
    output_dir : str
        Path to the directory where the reduced images will be saved.
    files : list of str
        List of input file paths corresponding to the reduced images.
    reduced_data : list of numpy.ma.MaskedArray
        List of flat-field corrected and masked image data arrays.
    headers : list of astropy.io.fits.Header
        List of FITS headers associated with the input files.

    Workflow
    --------
    1. Creates the output directory if it does not exist.
    2. Iterates through each reduced image:
       - Saves the image in FITS format:
         - Writes masked data using `filled(np.nan)` to replace masked values with NaNs.
         - Preserves the original header information in the FITS file.
       - Saves the image in NPY format:
         - Writes the same masked data array using NumPy's NPY format for fast access.
    3. Logs a confirmation message for each saved file in both formats.

    Output
    ------
    - '_reduced.fits': Flat-field corrected and masked images in FITS format.
    - '_reduced.npy': Flat-field corrected and masked images in NPY format.

    """

    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(reduced_data):
        # Save as FITS
        output_file_fits = os.path.join(output_dir, os.path.basename(files[i]).replace('.fits', '_reduced.fits'))
        fits.writeto(output_file_fits, img.filled(np.nan), header=headers[i], overwrite=config['output']['overwrite'])
        logging.info(f"Saved reduced image in FITS format: {output_file_fits}")

        # Save as NPY
        output_file_npy = os.path.join(output_dir, os.path.basename(files[i]).replace('.fits', '_reduced.npy'))
        np.save(output_file_npy, img.filled(np.nan))
        logging.info(f"Saved reduced image in NPY format: {output_file_npy}")

def reduce_images(rep_object, master_flat, mask, output_dir):
    """
    Main function for reducing astronomical images.

    Parameters
    ----------
    rep_object : str
        Path to the directory containing object FITS files to be reduced.
    master_flat : numpy.ndarray
        Master flat field for normalizing pixel intensities.
    mask : numpy.ndarray
        Boolean mask marking bad pixels (True) and good pixels (False).
    output_dir : str
        Path to the directory where the reduced images will be saved.

    Returns
    -------
    reduced_data : list of numpy.ma.MaskedArray
        List of reduced images with flat-field correction and masking applied.

    Workflow
    --------
    1. Loads object files:
       - Reads image data and headers from FITS files in the specified directory.
    2. Loads preprocessed dark frames:
       - Reads dark frame data and associated exposure times.
    3. Subtracts dark frames:
       - Matches each image with the closest dark frame based on exposure time.
       - Logs warnings if no exact match is found and uses the closest available dark.
    4. Applies flat-field correction:
       - Normalizes pixel intensities using the master flat.
       - Masks bad pixels using the provided mask.
    5. Saves reduced images:
       - Writes output in both FITS and NPY formats, preserving headers and masked data.

    """

    logging.info("Starting image reduction.")

    # Load object files
    files, data, headers = load_object_files(rep_object)

    # Load darks
    times_darks, darks = load_darks()

    # Subtract darks
    corrected_data = subtract_dark(data, headers, times_darks, darks)

    # Apply flat-field correction
    reduced_data = apply_flat_field_correction(corrected_data, master_flat, mask)

    # Save reduced images
    save_reduced_images(output_dir, files, reduced_data, headers)

    logging.info("Image reduction completed successfully.")

    return reduced_data


if __name__ == '__main__':
    # Example usage within module
    master_flat = np.load(os.path.join(config['paths']['output_dir'], 'master_flat.npy'))
    mask = np.load(os.path.join(config['paths']['output_dir'], 'mask.npy'))
    reduced_data = reduce_images(config['paths']['object_dir'], master_flat, mask, config['paths']['output_dir'])
