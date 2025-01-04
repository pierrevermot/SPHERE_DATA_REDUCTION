"""
Main Pipeline Script for Astronomical Data Processing
======================================================

Description:
------------
This script orchestrates the full data reduction pipeline for astronomical image processing.  
It integrates preprocessing, reduction, alignment, and both astrometric and photometric calibrations.  
The pipeline is designed to process SPHERE instrument data, ensuring scientifically accurate outputs  
with corrected alignment, astrometry, and photometry.

Workflow:
---------
1. **Preprocessing:**
   - Loads and processes flats and darks.
   - Generates master flats and masks for image correction.

2. **Reduction:**
   - Applies flat-field correction and dark subtraction to raw images.
   - Saves reduced images in FITS and NPY formats.

3. **Alignment:**
   - Splits and prepares images for alignment.
   - Computes shifts and aligns all frames using cross-correlation.

4. **Astrometric Calibration:**
   - Computes WCS (World Coordinate System) parameters based on pixel scale and reference coordinates.
   - Updates FITS headers with WCS information.

5. **Photometric Calibration:**
   - Downloads calibrated 2MASS images.
   - Matches flux between SPHERE data and 2MASS references.
   - Updates image flux values and FITS headers accordingly.

Output:
-------
- Processed and aligned FITS files saved in the output directory.
- Calibrated images with updated WCS and flux scaling factors.

Requirements:
-------------
- Python 3.x
- Dependencies: astropy, numpy, scipy, requests, matplotlib
- Configuration file: parameters.py

"""


import os
import logging
import numpy as np
from preprocessing import sort_flats, sort_darks, make_master_flat, create_masks, save_darks
from reduction import reduce_images
from alignment import load_reduced_images, split_and_prepare_images, compute_centers_gauss, align_images_subpix, save_aligned_images, average_left_right, split_mask
from astrometric_calibration import astrometric_calibration
from photometric_calibration import photometric_calibration
from parameters import config

# Setup logging
log_file = os.path.join(config['paths']['log_file'])
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO)


def main():
    """
    Main pipeline for astronomical data processing.
    This script orchestrates preprocessing, reduction, alignment, and calibration steps.
    """
    logging.info("Pipeline started.")

    # Preprocessing
    logging.info("Starting preprocessing...")
    times_flats, flats = sort_flats(config['paths']['flat_dir'])
    times_darks, darks = sort_darks(config['paths']['dark_dir'])
    save_darks(times_darks, darks)
    master_flat = make_master_flat(times_flats, flats, times_darks, darks)
    mask = create_masks(master_flat)
    logging.info("Preprocessing completed.")

    # Reduction
    logging.info("Starting reduction...")
    reduce_images(config['paths']['object_dir'], master_flat, mask, config['paths']['output_dir'])
    logging.info("Reduction completed.")

    # Alignment
    logging.info("Starting alignment...")
    files, data, headers = load_reduced_images(config['paths']['output_dir'])
    sub_images = split_and_prepare_images(data)
    mask_left, mask_right = split_mask(np.load(os.path.join(config['paths']['output_dir'], 'mask.npy')))
    images, mask = average_left_right(sub_images, mask_left, mask_right)
    y_indices, x_indices = compute_centers_gauss(images)
    aligned_data = align_images_subpix(images, y_indices, x_indices)
    save_aligned_images(config['paths']['output_dir'], aligned_data)
    logging.info("Alignment completed.")

    # Astrometric Calibration
    logging.info("Starting astrometric calibration...")
    aligned_file = os.path.join(config['paths']['output_dir'], 'aligned_datacube.fits')
    astrometric_calibration(aligned_file, config['paths']['output_dir'])
    logging.info("Astrometric calibration completed.")

    # Photometric Calibration
    logging.info("Starting photometric calibration...")
    output_dir = config['paths']['output_dir']
    input_files = [os.path.join(output_dir, 'aligned_datacube.fits'),
                   os.path.join(output_dir, 'median_image.fits')]
    for input_file in input_files:
        photometric_calibration(input_file, output_dir)
    logging.info("Photometric calibration completed.")
    logging.info("Pipeline finished successfully.")


if __name__ == '__main__':
    main()
