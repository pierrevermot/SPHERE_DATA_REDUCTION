"""
Astrometric Calibration Script for Astronomical Data
=====================================================

Description:
------------
This script performs astrometric calibration by defining a World Coordinate System (WCS)  
for astronomical images. It updates the FITS headers with WCS parameters, enabling accurate  
positioning and sky coordinates for each pixel in the image.

Workflow:
---------
1. **Input Handling:**
   - Processes both 2D and 3D FITS data.  
   - Computes the **median image** for 3D data and applies calibration on it.  
   - Saves the median image as a separate FITS file if input is 3D.

2. **Reference Positioning:**
   - Supports user-specified reference pixel coordinates (**x, y**) or uses the  
     **maximum emission** position in the image as the reference point.  
   - Handles maximum emission computation using **median filtering** for noise removal.

3. **WCS Setup:**
   - Uses input parameters from the configuration file:
     - Pixel scale (in milliarcseconds).  
     - Reference RA/DEC (in degrees).  
   - Computes WCS transformation:
     - Sets CRPIX, CRVAL, CDELT, and CTYPE values in the FITS header.

4. **Output:**
   - Saves the updated FITS file with WCS parameters.  
   - Generates and saves RA and DEC arrays for pixel-to-sky coordinate mapping.  
   - Outputs the median image (for 3D data) as both FITS and NPY files.

"""


import os
import sys
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import logging

# Add config directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
from parameters import config

# Setup logging
log_file = os.path.join(config['paths']['log_file'])
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO)


def astrometric_calibration(fits_file, output_dir):
    """
    Perform astrometric calibration and update WCS (World Coordinate System) in the FITS header.

    Parameters
    ----------
    fits_file : str
        Path to the FITS file to be calibrated.
    output_dir : str
        Path to the directory where the calibrated output will be saved.

    Workflow
    --------
    1. Loads the FITS file and retrieves image data and header.
    2. If input data is 3D, computes the **median image** and uses it for calibration.
       - Saves the median image as a separate FITS and NPY file.
    3. Retrieves calibration parameters from the configuration file (`parameters.py`):
       - Pixel scale (in milliarcseconds).
       - Reference RA/DEC coordinates.
       - Reference pixel coordinates (x, y).
    4. Automatically determines the reference pixel based on **maximum emission** if specified in config.
       - Uses median filtering to smooth noise before finding emission peaks.
    5. Sets up the WCS with computed parameters:
       - CRPIX, CRVAL, CDELT, and CTYPE fields are updated in the header.
    6. Generates and saves **RA** and **DEC** arrays corresponding to each pixel in the image.
    7. Saves the updated FITS file with WCS information.

    Output
    ------
    - Updated FITS file with WCS parameters.
    - 'median_image.fits': Median image (if input is 3D) saved in FITS format.
    - 'median_image.npy': Median image saved as an NPY file.
    - 'ra_array.npy': RA coordinates saved as an NPY file.
    - 'dec_array.npy': DEC coordinates saved as an NPY file.

    """

    hdu = fits.open(fits_file)
    header = hdu[0].header
    image_data = hdu[0].data

    # If data is 3D, compute the median image
    if image_data.ndim == 3:
        image_data = np.median(image_data, axis=0)
        logging.info("Computed median image for 3D data.")

    # Retrieve calibration parameters
    pixel_scale = config['calibration']['pixel_scale'] / 1000.0  # mas to arcsecond
    ref_x = config['calibration']['ref_x']
    ref_y = config['calibration']['ref_y']
    ref_ra = config['calibration']['ref_ra']
    ref_dec = config['calibration']['ref_dec']

    # Determine pixel coordinates for maximum emission if 'max' is specified
    if ref_x == 'max' or ref_y == 'max':
        filtered_data = np.nan_to_num(image_data)
        max_pos = np.unravel_index(np.argmax(filtered_data), filtered_data.shape)
        ref_x = max_pos[1]+1  # X-coordinate
        ref_y = max_pos[0]+1  # Y-coordinate
        logging.info(f"Using maximum emission position as reference: x={ref_x}, y={ref_y}")

    # Setup WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [ref_x, ref_y]
    w.wcs.cdelt = [-pixel_scale / 3600.0, pixel_scale / 3600.0]  # deg/pixel
    w.wcs.crval = [ref_ra, ref_dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Update header
    header.update(w.to_header())

    # Save median image if data was 3D
    if image_data.ndim == 2:
        median_file_fits = os.path.join(output_dir, 'median_image.fits')
        fits.writeto(median_file_fits, image_data, header, overwrite=True)
        np.save(os.path.join(output_dir, 'median_image.npy'), image_data)
        logging.info(f"Saved median image as {median_file_fits}")

    # Generate RA and DEC arrays
    y, x = np.mgrid[:image_data.shape[0], :image_data.shape[1]]
    ra, dec = w.all_pix2world(x, y, 0)
    np.save(os.path.join(output_dir, 'ra_array.npy'), ra)
    np.save(os.path.join(output_dir, 'dec_array.npy'), dec)
    logging.info("Saved RA and DEC arrays.")

    # Save updated FITS file
    output_file = os.path.join(output_dir, os.path.basename(fits_file))
    hdu.writeto(output_file, overwrite=True)
    logging.info(f"Updated WCS in {output_file}")


if __name__ == '__main__':
    # Example usage
    input_dir = config['paths']['output_dir']
    output_dir = config['paths']['output_dir']
    fits_files = sorted(glob.glob(os.path.join(input_dir, 'aligned_datacube.fits')))

    for fits_file in fits_files:
        # Astrometric calibration
        astrometric_calibration(fits_file, output_dir)

    logging.info("Astrometric calibration completed successfully.")
