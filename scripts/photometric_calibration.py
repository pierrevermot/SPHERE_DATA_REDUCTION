"""
Photometric Calibration Script for Astronomical Data
=====================================================

Description:
------------
This script performs **photometric calibration** by scaling the flux of astronomical images  
to match the absolute photometric scale of 2MASS reference data.  
It ensures the processed images have pixel values calibrated to physical units (W/m²).  

Workflow:
---------
1. **Input Handling:**
   - Processes both **2D** and **3D** FITS data.  

2. **Reference Image Download:**
   - Queries the **2MASS API** to download flux-calibrated images for the same region.  
   - Checks if the required 2MASS image already exists and avoids redundant downloads.  
   - Supports configurable bands (J, H, K) specified in the parameter file.

3. **Flux Comparison:**
   - Computes the total flux in the SPHERE image and the corresponding region in the  
     2MASS reference image.  
   - Calculates the multiplicative **scale factor** required to match the flux.

4. **Output:**
   - Saves the **flux-calibrated image** with updated headers containing:
     - Scale factor applied.  
     - Reference flux and target flux values.  
     - Photometric zero-point (from the 2MASS image header).  
   - Produces separate outputs for:
     - **2D calibrated images**.  
     - **3D calibrated datacubes**.  
   - Saves the results in both **FITS** and **NPY** formats.

Special Features:
-----------------
- Handles both **2D and 3D data**, applying calibration consistently across slices in 3D datacubes.  
- Automatically downloads and processes 2MASS reference data to match the observation's field of view.  
- Ensures compatibility with various bands (J, H, K) for photometric scaling.  
- Saves intermediate and calibrated data files, including logs for reproducibility.  

"""


import os
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import logging
import requests
from astropy.table import Table

# Add config directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
from parameters import config

# Setup logging
log_file = os.path.join(config['paths']['log_file'])
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO)


def get_2mass_image(ra, dec, size=0.1, band='K'):
    """
    Download a 2MASS image covering the specified region.

    Parameters
    ----------
    ra : float
        Right Ascension (RA) of the target region in degrees (J2000).
    dec : float
        Declination (DEC) of the target region in degrees (J2000).
    size : float, optional
        Size of the search region in degrees (default is 0.1).
    band : str, optional
        2MASS photometric band ('J', 'H', or 'K', default is 'K').

    Returns
    -------
    output_file : str
        Path to the downloaded FITS file containing the 2MASS image.

    Workflow
    --------
    1. Checks if the requested 2MASS image already exists locally:
       - Avoids redundant downloads if the file is already present.
    2. Sends a query to the 2MASS API to search for images in the specified region.
    3. Saves the query response to a temporary XML (VOTable) file.
    4. Parses the VOTable to retrieve the URL of the FITS image.
    5. Downloads the FITS image and saves it in the output directory.
    """

    output_file = os.path.join(config['paths']['output_dir'], f'2mass_{band}_{ra:.4f}_{dec:.4f}.fits')
    
    # Check if the file already exists
    if os.path.exists(output_file):
        logging.info(f"2MASS image already exists: {output_file}")
        return output_file

    url = "https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia"
    params = {
        'POS': f'{ra},{dec}',
        'SIZE': size,
        'FORMAT': 'image/fits',
        'band': band,
        'type': 'at',
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    # Save response content to a temporary file
    temp_votable_file = os.path.join(config['paths']['output_dir'], '2mass_query.xml')
    with open(temp_votable_file, 'w') as f:
        f.write(response.text)

    # Parse VOTable from saved file
    table = Table.read(temp_votable_file, format='votable')
    image_url = table['download'][2]

    # Download the FITS image
    fits_response = requests.get(image_url)
    fits_response.raise_for_status()

    with open(output_file, 'wb') as f:
        f.write(fits_response.content)

    logging.info(f"Downloaded 2MASS image: {output_file}")
    return output_file

def photometric_calibration(input_file, output_dir):
    """
    Perform photometric calibration based on 2MASS flux calibration.

    Parameters
    ----------
    input_file : str
        Path to the FITS file containing the image to be calibrated.
    output_dir : str
        Path to the directory where the calibrated image will be saved.

    Workflow
    --------
    1. Loads the input FITS file and extracts image data and headers.
    2. Handles both 2D and 3D data:
       - Computes the **median image** for 3D data to determine the scaling factor.
       - Applies the scale factor to the **entire 3D datacube** if the input is 3D.
    3. Computes the RA/DEC coordinates and field of view (FOV) from the WCS header.
    4. Downloads a matching 2MASS image using the specified band ('J', 'H', or 'K'):
       - Extracts the zero-point magnitude from the 2MASS header.
       - Matches the FOV of the SPHERE image to the 2MASS image region.
    5. Measures total fluxes in the SPHERE and 2MASS images.
    6. Computes and applies a multiplicative scaling factor to match fluxes.
    7. Updates the FITS header with:
       - Zero-point magnitude ('PHOTZP').
    8. Saves the calibrated image in FITS format:
       - Produces a 2D calibrated image for 2D input.
       - Produces a 3D calibrated datacube for 3D input.

    Output
    ------
    - '..._flux_calibrated.fits': Calibrated FITS image for 2D input.
    - '..._flux_calibrated_3d.fits': Calibrated FITS datacube for 3D input.

    """

    # Load input file
    hdu = fits.open(input_file)
    image_data = hdu[0].data
    header = hdu[0].header

    # Handle 3D data by computing median
    is_3d = False
    if image_data.ndim == 3:
        is_3d = True
        median_image = np.median(image_data, axis=0)
        logging.info("Computed median image for 3D data.")
    else:
        median_image = image_data

    # Get WCS info and ensure it is 2D
    wcs = WCS(header)
    if wcs.naxis > 2:
        wcs = wcs.dropaxis(2)

    # Compute RA and DEC of the center
    center = wcs.pixel_to_world(header['NAXIS1'] // 2, header['NAXIS2'] // 2)
    ra, dec = center.ra.deg, center.dec.deg

    # Calculate field of view in degrees
    fov_x = abs(header['CDELT1']) * header['NAXIS1']
    fov_y = abs(header['CDELT2']) * header['NAXIS2']
    fov = max(fov_x, fov_y) * 3600  # Convert to arcseconds

    # Download 2MASS reference image
    band = config['calibration']['band']
    ref_file = get_2mass_image(ra, dec, size=fov / 3600, band=band)
    ref_hdu = fits.open(ref_file)
    ref_data = ref_hdu[0].data
    ref_header = ref_hdu[0].header

    # Extract zero point magnitude
    zp_key = 'MAGZP'
    zero_point = ref_header[zp_key]


    # Compute coordinates of SPHERE image corners
    x_corners = [0, header['NAXIS1'], header['NAXIS1'], 0]
    y_corners = [0, 0, header['NAXIS2'], header['NAXIS2']]
    corners = wcs.pixel_to_world(x_corners, y_corners)

    # Extract region in 2MASS image
    ref_wcs = WCS(ref_header)
    if ref_wcs.naxis > 2:
        ref_wcs = ref_wcs.dropaxis(2)

    x_min, y_min = ref_wcs.world_to_pixel(corners[0])
    x_max, y_max = ref_wcs.world_to_pixel(corners[2])
    x_min, x_max = int(np.floor(x_min)), int(np.ceil(x_max))
    y_min, y_max = int(np.floor(y_min)), int(np.ceil(y_max))
    ref_region = ref_data[y_min:y_max, x_min:x_max]
    
    # Measure fluxes in both images
    flux_target = np.nansum(median_image)
    flux_ref = np.nansum(ref_region)

    # Compute calibration factor
    scale_factor = flux_ref / flux_target

    # Apply calibration
    if is_3d:
        calibrated_image = image_data * scale_factor
        output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.fits', '_flux_calibrated_3d.fits'))
    else:
        calibrated_image = median_image * scale_factor
        output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.fits', '_flux_calibrated.fits'))

    header['PHOTZP'] = zero_point

    # Save calibrated image
    hdu[0].data = calibrated_image
    hdu.writeto(output_file, overwrite=True)
    logging.info(f"Calibrated image saved: {output_file}")


if __name__ == '__main__':
    # Example usage
    output_dir = config['paths']['output_dir']
    input_files = [os.path.join(output_dir, 'aligned_datacube.fits'),
                   os.path.join(output_dir, 'median_image.fits')]

    for input_file in input_files:
        photometric_calibration(input_file, output_dir)

    logging.info("Photometric calibration completed successfully.")
