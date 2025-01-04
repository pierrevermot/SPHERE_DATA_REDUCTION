"""
Alignment Script for Astronomical Image Processing
===================================================

Description:
------------
This script performs the alignment of astronomical images to correct positional offsets  
between frames. It processes reduced images and aligns them based on maximum emission  
or Gaussian fitting techniques, ensuring consistent alignment for further analysis.

Workflow:
---------
1. **Load Reduced Images:**
   - Reads reduced FITS images and headers for alignment processing.

2. **Splitting and Preparation:**
   - Splits each image into two halves and prepares them for alignment.
   - Computes reference positions based on maximum emission or Gaussian fitting.

3. **Alignment Computation:**
   - Two shift measurement methods are supported:
     a) **Maximum Emission Method**:
        - Applies median filtering to remove noise.
        - Locates the position of maximum emission with **1-pixel precision**.
     b) **Gaussian Fitting Method**:
        - Fits a 2D Gaussian to the emission peak for **sub-pixel precision**.

   - Two realignment approaches are supported:
     a) **Pixel Precision Re-Alignment**:
        - Uses `np.roll` for efficient shifting by integer pixels.
     b) **Sub-Pixel Re-Alignment**:
        - Uses `scipy.ndimage.shift` for **sub-pixel precision** adjustments.

4. **Output:**
   - Saves aligned images in both FITS and NPY formats for further processing.
   - Produces a full aligned datacube when dealing with 3D data.

"""



import os
import sys
import glob
import numpy as np
from astropy.io import fits
import logging
from scipy.ndimage import shift, median_filter
from scipy.optimize import curve_fit

# Add config directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
from parameters import config

# Setup logging
log_file = os.path.join(config['paths']['log_file'])
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO)


def load_reduced_images(rep_reduced):
    """
    Load reduced images and their headers from the specified directory.

    Parameters
    ----------
    rep_reduced : str
        Path to the directory containing reduced FITS files with '_reduced.fits' suffix.

    Returns
    -------
    files : list of str
        List of file paths for the loaded reduced FITS files.
    data : numpy.ndarray
        Array of image data extracted from the FITS files.
    headers : list of astropy.io.fits.Header
        List of FITS headers associated with each file.

    Workflow
    --------
    1. Searches the specified directory for FITS files ending with '_reduced.fits'.
    2. Opens each FITS file and extracts:
       - Image data from the primary HDU.
       - Header metadata for reference.
    3. Logs the number of files successfully loaded.
    4. Returns the list of file paths, image data arrays, and headers for further processing.

    """

    files = sorted(glob.glob(rep_reduced + '/*_reduced.fits'))
    data = []
    headers = []
    for file in files:
        hdu = fits.open(file)
        data.append(hdu[0].data)
        headers.append(hdu[0].header)
        hdu.close()
    logging.info(f"Loaded {len(files)} reduced images.")
    return files, np.array(data), headers

def split_and_prepare_images(data):
    """
    Split images into two halves and prepare a list of sub-images.

    Parameters
    ----------
    data : numpy.ndarray
        Array of input images to be split and prepared (can be 3D).

    Returns
    -------
    sub_images : numpy.ndarray
        Array of sub-images created by splitting each input image into two halves.

    Workflow
    --------
    1. Flattens the input data if it contains multiple frames.
    2. Splits each image into two halves along the horizontal axis.
       - Left half and right half are treated as separate sub-images.
    3. Logs the total number of sub-images created.
    4. Returns the prepared sub-images for further processing.

    """

    flat_data = np.concatenate(data)
    sub_images = []
    for img in flat_data:
        left = img[:, :img.shape[1] // 2]
        right = img[:, img.shape[1] // 2:]
        sub_images.append(left)
        sub_images.append(right)
    logging.info(f"Split images into {len(sub_images)} sub-images.")
    return np.array(sub_images)

def split_mask(mask):
    """
    Split a mask into two halves for processing corresponding sub-images.

    Parameters
    ----------
    mask : numpy.ndarray
        Input mask array to be split (2D).

    Returns
    -------
    left : numpy.ndarray
        Left half of the mask.
    right : numpy.ndarray
        Right half of the mask.

    Workflow
    --------
    1. Splits the mask into two halves along the horizontal axis.
    2. Logs the operation to confirm the split.
    3. Returns the two halves for use with corresponding sub-images.

    """

    left = mask[:, :mask.shape[1] // 2]
    right = mask[:, mask.shape[1] // 2:]
    logging.info("Split mask into sub-images.")
    return left, right
def average_left_right(sub_images, mask_left, mask_right):
    """
    Compute the average of left and right sub-images after masking and alignment.

    Parameters
    ----------
    sub_images : numpy.ndarray
        Array of sub-images split from the original data (typically 3D).
    mask_left : numpy.ndarray
        Mask for the left half of the images, marking bad pixels.
    mask_right : numpy.ndarray
        Mask for the right half of the images, marking bad pixels.

    Returns
    -------
    average_images : numpy.ndarray
        Averaged images computed from left and right halves after masking and alignment.
    mask : numpy.ndarray
        Combined mask applied to the averaged images.

    Workflow
    --------
    1. Splits input sub-images into left and right halves.
    2. Rolls the right sub-images horizontally and vertically to align them with the left halves:
       - Uses roll offsets specified in the configuration file (`parameters.py`).
    3. Applies masks to the left and rolled-right halves to filter bad pixels.
    4. Stacks the masked left and right halves and computes their average.
    5. Ensures no zeros or negative values in the averaged images by replacing invalid values with epsilon.
    6. Returns the averaged images and their corresponding mask.

    """

    subs_left, subs_right = sub_images[::2], sub_images[1::2]
    subs_right = np.roll(subs_right, config['alignment']['x_window_roll'], axis=2)
    subs_right = np.roll(subs_right, config['alignment']['y_window_roll'], axis=1)
    mask_right_rolled = np.roll(mask_right, config['alignment']['x_window_roll'], axis=1)
    mask_right_rolled = np.roll(mask_right_rolled, config['alignment']['y_window_roll'], axis=0)
    subs_right = np.ma.masked_array(subs_right.data, 
            mask_right_rolled[None, :, :]*np.ones(len(subs_right))[:, None, None])
    subs_left = np.ma.masked_array(subs_left, 
            mask_left[None, :, :]*np.ones(len(subs_left))[:, None, None])
    stacked_ims = np.ma.masked_array([subs_left, subs_right])
    average_images = stacked_ims.mean(0)
    average_images = np.nan_to_num(average_images)
    average_images[average_images<=0] = sys.float_info.epsilon
    mask = average_images[0].mask
    return average_images, mask

def compute_centers(flat_data):
    """
    Compute the coordinates of maximum emission for each image.

    Parameters
    ----------
    flat_data : numpy.ndarray
        3D array of images (shape: n, height, width) for which emission centers need to be computed.

    Returns
    -------
    y_indices : numpy.ndarray
        Array of y-coordinates for the maximum emission points.
    x_indices : numpy.ndarray
        Array of x-coordinates for the maximum emission points.

    Workflow
    --------
    1. Replaces NaN values with zeros for stability during computation.
    2. Applies a median filter to smooth each image and reduce noise.
    3. Flattens each image and identifies the pixel with the highest intensity.
    4. Converts flat indices back to 2D coordinates (y, x).
    5. Returns arrays containing the y and x coordinates of emission maxima for all images.

    """

    # Assuming data has shape (n, 1024, 1024)
    n, h, w = flat_data.shape

    data_med = np.nan_to_num(flat_data)
    data_med = median_filter(data_med, (1,3,3))
    # Compute argmax for each image (2D) along flattened indices
    argmax_indices = np.argmax(data_med.reshape(n, -1), axis=1)

    # Convert to 2D coordinates (y, x)
    y_indices, x_indices = np.unravel_index(argmax_indices, (h, w))

    # Return indices
    return y_indices, x_indices


def compute_centers_gauss(flat_data):
    """
    Compute the sub-pixel centers of maximum emission using 2D Gaussian fitting.

    Parameters
    ----------
    flat_data : numpy.ndarray
        3D array of images (shape: n, height, width) to compute emission centers.

    Returns
    -------
    ys_gauss : list of float
        List of y-coordinates of the Gaussian-fitted emission centers for each image.
    xs_gauss : list of float
        List of x-coordinates of the Gaussian-fitted emission centers for each image.

    Workflow
    --------
    1. Computes the median image across all frames to locate the approximate emission center.
    2. Extracts a 40x40 pixel sub-region centered around the emission peak for finer analysis.
    3. Defines a 2D Gaussian function to fit the emission profile.
    4. Performs Gaussian fitting for each image to determine sub-pixel coordinates:
       - Initializes the fit using the pixel with the maximum intensity as a starting point.
       - Constrains the fit parameters within a small window around the initial guess.
       - Uses `scipy.optimize.curve_fit` to fit the Gaussian model.
    5. Appends the fitted x and y coordinates for each image.

    """

    med_flat_data = np.median(flat_data, 0)
    ym, xm = np.unravel_index(np.argmax(med_flat_data), med_flat_data.shape)
    flat_data_for_comp = flat_data[:, ym-20:ym+20, xm-20:xm+20]
    n_pix = len(flat_data_for_comp[0])
    x_coordinates = np.arange(n_pix)
    y_coordinates = np.arange(n_pix)
    mesh_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    xs_gauss, ys_gauss = [], []
    for im in flat_data_for_comp:
        amp = np.max(im)
        def gauss(coordinates, x0, y0):
            std = 3
            c = 0
            x, y = coordinates
            gauss_2D = amp*np.exp(-(x-x0)**2/(2*std**2))*np.exp(-(y-y0)**2/(2*std**2))+c
            return gauss_2D.flatten()
        
        p0_y, p0_x = np.unravel_index(np.argmax(im), im.shape)
        p0 = [p0_x, p0_y]
        bounds = ((p0_x-2, p0_y-2), (p0_x+2, p0_y+2))
        try:
            params, cov = curve_fit(gauss, mesh_coordinates, im.flatten(), p0=p0, bounds=bounds)
        except:
            params = p0
        x, y = params[0], params[1]
        xs_gauss.append(x)
        ys_gauss.append(y)
    return ys_gauss, xs_gauss

def align_images(flat_data, y_indices, x_indices, mask):
    """
    Align images based on the position of maximum emission with pixel precision.

    Parameters
    ----------
    flat_data : numpy.ndarray
        3D array of input images to be aligned (shape: n, height, width).
    y_indices : list of int
        List of y-coordinates for the emission centers in each image.
    x_indices : list of int
        List of x-coordinates for the emission centers in each image.
    mask : numpy.ndarray
        Boolean mask to filter bad pixels (True for bad pixels).

    Returns
    -------
    aligned_data : numpy.ma.MaskedArray
        Masked array of aligned images with bad pixels filtered.

    Workflow
    --------
    1. Computes integer shifts required to align each image relative to the first image.
    2. Uses `np.roll` to shift each image by the computed offsets (pixel-level precision).
    3. Applies the same shifts to the mask to ensure masking consistency.
    4. Logs the applied shifts for each image.
    5. Combines the aligned images into a masked array and returns them.

    """

    aligned_data = []
    for i, img in enumerate(flat_data):
        # Apply shift
        shift_y, shift_x = y_indices[i]-y_indices[0], x_indices[i]-x_indices[0]
        aligned_img = np.roll(img, shift=(-shift_y, -shift_x), axis=(0, 1))
        aligned_mask = np.roll(mask, shift=(-shift_y, -shift_x), axis=(0, 1))
        aligned_masked_img = np.ma.masked_array(aligned_img, aligned_mask)
        aligned_data.append(aligned_masked_img)
        logging.info(f"Image {i + 1} shifted by (y, x): ({shift_y}, {shift_x}).")
    return np.ma.array(aligned_data)


def align_images_subpix(flat_data, y_indices, x_indices):
    """
    Align images based on the position of maximum emission with sub-pixel precision.

    Parameters
    ----------
    flat_data : numpy.ndarray
        3D array of input images to be aligned (shape: n, height, width).
    y_indices : list of float
        List of y-coordinates for the emission centers (sub-pixel precision).
    x_indices : list of float
        List of x-coordinates for the emission centers (sub-pixel precision).

    Returns
    -------
    aligned_data : numpy.ndarray
        Array of aligned images with sub-pixel precision.

    Workflow
    --------
    1. Computes sub-pixel shifts required to align each image relative to the first image.
    2. Uses `scipy.ndimage.shift` to apply sub-pixel shifts with bilinear interpolation.
    3. Logs the applied shifts for each image.
    4. Combines the aligned images into an array and returns them.

    """

    aligned_data = []
    for i, img in enumerate(flat_data):
        # Apply shift
        shift_y, shift_x = y_indices[i]-y_indices[0], x_indices[i]-x_indices[0]
        aligned_img = shift(img, shift=(-shift_y, -shift_x), order=1)
        aligned_data.append(aligned_img)
        logging.info(f"Image {i + 1} shifted by (y, x): ({shift_y}, {shift_x}).")
    return np.array(aligned_data)

def save_aligned_images(output_dir, aligned_data):
    """
    Save the aligned datacube as single FITS and NPY files.

    Parameters
    ----------
    output_dir : str
        Path to the directory where the aligned datacube will be saved.
    aligned_data : numpy.ndarray or numpy.ma.MaskedArray
        Array of aligned images (can be masked) to save.

    Workflow
    --------
    1. Creates the output directory if it does not exist.
    2. Saves the aligned datacube in two formats:
       - FITS format for compatibility with astronomy tools.
       - NPY format for fast loading in Python.
    3. Logs a confirmation message for each saved file.

    Output
    ------
    - 'aligned_datacube.fits': Aligned images saved in FITS format.
    - 'aligned_datacube.npy': Aligned images saved in NPY format.

    """

    os.makedirs(output_dir, exist_ok=True)

    # Ensure data is masked array and save as FITS
    output_file_fits = os.path.join(output_dir, 'aligned_datacube.fits')
    fits.writeto(output_file_fits, aligned_data, overwrite=config['output']['overwrite'])
    logging.info(f"Saved aligned datacube in FITS format: {output_file_fits}")

    # Save as NPY
    output_file_npy = os.path.join(output_dir, 'aligned_datacube.npy')
    np.save(output_file_npy, aligned_data)
    logging.info(f"Saved aligned datacube in NPY format: {output_file_npy}")


if __name__ == '__main__':
    # Example usage within module
    files, data, headers = load_reduced_images(config['paths']['output_dir'])

    # Prepare sub-images
    sub_images = split_and_prepare_images(data)
    mask_left, mask_right = split_mask(np.load(os.path.join(config['paths']['output_dir'], 'mask.npy')))
    
    # Average left and right windows, interpolate missing pixels
    images, mask = average_left_right(sub_images, mask_left, mask_right)

    # Compute argmaxs
    y_indices, x_indices = compute_centers_gauss(images)

    # Align images
    aligned_data = align_images_subpix(images, y_indices, x_indices)

    # Save aligned datacube
    save_aligned_images(config['paths']['output_dir'], aligned_data)
