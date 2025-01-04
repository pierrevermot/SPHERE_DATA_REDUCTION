# Configuration File for Data Reduction Pipeline

config = {
    # Directory Paths
    'paths': {
        'input_dir': '../data/NGC_1068/',
        'output_dir': '../results/NGC_1068/',
        'flat_dir': '../data/NGC_1068/FLAT/',
        'dark_dir': '../data/NGC_1068/DARK/',
        'object_dir': '../data/NGC_1068/OBJECT/',
        'calibration_dir': '../data/NGC_1068/REF/',
        'log_file': '../logs/process.log'
    },

    # Processing Parameters
    'processing': {
        'flat_threshold': 6000,
        'mask_width': 64,
        'gaussian_filter': 5,
        'median_filter_size': 5,
        'gaussian_filter_size': 3,
        'n_med': 20
    },

    # Alignment Parameters
    'alignment': {
        'use_gaussian': True,
        'use_median_filter': True,
        'median_filter_size': 5,
        'x_window_roll': -1,
        'y_window_roll': 10
    },

    # Mask Parameters
    'mask': {
        'threshold_lower': 0.20,
        'threshold_upper': 0.99,
        'window_size': 256
    },
    
    'calibration': {
    'pixel_scale': 12.25,   # Pixel scale in milliarcseconds
    'ref_x': "max",          # Reference pixel position (x-coordinate)
    'ref_y': "max",          # Reference pixel position (y-coordinate)
    'band': 'K',
    'ref_ra': 40.669621,  # Reference RA (degrees)
    'ref_dec': -0.013318    # Reference DEC (degrees)
},


    # Shear Correction
    'shear': {
        'order': 3
    },

    # Reduction Parameters
    'reduction': {
        'scale_factor': 1.0
    },

    # Output Options
    'output': {
        'overwrite': True,
        'fits_output_name': 'processed_output.fits',
        'full_output_name': 'full_processed_output.fits'
    },

    # Debugging and Logging
    'logging': {
        'debug': True,
        'log_level': 'INFO'
    }
    
}
