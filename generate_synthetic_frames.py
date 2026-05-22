import os
import numpy as np
from astropy.io import fits
from datetime import datetime, timedelta

os.makedirs('fits', exist_ok=True)

center_ra = 280.50
center_dec = -20.0
start_time = datetime(2026, 5, 1, 12, 0, 0)

# Strictly symmetric square matrix dimensions
frame_size = 256

for i in range(5):
    # Establish base 256x256 double-precision float space
    data = np.zeros((frame_size, frame_size), dtype=np.float64)
    
    # Place a distinct 20x20 pixel feature square right in the center
    # This guarantees robust dynamic range values (> 0)
    data[118:138, 118:138] = 120.0
    
    frame_time = start_time + timedelta(seconds=i * 10)
    date_str = frame_time.strftime('%Y-%m-%d')
    time_str = frame_time.strftime('%H:%M:%S.%f')[:-3]

    hdu = fits.PrimaryHDU(data)
    
    # Metadata Header Configuration
    hdu.header['DATE-OBS'] = date_str
    hdu.header['TIME-OBS'] = time_str
    hdu.header['RA_TARG'] = center_ra
    hdu.header['DEC_TARG'] = center_dec
    
    # Center World Coordinate System (WCS) markers strictly on the central index (128)
    hdu.header['CRPIX1'] = 128.0
    hdu.header['CRPIX2'] = 128.0
    hdu.header['CRVAL1'] = center_ra
    hdu.header['CRVAL2'] = center_dec
    hdu.header['CDELT1'] = -0.0001
    hdu.header['CDELT2'] = 0.0001
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'

    filename = f"fits/frame_{i:03d}.fits"
    hdu.writeto(filename, overwrite=True)
    print(f"Generated clean symmetric frame: {filename}")

print("\nExecuting baseline processing...")