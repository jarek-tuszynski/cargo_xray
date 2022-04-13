# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:21:10 2020

@author: tuszynskij
"""
import vacis_xray as vacis
import numpy as np
from scipy import ndimage
import time
#from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Main program
def main():
    root_folder = 'C:/Projects/VACIS/'

    # read parameter file
    param = vacis.read_params(root_folder+'VACIS_params.xml')
    print('Content of parameter file:', param)

    # read and display xray file (pick any file)
    in_fname = '2020-03-10_10_44_41.tif'
    thick = vacis.read_xray(in_fname, param)
    vacis.display_xray(thick)

    # write calibrated xray and xray tiles
    out_fname = root_folder+'output_xray.jpg'
    vacis.write_xray(thick, out_fname, param)
    vacis.tile_xray2file(thick, out_fname, param)

	# resize image
	#s = (round(thick.shape[1]/3), round(thick.shape[0]/3))
	#thick = np.array(Image.fromarray(thick).resize(s))

    # Find container
    msk = np.ma.masked_where(thick>0.05, thick)
    msk = ndimage.binary_fill_holes(msk, structure=np.ones((5,5)))
    start = time.time()
    tlbr  = vacis.FindLargestRectangle(msk)
    print( time.time() - start, 'seconds runtime')
    fig,ax = plt.subplots(1)

    # show final mask and container location
    ax.imshow(msk) # Display the image
    w, h = tlbr[3]-tlbr[1]+1, tlbr[2]-tlbr[0]+1 # width and height
    rect = patches.Rectangle((tlbr[1],tlbr[2]),w,-h,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect) # Add the patch to the Axes
    plt.show()

    vacis.FindLargestRectangle_test(msk)

if __name__ == "__main__":
    main()
