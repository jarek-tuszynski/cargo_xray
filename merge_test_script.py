# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:21:10 2020

@author: tuszynskij
"""
import vacis_xray as vacis
import numpy as np
#from scipy import ndimage
import time
from PIL import Image
#import matplotlib.patches as patches
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Main program
def main():
    # Read and display image and segmentation map
    msk = Image.open("100860.png")
    msk = np.array(msk)

    # set up parameters
    param = {}
    scale = 1
    param['maxTileSize'] = round(700*scale)
    param['tileOverlap'] = 0.2
    param['minTileSize'] = round(64*scale)
    param['thresh_thickness'] = 0.25


    # resize image
    if scale!=1:
        s = (round(msk.shape[1]*scale), round(msk.shape[0]*scale))
        msk = np.array(Image.fromarray(msk).resize(s))

    fig, ax = plt.subplots(figsize=(16,8))
    ax.imshow(msk, aspect="equal")
    ax.axis('off')
    plt.show()

    # tile the image and show the tiles
    tiles, meta = vacis.tile_xray2memory(msk, param)
    pos = meta['pos']
    nRow, nCol = pos.shape
    f, axarr = plt.subplots(nRow, nCol, figsize=(16,8))
    for r in range(nRow):
        for c in range(nCol):
            ax = axarr[r,c]
            ax.imshow(tiles[pos[r,c]], aspect="equal")
            ax.axis('off')
    plt.show()

    # merge the tiles and show the results
    start = time.time()
    tiledImg = vacis.merge_tiles(tiles, meta)
    print( time.time() - start, 'seconds runtime')
    fig, ax = plt.subplots(figsize=(16,8))
    ax.imshow(tiledImg, aspect="equal")
    ax.axis('off')
    plt.show()

    vacis.merge_tiles_test(msk, param)

if __name__ == "__main__":
    main()
