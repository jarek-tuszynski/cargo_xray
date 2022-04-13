# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:21:10 2020

@author: tuszynskij
"""
import math
import os
import xmltodict
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from PIL import Image
import tifffile
from scipy import ndimage
import matplotlib.patches as patches
from scipy import stats
from sklearn.utils.extmath import weighted_mode

# --------------------------------------------------------------------------
# Read xml parameter file
# INPUTS:
#  - fname - filename of xml file with the data
# OUTPUT:
#  - param - a dictionary structure holding parameters
def read_params(fname):
	with open(fname) as fd:
		xml = xmltodict.parse(fd.read())
	param = xml['param']
	new_param = {}
	for field in param:
		if param[field].startswith('[') and param[field].endswith(']'):
			# parse string like '[0,0;...12,3.7043]' into np 2D array
			cal  = param[field].strip('[]').split(';') # get array of row strings
			val = np.zeros((len(cal),2))  # init 2D array
			for row in range(len(cal)):
				val[row,] = cal[row].split(',')
		else:
			val = float(param[field])
		new_param[field] = val
	return new_param

# --------------------------------------------------------------------------
# file reading and calibration
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Read and calibrate xray file
# INPUTS:
#  - fname - filename of xray file with the data
#  - param - a dictionary structure holding parameters
# OUTPUT:
#  - Thick = numpy 2D array holding image with pixels calibrated to inches of steel
def read_xray(fname, param):
	X  = tifffile.imread(fname, key=1)
	nCol2 = X.shape[1]
	X  = tifffile.imread(fname, key=0)
	nCol1 = X.shape[1]
	X = X[:, (nCol1-nCol2):-1]
	X  = np.clip(X, 1, param['air_transmission'])  # no zeros
	Y  = (1.0*X)/param['air_transmission'] # convert to transmission in 0-1 range
	T  = -np.log(Y)/0.766   # approximate thickness of steel in inches (Myron's formula)
	cy = param['CalibrationTable'][:,0]
	cx = param['CalibrationTable'][:,1]
	T  = np.clip(T, cx[0], cx[-1])  #clip pixels to range based on CalTable
	return interpolate.interp1d(cx, cy)(T)  # calibrate

# --------------------------------------------------------------------------
# display the image file
# INPUTS:
#  - image - numpy 2D array holding image
# OUTPUT:
#  - plotted image
def display_xray(image):
	fig, ax = plt.subplots()
	im = ax.imshow(image, aspect="equal")
	if str(image.dtype)!='bool':
			fig.colorbar(im, ax=ax)
	plt.show()

# --------------------------------------------------------------------------
# write the image file
# INPUTS:
#  * image - numpy 2D array holding image
#  * out_fname - output filename
#  * param - preference parameters including
#	 * param['max_thickness']
# OUTPUT:
#  - saved image
def write_xray(image, out_fname, param):
	img = np.int8(255*image/param['max_thickness'])
	im  = Image.fromarray(img)
	im.convert('RGB').save(out_fname)

# --------------------------------------------------------------------------
# preprocess xray image:
#  1. Read file
#  2. calibrate the data by conversion to inches of steel units
#  3. Find container in the image
# INPUTS:
#  - fname - filename of xray file with the data
#  - param - a dictionary structure holding parameters
# OUTPUT:
#  - container section of th image
def preprocess_xray(fname, param):
	thick = read_xray(fname, param)
	return crop_container(thick, param)

# --------------------------------------------------------------------------
# crop the container
# INPUTS:
#  - thick- xray image converted to thickness units (inches of steel)
#  - param - a dictionary structure holding parameters
# OUTPUT:
#  - container section of th image
def crop_container(thick, param):
	msk = np.ma.masked_where(thick>param.thresh_thickness, thick)
	msk = ndimage.binary_fill_holes(msk, structure=np.ones((5,5)))
	t, l, b, r, _ = FindLargestRectangle(msk)
	return thick[t:b,l:r]

# --------------------------------------------------------------------------
# tiling and tile merging
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# calculate grid parameters based on preferences and umage size
# INPUTS:
#  * N - number of pixels (rows or columns)
#  * param - preference parameters including
#	 * param['tileOverlap']
#	 * param['minTileSize']
#	 * param['maxTileSize']
# OUTPUTS:
#  * xMin, xMax - 2 same size arrays with min and max values of  grid calls
def tile_grid(N, param):
	tile_size	= param['maxTileSize']
	tile_overlap = round(tile_size * param['tileOverlap']) # 20% overlap
	d  = tile_size-tile_overlap
	if tile_size<N:
		nx   = math.floor(N/d)
		xMin = np.arange(nx+1)*d
		xMax = np.minimum(xMin + tile_size-1, N-1) + 1
		if (xMax[-1]-xMin[-1]<param['minTileSize']+tile_overlap):
			xMin = xMin[:-1]
			xMax = xMax[:-1]
	else: # single tile in that direction
		xMin = np.atleast_1d(0)
		xMax = np.atleast_1d(N)
	return xMin.astype('int'), xMax.astype('int') # use zero based index

# --------------------------------------------------------------------------
def tile_xray2file(image, filename, param):
# write the image file
# INPUTS:
#  * image - numpy 2D array holding image
#  * out_fname - output filename, which will be used as a source of folder, base
#		 name and extension. The actual filenames will have 2 digit grid
#		 numbers attached to the base name.
#  * param - preference parameters including
#	 * param['max_thickness']
# OUTPUT:
#  - saved images
	folder, base = os.path.split(filename)
	root  , ext  = os.path.splitext(base)
	size = image.shape
	x1, x2 = tile_grid(size[1], param) # calculate tile grid
	y1, y2 = tile_grid(size[0], param)
	for row in range(len(y1)):
		for col in range(len(x1)):
			out_fname = '%s\\%s_%02i_%02i%s' % (folder, root, row, col, ext)
			patch = image[ y1[row]:y2[row], x1[col]:x2[col] ]   # clip xray grayscale tile
			write_xray(patch, out_fname, param)

# --------------------------------------------------------------------------
def tile_xray2memory(image, param):
# write the image file
# INPUTS:
#  * image - numpy 2D array holding image
#  * out_fname - output filename, which will be used as a source of folder, base
#		 name and extension. The actual filenames will have 2 digit grid
#		 numbers attached to the base name.
#  * param - preference parameters including
#	 * param['max_thickness']
# OUTPUT:
#  - tiles - array of n 2D numpy arrays
#  - meta  - tile matadata dictionary:
#	 - meta.bbox - n x 4 numpy array of tile bounding boxes. One tile per row,
#            with columns storing: top, left, bottom and right edge of each tile
#    - meta.pos - 2D array storing for each tile row and column the tile number
	size = image.shape
	x1, x2 = tile_grid(size[1], param) # calculate tile grid
	y1, y2 = tile_grid(size[0], param)
	n = len(x1)*len(y1)
	tiles = [None] * n
	pos   = np.zeros((len(y1), len(x1)), dtype=int)
	bbox  = np.zeros((n,4), dtype=int)
	k = 0;
	for row in range(len(y1)):
		for col in range(len(x1)):
			patch = image[ y1[row]:y2[row], x1[col]:x2[col] ]   # clip xray grayscale tile
			tiles[k] = patch
			bbox [k] = [ y1[row], x1[col], y2[row], x2[col] ]
			pos[row,col] = k
			k+=1
	meta = {'pos':pos, 'bbox':bbox}
	return tiles, meta

# --------------------------------------------------------------------------
def merge_tiles(tiles, meta):
# merge_tiles conflicts at overlapping regions are resolved by picking pixels
#   further from the boundary
# INPUTS:
#  - tiles - array of n 2D numpy arrays
#  - meta  - tile matadata dictionary, created by tile_xray2memory:
#	 - meta.bbox - n x 4 numpy array of tile bounding boxes. One tile per row,
#            with columns storing: top, left, bottom and right edge of each tile
#    - meta.pos - 2D array storing for each tile row and column the tile number
# OUTPUT:
#  - tiledImg - image merged from all the tiles

    def distance_from_edge(x):
        x = np.pad(np.ones(x.shape), 1, mode='constant') # pad edge with 0s
        dist = ndimage.distance_transform_cdt(x, metric='taxicab')
        return dist[1:-1, 1:-1] # trim edge

    pos  = meta['pos']
    bbox = meta['bbox']
    nRow, nCol = pos.shape  # number or rows and columns in tile grid
    bbMax = np.max(bbox, axis=0)
    mRow, mCol = bbMax[2:4,] # number of rows and columns in tiled image

    # create structure combining all the tiles
    # also keep track of a discance of each pixel to tiles edge, According to
    # https://arxiv.org/abs/1805.12219 paper acuracy of labels decreases
    # towards the edges, se we will use this distance as a weigh
    dist = distance_from_edge(tiles[0])
    W = np.zeros((mRow,mCol), dtype=int)
    T = np.zeros((mRow,mCol), dtype=int)
    for row in range(nRow):
        for col in range(nCol):
            k = pos[row,col]
            y1,x1,y2,x2 = bbox[k,:]
            t1 = tiles[k]
            w1 = dist[:t1.shape[0], :t1.shape[1]]
            t2 = T[ y1:y2, x1:x2]
            w2 = W[ y1:y2, x1:x2]
            msk = w2>w1
            t1[msk] = t2[msk]
            w1[msk] = w2[msk]
            T[ y1:y2, x1:x2] = t1
            W[ y1:y2, x1:x2] = w1

    #display_xray(W)
    return T

# --------------------------------------------------------------------------
def merge_tiles_test(image, param):
# testing of tile_xray2memory and merge_tiles functions:
# test #1 : make sure that tiling and tile marge restores original image
    tiles, meta = tile_xray2memory(image, param)
    out = merge_tiles(tiles, meta)
    assert(np.all(image.shape==out.shape)), 'merge_tiles_test failed: sizes differ'
    assert(np.all(image==out)), 'merge_tiles_test failed: values differ'

# --------------------------------------------------------------------------
# Container Finding
# --------------------------------------------------------------------------
def FindLargestRectangle(msk):
# finds rectangle with longest perimeter with all points set to 1
# input: msk   - B/W boolean numpy matrix
# output: tlbr - [top, left, bottom, right] edges of the bounding box
	nr, nc = msk.shape     # number of rows and columns
	heights = [0] * nc	   # initialize a list storing heights if ones above
	tlbr = [0,0,0,0,0]	   # initialize array with bounding box and optimization criteria
	for bRow in range(nr): # for each bottom row
		# Update the heights of columns of 1's with a bottom at row "bRow"
		for col in range(nc):
			heights[col] = heights[col]+1 if msk[bRow,col] else 0
		# Find the largest rectangle in the histgrams
		stack = [-1,0]                    # initialize stack
		for rCol in range(nc):			  # for each right column
			while (len(stack) >= 2) and ((rCol==nc-1) or (heights[stack[-1]] >= heights[rCol+1])):
				height = heights[stack.pop()]
				lCol   = stack[-1] + 1	  # look up left column
				width  = rCol - lCol + 1  # calculate width
				crit   = height + width   # optimizaton criteria area (*) or perimeter(+)
				if (crit>tlbr[4]):		  # check if it produces larger Criteria
					tRow = bRow-height+1  # top row
					tlbr = [tRow, lCol, bRow, rCol, crit]
			stack.append(rCol+1)
	return tlbr

# --------------------------------------------------------------------------
def FindLargestRectangle_test(msk):
# Perform test of FindLargestRectangle function:
# test #1: operations like transpose, and vertical and horizontal flip should
#          not change the maximal size of the rectangle
# input: msk   - B/W boolean numpy matrix
	size = [0]*4
	t,l,b,r,size[0] = FindLargestRectangle(msk)
	_,_,_,_,size[1] = FindLargestRectangle(np.fliplr(msk))
	_,_,_,_,size[2] = FindLargestRectangle(np.flipud(msk))
	_,_,_,_,size[3] = FindLargestRectangle(np.transpose(msk))
	assert(min(size)==max(size)), 'Error in Test_FindLargestRectangle: image reflections should no change output size'

# --------------------------------------------------------------------------
# Plots mask used for largest rectangle finding
def FindLargestRectangle_plot(msk):
	t,l,b,r,_ = FindLargestRectangle(msk)

	fig,ax = plt.subplots(1)
	ax.imshow(msk) # Display the image
	rect = patches.Rectangle((l,b),r-l, b-t,linewidth=1,edgecolor='r',facecolor='none')
	ax.add_patch(rect) # Add the patch to the Axes
	plt.show()
