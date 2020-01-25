import sys
import glob
import os
import pandas as pd
import re
import math
import numpy as np

# Default definitions
pixel_len = 1.52
time_scalar = 2.58
num_re = re.compile('([0-9]+)')
def numeric_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(num_re, s)]


def dist(p1, p2):
    return math.sqrt(((p2[0] - p1[0]) * pixel_len)**2 + ((p2[1] - p1[1]) * pixel_len)**2)


def in_range(ref_x, ref_y, c_x, c_y, l_bound, u_bound):
    return l_bound <= dist((ref_x, ref_y), (c_x, c_y)) < u_bound


# r_max should be the exclusive maximum.
def make_circle_lists(ref_x, ref_y, r_max):
    """dists_list = []
    squares = [(x, y) for y in range(ref_y - r_max, ref_y + r_max) for x in range(ref_x - r_max, ref_x + r_max)]
    for d in range(0, r_max):
        cur_list  = []
        for x, y in squares:
                if in_range(ref_x, ref_y, x, y, r_max, r_max + 1):
                    cur_list.append((x,y))
        dists_list.append(cur_list)

    return dists_list"""
    circles = []
    for d in range(r_max):
        cur = []
        for p in [(x,y) for x in range(512) for y in range(512)]:
            if in_range(ref_x, ref_y, p[0], p[1], d, d+1):
                cur.append(p)
        circles.append(cur)

    return circles

def test_collisions(circles_list):
    d = {}
    for cicle in circles_list:
        for p in cicle:
            d[p] = d.get(p, 0) + 1
            if d[p] > 1:
                print('Collision at', p, sep=' ')


#path = input('Please enter the path to your data folder: ')
#path = '../../raw_data/d'
if len(sys.argv) < 5:
    print('Missing arguments.\nPlease include \'src\' \'dest\' \'x_ref\' \'y_ref\'\nIn that order.')
else:
    #path = '../../raw_data/121319 Force measurements morning.lif_Series012_Crop001'
    path = sys.argv[1]
    print('Source: ' + path)
    dest = sys.argv[2]
    print('Destination: ' + dest)
    #x_y = input('Please enter the (X, Y) coordinates of your reference pixel')
    #x, y = tuple(x_y.split(','))
    #ref_x, ref_y = (217, 264)
    ref_x, ref_y = (int(sys.argv[3]), int(sys.argv[4]))
    print('(x_ref, y_ref): (', ref_x, ',', ref_y, ')\n', sep=' ')

    print('Loading all movie data...')
    files = glob.glob(os.path.join(path, "*.txt"))
    files.sort(key=numeric_sort_key)

    data = [pd.read_csv(path, sep=' ', header=None).drop(columns=[512]) for path in files]
    print('Done')
    # Step 1 get list of circle coordinates for each distance from reference pixel
    x_max, y_max = (max([i.shape[0] for i in data]), max([i.shape[1] for i in data]))
    r_max = min([ref_x, x_max - ref_x, ref_y, y_max - ref_y])

    print('Calculating circle indices...')
    circles = make_circle_lists(ref_x, ref_y, r_max)
    print('Done')
    # Step 2 create new DF where each col is a distance and row is a timestamp. Entries are circle average for dist and time
    print('Averaging circle values...')
    dist_averages = pd.DataFrame([[0 for i in range(r_max)] for j in range(len(data))])
    for t in range(len(data)):
        # Time t; average the cells of each circle and insert them into t,radius
        for r in range(r_max):
            #dist_averages = dist_averages.append(pd.DataFrame(np.mean([data[t][x][y] for (x,y) in circles[r]])), ignore_index=True)
            dist_averages[r][t] = np.mean([data[t][x][y] for (x,y) in circles[r]], dtype=np.float64)


    print('Done.')


    #dist_averages.to_csv('../processed_data')
    dist_averages.transpose().to_csv(dest, header=np.arange(dist_averages.shape[0])*time_scalar, index=False)
