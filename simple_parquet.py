from numpy import random, ceil, floor,cos, sin, pi,sqrt, round
import random
import numpy as np
from PIL import Image
from scipy.ndimage import rotate
import wang_tile as wt

def create_tile(tile_set, size):
    color = tile_set[0]
    wall = tile_set[1]
    n = len(color[0])
    tile = np.zeros((size*2,size*2,n), np.uint8)
    ind = 0

    for i in range(2):
        for j in range(2):
            tile[i*size:(i+1)*size, j*size:(j+1)*size] = color[ind]
            ind += 1
    
    rotate_tile = rotate(tile, angle=-45, reshape=False)
    return rotate_tile[int(size*0.5):int(size*1.5), int(size*0.5):int(size*1.5)], wall

def wallcolors(color=True, typ = None):
    dict_color = {}
    if color == True:
        R = (162, 65, 204 )
        dict_color['R'] = R
        B = (31, 183, 185 )
        dict_color['B'] = B
        G = (150, 172, 173 )
        dict_color['G'] = G
        Y = (205, 43, 107 )
        dict_color['Y'] = Y
        color = [R,B,G,Y]
    
    if typ == None:
        list_tiles = []
        #numbers = np.random.default_rng().choice(20, size=10, replace=False)  
        for up in color:
            for right in color:
                for down in color:
                    for left in color:
                        list_tiles.append([up, right, down, left])

        for col in color:
            list_tiles.remove([col]*4)
        
    else:
        list_tiles = []
        colors_v = [(dict_color[key], key) for key in typ[0]]
        colors_h = [(dict_color[key], key) for key in typ[1]]
        for up, cup in colors_v:
            for right, cr in colors_h:
                for down, cd in colors_v:
                    for left, cl in colors_h:
                        list_tiles.append([[up, right, left, down], [cup, cr, cd, cl]])


    return list_tiles

colors = wallcolors(True, typ = [['R','Y','B'],['G','B']])
tiles = [create_tile(set, 80) for set in colors]

p1 = wt.Wang_Tile(40,50,random.sample(tiles,12))

x = p1.create_canvas()

image = Image.fromarray(x.astype(np.uint8), mode = 'RGB')
image.show()