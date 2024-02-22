from numpy import random, ceil, floor,cos, sin, pi,sqrt, round
import random
import numpy as np
from PIL import Image
from scipy.ndimage import rotate

'''
Here created functions, which are using to create parquet.
'''

class Wang_Tile():

    def __init__(self,amountv, amounth, set):
        if len(set[0]) != 2:
            raise ValueError('Wrong size') 
        if set[0][0].shape[0] != set[0][0].shape[1] and len(set[0][0]) != 4:
            raise ValueError('Screw up') 
        self.dimension = set[0][0].shape[2]
        self.tiles = set
        self.tsize = set[0][0].shape[0]
        self.N = amountv
        self.M = amounth
        self.height = amountv*self.tsize
        self.width = amounth*self.tsize
        if len(set[0][0].shape) == 2:
            self.shape = (self.height, self.width)
        else:
            self.shape = (self.height, self.width, self.dimension)



    def choose_samples_tile_noise(self, tile, class_):
        finding = [False]*len(self.tiles)
        for ind in range(len(self.tiles)):

            if class_ == 'left':
                sample = self.tiles[ind][1]
                #print(sample[3],tile[1])
                if sample[3] == tile[1]:
                    finding[ind] = True

            if class_ == 'up':
                sample = self.tiles[ind][1]
                if sample[0] == tile[2]:
                    finding[ind] = True
        
        return [ ind for ind, condition in zip(range(len(self.tiles)), finding) if condition == True]

    def create_canvas(self):
        #create canvas (array)
        canvas = np.zeros((self.shape))
        color_preserve = np.zeros((self.N,self.M), dtype=list)
        for i in range(self.N):
            row_tiles = np.zeros((self.tsize,self.shape[1],self.dimension))
            row_wall = np.zeros((1,self.M),dtype=list)
            j = 0
            while j<self.M:
                if i == 0 and j == 0:
                        sample = random.sample(self.tiles, k = 1)[0]
                        row_tiles[:,j*self.tsize:(j+1)*self.tsize] = sample[0].astype(np.uint8)
                        row_wall[0,j] = sample[1]
                        j += 1
                        
                else:
                    if i == 0 and j != 0:
                        tile_left = row_tiles[:,(j-1)*self.tsize:j*(self.tsize)].astype(np.uint8)
                        wall_right = row_wall[0,j-1]
                        samples = self.choose_samples_tile_noise(wall_right, 'left')

                        if len(samples) == 0:
                            j = max(0, j-np.random.randint(2,7))

                        else:

                            ind = random.sample(samples, k = 1)[0]
                            row_tiles[:,j*self.tsize:(j+1)*self.tsize]  = self.tiles[ind][0].astype(np.uint8)
                            row_wall[0,j] = self.tiles[ind][1]
                            j += 1
                            
                    elif  j == 0 and i !=0:
                        
                        tile_up = canvas[(i-1)*self.tsize:(i)*self.tsize,(j)*self.tsize:(j+1)* self.tsize].astype(np.uint8)
                        wall_up = color_preserve[i-1,j]
                        samples = self.choose_samples_tile_noise(wall_up, 'up')
                        if len(samples) == 0:
                            j = max(0, j-np.random.randint(2,7))

                        else:

                            ind = random.sample(samples, k = 1)[0]
                            sample = self.tiles[ind]
                            row_tiles[:,j*self.tsize:(j+1)*self.tsize]  = sample[0].astype(np.uint8)
                            row_wall[0,j] = sample[1]  
                            j += 1

                    else:
                        tile_up = canvas[(i-1)*self.tsize:(i)*self.tsize,(j)*self.tsize:(j+1)*self.tsize].astype(np.uint8)
                        wall_up = color_preserve[i-1,j]
                        samples_up = self.choose_samples_tile_noise(wall_up, 'up')
                        tile_left = row_tiles[:,(j-1)*self.tsize:j*(self.tsize)].astype(np.uint8)
                        wall_right = row_wall[0,j-1]
                        samples_left = self.choose_samples_tile_noise(wall_right, 'left')
                        if len(samples_left) == 0 or len(samples_up) == 0:
                            samples = []
                        else:
                            samples = []
                            for left in samples_left:
                                samples += [left for up in samples_up if self.tiles[up][1] == self.tiles[left][1]]
                        #print('merge:',len(samples),' up:',len(samples_up), ' left:', len(samples_left))
                        if len(samples) == 0:
                            j = max(0, j-np.random.randint(0,ceil(self.M*0.1)))

                        else:
                            ind = random.sample(samples, k = 1)[0]
                            sample = self.tiles[ind]
                            row_tiles[:,j*self.tsize:(j+1)*self.tsize]  = sample[0].astype(np.uint8)
                            row_wall[0,j] = sample[1]  
                            j += 1
            
            canvas[i*self.tsize:(i+1)*self.tsize,:] = row_tiles
            color_preserve[i,:] = row_wall
        return canvas

#    def size(self):
#        x = int(ceil(self.height/self.tsize)*self.tsize)
#        y = int(ceil(self.width/self.tsize)*self.tsize)
#        noise = create_canvas_noise([x,y], tile_size, amount, typ)

#        return noise[:n, :m]
# skup się na ogólnym podejściu. Masz tylko dać algorytm do generowanie przestrzenie, 
# generowanie zestawu ma być domyślne, a jeżeli użytkownik będzie chciał zmienić samemu zestaw, to ma takie prawo
    
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

p1 = Wang_Tile(40,50,random.sample(tiles,12))

x = p1.create_canvas()

image = Image.fromarray(x.astype(np.uint8), mode = 'RGB')
image.show()
