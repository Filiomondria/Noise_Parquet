from numpy import random, ceil, floor,cos, sin, pi,sqrt, round
import random
import numpy as np


'''
Here created functions, which are using to create parquet.
'''

class Wang_Tile():

    def __init__(self,amountv, amounth, set):

        if len(set[0]) != 2:
            raise ValueError('Wrong size') 
        if set[0][0].shape[0] != set[0][0].shape[1] and len(set[0][0]) != 4:
            raise ValueError('Screw up') 
        
        try: 
            set[0][0].shape[2]
        except:
            self.dimension = False
        else:
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
                if sample[3] == tile[1]:
                    finding[ind] = True

            if class_ == 'up':
                sample = self.tiles[ind][1]
                if sample[0] == tile[2]:
                    finding[ind] = True
        
        return [ ind for ind, condition in zip(range(len(self.tiles)), finding) if condition == True]

    def create_canvas(self):
        canvas = np.zeros((self.shape))
        color_preserve = np.zeros((self.N,self.M), dtype=list)
        for i in range(self.N):
            if not self.dimension:
               row_tiles = np.zeros((self.tsize,self.shape[1])) 
            else:
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

    # def size(self):
    #    x = int(ceil(self.height/self.tsize)*self.tsize)
    #    y = int(ceil(self.width/self.tsize)*self.tsize)
    #    noise = create_canvas_noise([x,y], tile_size, amount, typ)

    #    return noise[:n, :m]
# skup się na ogólnym podejściu. Masz tylko dać algorytm do generowanie przestrzenie, 
# generowanie zestawu ma być domyślne, a jeżeli użytkownik będzie chciał zmienić samemu zestaw, to ma takie prawo
    

