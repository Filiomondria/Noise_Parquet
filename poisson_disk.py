from numpy import random, ceil, floor,cos, sin, pi,sqrt, round
import random
import numpy as np
from PIL import Image

class Disk:
    '''
    Klasa stworzona do tworznia szumów
    '''
    def __init__(self, height, width, sample_size, r = 4, amount = 10, k = 4):
        self.height = height
        self.width = width
        self.r = r
        self.amount = amount
        self.k = k
        self.sample_size = sample_size
  
    def poisson_disc_samples(self, list_points):
        '''
        Funkcja tworzy dyski poissona na zadanej płaszczyźnie
        '''
        N = 2 # bazujemy na wymiarze 2D
        cellsize = self.r / sqrt(N)
        grid_width = int(ceil(self.width / cellsize)) # ilość kratek w szerokości
        grid_height = int(ceil(self.height / cellsize)) # ilość kratek w wysokości
        grid = np.empty((grid_height,grid_width,),dtype=object)

        # Patrz na współrzędne punktu p
        def grid_coords(p):
            return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize)) 
        
        #Funkcja do sprawdzania kolizji 
        def valid_point(p,gx,gy):

            if not (0 <= p[0] < self.height and 0 <= p[1] < self.width):
                    return False
            
        # wektor 
            # napewno trzeba dać te int(ceil(self.r)?
            xl, xr = max(gx - int(ceil(self.r)+3), 0), min(gx + int(ceil(self.r)+3), grid_width)
            yt,yb = max(gy - int(ceil(self.r)+3), 0), min(gy + int(ceil(self.r)+3), grid_height)
            for x in range(xl,xr):
                for y in range(yt,yb):
                    
                    g = grid[x,y]
                    if g is None:
                        continue
                    if sqrt((g[0] - p[0]) ** 2 + (g[1] - p[1]) ** 2) <= self.r:
                        
                        return False

            return True
        
        # generujemy początkową próbkę(punkt) i dodajemy go do listy aktywnych

        if len(list_points) == 0: 
            p = random.random() * self.height , random.random() * self.width#, r
            active_points = [p]
            grid_x, grid_y = grid_coords(p[:2])
            grid[grid_x, grid_y] = p

        else:
            active_points = list_points
            for ele in list_points:
                grid_x, grid_y = grid_coords(ele[:2])
                grid[grid_x, grid_y] = ele

        while len(active_points) > 0:
            # bierzemy losową próbkę z listy aktywnych próbek
            qi = int(random.random()*len(active_points))
            qx, qy = active_points[qi][:2]
            check = len(active_points)
            for tries in range(self.k):
                angle = 2 * pi * random.random()
                # losujem pod jakim kąte i o losową odległość od próbki ustawić koejną próbkę
                # odległość od punktu badanego jest od r do 2r
                d = self.r * (random.random() + 1)  
                new_px = qx + d * cos(angle)
                new_py = qy + d * sin(angle)
                new_p = (new_px,new_py)
                grid_x, grid_y = grid_coords(new_p)    
                if not valid_point(new_p, grid_x, grid_y):
                    continue
                active_points.append(new_p)
                grid[grid_x,grid_y] = new_p
            if check + self.k > len(active_points):
                active_points.remove(active_points[qi])

        grid = grid.flatten()      
        return [p for p in grid if p is not None]


    def relaxation_method(self):
        '''
        Funkcja służy do generowania dysków poisson na danej płaszczyźnie i zapewnia odpowiednią ilość dysków.
        '''
        points = []
        check = True
        while check:
            created_points = self.poisson_disc_samples(list_points = points)
            points = created_points

            if len(points) < self.amount:
                self.r = self.r*0.9
                self.k = int(np.ceil(self.k/0.9))

            else:
                check = False

        return points

    def noise_matrix(self):
        '''
        Funkcja która transformuje szum z listy do matrixa
        '''
        matrix = np.zeros((self.width,self.height), dtype=np.int8)
        self.points = self.relaxation_method()
        for i, j in self.points: 

            matrix[int(i),int(j)] = 1

        return matrix #czy raczej self.matrix

    
    def sample_noise(self):
        '''
        punkt to lewy górny wierzchołek próbki
        1 Etap. Losujemy punkt  z index row and col z przedziału [0,size - sample_size]
        2 Etap. Losujemy punkt z uwzględnieniem próbki 1, tak więc bierzemy dolny index row próbki i mamy ograniczenie dla przedziału row, przedział dla kolumn się nie zmienia.
        3 Etap. Losujemy punkt z uwzględnieniem próbki 1 i 2, najperw sprawdzamy czy punkt próbki 2 jest równy size - sample_size, jeśli tak to ograniczamy 
        '''

        amount = 4
        size = (self.height, self.width)
        matrix_noise = self.noise_matrix()
        sample_list = []

        # i to kolumna a j to wiersz

        # zedytować to trzeba, zapomniałem o granicach
        mask = np.zeros(size)
        for _ in range(amount):
            if _ == 0:
                mask[4:size[0] - self.sample_size, 4:size[1] - self.sample_size] = 1
            if _ == 1:
                mask[size[0] - self.sample_size:-5, 4:size[1] - self.sample_size] = 1
            if _ == 2:
                mask[4:size[0] - self.sample_size, size[1] - self.sample_size:-5] = 1
            if _ == 3:
                mask[size[0] - self.sample_size:-5, size[1] - self.sample_size:-5] = 1
            not_found = True
            points_active = np.argwhere(mask == 1)
            while not_found:

                random_index = np.random.choice(len(points_active))
                point = points_active[random_index]
                if point[0] + self.sample_size < size[0] and point[1] + self.sample_size < size[1]:
                    if mask[point[0] + self.sample_size, point[1]] == 1 and mask[point[0], point[1] + self.sample_size] == 1 and mask[point[0] + self.sample_size,point[1] + self.sample_size] == 1:
                        not_found = False
                        sample_list.append(matrix_noise[point[0]:point[0] + self.sample_size, point[1]:point[1] + self.sample_size])
                        mask[point[0]:point[0] + self.sample_size, point[1]:point[1] + self.sample_size] = 0
                    else:
                        points_active = np.delete(points_active, random_index, axis = 0)
                else:
                    points_active = np.delete(points_active, random_index, axis = 0)

        return sample_list
    
    def binar_map(self, name):
        noise = self.noise_matrix()
        map = Image.fromarray(((1-noise)*255).astype(np.uint8), mode='L')
        map.save(name + '.png')


