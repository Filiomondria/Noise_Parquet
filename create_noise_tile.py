from numpy import random, ceil, floor,cos, sin, pi,sqrt, round, longdouble
import random
import numpy as np
from PIL import Image
from scipy.ndimage import rotate
import poisson_disk as pdisk
from scipy import signal
import PIL
from PIL import ImageFilter

class Noise_Tile:

    def __init__(self, patchs, overlap, size, tile_size = None):
        '''
        Overlap - size of overlap, overlap x size patch
        '''
        self.patchs = patchs

        if overlap%2 == 0:
            self.overlap = overlap
        else:
            self.overlap = overlap+1
        self.s = size
        self.stile = tile_size

    def kernel_mask(self,overlap,filtr):
        '''
        Do poprawy, importuj funkcję
        Ogólnie to ambiguous dla zmiennej overlap XD
        '''
        img = PIL.Image.fromarray((overlap).astype(np.uint8), mode = 'L')
        img = img.filter(ImageFilter.Kernel((5,5), filtr))
        output = np.asarray(img)
        return output

    

    def boundary(self, patch_old,patch_n, patch_size,overlap_type):

        """
        Tutaj można coś poprawić
        Function computes a minimum boundary cut (another path) and return list of position where boudary lies.
        Parameters: overlap_type 
        """

        wps = np.arange(patch_size - 2, -1, -1)

        # case vertical
        if overlap_type == 'V':

            E = np.zeros((patch_size,self.overlap),dtype = np.int8)
            T = np.zeros((patch_size,self.overlap),dtype = tuple)
            gamma = [None]*patch_size
            # można zrobił z macierzą wagi Q jak w wyborze łatek ale na razie to odpuścimy
            
            patch_old_overlap = patch_old[:,-self.overlap:]
            patch_n_overlap = patch_n[:,:self.overlap]
            e = np.abs(patch_old_overlap + patch_n_overlap)
            E[-1:] = e[-1:]
            # teraz tak dobrać by było mniej więcej po środku

            for i in wps:
                j = 0
                min_ = np.min([E[i + 1, j], E[i + 1, j + 1]])
                E[i,0] = e[i,j] + min_
                T[i,0] = (i+1,np.argwhere(E[i+1,j:j+2] == min_)[0,0]+j)

                for j in range(1,self.overlap-1):
                    min_ = np.min([E[i + 1, j - 1], E[i + 1, j], E[i + 1, j + 1]])
                    E[i,j] = e[i,j] + min_
                    T[i,j] = (i+1, np.argwhere(E[i+1,j-1:j+2] == min_)[0,0]+j-1)

                j = self.overlap-1
                min_ = np.min([E[i + 1, j - 1], E[i + 1, j]])
                E[i,self.overlap-1] = e[i,j] + min_
                T[i,self.overlap-1] = (i+1, np.where(E[i+1,j-1:j+1] == min_)[0][0]+j-1)

            j_star = self.overlap//2
            gamma[patch_size-1] = (0,j_star)
            for i in wps:
                gamma[i] = T[gamma[i + 1]]
            gamma[0] = T[gamma[1]]

        #case horizontal
        if overlap_type == 'H':
            E = np.zeros((self.overlap,patch_size),dtype = np.int8)
            T = np.zeros((self.overlap,patch_size),dtype = tuple)
            gamma = [None]*patch_size
            # można zrobił z macierzą wagi Q jak w wyborze łatek ale na razie to odpuścimy
            patch_old_overlap = patch_old[-self.overlap:,:]
            patch_n_overlap = patch_n[:self.overlap,:]
            e = np.abs(patch_old_overlap + patch_n_overlap)
            
            E[:-1] = e[:-1]

            for j in wps:
                i = 0
                min_ = np.min([E[i, j + 1], E[i + 1, j + 1]])
                E[0,j] = e[i,j] + min_
                T[0,j] = (np.argwhere(E[i:i + 2,j + 1] == min_)[0,0] + i, j + 1)

                for i in range(1,self.overlap-1):
                    min_ = np.min([E[i - 1, j + 1], E[i , j + 1], E[i + 1, j + 1]])
                    E[i,j] = e[i,j] + min_
                    T[i,j] = (np.argwhere(E[i - 1: i + 2, j + 1] == min_)[0,0] + i - 1, j + 1)
                
                i = self.overlap-1
                min_ = np.min([E[i - 1, j + 1], E[i, j + 1]])
                E[self.overlap-1, j] = e[i, j] + min_
                T[self.overlap-1, j] = (np.where(E[i - 1: i + 1, j + 1] == min_)[0][0] + i - 1, j + 1)
            
            i_star = self.overlap//2 #np.argmin(E[:,0])
            
            gamma[patch_size-1] = (i_star,0)
            for j in wps:
                gamma[j] = T[gamma[j + 1]]
            gamma[0] = T[gamma[1]]

        if overlap_type == 'L':
            
            if not len(patch_old) == 2:
                return "Smile"
            
            # compute horizontal boundaru
            patch_h = patch_old[1]
            Eh = np.zeros((patch_size,patch_size),dtype = np.int8)
            Th = np.zeros((patch_size,patch_size),dtype = tuple)
            
            # można zrobił z macierzą wagi Q jak w wyborze łatek ale na razie to odpuścimy

            eh = np.abs(patch_h + patch_n)
            eh[self.overlap//2,-1] = -100
            Eh[:-1] = eh[:-1]

            for j in wps:
                i = 0
                min_ = np.min([Eh[i, j + 1], Eh[i + 1, j + 1]])
                Eh[0,j] = eh[i,j] + min_
                Th[0,j] = (np.argwhere(Eh[i:i + 2,j + 1] == min_)[0,0] + i, j + 1)

                for i in range(1,self.overlap-1):
                    min_ = np.min([Eh[i - 1, j + 1], Eh[i , j + 1], Eh[i + 1, j + 1]])
                    Eh[i,j] = eh[i,j] + min_
                    Th[i,j] = (np.argwhere(Eh[i - 1: i + 2, j + 1] == min_)[0,0] + i - 1, j + 1)
                
                i = self.overlap-1
                min_ = np.min([Eh[i - 1, j + 1], Eh[i, j + 1]])
                Eh[self.overlap-1, j] = eh[i, j] + min_
                Th[self.overlap-1, j] = (np.where(Eh[i - 1: i + 1, j + 1] == min_)[0][0] + i - 1, j + 1)
            
            # można zrobił z macierzą wagi Q jak w wyborze łatek ale na razie to odpuścimy

            patch_v = patch_old[0]
            Ev = np.zeros((patch_size,patch_size),dtype = np.int8)
            Tv = np.zeros((patch_size,patch_size),dtype = tuple)

            #zagadaka na przyszłość: czemu do cholery podczas wykonywania np.abs(patch_v - patch_n) dostaję 255

            ev = np.abs(patch_v + patch_n)

            ev[-1,self.overlap//2] = -100
            
            Ev[-1:] = ev[-1:]

            for i in wps:
                j = 0
                min_ = np.min([Ev[i + 1, j], Ev[i + 1, j + 1]])
                Ev[i,0] = ev[i,j] + min_
                Tv[i,0] = (i+1,np.argwhere(Ev[i+1,j:j+2] == min_)[0,0]+j)

                for j in range(1,self.overlap-1):
                    min_ = np.min([Ev[i + 1, j - 1], Ev[i + 1, j], Ev[i + 1, j + 1]])
                    Ev[i,j] = ev[i,j] + min_
                    Tv[i,j] = (i+1, np.argwhere(Ev[i+1,j-1:j+2] == min_)[0,0]+j-1)

                j = self.overlap-1
                min_ = np.min([Ev[i + 1, j - 1], Ev[i + 1, j]])
                Ev[i,self.overlap-1] = ev[i,j] + min_
                Tv[i,self.overlap-1] = (i+1, np.where(Ev[i+1,j-1:j+1] == min_)[0][0]+j-1)
                e = eh +ev
            Ev = Ev.astype(longdouble)
            Eh = Eh.astype(longdouble)
            e  = e.astype(longdouble)
            i_star = np.argmin([Ev[i,i] + Eh[i,i] - e[i,i] for i in range(self.overlap)])
            
            gamma_h = [None]*(patch_size-i_star)
            gamma_v = [None]*(patch_size-i_star)
            gamma_h[0] = (i_star,i_star)
            gamma_v[-1] = (i_star,i_star)
            # plan jest taki, że od i startuje nasza lista punktów dla gamma_h i kończy się na końcu listy gammy_v
            for i in wps[i_star:]:
                gamma_v[i] = Tv[gamma_v[i + 1]]

            #wph = np.arange(patch_size-i_star,2*(patch_size-i_star),1)
            wpv = np.arange(1,patch_size-i_star,1)
            for j in wpv:
                gamma_h[j] = Th[gamma_h[j-1]]

            gamma = [gamma_v, gamma_h]

        return gamma

    def create_tile_noise(self,samples, patch_size):
        samples_color = [ ele[1] for ele in samples]
        wall_color = [ele[0] for ele in samples]
        amount_patch = 2
        # create framework for gray scale and color scale
        array_output_color = np.zeros([(amount_patch-1) * (patch_size - self.overlap) + patch_size,
                                    (amount_patch-1) * (patch_size - self.overlap) + patch_size],dtype = np.uint8)
        
        black_array = np.zeros([(amount_patch-1) * (patch_size - self.overlap) + patch_size,
                                    (amount_patch-1) * (patch_size - self.overlap) + patch_size],dtype = np.uint8)
        
        filtr = np.array([[0.25, 0.25, 0.25,0.25,0.25],
                        [0.25, 0.5, 0.5, 0.5, 0.25],
                        [0.25, 0.5, 1, 0.5, 0.25],
                        [0.25, 0.5, 0.5, 0.5, 0.25],
                        [0.25, 0.25, 0.25,0.25,0.25]])
        
        filtr = tuple(filtr.reshape(1, -1)[0])
        # fill first position
        
        array_output_color[:patch_size, :patch_size] = samples_color[0]
        # dobra n arazie po polsku
        for i in range(0, amount_patch):
            
            for j in range(0, amount_patch):
                # zacznijmy od wygenerowania dla pierwszych amount_patch poziomych( łączenie pionowe) i=0
            
                if i == 0:
            
                    if i == 0 and j == 0:
                        continue
                    patch_old = array_output_color[i*patch_size:(i+1) * patch_size, (j-1)*(patch_size-self.overlap):(j-1)*(patch_size-self.overlap)+patch_size]
                    patch_mono = self.kernel_mask(patch_old, filtr)
                    neighbour = samples_color[i + j*amount_patch]
                    neighbour_mono = self.kernel_mask(neighbour, filtr)
                    path = self.boundary(patch_old = patch_mono, patch_n = neighbour_mono,
                                patch_size = patch_size, overlap_type = 'V')
                    # tworzę maskę 
                    mask_c = np.zeros((patch_size, self.overlap))
                    mask_b = np.zeros((patch_size, self.overlap))
                    # wyodrębniam miejsca nachodzenia się sąsiadujących łatek
                    overlap_old_c = patch_old[:,-self.overlap:]
                    overlap_n_c = neighbour[:,:self.overlap]
                # uzuprłniam maskę (weight matrix)
                    for ip,jp in path:
                        mask_c[ip,:jp] = 1
                        mask_b[ip, jp] = 1
                    overlap_area_c = overlap_old_c*mask_c + overlap_n_c*(1-mask_c)

                    new_patch_c = np.zeros((patch_size,patch_size))
                    new_patch_c[:,:self.overlap] = overlap_area_c
                    new_patch_c[:,self.overlap:] = neighbour[:,self.overlap:]
                    
                    array_output_color[i*patch_size:(i+1) * patch_size, j*(patch_size-self.overlap):(j)*(patch_size-self.overlap)+patch_size] = new_patch_c
                    black_array[i*patch_size:(i+1) * patch_size, j*(patch_size-self.overlap):(j)*patch_size] = mask_b

                # zacznijmy od wygenerowania dla pierwszych amount_patch pionowych( łączenie poziome) j=0
                elif j==0 :
                    patch_old = array_output_color[(i-1)*(patch_size-self.overlap):(i-1)*(patch_size-self.overlap)+patch_size,j*patch_size:(j+1)*patch_size]
                    patch_mono = self.kernel_mask(patch_old, filtr)
                    neighbour = samples_color[i + j*amount_patch]

                    neighbour_mono = self.kernel_mask(neighbour, filtr)
            
                    path = self.boundary(patch_old = patch_mono, patch_n = neighbour_mono,
                                patch_size = patch_size, overlap_type = 'H')
                    # tworzę maskę 
                    mask_c = np.zeros((self.overlap, patch_size))
                    mask_b = np.zeros((self.overlap, patch_size))
                    # wyodrębniam miejsca nachodzenia się sąsiadujących łatek
                    
                    overlap_old_c = patch_old[-self.overlap:,:]
                    overlap_n_c = neighbour[:self.overlap,:]
                    
                # uzuprłniam maskę (weight matrix)
                    for ip,jp in path:
                        mask_c[:ip,jp] = 1
                        mask_b[ip,jp] = 1
                    # color
                    overlap_area_c = overlap_old_c*mask_c + overlap_n_c*(1-mask_c)
                    
                    new_patch_c = np.zeros((patch_size,patch_size))
                    new_patch_c[:self.overlap, :] = overlap_area_c
                    new_patch_c[self.overlap:, :] = neighbour[self.overlap:, :]
                    
                    array_output_color[i*(patch_size-self.overlap):(i)*(patch_size-self.overlap)+patch_size,j*patch_size:(j+1)*patch_size] = new_patch_c
                    black_array[i*(patch_size-self.overlap):(i)*patch_size,j*patch_size:(j+1)*patch_size] = mask_b
                else:
                    patch_old_h = array_output_color[(i) * (patch_size - self.overlap)+self.overlap-patch_size: (i) * (patch_size - self.overlap)+self.overlap, j*(patch_size-self.overlap):(j)*(patch_size-self.overlap)+patch_size]
                    patch_old_v = array_output_color[i*(patch_size-self.overlap):(i)*(patch_size-self.overlap)+patch_size, (j) * (patch_size - self.overlap)+self.overlap-patch_size: (j) * (patch_size - self.overlap)+self.overlap]
                    
                    
                    # patch_old_v oznacz lewo, patch_old_h oznacza góra

                    patch_mono_h = self.kernel_mask(patch_old_h, filtr)
                    patch_mono_v = self.kernel_mask(patch_old_v, filtr)
                    neighbour = samples_color[i + j*amount_patch]
                    neighbour_mono = self.kernel_mask(neighbour, filtr)

                    path = self.boundary(patch_old = [patch_mono_v,patch_mono_h], patch_n = neighbour_mono,
                                patch_size = patch_size, overlap_type = 'L')
                    # tworzę maski na szary i kolorowy i na łączenia horizontal and vertical
    
                    mask_c = np.zeros((patch_size,patch_size))
                    mask_b = np.zeros((patch_size,patch_size))
                    # fill mask 
                    # uzupełniam kwadrat dla narożnika
                    mask_c[:path[0][-1][0],:path[0][-1][0]] = 1
                    for ip,jp in path[0]:
                        mask_c[ip,:jp] = 1
                        mask_b[ip,jp] = 1
                    for ip,jp in path[1]:
                        mask_c[:ip,jp] = 1
                        mask_b[ip,jp] = 1
                    # cut mask into horizontal and vertical space
                    mask_c_h = mask_c[:self.overlap,:]
                    mask_c_v = mask_c[self.overlap:,:self.overlap]

                    mask_b_h = mask_b[:self.overlap,:]
                    mask_b_v = mask_b[self.overlap:,:self.overlap]

                    #color

                    overlap_old_c_h = patch_old_v[self.overlap:,-self.overlap:]
                    overlap_n_c_v = neighbour[self.overlap:,:self.overlap]

                    overlap_old_c_v = patch_old_h[-self.overlap:,:]
                    overlap_n_c_h = neighbour[:self.overlap,:]

                    # używamy maski jako matrix weight
                    
                    overlap_area_c_h = overlap_old_c_v*mask_c_h + overlap_n_c_h*(1-mask_c_h)
                    overlap_area_c_v = overlap_old_c_h*mask_c_v + overlap_n_c_v*(1-mask_c_v)
                    
                    # teraz do array_outpu_image
                    
                    new_patch_c = np.zeros((patch_size,patch_size))
                    new_patch_c[self.overlap:,:self.overlap] = overlap_area_c_v
                    new_patch_c[:self.overlap,:] = overlap_area_c_h
                    new_patch_c[self.overlap:,self.overlap:] = neighbour[self.overlap:,self.overlap:]
                    #patch_old = new_patch_c

                    array_output_color[i*(patch_size-self.overlap):(i)*(patch_size-self.overlap)+patch_size, j*(patch_size-self.overlap):(j)*(patch_size-self.overlap)+patch_size] = new_patch_c
                    
                    black_array[i*(patch_size) : (i+1)*patch_size - self.overlap, j*(patch_size - self.overlap) : j*(patch_size)] = mask_b_v
                    black_array[i*patch_size - self.overlap : (i)*patch_size, j*(patch_size - self.overlap): (j+1)*patch_size - self.overlap] = mask_b_h


        wall_color[2], wall_color[3] = wall_color[3], wall_color[2]
        rotate_tile = rotate(array_output_color, angle=-45, reshape=True)
        rotate_black = rotate(black_array, angle=-45, reshape=True)
        x,y = np.shape(rotate_tile)[:2]
# i have trouble in here
        # print(black_array.shape)
        # print(rotate_black.shape)

        if x % 4 != 0:
            return 'Odd' 
        
        x, y = x/2, y/2      
        size = int(x)
        b = int(x//2)
        aha = rotate_tile[b: b+size, b:b+size], wall_color

        return aha

# import poisson_disk as poisdisk

# p1 = poisdisk.Disk(240,240, 100)

# samples = p1.sample_noise()#(image=r'C:\Users\foszc\OneDrive\Pulpit\engineering thesis\my_image.png')


# def noise_tile(noises, typ):

#     samples = noises
    
#     R, B, G, Y = samples

#     dict_color = {}
    
#     dict_color['R'] = R
#     dict_color['B'] = B
#     dict_color['G'] = G
#     dict_color['Y'] = Y
        
#     list_tiles = []
#     colors_v = [key for key in typ[0]]
#     colors_h = [key for key in typ[1]]
#     for up in colors_v:
#         for right in colors_h:
#             for down in colors_v:
#                 for left in colors_h:

#                     dictup = [up,dict_color[up]]
#                     dictright = [right,dict_color[right]]
#                     dictdown = [down,dict_color[down]]
#                     dictleft = [left,dict_color[left]]

#                     list_tiles.append([dictup, dictright, dictleft, dictdown])
        
#     return list_tiles


# print('-'*40)


# walls  = noise_tile(samples,[['R','G'],['B','Y']])


# p2 = Noise_Tile(walls[0], 16, 100,80)

# tile = p2.create_tile_noise(samples=walls[0],patch_size=100)

# print(type(tile[0]))

# print(tile[0,0])
# noisew = PIL.Image.fromarray((tile*255).astype(np.uint8), mode='L')
# noisew.show()