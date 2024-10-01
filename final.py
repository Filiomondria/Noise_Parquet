import PIL
from PIL import ImageFilter
import numpy as np
from scipy.stats import lognorm, bernoulli, poisson, gamma, expon
import random
from colorsys import rgb_to_hls, hls_to_rgb 

import poisson_disk as poisdisk
import wang_tile as wt
import create_noise_tile as cnt


class Inhomogeneous_Noise:

    def __init__(self, image,tile_size = 100, amount = 10, typy = [['R','G'],['Y','B']], size_pixel=1):
        self.image_in = PIL.Image.open(image)
        self.matrix_input = np.copy(np.asarray(self.image_in.convert('RGB')))
        self.height = self.matrix_input.shape[0]
        self.width = self.matrix_input.shape[1]
        self.amount = amount
        self.typy = typy
        # maybe later, after that we calculate size of sample
        self.tile_size = tile_size
        self.pixel = size_pixel

    def noise_tile(self, noises):

        samples = noises
        
        R, B, G, Y = samples

        dict_color = {}
        
        dict_color['R'] = R
        dict_color['B'] = B
        dict_color['G'] = G
        dict_color['Y'] = Y
            
        list_tiles = []
        colors_v = [key for key in self.typy[0]]
        colors_h = [key for key in self.typy[1]]
        for up in colors_v:
            for right in colors_h:
                for down in colors_v:
                    for left in colors_h:

                        dictup = [up,dict_color[up]]
                        dictright = [right,dict_color[right]]
                        dictdown = [down,dict_color[down]]
                        dictleft = [left,dict_color[left]]

                        list_tiles.append([dictup, dictright, dictleft, dictdown])
            
        return list_tiles


    def homogeneous_noise(self):
        n,m,z = self.matrix_input.shape
        # amount of tiles in rows and columns
        x = int(np.ceil(n/self.tile_size))
        y = int(np.ceil(m/self.tile_size))
        
        # here we can estimate size of sample
        #sample_size = self.tile_size/np.sqrt(2) + ...

        p1 = poisdisk.Disk(240,240, 100)

        samples = p1.sample_noise()

        all_walls  = self.noise_tile(samples)
        walls = random.sample(all_walls, k = self.amount)

        #create chosen tiles
        tiles = []
        for wall in walls:
            p2 = cnt.Noise_Tile(walls[0], 16, 100,80)
            tile = p2.create_tile_noise(samples=wall,patch_size=100)
            tiles.append(tile)

        parquet = wt.Wang_Tile(x,y,tiles)
        
        canvas = parquet.create_canvas()

        return canvas
    
    def hsv_to_hsl(self, hsv):
        """
        Convert HSV (Hue, Saturation, Value) to HSL (Hue, Saturation, Lightness).
        Input and output ranges are in [0, 1].
        """
        hsv = hsv/255
        h, s, v = hsv

        l = (2 - s) * v / 2

        if l == 0 or l == 1:
            s = 0
        else:
            s = (v-l)/min(l,1-l)

        return [h, np.round(s,2), np.round(l,2)]
    
    def hsl_to_hsv(self,hsl):
        """
        Convert HSL (Hue, Saturation, Lightness) to HSV (Hue, Saturation, Value).
        Input and output ranges are in [0, 1].
        """
        h, s, l = hsl

        v = l + s * min(l, 1 - l)

        if v == 0:
            s = 0
        else:
            s = 2 * (1 - l / v)

        return h, np.round(s,2)*255, np.round(v,2)*255
    
    def hsltorgb(self,hsl):
        h, s, l = hsl

        h /= 360.0 

        r, g, b = hls_to_rgb(h, l, s)

        r *= 255.0
        g *= 255.0
        b *= 255.0

        return np.round(r), np.round(g), np.round(b)
    
    def rgbtohsl(self,rgb):
        r, g, b = rgb
        r /= 255.0
        g /= 255.0
        b /= 255.0

        h, l, s = rgb_to_hls(r, g, b)

        h *= 360.0 
        return np.round(h,0), np.round(s,2), np.round(l,2)
    
    def accept_reject(self, arg, mode, x):
        if mode == 'lognorm':
            # sigma, nie sigma^2
            lognorm_dist = lognorm(arg[1], scale=arg[0])
            cdf_value = lognorm_dist.cdf(x)
            threshold = 0.95
            if cdf_value > threshold:
                return 1
            else:
                return 0
            
    def inhomogeneous_noise(self,light, typ = None):

        if not light<=1 and light>=0:
            p = bernoulli.cdf(1, light)
        else:
            p = light

        if typ == 'poisson':
            light *= 20
            p = poisson.cdf(light,10)

        if typ == 'lognorm':
            light *= 2
            p = lognorm.cdf(light, s= 0.25)

        if typ == 'gamma':
            light *= 10
            p = gamma.cdf(light, a = 3.25)

        if typ == 'expon':
            light *= 5
            p = expon.cdf(light)

        if typ == 'tanh':
            light *= np.pi
            p = np.tanh(light)
        
        wynik = 1 if np.random.random() < p else 0
        
        if wynik == 1:
            return False
        else:
            return True

    def overlap_noise(self, noise):

        blurred_image = self.image_in.filter(ImageFilter.GaussianBlur(radius=10))
        matrix_blurr = np.round(np.asarray(blurred_image.convert('RGB')),2)

        matrix_out = np.copy(self.matrix_input)
        indices = np.where(noise == 1)
        coordinates = list(zip(indices[0], indices[1]))

        for c in coordinates:
            c_blurr = matrix_blurr[c]
            hsl_blurr = list(self.rgbtohsl(c_blurr))

            if self.inhomogeneous_noise(hsl_blurr[2], 'expon'):
                if self.pixel == 1:
                    c_rgb = self.matrix_input[c]
                    hsl = list(self.rgbtohsl(c_rgb))
                    hsl[2] = (hsl[2] + 1)/2
                    rgb = self.hsltorgb(hsl)
                    matrix_out[c] = rgb
                # else:
                #     if c - self
                #     c_rgb = self.matrix_input[c-self.pixel : c + self.pixel + 1,c-self.pixel : c + self.pixel + 1]
                #     hsl = list(self.rgbtohsl(c_rgb))
                #     c - self.pixel

            else:
                pass



        return matrix_out


    def final(self, title):
        noise_parquet = self.homogeneous_noise()
        noise = noise_parquet[:self.height,:self.width]
        print(self.width,self.height)
        matrix_output = self.overlap_noise( noise)

        image_output = PIL.Image.fromarray(matrix_output.astype(np.uint8), mode='RGB')
        image_output.save(title +'.png')

        noisew = PIL.Image.fromarray((noise*255).astype(np.uint8), mode='L')
        noisew.save(title +'_noise.png')




p = Inhomogeneous_Noise(r'my_image.png')

p.final('Check')

