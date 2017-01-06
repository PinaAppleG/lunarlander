from skimage.measure import block_reduce
import numpy as np

def rgb2gray(rgb):
    '''
    Converts RGB array to grayscale.
    '''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def preprocess_env(env):
    '''
    Renders environment and preprocesses the corresponding image by downsampling the image to a 80*120 size  from 400*800
    and converting to grayscale.
    '''

    array = env.render('rgb_array')
    env.render(close=True)
    array = rgb2gray(array)
    array = block_reduce(array, block_size=(5, 5), func=np.mean)
    return array