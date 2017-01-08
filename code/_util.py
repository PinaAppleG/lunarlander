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
    array = rgb2gray(array)
    array = block_reduce(array, block_size=(5, 5), func=np.mean)
    return array

def phi(states):
    '''
    Function from algorithm 1 applies this preprocessing to the last 4 frames of 
    a history and stacks them to produce the input to the Q-function
    '''
    phi_t = []
    for k in range(0,4):
        try:
            phi_t.append(states[-1-2*k])
        except IndexError:
            # If not enough frames, add initial fram
            phi_t.append(states[0])
    return np.array(phi_t)