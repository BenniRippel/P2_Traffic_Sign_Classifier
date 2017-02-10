# Load pickled data
import pickle
from sklearn.utils import shuffle
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import random

# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


### Preprocess the data here.
### Feel free to use as many code cells as needed.


# grayscale ? klassengroessen gleich? equal hist? ....

# convert to grayscale and normalize

def grayscale_and_normalize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray=cv2.equalizeHist(gray)
    ret = (gray-np.mean(gray)) / np.std(gray)
    return ret.reshape((image.shape[0], image.shape[1], 1))

def normalize_and_center(image):
    return (image/255.0)-0.5

def convert_data(dataset):
    ret = np.zeros((dataset.shape[0], 32, 32, 3))
    for i in range(dataset.shape[0]): 
        ret[i] = normalize_and_center(dataset[i])
    return ret

X_train = convert_data(X_train)
X_test = convert_data(X_test)

### Generate data additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.


#get additional data:
# rotate, shift, skew, change contrast

def shift_img(x_shift, y_shift):
    '''Calculate matrix to shift image by x_shift and y_shift pixels. Use matrix with cv2.warpAffine'''
    return np.float32([[1,0,x_shift],[0,1,y_shift]])

def rotate_img(img, deg):
    '''Calculate matrix to rotate image by deg degrees. Use matrix with cv2.warpAffine'''
    center = (img.shape[0]/2, img.shape[1]/2)
    return cv2.getRotationMatrix2D(center, deg, 1)

def shear_img(img, shear):
    '''Calculates a matrix to shear an image, based on a random variation of a given set of points, 
    with a maximum distance of 'shear from the original point, for every element of the new point. 
    Use matrix with cv2.warpAffine'''
    ori_pts = np.float32([[int(img.shape[0]*0.5) ,int(img.shape[1]*0.3) ],
                          [int(img.shape[0]*0.3) ,int(img.shape[1]*0.7) ],  
                          [int(img.shape[0]*0.7) ,int(img.shape[1]*0.3) ]])
    add = np.array(np.random.uniform(-1.0, 1.0, (3,2))*shear*1.0).astype(int)
    shear_matrix = cv2.getAffineTransform(ori_pts, np.add(ori_pts, add).astype('float32'))
    return shear_matrix

def contrast_img(img, gain, bias):
    '''Multiply grayscale image with gain and add bias for contrast change'''
    return np.add(np.multiply(img, 1.0+gain), bias)

def generate_images(img):
    '''Apply several affine transformations with random parameters as well as a contrast shift, to generate data'''
    # define parameters
    max_shear = 2
    max_rot = 25
    max_shift = 5
    max_cont_gain = 0.1
    max_cont_bias = 0.1
    # apply transformations
    # change contrast
    out = contrast_img(img, np.random.uniform(-1.0, 1.0)*max_cont_gain, np.random.uniform(-1.0, 1.0)*max_cont_bias)
    
    # get matrices for affine transformation
    m_shear = shear_img(img, np.random.uniform()*max_shear)
    m_rot = rotate_img(img, np.random.uniform(-1.0, 1.0)*max_rot)
    m_shift = shift_img(np.random.uniform(-1.0, 1.0)*max_shift, np.random.uniform(-1.0, 1.0)*max_shift)
    # combine matrices
    comb_matrix = combineTransformationMatrices(m_shear, m_rot, m_shift)
    #apply warpAffine
    out = cv2.warpAffine(img, comb_matrix, (32, 32))
    return out.reshape((1,32,32,3))

def combineTransformationMatrices(*args):
    # vstack [0, 0, 1] for matrix multiplication
    matrices = [np.vstack([args[i], np.array([0, 0, 1])]) for i in range(len(args))]
    ret = args[0]
    for idx in range(1, len(args)):
        ret = np.matmul(ret, matrices[idx])
    
    return ret[:2, :3]
    
def augment_data(featureset, labelset, iterations, verbose=False):
    if verbose:
        print('Augmenting {0!s} images with {1!s} additional images...'.format(featureset.shape[0], iterations))
    img_indices = range(featureset.shape[0])
    for img_idx in img_indices:
        if verbose and img_idx % 500 ==0:
            print('Processing number: ', img_idx)
        for iters in range(iterations):
            featureset=np.vstack([featureset, generate_images(featureset[img_idx])])
            labelset=np.hstack([labelset, labelset[img_idx]])
    if verbose:
        print('.. Done!')
    return featureset, labelset


# split X_train in training and validation sets
X_train, y_train = shuffle(X_train, y_train)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# generate additional images for training data
X_aug, y_aug = augment_data(X_train, y_train, iterations=5, verbose=True)

data = {'X_train':X_aug, 'y_train':y_aug, 'X_validation':X_validation, 'y_validation':y_validation}

# save as pickle
with open('AugmentedData.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
