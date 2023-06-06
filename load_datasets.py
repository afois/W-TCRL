import gzip
import numpy as np
from scipy.io import loadmat
import PIL.Image
if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
    PIL.Image.Resampling = PIL.Image

def resize_images(images,
                  new_size,
                  dtype=np.float32):

    if type(new_size) is tuple:
        new_shape = (images.shape[0], *new_size)
        # array of resized images
        res_imgs = np.empty(new_shape, dtype=dtype)
        for idx, img in enumerate(images):
            im = PIL.Image.fromarray(img)
            im_res = im.resize(new_size, PIL.Image.Resampling.LANCZOS)
            res_imgs[idx] = np.asarray(im_res)

    return res_imgs

def load_images(filename):
    with gzip.open(filename) as f:
        a = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
        return a


def load_labels(filename):
    with gzip.open(filename) as f:
        a = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return a

# code from http://jamesgregson.ca/extract-image-patches-in-python.html
def extract_grayscale_patches(img, shape, offset=(0,0), stride=(1,1)):
    """Extracts (typically) overlapping regular patches from a grayscale image

    Changing the offset and stride parameters will result in images
    reconstructed by reconstruct_from_grayscale_patches having different
    dimensions! Callers should pad and unpad as necessary!

    Args:
        img (HxW ndarray): input image from which to extract patches

        shape (2-element arraylike): shape of that patches as (h,w)

        offset (2-element arraylike): offset of the initial point as (y,x)

        stride (2-element arraylike): vertical and horizontal strides

    Returns:
        patches (ndarray): output image patches as (N,shape[0],shape[1]) array
    """
    px, py = np.meshgrid( np.arange(shape[1]),np.arange(shape[0]))
    l, t = np.meshgrid(
        np.arange(offset[1],img.shape[1]-shape[1]+1,stride[1]),
        np.arange(offset[0],img.shape[0]-shape[0]+1,stride[0]) )
    l = l.ravel()
    t = t.ravel()
    x = np.tile( px[None,:,:], (t.size,1,1)) + np.tile( l[:,None,None], (1,shape[0],shape[1]))
    y = np.tile( py[None,:,:], (t.size,1,1)) + np.tile( t[:,None,None], (1,shape[0],shape[1]))

    return img[y.ravel(),x.ravel()].reshape((t.size,shape[0],shape[1]))

def getReconstructedImage(patches, patch_size, indices, original_img_size):
    offset_y = patch_size[0]
    offset_x = patch_size[1]

    reconstr_img = np.zeros(original_img_size)
    for i, idx in enumerate(indices):
        y = idx[0]
        x = idx[1]

        reconstr_img[y: y + offset_y, x: x + offset_x] = patches[i]

    return reconstr_img


def load_MNIST(seed=0,
               nb_training_img=15000,
               nb_testing_img=1000,
               nb_samples_train=120000,
               nb_samples_test=10000,
               patch_size=(5, 5),
               path_train_img='datasets/MNIST/train-images-idx3-ubyte.gz',
               path_test_img='datasets/MNIST/t10k-images-idx3-ubyte.gz',
               new_im_size=None):

    np.random.seed(seed)

    #### MNIST DATASET ####
    original_size = (28, 28)
    nb_pixels = np.prod(patch_size)

    # load all train images
    all_train_images = load_images(path_train_img)
    all_train_images = all_train_images.reshape(
        all_train_images.shape[0], *original_size)

    if type(new_im_size) is tuple:
        all_train_images = (resize_images(all_train_images, new_im_size, np.float32) / 255).astype(np.float16)

    else:
        all_train_images = (all_train_images / 255).astype(np.float16)

    # select random subset train images
    indices_train_img = np.random.choice(all_train_images.shape[0], nb_training_img, replace=False)
    train_images = all_train_images[indices_train_img]

    # load all test images
    all_test_images = load_images(path_test_img)
    all_test_images = all_test_images.reshape(
        all_test_images.shape[0], *original_size)

    if type(new_im_size) is tuple:
        all_test_images = (resize_images(all_test_images, new_im_size, np.float32) / 255).astype(np.float16)

    else:
        all_test_images = (all_test_images / 255).astype(np.float16)

    # select random subset test images
    indices_test_img = np.random.choice(all_test_images.shape[0], nb_testing_img, replace=False)
    test_images = all_test_images[indices_test_img]

    ### EXTRACT TRAIN PATCHES ###
    # extract all train patches
    all_train_patches = np.array([extract_grayscale_patches(img, shape=patch_size, stride=patch_size) for img in train_images], dtype = np.float16)
    shape = all_train_patches.shape
    all_train_patches = all_train_patches.reshape(shape[0] * shape[1], *shape[2:])
    # select randomly a given nb of samples of train patches
    idx_train_p = np.random.choice(all_train_patches.shape[0], nb_samples_train, replace=True)
    train_patches = all_train_patches[idx_train_p]

    ### EXTRACT TEST PATCHES ###
    all_test_patches = np.array(
        [extract_grayscale_patches(img, shape=patch_size, stride=patch_size) for img in test_images], dtype = np.float16)
    shape = all_test_patches.shape
    all_test_patches = all_test_patches.reshape(shape[0] * shape[1], *shape[2:])

    if nb_samples_test is None:
        test_patches = all_test_patches
    else:
        idx_test_p = np.random.choice(all_test_patches.shape[0], nb_samples_test, replace=False)
        test_patches = all_train_patches[idx_test_p]

    train_patches = train_patches.reshape(train_patches.shape[0], nb_pixels)
    test_patches = test_patches.reshape(test_patches.shape[0], nb_pixels)

    dataset = {"train": train_patches,
               "test": test_patches}

    return dataset


def load_natural_images(seed=0,
                        nb_training_img=10,
                        nb_testing_img=10,
                        nb_samples_train=120000,
                        nb_samples_test = 10000,
                        patch_size=(8, 8),
                        path_images='datasets/natural_images/IMAGES_RAW.mat'):

    np.random.seed(seed)

    #### NATURAL IMAGES DATASET ####
    idx_start_test_img = 0
    original_size = (512, 512)
    nb_pixels = np.prod(patch_size)

    all_images = loadmat(path_images, variable_names='IMAGESr', appendmat=True).get('IMAGESr')
    all_images = all_images.transpose()

    data_min = np.min(all_images)
    data_max = np.max(all_images)

    # normalize data
    all_images = (all_images - data_min) / (data_max - data_min)

    ### EXTRACT TRAIN PATCHES ###
    # extract all train patches
    if isinstance(nb_training_img, int):
        all_train_patches = np.array([extract_grayscale_patches(img, shape = patch_size, stride=patch_size) for img in all_images[0:nb_training_img]], dtype = np.float16)

    if isinstance(nb_training_img, list) or isinstance(nb_training_img, np.ndarray):
        all_train_patches = np.array([extract_grayscale_patches(img, shape = patch_size, stride=patch_size) for img in all_images[nb_training_img]], dtype = np.float16)

    shape = all_train_patches.shape
    all_train_patches = all_train_patches.reshape(shape[0] * shape[1], *shape[2:])
    # select randomly a given nb of samples of train patches
    idx_train_p = np.random.choice(all_train_patches.shape[0], nb_samples_train, replace=True)
    train_patches = all_train_patches[idx_train_p]

    ### EXTRACT TEST PATCHES ###
    if isinstance(nb_testing_img, int) :
        all_test_patches = np.array([extract_grayscale_patches(img, shape = patch_size, stride = patch_size) for img in all_images[0:nb_testing_img]], dtype = np.float16)

    if isinstance(nb_training_img, list) or isinstance(nb_training_img, np.ndarray):
        all_test_patches = np.array([extract_grayscale_patches(img, shape = patch_size, stride = patch_size) for img in all_images[nb_testing_img]], dtype = np.float16)

    shape = all_test_patches.shape
    all_test_patches = all_test_patches.reshape(shape[0] * shape[1], *shape[2:])

    if nb_samples_test is None :
        test_patches = all_test_patches
    else :
        idx_test_p = np.random.choice(all_test_patches.shape[0], nb_samples_test, replace=False)
        test_patches = all_train_patches[idx_test_p]

    train_patches = train_patches.reshape(train_patches.shape[0], nb_pixels)
    test_patches = test_patches.reshape(test_patches.shape[0], nb_pixels)

    dataset = {"train": train_patches,
               "test": test_patches}

    return dataset