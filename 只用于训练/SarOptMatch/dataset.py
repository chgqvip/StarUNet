import glob
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import math
from sklearn.model_selection import train_test_split

from . import utils


@utils.measure_time
def sen1_2(data_path, seed, ims_per_folder = 100):
    """
    creates a subset of the SEN1-2 dataset.
    Takes ims_per_folder images from each folder in the original dataset.
    Returns three arrays: sar images, optical images, and ground truth offsets.
    """
    
    np.random.seed(seed)
    data = glob.glob(data_path + '/*')
    sar = []
    opt = []
    for season in range(len(data)):
        sar_paths = glob.glob(data[season] + '/s1_*')
        opt_paths = glob.glob(data[season] + '/s2_*')
        assert len(sar_paths) == len(opt_paths)
        for i in range(len(sar_paths)):
            sar_folder = np.array(glob.glob(sar_paths[i]+'/*'))
            opt_folder = np.array(glob.glob(opt_paths[i]+'/*'))
            indices = np.random.choice(len(sar_folder), size = ims_per_folder, replace = False)
            sar.append([np.array(k) for k in sar_folder[indices]])
            opt.append([np.array(k) for k in opt_folder[indices]])
    sar = np.concatenate([i for i in sar])
    opt = np.concatenate([i for i in opt])
    offsets = tf.random.uniform([len(sar),2], minval = 0, maxval = 65, dtype = tf.int32, seed = 10) # operation seed for deterministic results
    return sar, opt, offsets # (sar,opt,ground truth offset)


def lee(im):
    """
    Tensorflow implementation of the Lee filter for speckle noise reduction in SAR imagery
    """    
    im_mean = tfa.image.mean_filter2d(image = im, filter_shape = (3,3))
    im_sqr_mean = tfa.image.mean_filter2d(image = im**2, filter_shape = (3,3))
    im_var = im_sqr_mean - im_mean

    overall_var = tf.image.total_variation(im)

    im_weights = im_var / (im_var + overall_var)
    im_out = im_mean + im_weights * (im - im_mean)
    return im_out



def gaussian_blur(threshold: tf.float32,offsets):
    """
    Creates a soft ground truth label (65x65 heatmap) from the ground truth offset coordinate.
    The heatmap is normalized so that all elements sum to approximately 1.

    Args:
        threshold: Float, the radius within which the Gaussian is non-zero.
        offsets: tf.Tensor of shape [2], the (dy, dx) integer coordinates of the center (0-64).

    Returns:
        A tf.Tensor of shape [65, 65] representing the normalized heatmap.
    """
    # Ensure offsets is float32 for calculations
    offsets_float = tf.cast(offsets, dtype=tf.float32)

    # Create a grid of indices (y, x) for the 65x65 grid
    y_indices, x_indices = tf.meshgrid(tf.range(65, dtype=tf.float32),
                                       tf.range(65, dtype=tf.float32),
                                       indexing='ij') # ij indexing: first dim is y, second is x
    # Stack indices to get shape [65, 65, 2]
    indices = tf.stack([y_indices, x_indices], axis=-1)

    # Calculate squared Euclidean distances from the center (offsets) to all grid points
    # Broadcasting offsets_float [2] with indices [65, 65, 2]
    squared_dists = tf.reduce_sum(tf.square(indices - offsets_float), axis=-1) # Shape [65, 65]
    dists = tf.sqrt(squared_dists) # Shape [65, 65]

    # Calculate Gaussian values (unnormalized PDF for sigma=1)
    # Using 1.0 for sigma. The constant 1/(2*pi*sigma^2) can be ignored as we normalize later.
    gaussian_values = tf.exp(-squared_dists / 2.0) # Shape [65, 65]

    # Apply thresholding: set values outside the threshold radius to zero
    mask = tf.where(dists < threshold, gaussian_values, tf.zeros_like(gaussian_values)) # Shape [65, 65]

    # --- Correct Global Normalization (Sum to 1 over the whole grid) ---
    mask_sum = tf.reduce_sum(mask)

    # Avoid division by zero if the sum is zero (e.g., threshold is too small or offset is invalid)
    # Add epsilon to the denominator for numerical stability
    normalized_mask = mask / (mask_sum + tf.keras.backend.epsilon())

    return normalized_mask # Shape [65, 65], sum of all elements is approx 1.

# --- How to use it in _parse_files_grayscale/_parse_files_RGB ---
# Replace the old call:
# mask = gaussian_blur(tf.constant(2, dtype = tf.float32), offsets)
# With the new call:
# mask = gaussian_blur_corrected(tf.constant(2.0, dtype=tf.float32), offsets) # Use float threshold



@tf.autograph.experimental.do_not_convert
def _parse_files_grayscale(opt, sar, offsets):

    opt_string = tf.io.read_file(opt)
    opt_decoded = tf.image.decode_png(opt_string, channels=3)
    opt_rescaled = tf.image.convert_image_dtype(opt_decoded, dtype = tf.float32)

    opt_grayscale = tf.image.rgb_to_grayscale(opt_rescaled)

    sar_string = tf.io.read_file(sar)
    sar_decoded = tf.image.decode_png(sar_string, channels = 1)
    sar_rescaled = tf.image.convert_image_dtype(sar_decoded, dtype = tf.float32)
    sar_filtered = lee(sar_rescaled)
    sar_cropped = tf.image.crop_to_bounding_box(sar_filtered, offsets[0], offsets[1], 192,192)
    mask = gaussian_blur(tf.constant(2, dtype = tf.float32), offsets)
    return (opt_grayscale,sar_cropped), mask




def _parse_files_RGB(opt, sar, offsets):
    opt_string = tf.io.read_file(opt)
    opt_decoded = tf.image.decode_png(opt_string, channels=3)
    opt_rescaled = tf.image.convert_image_dtype(opt_decoded, dtype = tf.float32)
    
    opt_grayscale = opt_rescaled
    
    sar_string = tf.io.read_file(sar)
    sar_decoded = tf.image.decode_png(sar_string, channels = 1)
    sar_rescaled = tf.image.convert_image_dtype(sar_decoded, dtype = tf.float32)
    sar_filtered = lee(sar_rescaled)
    sar_cropped = tf.image.crop_to_bounding_box(sar_filtered, offsets[0], offsets[1], 192,192)
    mask = gaussian_blur(tf.constant(2, dtype = tf.float32), offsets)
    return (opt_grayscale,sar_cropped), mask



def files_to_dataset(filenames_opt, filenames_sar, offsets, masking_strategy, batch_size, convert_to_grayscale):
    assert len(filenames_opt) == len(filenames_sar)
    filenames_opt = tf.convert_to_tensor(filenames_opt)
    filenames_sar = tf.convert_to_tensor(filenames_sar)
    offsets = tf.convert_to_tensor(offsets)
    dataset = tf.data.Dataset.from_tensor_slices((filenames_opt, filenames_sar, offsets))
    
    if convert_to_grayscale:
        dataset = dataset.map(_parse_files_grayscale)
    else:
        dataset = dataset.map(_parse_files_RGB)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


@utils.measure_time
def split_data(sar_files, opt_files, offsets, batch_size = 4, seed = 42, masking_strategy = "",):
    """
    Split data in training and validation sets.
    These sets are tf.data.Dataset objects.
    """
    
    off_x, off_y = tf.transpose(offsets, perm=[1,0]).numpy()
    sar_train, sar_validation, opt_train, opt_validation, offsets_x_train, offsets_x_val, offsets_y_train, offsets_y_val = \
        train_test_split(sar_files, opt_files, off_x, off_y, test_size=0.3, random_state=seed)
    
    #Offset
    offsets_train = np.stack((offsets_x_train, offsets_y_train)).transpose(1,0)
    offsets_val = np.stack((offsets_x_val, offsets_y_val)).transpose(1,0)



    # #Images: fetch images from filenames
    # training_data = files_to_dataset(opt_train, sar_train, offsets_train, masking_strategy, batch_size, convert_to_grayscale = False)
    # validation_data = files_to_dataset(opt_validation, sar_validation, offsets_val, masking_strategy, batch_size, convert_to_grayscale = False)
    # validation_data_RGB = files_to_dataset(opt_validation, sar_validation, offsets_val, masking_strategy, batch_size, convert_to_grayscale = False)
    # return training_data, validation_data, validation_data_RGB #sar_validation, opt_validation, sar_train, opt_train
    #Images: fetch images from filenames
    training_data = files_to_dataset(opt_train, sar_train, offsets_train, masking_strategy, batch_size, convert_to_grayscale = True)
    validation_data = files_to_dataset(opt_validation, sar_validation, offsets_val, masking_strategy, batch_size, convert_to_grayscale = True)
    validation_data_RGB = files_to_dataset(opt_validation, sar_validation, offsets_val, masking_strategy, batch_size, convert_to_grayscale = False)
    return training_data, validation_data, validation_data_RGB #sar_validation, opt_validation, sar_train, opt_train

      
                
                
   
                