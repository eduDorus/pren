# Load path/class_id image file:
data_folder = 'data/pren_dataset_large'

# Build a HDF5 dataset (only required once)
from tflearn.data_utils import build_hdf5_image_dataset
build_hdf5_image_dataset(data_folder, image_shape=(128, 128), mode='folder', output_path='data/dataset_large.h5', categorical_labels=True, normalize=True, grayscale=True)