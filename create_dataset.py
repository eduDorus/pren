# Load path/class_id image file:
dataset_folder = 'pren_dataset'

# Build a HDF5 dataset (only required once)
from tflearn.data_utils import build_hdf5_image_dataset
build_hdf5_image_dataset(dataset_folder, image_shape=(128, 128), mode='folder', output_path='dataset.h5', categorical_labels=True, normalize=True)
