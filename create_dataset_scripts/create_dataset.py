# Load path/class_id image file:
data_folder = '../data/final_dataset'

# Build a HDF5 dataset (only required once)
from tflearn.data_utils import build_hdf5_image_dataset
build_hdf5_image_dataset(data_folder, image_shape=(480, 640), mode='folder', output_path='../data/final_dataset.h5', categorical_labels=True, normalize=True)
