"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

from PIL import Image

def resize_image(a, new_size):
    image = Image.fromarray(a)
    image = image.resize(new_size, Image.ANTIALIAS)
    return np.asarray(image)

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        new_shape = (volume.shape[0], self.patch_size, self.patch_size)
        print(f'volume.shape={volume.shape}, new_shape={new_shape}')
        reshaped_volume = np.zeros(new_shape)
        reshaped_volume[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]] = volume    
        return self.single_volume_inference(reshaped_volume)
    

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # <YOUR CODE HERE>
        
        max_volume = np.max(volume)
        min_volume = np.min(volume)
        print(f'max_volume={max_volume}, min_volume={min_volume}')
        volume = (volume - min_volume) / (max_volume - min_volume)
        slices = np.zeros(volume.shape)
        for slice_index in range(volume.shape[0]):
            s = volume[slice_index,:,:]
            s = s.astype(np.single)
            #s = s / 255.0
            #s = s / np.max(s)
            s = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
            s = s.to(self.device)
            prediction = self.model(s)
            prediction = np.squeeze(prediction.cpu().detach())
            prediction = torch.argmax(prediction, dim = 0)
            slices[slice_index,:,:] = prediction
            print(f'slice_index={slice_index + 1}')
        return slices
