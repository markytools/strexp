import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map
import copy
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from modules.guided_backprop import GuidedBackprop
import sys,os
sys.path.append(os.getcwd())

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    org_im = np.uint8(org_im.detach().to("cpu").numpy()[0][0]*255)
    org_im = Image.fromarray(org_im)
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): Full filename including directory and png
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = file_name
    # print("gradient save shape: ", gradient.shape)
    save_image(gradient, path_to_file)

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)

def module_output_to_numpy(tensor):
    return tensor.data.to('cpu').numpy()

def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im
class SaveOutput:
    def __init__(self, totalFeatMaps):
        self.layer_outputs = []
        self.grad_outputs = []
        self.first_grads = []
        self.totalFeatMaps = totalFeatMaps
        self.feature_ext = None
    ### Used on register_forward_hook
    ### Output up to totalFeatMaps
    def append_layer_out(self, module, input, output):
        self.layer_outputs.append(output[0]) ### Appending with earlier index pertaining to earlier layers
    ### Used on register_backward_hook
    ### Output up to totalFeatMaps
    def append_grad_out(self, module, grad_input, grad_output):
        self.grad_outputs.append(grad_output[0][0]) ### Appending with last-to-first index pertaining to first-to-last layers
    ### Used as guided backprop mask
    def append_first_grads(self, module, grad_in, grad_out):
        self.first_grads.append(grad_in[0])
    def clear(self):
        self.layer_outputs = []
        self.grad_outputs = []
        self.first_grads = []
    def set_feature_ext(self, feature_ext):
        self.feature_ext = feature_ext
    def getGuidedGradImg(self, layerNum, input_img):
        # print("layer outputs shape: ", self.layer_outputs[0].shape)
        # print("layer grad_outputs shape: ", self.grad_outputs[0].shape)
        conv_output_img = module_output_to_numpy(self.layer_outputs[layerNum])
        grad_output_img = module_output_to_numpy(self.grad_outputs[len(self.grad_outputs)-layerNum-1])
        first_grad_output = self.first_grads[0].data.to('cpu').numpy()[0]
        print("conv_output_img output shape: ", conv_output_img.shape)
        print("grad_output_img output shape: ", grad_output_img.shape)
        print("first_grad_output output shape: ", first_grad_output.shape)
        print("target min max: {}, {}".format(conv_output_img.min(), conv_output_img.max()))
        print("guided_gradients min max: {}, {}".format(grad_output_img.min(), grad_output_img.max()))
        weights = np.mean(grad_output_img, axis=(1, 2))  # Take averages for each gradient
        print("weights shape: ", weights.shape)
        print("weights min max1: {}, {}".format(weights.min(), weights.max()))
        # Create empty numpy array for cam
        # conv_output_img = np.clip(conv_output_img, 0, conv_output_img.max())
        cam = np.ones(conv_output_img.shape[1:], dtype=np.float32)
        print("cam min max1: {}, {}".format(cam.min(), cam.max()))
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * conv_output_img[i, :, :]
        # cam = np.maximum(cam, 0)
        print("cam min max2: {}, {}".format(cam.min(), cam.max()))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_img.shape[3],
                       input_img.shape[2]), Image.ANTIALIAS))/255
        # cam_gb = np.multiply(cam, first_grad_output)
        # grayscale_cam_gb = convert_to_grayscale(cam)
        return cam
    def getGuidedGradTimesImg(self, layerNum, input_img):
        grad_output_img = module_output_to_numpy(self.grad_outputs[len(self.grad_outputs)-layerNum-1])
        print("grad_output_img output shape: ", grad_output_img.shape)
        grad_times_image = grad_output_img[0]*input_img.detach().to("cpu").numpy()[0]
        return grad_times_image
    ### target_output -- pass a created output tensor with one hot (1s) already in placed, used for guided gradients (first layer)
    def output_feature_maps(self, targetDir, input_img):
        # GBP = GuidedBackprop(self.feature_ext, 'resnet34')
        # guided_grads = GBP.generate_gradients(input_img, one_hot_output_guided, text_for_pred)
        # print("guided_grads shape: ", guided_grads.shape)
        for layerNum in range(self.totalFeatMaps):
            grad_times_image = self.getGuidedGradTimesImg(layerNum, input_img)
            # save_gradient_images(cam_gb, targetDir + 'GGrad_Cam_Layer{}.jpg'.format(layerNum))
            # save_gradient_images(grayscale_cam_gb, targetDir + 'GGrad_Cam_Gray_Layer{}.jpg'.format(layerNum))
            ### Output heatmaps
            grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
            save_gradient_images(grayscale_vanilla_grads, targetDir + 'Vanilla_grad_times_image_gray{}.jpg'.format(layerNum))
