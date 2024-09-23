import torch
from skimage import measure

import os
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt
import torch

import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_and_save_image(image_tensor, save_path="saved_image.png"):
    image_tensor = image_tensor[0]
    print(f"image_tensor shape: {image_tensor.shape}")
    
    # Extract the directory from the save_path
    directory = os.path.dirname(save_path)
    
    # If the directory does not exist, create it
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    
    # Ensure the tensor is on the CPU and normalized (if necessary)
    image = image_tensor.cpu() if image_tensor.is_cuda else image_tensor
    print(f"image shape {image.shape}")
    
    # Normalize the image to the range [0, 1] (optional)
    image = (image - image.min()) / (image.max() - image.min())
    print(f"image shape after normalize {image.shape}")
    
    # Select the first 3 channels if the image has more than 3 (for RGB)
    #if image.shape[0] > 3:
    image = image[:3, :, :]
    
    # Permute the dimensions to (Height, Width, Channels)
    image = image.permute(1, 2, 0)
    print(f"image shape after permuted : {image.shape}")
    
    # Plot the image using matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))  # Create the figure and axis
    ax.imshow(image.numpy())  # Convert tensor to NumPy and display the image
    plt.axis('off')  # Turn off axis for cleaner image display

    # Save the image to the specified path
    plt.savefig(save_path, bbox_inches='tight')  # Save with tight bounding box
    print(f"Image saved at {save_path}")
    
    # Optionally show the image as well
    plt.show()

    # Close the figure to prevent memory issues
    plt.close(fig)




def AOLM(fms, fm1):
    #print(f"fms shape: {fms.shape}, fm1 shape :{fm1.shape}")
    plot_and_save_image(fms, save_path="saved_image.png")
    A = torch.sum(fms, dim=1, keepdim=True)
    #print("A shape", A.shape)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    #print("a shape",a.shape)
    M = (A > a).float()
    #print("M shape", M.shape)

    A1 = torch.sum(fm1, dim=1, keepdim=True)
    #print("A1 shape", A1.shape)
    a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
    #print("a1 shape", a1.shape)
    M1 = (A1 > a1).float()
    #print("M1 shape",M1.shape)


    coordinates = []
    lam_1, lam_2 = [], []
    for i, (m, fms_2) in enumerate(zip(M, fms)):
        #print(f"m shape inside the loop :{m.shape}")
        mask_np = m.cpu().numpy().reshape(14, 14)
        print(f"shape of the fms_2 :{fms_2.shape}")
        plot_and_save_image(fms_2.detach(), save_path="saved_image.png")
        #fms_2 = fms_2.cpu().numpy().reshape(14, 14)
       
        component_labels = measure.label(mask_np)
        #print(f"component_labels shape {component_labels.shape}")

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))
        intersection = ((component_labels==(max_idx+1)).astype(int) + (M1[i][0].cpu().numpy()==1).astype(int)) ==2
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 14, 14]
            print('there is one img no intersection')
        else:
            bbox = prop[0].bbox
        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1

        #print(f"x_lefttop: {x_lefttop}, y_lefttop: {y_lefttop}, x_rightlow: {x_rightlow}, y_rightlow: {y_rightlow}")

        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        
        coordinates.append(coordinate)
    return coordinates

