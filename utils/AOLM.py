from skimage import measure
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


# def AOLM(fms, fm1):
#     # Plot the feature map for visualization if needed
#     plot_and_save_image(fms, save_path="saved_image.png")
    
#     A = torch.sum(fms, dim=1, keepdim=True)
#     a = torch.mean(A, dim=[2, 3], keepdim=True)
#     M = (A > a).float()

#     A1 = torch.sum(fm1, dim=1, keepdim=True)
#     a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
#     M1 = (A1 > a1).float()

#     coordinates = []
#     for i, (m, fms_2) in enumerate(zip(M, fms)):
#         # Check the shape of fms_2 before reshaping
#         print(f"Original fms_2 shape: {fms_2.shape}")
        
#         mask_np = m.cpu().numpy().reshape(14, 14)

#         # Reduce fms_2 over the channel dimension if needed
#         # Taking the mean of the channels to reduce it to (14, 14)
#         fms_2_reduced = torch.mean(fms_2, dim=0)  # Now fms_2_reduced should be (14, 14)
#         print(f"Reduced fms_2 shape: {fms_2_reduced.shape}")
        
#         fms_2_np = fms_2_reduced.cpu().numpy()  # Convert to NumPy array for further processing

#         component_labels = measure.label(mask_np)

#         properties = measure.regionprops(component_labels)
#         areas = [prop.area for prop in properties]

#         if len(areas) == 0:
#             print("No regions found in component_labels.")
#             continue  # Skip if no areas found
        
#         max_idx = areas.index(max(areas))
#         intersection = ((component_labels == (max_idx + 1)).astype(int) + (M1[i][0].cpu().numpy() == 1).astype(int)) == 2
#         prop = measure.regionprops(intersection.astype(int))

#         if len(prop) == 0:
#             bbox = [0, 0, 14, 14]
#             print('No intersection found')
#         else:
#             bbox = prop[0].bbox
        
#         x_lefttop = bbox[0] * 32 - 1
#         y_lefttop = bbox[1] * 32 - 1
#         x_rightlow = bbox[2] * 32 - 1
#         y_rightlow = bbox[3] * 32 - 1

#         # Ensure the coordinates are valid
#         if x_lefttop < 0:
#             x_lefttop = 0
#         if y_lefttop < 0:
#             y_lefttop = 0
        
#         coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]

#         # Correct the `lam` calculation by using the reduced (14, 14) feature map
#         lam = torch.sum(fms_2_reduced / ((x_rightlow - x_lefttop) * (y_rightlow - y_lefttop)))
#         print(f"lam: {lam}")
#         coordinates.append(coordinate)

#     return coordinates

def AOLM(fms, fm1):
    # Plot the feature map for visualization if needed
    #plot_and_save_image(fms, save_path="saved_image.png")
    
    A = torch.sum(fms, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    M = (A > a).float()

    A1 = torch.sum(fm1, dim=1, keepdim=True)
    a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
    M1 = (A1 > a1).float()

    coordinates = []
    lamda = []
    for i, (m, fms_2) in enumerate(zip(M, fms)):
        # Check the shape of fms_2 before reshaping
        #print(f"Original fms_2 shape: {fms_2.shape}")
        
        mask_np = m.cpu().numpy().reshape(14, 14)

        # Reduce fms_2 over the channel dimension to get (14, 14)
        fms_2_reduced = torch.mean(fms_2, dim=0)  # Now fms_2_reduced is (14, 14)
        #print(f"Reduced fms_2 shape: {fms_2_reduced.shape}")

        fms_2_np = fms_2_reduced.cpu().numpy()  # Convert to NumPy array for further processing

        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = [prop.area for prop in properties]

        if len(areas) == 0:
            print("No regions found in component_labels.")
            continue  # Skip if no areas found

        max_idx = areas.index(max(areas))
        intersection = ((component_labels == (max_idx + 1)).astype(int) + (M1[i][0].cpu().numpy() == 1).astype(int)) == 2
        prop = measure.regionprops(intersection.astype(int))

        if len(prop) == 0:
            bbox = [0, 0, 14, 14]
            print('No intersection found')
        else:
            bbox = prop[0].bbox

        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1

        # Ensure the coordinates are valid
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0

        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]

        # Correct the `lam` calculation by using the reduced (14, 14) feature map
        # Normalize the values to be between 0 and 1
        fms_2_reduced_flat = fms_2_reduced.view(-1)
        fms_2_reduced_min = torch.min(fms_2_reduced_flat)
        fms_2_reduced_max = torch.max(fms_2_reduced_flat)
        fms_2_reduced_norm = (fms_2_reduced_flat - fms_2_reduced_min) / (fms_2_reduced_max - fms_2_reduced_min + 1e-6)

        # Compute lam as the mean of the normalized feature map
        lam = torch.mean(fms_2_reduced_norm).item()

        # Scale lam to be between 0.1 and 0.9
        lam = 0.8 * lam + 0.1
        lam = max(0.1, min(lam, 0.9))  # Ensure lam is within [0.1, 0.9]

        #print(f"lam: {lam}")
        coordinates.append(coordinate)
        #lamda.append(lam)

    return coordinates, lam



