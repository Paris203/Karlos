import torch
from torch import nn
import torch.nn.functional as F
from networks import resnet
from config import pretrain_path, coordinates_cat, iou_threshs, window_nums_sum, ratios, N_list
import numpy as np
from utils.AOLM import AOLM
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib.pyplot as plt


# def extract_important_region(feature_maps,input_image, threshold=0.5):
#     # Step 1: Generate Feature Maps
#     #feature_maps = model.forward(input_image)  # Replace with your CNN model
    
#     # Step 2: Aggregate Feature Maps into Activation Map
#     activation_map = torch.mean(feature_maps, dim=1)  # Average pooling along channels
    
#     # Step 3: Threshold to create binary mask
#     binary_mask = activation_map > threshold
    
#     # Step 4: Find bounding box coordinates
#     mask_np = binary_mask.cpu().numpy()[0]  # Convert to NumPy and select the first batch
#     coords = np.argwhere(mask_np)
#     top_left = coords.min(axis=0)
#     bottom_right = coords.max(axis=0)
    
#     # Step 5: Crop the important region
#     cropped_image = input_image[:, :, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
#     print("cropped image",cropped_image.shape)
    
#     # Visualize cropped region
#     #plt.imshow(cropped_image[0].permute(1, 2, 0).cpu().numpy())
#     #plt.show()
    
    
#     return cropped_image

import torch
from skimage import measure  # for region properties and labeling
import matplotlib.pyplot as plt

# Assuming you have defined the AOLM function you provided

# Function to crop the most important region
def crop_using_bounding_box(input_image, coordinates):
    # Step 1: Forward pass to get feature maps
    #fms = model.forward(input_image)  # First set of feature maps
    #fm1 = fms  # You can use a different set of feature maps if needed

    # Step 2: Get bounding box coordinates using AOLM
    #coordinates = AOLM(fms, fm1)

    # Step 3: Loop through each image in the batch and crop
    cropped_images = []
    for i, (x_lefttop, y_lefttop, x_rightlow, y_rightlow) in enumerate(coordinates):
        # Convert coordinates to integer values
        x_lefttop, y_lefttop = int(x_lefttop), int(y_lefttop)
        x_rightlow, y_rightlow = int(x_rightlow), int(y_rightlow)

        # Crop the image using the coordinates
        cropped_image = input_image[i, :, x_lefttop:x_rightlow, y_lefttop:y_rightlow]

        # Append the cropped image to the list
        cropped_images.append(cropped_image)

        # Optionally visualize the cropped region
        plt.imshow(cropped_image.permute(1, 2, 0).cpu().numpy())
        plt.show()

    return cropped_images

# Example usage:
# Assuming `input_image` is your batch of images and `model` is your pre-trained CNN
# cropped_images = crop_using_bounding_box(input_image, model)




def plot_and_save_images(batch_tensor, save_dir="./images/"):
    # Ensure the batch is on the CPU
    #batch = batch_tensor.cpu() if batch_tensor.is_cuda else batch_tensor
    print(f"Batch shape: {batch_tensor}")  # Expecting shape (batch_size, channels, height, width)
    
    # Create the save directory if it doesn't exist
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = batch_tensor[0]
    print(batch_size)
    
    for i in range(6):
        image = batch[i]
        
        # Normalize the image to the range [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        # Select the first 3 channels if the image has more than 3 (for RGB)
        if image.shape[0] > 3:
            image = image[:3, :, :]
        
        # Permute the dimensions to (Height, Width, Channels)
        image = image.permute(1, 2, 0)
        
        # Plot the image using matplotlib
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image.numpy())
        plt.axis('off')  # Turn off axis for cleaner image display
        
        # Save the image to the specified path
        save_path = os.path.join(save_dir, f"image_{i+1}.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Image saved at {save_path}")
        
        # Close the figure to prevent memory issues
        plt.close(fig)

# Example usage:
# plot_and_save_images(batch_of_images, save_dir="./saved_images/")






def nms(scores_np, proposalN, iou_threshs, coordinates):
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0]
    indices_coordinates = np.concatenate((scores_np, coordinates), 1)

    indices = np.argsort(indices_coordinates[:, 0])
    indices_coordinates = np.concatenate((indices_coordinates, np.arange(0,windows_num).reshape(windows_num,1)), 1)[indices]                  #[339,6]
    indices_results = []

    res = indices_coordinates

    while res.any():
        indice_coordinates = res[-1]
        indices_results.append(indice_coordinates[5])

        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1,proposalN).astype(np.int64)
        res = res[:-1]

        # Exclude anchor boxes with selected anchor box whose iou is greater than the threshold
        start_max = np.maximum(res[:, 1:3], indice_coordinates[1:3])
        end_min = np.minimum(res[:, 3:5], indice_coordinates[3:5])
        lengths = end_min - start_max + 1
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1] + 1) * (res[:, 4] - res[:, 2] + 1) +
                                      (indice_coordinates[3] - indice_coordinates[1] + 1) *
                                      (indice_coordinates[4] - indice_coordinates[2] + 1) - intersec_map)
        res = res[iou_map_cur <= iou_threshs]

    while len(indices_results) != proposalN:
        indices_results.append(indice_coordinates[5])

    return np.array(indices_results).reshape(1, -1).astype(np.int)

class APPM(nn.Module):
    def __init__(self):
        super(APPM, self).__init__()
        self.avgpools = [nn.AvgPool2d(ratios[i], 1) for i in range(len(ratios))]

    def forward(self, proposalN, x, ratios, window_nums_sum, N_list, iou_threshs, DEVICE=device):
        batch, channels, _, _ = x.size()
        avgs = [self.avgpools[i](x) for i in range(len(ratios))]
        #print(f"avgs shape {avgs[0].shape}")

        # feature map sum
        fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(ratios))]
        #print(f"feature map shape {fm_sum[0].shape}")
  

        all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(ratios))], dim=1)
        windows_scores_np = all_scores.data.cpu().numpy()
        window_scores = torch.from_numpy(windows_scores_np).to(DEVICE).reshape(batch, -1)

        # nms
        proposalN_indices = []
        for i, scores in enumerate(windows_scores_np):
            indices_results = []
            for j in range(len(window_nums_sum)-1):
                indices_results.append(nms(scores[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])], proposalN=N_list[j], iou_threshs=iou_threshs[j],
                                           coordinates=coordinates_cat[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])]) + sum(window_nums_sum[:j+1]))
            # indices_results.reverse()
            proposalN_indices.append(np.concatenate(indices_results, 1))   # reverse

        proposalN_indices = np.array(proposalN_indices).reshape(batch, proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).to(DEVICE)
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in enumerate(all_scores)], 0).reshape(
            batch, proposalN)

        return proposalN_indices, proposalN_windows_scores, window_scores

class MainNet(nn.Module):
    def __init__(self, proposalN, num_classes, channels):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.proposalN = proposalN
        self.pretrained_model = resnet.resnet50(pretrained=True, pth_path=pretrain_path)
        #self.pretrained_model = resnet.resnet18()
        self.rawcls_net = nn.Linear(channels, num_classes)
        self.APPM = APPM()

    def forward(self, x, epoch, batch_idx, status='test', DEVICE= device):
        fm, embedding, conv5_b = self.pretrained_model(x)
        #plot_and_save_image(fm[0].detach())
        batch_size, channel_size, side_size, _ = fm.shape
        assert channel_size == 2048 # 512 change by diallo

        # raw branch
        raw_logits = self.rawcls_net(embedding)

        #SCDA
        coordinates = torch.tensor(AOLM(fm.detach(), conv5_b.detach()))
        #x_lefttop, y_lefttop, x_rightlow, y_rightlow


        local_imgs = torch.zeros([batch_size, 3, 448, 448]).to(DEVICE)  # [N, 3, 448, 448]
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(448, 448),
                                                mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
        #plot_and_save_images(local_imgs)
        local_fm, local_embeddings, _ = self.pretrained_model(local_imgs.detach())  # [N, 2048]
        image_crop = crop_using_bounding_box(local_imgs,coordinates)
        #image_crop = extract_important_region(local_fm, local_imgs, threshold=0.5)
        plot_and_save_images(image_crop)
        #plot_and_save_images(local_fms.detach())
        local_logits = self.rawcls_net(local_embeddings)  # [N, 200]

        proposalN_indices, proposalN_windows_scores, window_scores \
            = self.APPM(self.proposalN, local_fm.detach(), ratios, window_nums_sum, N_list, iou_threshs, DEVICE)

        if status == "train":
            # window_imgs cls
            window_imgs = torch.zeros([batch_size, self.proposalN, 3, 224, 224]).to(DEVICE)  # [N, 4, 3, 224, 224]
            for i in range(batch_size):
                for j in range(self.proposalN):
                    [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                    window_imgs[i:i + 1, j] = F.interpolate(local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(224, 224),
                                                                mode='bilinear',
                                                                align_corners=True)  # [N, 4, 3, 224, 224]

            window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 224, 224)  # [N*4, 3, 224, 224]
            _, window_embeddings, _ = self.pretrained_model(window_imgs.detach())  # [N*4, 2048]
            proposalN_windows_logits = self.rawcls_net(window_embeddings)  # [N* 4, 200]
        else:
            proposalN_windows_logits = torch.zeros([batch_size * self.proposalN, self.num_classes]).to(DEVICE)

        return proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, \
               window_scores, coordinates, raw_logits, local_logits, local_imgs
