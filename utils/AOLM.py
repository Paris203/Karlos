import torch
from skimage import measure


def AOLM(fms, fm1):
    #print(f"fms shape: {fms.shape}, fm1 shape :{fm1.shape}")
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
    for i, m in enumerate(M):
        #print(f"m shape inside the loop :{m.shape}")
        mask_np = m.cpu().numpy().reshape(14, 14)
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

