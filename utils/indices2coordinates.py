import numpy as np


def ComputeCoordinate(image_size, stride, indice, ratio):
    size = int(image_size / stride)
    column_window_num = (size - ratio[1]) + 1
    x_indice = indice // column_window_num
    y_indice = indice % column_window_num
    x_lefttop = int(x_indice * stride - 1)  # Convert to integer
    y_lefttop = int(y_indice * stride - 1)  # Convert to integer
    x_rightlow = int(x_lefttop + ratio[0] * stride)  # Convert to integer
    y_rightlow = int(y_lefttop + ratio[1] * stride)  # Convert to integer
    #print(f"x_lefttop: {x_lefttop}, y_lefttop: {y_lefttop}, x_rightlow: {x_rightlow}, y_rightlow: {y_rightlow}")

    # for image
    if x_lefttop < 0:
        x_lefttop = 0
    if y_lefttop < 0:
        y_lefttop = 0

    coordinate = np.array([x_lefttop, y_lefttop, x_rightlow, y_rightlow]).reshape(1, 4)  # Use list notation here

    return coordinate



def indices2coordinates(indices, stride, image_size, ratio):
    batch, _ = indices.shape
    coordinates = []

    for j, indice in enumerate(indices):
        coordinates.append(ComputeCoordinate(image_size, stride, indice, ratio))

    coordinates = np.array(coordinates).reshape(batch,4).astype(int)       # [N, 4]
    return coordinates

