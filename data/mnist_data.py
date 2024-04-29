import numpy as np
import random

import torchvision


def generate_image_with_noise(img, dim=28, noise_max=10):
    noise_level = random.randint(3, noise_max)
    img_noise = img
    # Add lattice noise
    # Random shift for the entire noise lattice
    shift_x = random.randint(0, noise_level)
    shift_y = random.randint(0, noise_level)

    # Add lattice noise with the shift applied to the entire pattern
    for x in range(shift_x, dim, noise_level):
        for y in range(shift_y, dim, noise_level):
            if x < dim and y < dim:
                img_noise[0, x, y] = 1
    return img_noise

def generate_image_with_noise2(img, dim = 28,noise_max=10):
    noise_level = random.randint(8, noise_max)
    img_noise = img
      # Add lattice noise
    # Random shift for the entire noise lattice
    shift_x = random.randint(0, noise_level)
    shift_y = random.randint(0, noise_level)

    # Add lattice noise with the shift applied to the entire pattern
    for x in range(shift_x, dim, noise_level):
        for y in range(shift_y, dim, noise_level):
            if x < dim and y < dim-1:
                img_noise[0, x, y] = 1
                img_noise[0, x, y+1] = 1
    return img_noise


def generate_rotated_mnist_samples(dataset, num_samples, dim=28, noise_max=6, noise_type=0):
    X_data = []
    y_data = []
    y_label = []
    
    rotation_angles=[0, 90]

    if num_samples > 10000:
        selected_indices = np.arange(0, num_samples)
    else:
        selected_indices = np.arange(45000, 45000 + num_samples)
    # selected_indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in selected_indices:
        image, label = dataset[idx]
        if noise_max != 0:
            if noise_type==0:
                image = generate_image_with_noise(image, dim=dim, noise_max=noise_max)
            elif noise_type==1:
                image = generate_image_with_noise2(image, dim=dim, noise_max=noise_max)
        # Apply random rotation
        rotation_angle = random.choice(rotation_angles)
        rotated_image = torchvision.transforms.functional.rotate(image, rotation_angle)
        # Resize or preprocess the rotated MNIST image to your desired dimensions (e.g., 64x64)
        # rotated_image = preprocess_rotated_mnist_image(rotated_image)
        X_data.append(rotated_image)
        y_data.append(rotation_angle // 90)
        y_label.append(label)
        
    
    X_data_np = [tensor.numpy() for tensor in X_data] 
    X_data_combined = np.stack(X_data_np)
    
    y_data = np.array(y_data)
    y_label = np.array(y_label)

    return X_data_combined, y_data, y_label


def generate_labled_rotated_mnist_samples(dataset, num_samples, dim=28, noise_max=6,noise_type=0):
    X_data = []
    y_label = []

    if num_samples > 10000:
        selected_indices = np.arange(0, num_samples)
    else:
        selected_indices = np.arange(45000, 45000 + num_samples)

    for idx in selected_indices:
        image, label = dataset[idx]
        if noise_max != 0:
            if noise_type == 0:
                image = generate_image_with_noise(image, dim=dim, noise_max=noise_max)
            elif noise_type == 1:
                image = generate_image_with_noise2(image, dim=dim, noise_max=noise_max)
        # Apply random rotation
        rotated_image = torchvision.transforms.functional.rotate(image, 90)
        # Resize or preprocess the rotated MNIST image to your desired dimensions (e.g., 64x64)
        X_data.append(rotated_image)
        y_label.append(label)

        X_data.append(image)
        y_label.append(label)

    X_data_np = [tensor.numpy() for tensor in X_data] 
    X_data_combined = np.stack(X_data_np)
    y_label = np.array(y_label)

    return X_data_combined, y_label
