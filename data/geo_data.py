import numpy as np
import random

import skimage
from skimage.draw import polygon
from PIL import Image, ImageDraw



def generate_triangle_image(dim=64):
    img = np.zeros((dim, dim))

    # Randomize size and location
    base_length = np.random.randint(dim // 8, dim / 2)
    height = int((np.sqrt(3) / 2) * base_length)  # Height of an equilateral triangle

    # Random position for the top vertex
    top_vertex_x = np.random.randint(0 + base_length // 2, dim - base_length // 2)
    top_vertex_y = np.random.randint(0, dim - height)

    # Calculate other two vertices based on top vertex
    bottom_left_vertex = (top_vertex_x - base_length // 2, top_vertex_y + height)
    bottom_right_vertex = (top_vertex_x + base_length // 2, top_vertex_y + height)

    # Draw the triangle using polygon
    rr, cc = polygon([top_vertex_y, bottom_left_vertex[1], bottom_right_vertex[1]],
                     [top_vertex_x, bottom_left_vertex[0], bottom_right_vertex[0]],
                     shape=img.shape)
    img[rr, cc] = 255

    return img


def generate_pentagon_image(dim=64):
    img = np.zeros((dim, dim))

    # Define a function to get the pentagon vertices
    def get_pentagon_vertices(center_x, center_y, radius):
        vertices = []
        for i in range(5):
            # Note: The angle offset (np.pi / 10) is to ensure the flat side is at the bottom
            angle = 2 * np.pi * i / 5 - np.pi / 10
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            vertices.append((y, x))  # Note: switched the order to (row, col)
        return vertices

    # Randomize size and location
    radius = np.random.randint(dim // 8, dim // 3)
    center_x = np.random.randint(radius, dim - radius)
    center_y = np.random.randint(radius, dim - radius)

    vertices = get_pentagon_vertices(center_x, center_y, radius)
    rows, cols = zip(*vertices)  # Unzip to separate row coordinates and column coordinates

    # Fill the pentagon
    rr, cc = skimage.draw.polygon(rows, cols)
    img[rr, cc] = 255

    return img


def generate_triangle_image_with_noise(dim=64, noise_max=32):
    img = generate_triangle_image(dim)
    if noise_max == 0:
        return img
    noise_level = random.randint(8, noise_max)
    # Add lattice noise
    # Random shift for the entire noise lattice
    shift_x = random.randint(0, noise_level)
    shift_y = random.randint(0, noise_level)

    # Add lattice noise with the shift applied to the entire pattern
    for x in range(shift_x, dim, noise_level):
        for y in range(shift_y, dim, noise_level):
            if x < dim and y < dim:
                img[x, y] = 255
    return img


def generate_number_8_image_with_noise(image_size=64, min_radius=5, max_radius=10, noise_max=32):

    image = Image.new("L", (image_size, image_size), 0)  # "L" mode for grayscale, 0 for black background

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Randomly determine the radius
    radius = random.randint(min_radius, max_radius)

    # Calculate the maximum y-coordinate for the first circle
    max_y1 = (image_size - 4 * radius)

    # Ensure that y1 is within the bounds
    y1 = random.randint(radius, max_y1)

    # Position the second circle directly below the first one, making them tangent
    y2 = y1 + 2 * radius

    # The x-coordinate is the same for both circles
    x = random.randint(radius, image_size - radius)

    # Draw the two circles
    draw.ellipse([(x - radius, y1 - radius), (x + radius, y1 + radius)], fill=255)  # White circle
    draw.ellipse([(x - radius, y2 - radius), (x + radius, y2 + radius)], fill=255)  # White circle

    if noise_max == 0:
        return np.array(image)

    noise_level = random.randint(8, noise_max)

    shift_x = random.randint(0, noise_level)
    shift_y = random.randint(0, noise_level)
    # Add lattice noise with random shift
    # Apply shifted lattice noise
    for x in range(shift_x, image_size, noise_level):
        for y in range(shift_y, image_size, noise_level):
            if image.getpixel((x, y)) == 0:
                image.putpixel((x, y), 255)
    return np.array(image)


def rotate_image(img, angle):
    if angle == 0:
        return img
    else:
        return np.rot90(img)


def generate_triangle_image_with_noise2(dim=64, noise_max=32):
    
    img = generate_triangle_image(dim)
    if noise_max == 0:
        return img
    
    noise_level = random.randint(8, noise_max)
    # Add lattice noise
    # Random shift for the entire noise lattice
    shift_x = random.randint(0, noise_level)
    shift_y = random.randint(0, noise_level)

    # Add lattice noise with the shift applied to the entire pattern
    for x in range(shift_x, dim, noise_level):
        for y in range(shift_y, dim, noise_level):
            if x < dim and y < dim - 2:
                img[x, y] = 255
                img[x, y + 1] = 255
                img[x, y + 2] = 255
    return img


def generate_number_8_image_with_noise2(image_size=64, min_radius=5, max_radius=10, noise_max=32):
    


    image = Image.new("L", (image_size, image_size), 0)  # "L" mode for grayscale, 0 for black background

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Randomly determine the radius
    radius = random.randint(min_radius, max_radius)

    # Calculate the maximum y-coordinate for the first circle to ensure the entire "8" fits within the image
    max_y1 = (image_size - 4 * radius)

    # Ensure that y1 is within the bounds
    y1 = random.randint(radius, max_y1)

    # Position the second circle directly below the first one, making them tangent
    y2 = y1 + 2 * radius

    # The x-coordinate is the same for both circles
    x = random.randint(radius, image_size - radius)

    # Draw the two circles
    draw.ellipse([(x - radius, y1 - radius), (x + radius, y1 + radius)], fill=255)  # White circle
    draw.ellipse([(x - radius, y2 - radius), (x + radius, y2 + radius)], fill=255)  # White circle
    
    if noise_max == 0:
        return np.array(image)
    
    noise_level = random.randint(8, noise_max)

    shift_x = random.randint(0, noise_level)
    shift_y = random.randint(0, noise_level)
    # Add lattice noise with random shift
    # Apply shifted lattice noise
    for x in range(shift_x, image_size - 2, noise_level):
        for y in range(shift_y, image_size, noise_level):
            if image.getpixel((x, y)) == 0:
                image.putpixel((x, y), 255)
            if image.getpixel((x + 1, y)) == 0:
                image.putpixel((x + 1, y), 255)
            if image.getpixel((x + 2, y)) == 0:
                image.putpixel((x + 2, y), 255)
    return np.array(image)


def generate_data(n = 20000,dim=64 ,noise_level=32, pair = 0, noise_type = 0):
    X_data = []
    y_data = []
    y_shape = []
    
    for i in range(n):
        a = np.random.rand()
        if a > 0.5:
            shape = 0
            if noise_type == 0:
                img = generate_triangle_image_with_noise(dim=dim ,noise_max=noise_level)
            elif noise_type == 1:
                img = generate_triangle_image_with_noise2(dim=dim, noise_max=noise_level)
            else:
                img = generate_triangle_image_with_noise(dim=dim ,noise_max=0)
        else:
            shape =1
            if pair == 0:
                if noise_type == 0:
                    img = generate_number_8_image_with_noise(image_size=dim ,noise_max=noise_level)
                elif noise_type == 1:
                    img = generate_number_8_image_with_noise2(image_size=dim, noise_max=noise_level)
                else:
                    img = generate_number_8_image_with_noise(image_size=dim ,noise_max=0)
            else:
                img = generate_pentagon_image(dim= dim)
            


        angle = 0 if np.random.rand() > 0.5 else 90
        rotated_img = rotate_image(img, angle)

        X_data.append(rotated_img)
        y_data.append(angle // 90)
        y_shape.append(shape)
        # 0 for 0°, 1 for 90°

    X_data = np.array(X_data).reshape(-1, 1, dim, dim)
    y_data = np.array(y_data)
    y_shape = np.array(y_shape)

    return X_data ,y_data ,y_shape


# Since the labels of the rotated images are known as well, we include rotated images in lableled data

def generate_labeled_data(n ,dim=64 ,noise_level=32,pair = 0, noise_type=0):

    X_data = []
    y_data = []
    y_shape = []

    for i in range(n):
        a = np.random.rand()
        if a > 0.5:
            shape = 0
            if noise_type == 0:
                img = generate_triangle_image_with_noise(dim=dim, noise_max=noise_level)
            elif noise_type == 1:
                img = generate_triangle_image_with_noise2(dim=dim, noise_max=noise_level)
        else:
            shape = 1
            if pair == 0:
                if noise_type == 0:
                    img = generate_number_8_image_with_noise(image_size=dim, noise_max=noise_level)
                elif noise_type == 1:
                    img = generate_number_8_image_with_noise2(image_size=dim, noise_max=noise_level)
            else:
                img = generate_pentagon_image(dim = dim)
            

        angle = 0 if np.random.rand() > 0.5 else 90
        rotated_img = rotate_image(img, angle)

        X_data.append(rotated_img)
        y_data.append(angle // 90)
        y_shape.append(shape)

        X_data.append(img)
        y_data.append(0)
        y_shape.append(shape)
        # 0 for 0°, 1 for 90°

    X_data = np.array(X_data).reshape(-1, 1, dim, dim)
    y_data = np.array(y_data)
    y_shape = np.array(y_shape)

    return X_data ,y_data ,y_shape