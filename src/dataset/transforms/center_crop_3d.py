def center_crop_3d(img, output_size):
    """
    Center crop a 3D image to the desired output size.
    :param img: The 3D image to be cropped.
    :param output_size: The desired output size.
    :return: The cropped 3D image.
    """
    # Calculate the start indices for each dimension
    start_x = (img.shape[0] - output_size[0]) // 2
    start_y = (img.shape[1] - output_size[1]) // 2
    start_z = (img.shape[2] - output_size[2]) // 2

    # Calculate the end indices for each dimension
    end_x = start_x + output_size[0]
    end_y = start_y + output_size[1]
    end_z = start_z + output_size[2]

    # Crop the image
    cropped_img = img[start_x:end_x, start_y:end_y, start_z:end_z]

    return cropped_img
