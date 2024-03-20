# We will create a function that applies a high-pass filter to preserve high frequency components
# and attenuate low frequency components. This can be achieved by creating a mask that is 1 in the
# areas we want to keep and smoothly transitions to a lower value (e.g., 0.5) in the areas we want
# to compress.

from matplotlib import pyplot as plt
import numpy as np
import cv2

def get_fft_spectrum(image_channel):
    # image_channel should be a single channel image
        # Find the Fourier transform using numpy
    f_transform = np.fft.fft2(image_rgb[:, :, i])
    # Shift the zero frequency component to the center
    f_shift = np.fft.fftshift(f_transform)
    # Get the magnitude spectrum (log for better visualization)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    return f_shift, magnitude_spectrum

def high_pass_filter(image_channel, cutoff_frequency_ratio=0.1, filter_strength=0.5):
    """
    Apply a high-pass filter to an image channel to preserve high frequency components
    and attenuate low frequency components.

    Parameters:
    - image_channel: single color channel of an image
    - cutoff_frequency_ratio: the ratio of the distance from the center to start applying the filter
    - filter_strength: the strength to which low frequency components are reduced

    Returns:
    - filtered_image_channel: the filtered image channel
    """
    # Find the Fourier transform of the image
    f_transform = np.fft.fft2(image_channel)
    f_shift = np.fft.fftshift(f_transform)

    # Get the dimensions of the image
    rows, cols = image_channel.shape
    crow, ccol = rows//2, cols//2  # Center of the image

    # Create a mask with high values at the edges (high frequencies) and low value in the center (low frequencies)
    mask = np.ones((rows, cols), np.float32)
    center_square = int(crow * cutoff_frequency_ratio), int(ccol * cutoff_frequency_ratio)
    mask[crow - center_square[0]:crow + center_square[0], ccol - center_square[1]:ccol + center_square[1]] = filter_strength

    # Apply the mask to the shifted Fourier transform
    f_shift_filtered = f_shift * mask

    # Inverse shift and inverse Fourier transform to get the filtered image back in the spatial domain
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back


if __name__ == "__main__":

    # Apply the high-pass filter to each channel
    image_filtered_rgb = np.zeros_like(image_rgb)
    for i in range(3):  # Loop over the RGB channels
        image_filtered_rgb[:, :, i] = high_pass_filter(image_rgb[:, :, i])

    # Plot the original image and the filtered image
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Filtered image
    axes[1].imshow(np.clip(image_filtered_rgb, 0, 255).astype(np.uint8))
    axes[1].set_title('Filtered Image')
    axes[1].axis('off')

    # Show the plots
    plt.tight_layout()
    plt.show()
