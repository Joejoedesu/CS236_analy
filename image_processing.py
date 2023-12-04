import cv2
import numpy as np
import matplotlib.pyplot as plt

def frequency_analysis(n):
    # Load the image
    original_image = cv2.imread(f'..\\Lora_experiment\\{n}.png', cv2.IMREAD_GRAYSCALE)

    # Perform Fourier Transform
    f_transform = np.fft.fft2(original_image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)

    sub_mag = magnitude_spectrum[0:256, 0:256]
    sub_sub_mag = sub_mag[128:256, 128:256]

    # Display the original and magnitude spectrum images
    # plt.subplot(121), plt.imshow(original_image, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(121), plt.imshow(np.log(1 + sub_mag), cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    #sum of sub_mag
    sum_mag = np.sum(sub_mag) - np.sum(sub_sub_mag)
    print(n, " ", sum_mag/(256*256-128*128))

    plt.show()

def compute_color_gradients(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to float32 for gradient calculation
    image_float = image.astype(np.float32) / 255.0

    # Compute the gradients for each color channel
    gradient_x = cv2.Sobel(image_float, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image_float, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of the gradients
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    # print(gradient_magnitude.shape)
    # print(gradient_magnitude[0][0])
    return gradient_magnitude

def plot_gradients(image):
    image_path = f'..\\Lora_experiment\\{image}.png'
    gradient_magnitude = compute_color_gradients(image_path)

    # create a np arrary the same shape as gradient_magnitude
    track = np.zeros((gradient_magnitude.shape[0], gradient_magnitude.shape[1]))
    s = 0
    # print(track.shape)
    for i in range(gradient_magnitude.shape[0]):
        for j in range(gradient_magnitude.shape[1]):
            if gradient_magnitude[i][j][0] + gradient_magnitude[i][j][1] + gradient_magnitude[i][j][2] >= 0.09:
                # print(gradient_magnitude[i][j][0] + gradient_magnitude[i][j][1] + gradient_magnitude[i][j][2])
                track[i][j] = 1
                s += min(min(i%8, 8 - i%8), min(j%8, 8 - j%8))

    print(image, " ", s/(256*256))
    # sum = np.sum(track)
    # print(image, " ", sum)

    # Display the original image and the gradient magnitude
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(122)
    plt.imshow(gradient_magnitude, cmap='gray')
    # plt.title('Color Gradient Magnitude')
    plt.colorbar()

    plt.show()


plot_gradients('sample')
# sample = ['v1', 'v2', 'con1', 'con2', 'chill1', 'chill2']
# for s in sample:
#     plot_gradients(s)
# frequency_analysis('con_back')


# name = ['apple_con_pixel', 'apple_chill_pixel', 'apple_v15_pixel', 'apple_con_pure', 'apple_chill_pure', 'apple_v15_pure']
# for n in name:
#     frequency_analysis(n)

# apple_con_pixel   3198.908552923458
# apple_chill_pixel   6833.618990467027
# apple_v15_pixel   8006.933639675619
# apple_con_pure   1942.862489584301
# apple_chill_pure   1543.8009915243845
# apple_v15_pure   1596.1488964223827