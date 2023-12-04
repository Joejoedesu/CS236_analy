# CS236_analy
the analysis code for CS236 project

The machine learning training and inference were conducted on the Web UI platform: https://github.com/AUTOMATIC1111/stable-diffusion-webui

##image_processing.py
frequency_analysis: compute the intensity of the high-frequency signals in the image using FFT. The input is the image name of the analysis target.

compute_gradient: use the Sobel filter to detect the edge for the given image.

plot_gradient: process, analyze, and visualize the edges detected in the image.

##LoRA_analysis.py
It processes the LoRA training results in the safetensor file formate and computes the subspace similarity between the selected targets.

##Pixel_style_process.py
It converts a 512x512 image generated by the diffusion model into a 64x64 image usable as the game asset.
