#AI Usage Statement: AI was used to help debug and increase efficiency, as well as the plot functions

from termcolor import colored
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import cv2
import numpy as np
import pandas as pd
import time


start_time = time.time()

filenames = [r"C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-3-Fibrosis\images\MASK_Sk658 Llobe ch010021.jpg",
             r"C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-3-Fibrosis\images\MASK_Sk658 Llobe ch010030.jpg",
             r"C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-3-Fibrosis\images\MASK_Sk658 Llobe ch010051.jpg",
             r"C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-3-Fibrosis\images\MASK_Sk658 Llobe ch010024.jpg",
             r"C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-3-Fibrosis\images\MASK_Sk658 Slobe ch010158.jpg",
             r"C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-3-Fibrosis\images\MASK_Sk658 Llobe ch010067.jpg",
             r"C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-3-Fibrosis\images\MASK_Sk658 Llobe ch010164.jpg",
             r"C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-3-Fibrosis\images\MASK_Sk658 Llobe ch010174.jpg",
             r"C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-3-Fibrosis\images\MASK_Sk658 Slobe ch010134.jpg",
             ]

# the depth in microns that each image was taken at
depths = [30, 200, 400, 600, 920, 1500, 2200, 3100, 4500]

# empty list to store the white pixel percentages as we calculate them
white_percents = []

print(colored("Counts of pixel color in each image", "yellow"))

# loop through each image and its corresponding depth at the same time
for filename, depth in zip(filenames, depths):
    # read the image in grayscale (0 flag = grayscale)
    img = cv2.imread(filename, 0)

    # convert grayscale to pure black and white using a threshold of 127
    # anything above 127 becomes 255 (white), anything below becomes 0 (black)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # count the number of white pixels (value = 255)
    white = np.count_nonzero(binary == 255)
    # .size gives total number of pixels (rows * cols), NOT just rows
    total = binary.size
    # everything that isnt white must be black
    black = total - white

    # calculate what percent of the image is white
    white_percent = 100 * white / total
    white_percents.append(white_percent)

    # print the results for this image
    print(colored(f"\n{filename}:", "red"))
    print(colored(f"White pixels: {white}", "white"))
    print(colored(f"Black pixels: {black}", "grey"))
    print(f"{white_percent:.2f}% White | Depth: {depth} microns")

# store everything in a dataframe so we can save it as a csv
df = pd.DataFrame({
    'Filenames': filenames,
    'Depths': depths,
    'White percents': white_percents
})

# save the dataframe to a csv file
df.to_csv('Percent_White_Pixels.csv', index=False)
print("\nThe .csv file 'Percent_White_Pixels.csv' has been created.")

# stop the timer and print how long everything took
end_time = time.time()
print(f"\nExecution time: {end_time - start_time:.4f} seconds")

# Interpolation 

x = depths
y = white_percents

# create two interpolation functions using scipy - one quadratic, one linear
# these let us estimate white % at depths we didnt actually image
i_quadratic = interp1d(x, y, kind='quadratic')
i_linear    = interp1d(x, y, kind='linear')

# these are the three depths we want to predict - they match our ground truth images
query_depths = [60, 500, 2600]
query_labels = ['Depth 60', 'Depth 500', 'Depth 2600']

# run each "query" depth through both interpolators
quad_points   = [float(i_quadratic(d)) for d in query_depths]
linear_points = [float(i_linear(d))    for d in query_depths]

# print a nicely formatted table comparing the two methods
print(colored("\n--- Interpolation Results ---", "yellow"))
print(f"{'Label':<15} {'Depth':>8}  {'Quadratic':>12}  {'Linear':>12}")
print("-" * 52)
for lbl, d, q, l in zip(query_labels, query_depths, quad_points, linear_points):
    print(colored(f"{lbl:<15} {d:>8.1f}  {q:>12.4f}  {l:>12.4f}", "green"))

# Plots are below - they show the original data points, the interpolation curves, and the interpolated points for visual comparison

# generate 500 evenly spaced depth values across our range for smooth curve plotting
depth_smooth = np.linspace(min(x), max(x), 500)
curve_quad   = i_quadratic(depth_smooth)
curve_linear = i_linear(depth_smooth)

# colors for the interpolated points on the plots
point_colors = ['red', 'orange', 'cyan']

# 3 subplots stacked vertically
fig, axs = plt.subplots(3, 1, figsize=(9, 14))

# Plot 1: just the raw data, no interpolation ---
axs[0].scatter(x, y, marker='o', color='blue', zorder=5, label='Data')
axs[0].set_title('Depth vs. Percentage White Pixels (Raw Data)')
axs[0].set_xlabel('Depth (microns)')
axs[0].set_ylabel('White pixels (% of total)')
axs[0].grid(True)
axs[0].legend()

# Plot 2: quadratic curve + the interpolated points highlighted ---
axs[1].plot(depth_smooth, curve_quad, color='blue', linewidth=1.5, label='Quadratic fit')
axs[1].scatter(x, y, marker='o', color='blue', zorder=5)
for d, p, lbl, col in zip(query_depths, quad_points, query_labels, point_colors):
    axs[1].scatter(d, p, color=col, s=100, zorder=6, label=f'Quad – {lbl}: {p:.2f}%')
axs[1].set_title('Quadratic Interpolation with Interpolated Points')
axs[1].set_xlabel('Depth (microns)')
axs[1].set_ylabel('White pixels (% of total)')
axs[1].grid(True)
axs[1].legend(fontsize=8)

# --- Plot 3: linear curve + the interpolated points highlighted ---
axs[2].plot(depth_smooth, curve_linear, color='purple', linewidth=1.5, label='Linear fit')
axs[2].scatter(x, y, marker='o', color='purple', zorder=5)
for d, p, lbl, col in zip(query_depths, linear_points, query_labels, point_colors):
    axs[2].scatter(d, p, color=col, s=100, zorder=6, label=f'Linear – {lbl}: {p:.2f}%')
axs[2].set_title('Linear Interpolation with Interpolated Points')
axs[2].set_xlabel('Depth (microns)')
axs[2].set_ylabel('White pixels (% of total)')
axs[2].grid(True)
axs[2].legend(fontsize=8)

plt.tight_layout()
plt.show()

# Verification of interpolation accuracy using true images
# these are the actual images taken at the depths we interpolated above
# we use them to check how accurate our interpolations were

after_filenames = [r"C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-3-Fibrosis\images\MASK_Sk658 Llobe ch010019.jpg",
                   r"C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-3-Fibrosis\images\MASK_Sk658 Llobe ch010032.jpg",
                   r"C:\Users\Jmarc\Desktop\Comp BME\module-2-jackmarchesi\Module-3-Fibrosis\images\MASK_SK658 Slobe ch010143.jpg"]

after_depths = [60, 500, 2600]
after_white_percents = []

# Same pixel counting process as before, just for the after images
for filename, depth in zip(after_filenames, after_depths):
    img = cv2.imread(filename, 0)

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    white = np.count_nonzero(binary == 255)
    total = binary.size
    black = total - white

    white_percent = 100 * white / total
    after_white_percents.append(white_percent)

    print(colored(f"\n{filename}:", "red"))
    print(colored(f"White pixels: {white}", "white"))
    print(colored(f"Black pixels: {black}", "grey"))
    print(f"{white_percent:.2f}% White | Depth: {depth} microns")

# Percent Error calculations
# percent error formula: |predicted - actual| / actual * 100
# this tells us how far off each interpolation method was from the real image

print(colored("\n--- Percent Error vs. Ground Truth ---", "yellow"))
print(f"{'Label':<15} {'Depth':>8}  {'Actual':>10}  {'Quad':>10}  {'Quad Err%':>10}  {'Linear':>10}  {'Lin Err%':>10}")
print("-" * 80)

quad_errors   = []
linear_errors = []

for lbl, d, actual, q, l in zip(query_labels, after_depths, after_white_percents, quad_points, linear_points):
    quad_err   = abs(q - actual) / actual * 100
    linear_err = abs(l - actual) / actual * 100
    quad_errors.append(quad_err)
    linear_errors.append(linear_err)
    print(colored(
        f"{lbl:<15} {d:>8}  {actual:>10.4f}  {q:>10.4f}  {quad_err:>9.2f}%  {l:>10.4f}  {linear_err:>9.2f}%",
        "green"))

# print the average error for each method so we can compare them overall
print("-" * 80)
print(colored(
    f"{'Mean Error':<15} {'':>8}  {'':>10}  {'':>10}  {np.mean(quad_errors):>9.2f}%  {'':>10}  {np.mean(linear_errors):>9.2f}%",
    "cyan"))