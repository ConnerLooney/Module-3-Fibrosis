#AI Usage Statement: AI was used to help debug and increase efficiency

from termcolor import colored
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import cv2
import numpy as np
import pandas as pd
import time

start_time = time.time()





filenames = [
    r"/Users/connerlooney/Documents/GitHub/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010021.jpg",
    r"/Users/connerlooney/Documents/GitHub/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010021.jpg",
    r"/Users/connerlooney/Documents/GitHub/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010021.jpg",
    r"/Users/connerlooney/Documents/GitHub/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010021.jpg",
    r"/Users/connerlooney/Documents/GitHub/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010021.jpg",
    r"/Users/connerlooney/Documents/GitHub/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010021.jpg",
    r"/Users/connerlooney/Documents/GitHub/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010021.jpg",
    r"/Users/connerlooney/Documents/GitHub/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010021.jpg",
    r"/Users/connerlooney/Documents/GitHub/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010021.jpg",
    r"/Users/connerlooney/Documents/GitHub/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010021.jpg",
    r"/Users/connerlooney/Documents/GitHub/Module-3-Fibrosis/images/MASK_Sk658 Llobe ch010021.jpg",
]


depths = [30, 200, 330, 400, 600, 750, 920, 1500, 2200, 3100, 4500]

white_percents = []

print(colored("Counts of pixel color in each image", "yellow"))

for filename, depth in zip(filenames, depths):
    img = cv2.imread(filename, 0)

    if img is None:
        print(f"Error loading image: {filename}")
        continue

    # Threshold
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Count pixels efficiently
    white = np.count_nonzero(binary == 255)
    total = binary.size
    black = total - white

    # Compute percent
    white_percent = 100 * white / total
    white_percents.append(white_percent)

    # Print (same style as before)
    print(colored(f"\n{filename}:", "red"))
    print(colored(f"White pixels: {white}", "white"))
    print(colored(f"Black pixels: {black}", "grey"))
    print(f"{white_percent:.2f}% White | Depth: {depth} microns")

# Create DataFrame (UNCHANGED STRUCTURE)
df = pd.DataFrame({
    'Filenames': filenames,
    'Depths': depths,
    'White percents': white_percents
})

# Save CSV (same as before)
df.to_csv('Percent_White_Pixels.csv', index=False)

print("\nThe .csv file 'Percent_White_Pixels.csv' has been created.")


end_time = time.time()
print(f"\nExecution time: {end_time - start_time:.4f} seconds")



# LECTURE 2: UNCOMMENT BELOW

# # Interpolate a point: given a depth, find the corresponding white pixel percentage

interpolate_depth = float(input(colored(
     "Enter the depth at which you want to interpolate a point (in microns): ", "yellow")))

x = depths
y = white_percents

 # You can also use 'quadratic', 'cubic', etc.
i = interp1d(x, y, kind='quadratic')
interpolate_point = i(interpolate_depth)
print(colored(
    f'The interpolated point is at the x-coordinate {interpolate_depth} and y-coordinate {interpolate_point}.', "green"))

depths_i = depths[:]
depths_i.append(interpolate_depth)
white_percents_i = white_percents[:]
white_percents_i.append(interpolate_point)


# # make two plots: one that doesn't contain the interpolated point, just the data calculated from your images, and one that also contains the interpolated point (shown in red)
fig, axs = plt.subplots(2, 1)

axs[0].scatter(depths, white_percents, marker='o', linestyle='-', color='blue')
axs[0].set_title('Plot of depth of image vs percentage white pixels')
axs[0].set_xlabel('depth of image (in microns)')
axs[0].set_ylabel('white pixels as a percentage of total pixels')
axs[0].grid(True)


axs[1].scatter(depths_i, white_percents_i, marker='o',
               linestyle='-', color='blue')
axs[1].set_title(
    'Plot of depth of image vs percentage white pixels with interpolated point (in red)')
axs[1].set_xlabel('depth of image (in microns)')
axs[1].set_ylabel('white pixels as a percentage of total pixels')
axs[1].grid(True)
axs[1].scatter(depths_i[len(depths_i)-1], white_percents_i[len(white_percents_i)-1],
               color='red', s=100, label='Highlighted point')


# # Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
