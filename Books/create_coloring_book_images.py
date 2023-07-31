import os
import numpy as np
import skimage
from PIL import Image
from PIL.Image import Image as PilImage

# create a sketch for coloring based in an input image
def generate_coloring_page(input: PilImage) -> PilImage:
    # Convert to grayscale if needed
    if input.mode != "L":
        input = input.convert("L")
    np_image = np.asarray(input)
    
    # detect edges
    np_image = skimage.filters.sobel(np_image)
    # convert to 8 bpp
    np_image = skimage.util.img_as_ubyte(np_image)
    # Invert to get dark edges on a light background
    np_image = 255 - np_image
    # Improve the contrast
    np_image = skimage.exposure.rescale_intensity(np_image)
    
    return Image.fromarray(np_image)

# specify directories
input_dir = './images'  # replace with your input directory
output_dir = './images/coloring_pages'  # replace with your output directory

# create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# iterate over all .jpg images in the specified directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        # generate coloring page
        image = Image.open(os.path.join(input_dir, filename))
        coloring_page = generate_coloring_page(image)
        
        # save coloring page in the output directory
        output_filename = os.path.splitext(filename)[0] + '_cp.jpg'
        coloring_page.save(os.path.join(output_dir, output_filename))