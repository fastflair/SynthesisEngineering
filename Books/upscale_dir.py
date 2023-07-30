import sys
import os
import shutil

image_folder = "./images"
upload_folder = image_folder+"/upscaled"

if os.path.isdir(upload_folder):
    shutil.rmtree(upload_folder)
os.makedirs(upload_folder, exist_ok=True)

os.system(f"python ./../../GFPGAN/inference_gfpgan.py -i {image_folder} -o {upload_folder} -v 1.3 -s 4 --bg_upsampler realesrgan")
