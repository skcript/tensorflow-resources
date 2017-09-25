from resizeimage import resizeimage
import Image
import os
from config import *

print IMAGE_FOLDER

files = [file for file in os.listdir(IMAGE_FOLDER) if file.endswith(".jpg") ]

for file in files:
  with Image.open(os.path.join(IMAGE_FOLDER, file)).convert('LA') as image:
    resizeimage.resize_contain(image, [128,128]).save(os.path.join(RESIZED_IMAGE_FOLDER, file))
