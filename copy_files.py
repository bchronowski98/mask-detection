import os
from shutil import copyfile

photos_dir = r'C:\Users\Bartek\Desktop\MaskProject\UTKFace'
mask_photos_dir = r'C:\Users\Bartek\Desktop\MaskProject\Data\mask_to_loop'
no_mask_photos_dir = r'C:\Users\Bartek\Desktop\MaskProject\Data\no_mask'


count1 = 0
count2 = 0

print('copy_files started...')

for count1, filename in enumerate(os.listdir(photos_dir), 1):
    if count1 % 10 == 0:
        copyfile(os.path.join(photos_dir, filename), os.path.join(no_mask_photos_dir, filename))

for count2, filename in enumerate(os.listdir(photos_dir), 2):
    if count2 % 10 == 0:
        copyfile(os.path.join(photos_dir, filename), os.path.join(mask_photos_dir, filename))


print('copy_files successfully ended')