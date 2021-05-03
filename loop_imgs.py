import os
from put_mask import create_mask

print('Looping over mask_to_loop folder started')

folder_path = r"C:\Users\Bartek\Desktop\MaskProject\Data\mask_to_loop"


images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for i in range(len(images)):
    create_mask(images[i])

print('loop_imgs.py ended')

