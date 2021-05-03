# mask-detection

## UTKFace dataset
1. download dataset from https://drive.google.com/file/d/0BxYys69jI14kYVM3aVhKS1VhRUk/view?usp=sharing 

2. create folder for project -> inside this foler create Data folder -> 
inside it mask, mask_to_loop, no_mask folders

 3.update links in py files

4.run copy_files.py (copy 10% of dataset)
### mask generator from https://github.com/prajnasb/observations/tree/master/mask_classifier/Data_Generator
5.run loop_imgs.py (looping through mask_to_loop folder to put masks on imgs)
6.run model(mask_model.py)
7.detection.py (predicting mask from live video)

