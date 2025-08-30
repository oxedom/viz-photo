# import open_clip
import os
import pathlib
import PIL

# Current dir 
imagesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')
# List images
images = os.listdir(imagesPath)
for i in images:
    img_path = (os.path.join(imagesPath, i))
    print(img_path)
    PIL
# print(images)
exit(1)

pathlib.Path()

open_clip.list_pretrained()
model, preprocess  = open_clip.create_model_from_pretrained('ViT-B-16', 'laion2b_s34b_b88k' )
model.eval('cpu')

