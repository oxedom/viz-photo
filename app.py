import open_clip
import os
import pathlib
from PIL import Image
import json

device = 'cpu'
open_clip.list_pretrained()
model, preprocess  = open_clip.create_model_from_pretrained('ViT-B-16', 'laion2b_s34b_b88k' )
model.eval()


# Current dir 
imagesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')
# List images
images = os.listdir(imagesPath)
final_res = {}
for idx, i in enumerate(images):
    img_path = (os.path.join(imagesPath, i))
    processed_img = preprocess(Image.open(img_path)).unsqueeze(0)
    res = model.encode_image(processed_img)
    final_res[idx] = res
    # open_image.

breakpoint()


print('pizza')
# with open('vectors.json', 'w') as file:
#     file.write(json.dumps(final_res))
# # print(images)
# exit(1)


