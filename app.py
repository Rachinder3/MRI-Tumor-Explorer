####################### Loading Libraries ##########################3

import streamlit as st
from PIL import Image
import numpy as np
import os
import torch
from torch_snippets import *
import segmentation_models_pytorch as smp
from app_utils import inferencing, mask_on_image

######################### Setting Configurations ############################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

models_dir = "models"
model_file_name = "unet_vgg16bn.pth"
model_file_path = os.path.join(models_dir,model_file_name)

ENCODER = 'vgg16_bn'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
CLASSES = ['tumor']
prepro_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
########################  Loading Model  ############################

model = torch.load(model_file_path,map_location=torch.device(device))

#########################  Streamlit application  ###########################
 
"""
# Brain-Tumor-Semantic-Segmentation
"""




image_file = st.file_uploader("Choose a file (Brain MRI).", type=["png","jpg","jpeg"])
if image_file is not None:
    # To read file as bytes:
    bytes_data = image_file.getvalue()
    file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
    #st.write(file_details)

    
    img = Image.open(image_file)
    img = img.convert('RGB')
    img = np.array(img)
    
    # st.image(img)
    # st.write(img.shape)
    # st.write(device)
    # st.write(model)
    
    pr_mask, img_inference = inferencing(model, img, prepro_fn, device)
    overlayed_img = mask_on_image(pr_mask, img_inference)
    
    
    cols = st.columns(3)
    
    # st.image(img_inference,channels="RGB", caption="Original Brain MRI")
    # st.image(pr_mask,channels="RGB", caption='Predicted Mask')
    # st.image(overlayed_img,channels="RGB", caption = "Mask overlayed on Brain MRI")
    
    cols[0].image(img_inference,channels="RGB", caption="Original Brain MRI")
    cols[1].image(pr_mask,channels="RGB", caption='Predicted Mask')
    cols[2].image(overlayed_img,channels="RGB", caption = "Mask overlayed on Brain MRI")
    