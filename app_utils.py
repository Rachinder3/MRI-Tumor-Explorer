from datetime import datetime
from PIL import Image
from io import BytesIO
import re, time, base64
import cv2
import matplotlib.pyplot as plt
import time
import torch
import numpy as np




current_time_stamp = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"



def to_tensor(x, **kwargs):
    """
    converts numpy array to tensor
    """
    return x.transpose(2, 0, 1).astype('float32')


def inferencing(model, image, prepro_fn,device):
    
    """
    Performs Inferencing on Brain MRI
    """
    
    img_inference = cv2.resize(image, (224,224))
    img = img_inference.copy()
    img = img/255.
    img = prepro_fn(img)
    img = to_tensor(img)
    x_tensor = torch.from_numpy(img).to(device).unsqueeze(0)
    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    return pr_mask, img_inference

def mask_on_image(pr_mask, img_inference):
    
    """
    Overlays Mask on Brain MRI
    
    """
    
    msk_stack = np.stack((pr_mask,pr_mask,pr_mask),axis = 2)
    overlayed_img = np.where(msk_stack, (255,0,0), img_inference)
    return overlayed_img
    



def base64_2_img(img, path):
    """
    converts base 64 format to image
    
    arguements:
        img: image in base 64 format
    
        path: path to store the image
    
    """
    try:
        # ensuring img is string
        if type(img) == bytes:
            img = img.decode("utf-8")
        
        
            
            
        
        base64_data = re.sub('^data:image/.+;base64,', '', img)
        
        byte_data = base64.b64decode(base64_data)
        
        image_data = BytesIO(byte_data)
        img = Image.open(image_data)
        img.save(path)
        #cv2.imwrite(filename=path, img=img)
            
            
    except Exception as e:
        print(str(e))
        
        
def image_2_base64(path):
    '''
    converts image to base 64 format
    
    
    arguements: 
        path: path of the image
        
    returns:
        image in base 64 format
    
    '''
    
    try:
        image = open(path,'rb')
        image_read = image.read()
        image_64_encode = base64.b64encode(image_read)
        image.close()
        return image_64_encode
    
        
    
    except Exception as e:
        print(str(e))        
            
        
    
    
    
    