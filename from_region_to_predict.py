import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import os
from tensorflow import keras

def load_model(modelname):
    new_model = keras.models.load_model(modelname)
    return new_model

def from_region_to_preict(img,modelname):
    img = img.convert('L')  # Ensure the image is in gray format
    img = img.resize((128, 128))  # Resize image to a fixed size
    img_array = np.array(img)
    x_images = img_array.astype('float32') / 255.0
    x_images = x_images.reshape((1, 128, 128))

    new_model = load_model(modelname)
    encoded_imgs = new_model.encoder(x_images).numpy()
    result = new_model.decoder(encoded_imgs).numpy()
    result = (result * 255).astype(np.uint8)

    # Remove the batch dimension and reshape if necessary
    result = result.reshape((128, 128)) 
    # Convert to an image and save
    output_image = Image.fromarray(result)
    output_image.save(output_path+'/'+"outcome.jpg")
    return result


if __name__ == '__main__':
    folder = 'C:/Users/User/makeup_project_finaltest/region'
    filename = 'region.jpg'
    output_path='C:/Users/User/makeup_project_finaltest/predict'
    output_filename= "predict.jpg"
    modelname = 'saved_model'

    img_path = os.path.join(folder, filename)
    img = Image.open(img_path)


    result = from_region_to_preict(img,modelname)
    output_image = Image.fromarray(result)
    output_image.save(output_path+'/'+output_filename)
