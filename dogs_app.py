#%%writefile my_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
@st.cache 
st.title("Dogs Image Classification App")

my_content = open("races.txt", "r")
dog_names = my_content.read()
dogs_list = dog_names.split('\n')
my_content.close()

upload_file = st.sidebar.file_uploader("Importez une image", type = 'jpg')
generate_pred = st.sidebar.button("Prédire une race")
model = tf.keras.models.load_model('final_model.h5')
def import_n_pred(image_data, model):
    size = (224,224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape = img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred
if generate_pred:
    image = Image.open(upload_file)
    with st.expander('image', expanded=True):
        st.image(image, use_column_width=True)
    pred = import_n_pred(image, model)
    st.title("La race de chien prédite est : {}".format(dogs_list[np.argmax(pred)]))
    
st.markdown("*Created by Cheikh Badiane, Machine Learning Engineer.*")
