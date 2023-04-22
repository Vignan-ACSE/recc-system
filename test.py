import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from funcs import *
import streamlit.components.v1 as components


models = ['Similar items based on image embeddings', 
              'Similar items based on text embeddings', 
              'Similar items based discriptive features', 
              'Similar items based on embeddings from TensorFlow Recommendrs model',
              'Similar items based on a combination of all embeddings']
    
model_descs = ['Image embeddings are calculated using VGG16 CNN from Keras', 
                  'Text description embeddings are calculated using "universal-sentence-encoder" from TensorFlow Hub',
                  'Features embeddings are calculated by one-hot encoding the descriptive features provided by H&M',
                  'TFRS model performes a collaborative filtering based ranking using a neural network', 
                  'A concatenation of all embeddings above is used to find similar items']

def get_item_image(item_id, resize=True, width=100, height = 150):
    
    path = 'results/images/'+item_id+'.jpeg'
    image = Image.open(path)
    
    if resize:
        basewidth = width
        wpercent = (basewidth / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(wpercent)))
        image = image.resize((width, height), Image.ANTIALIAS)
    image = ImageOps.expand(image, 2)
        
    return image


# Load your dataset
df = pd.read_csv("./results/profiles.csv")

# Get a list of unique values for a column
gender = st.radio("Select your gender:", ('Male', 'Female'))

age = st.number_input("Enter your age", min_value=0, max_value=120)

if gender == 'Male':
    if age <= 10:
        clothing_options = ['Kids Boy Denim', 'Kids Boy Jersey Basic', 'Young Boy UW/NW',
        'Young Boy Big Acc', 'Baby Boy Jersey Fancy',
       'Kids Boy Jersey Fancy', 'Kids Boy Trouser',
       'Young Boy Jersey Fancy', 'Kids Boy Outdoor', 'Kids Boy Exclusive']
    else: 
        clothing_options = ['Shirt', 'Light Basic Jersey', 'EQ H&M Man', 'Swimwear',
       'Jersey Fancy', 'Jacket Smart', 'Belts', 'Underwear Jersey',
       'Knitwear']

else:
    if age <= 10:
        clothing_options = ['Kids Girl S&T', 'Kids Girl Big Acc', 'Kids Girl Denim',
       'Kids Girl Knitwear', 'Kids Girl Dresses', 'Girls Small Acc/Bags',
       'Baby Girl Woven', 'Young Girl Big Acc', 'Young Girl Jersey Basic',
       'Young Girl Jersey Fancy', 'Young Girl Swimwear',
       'Young Girl Knitwear', 'Young Girl Shoes',
       'Kids Girl Jersey Fancy']
    
    else:
        clothing_options = ['Casual Lingerie','Knitwear Basic', 'Woven top', 'Ladies Sport Bottoms',
       'Woven Occasion', 'Knitwear', 'Outwear',
       'Swimwear', 'Ladies Sport Bras',
       'Trouser', 'Blouse & Dress','Jersey Basic',
       'Jersey fancy', 'Dresses', 'Jersey',
       'Other Accessories', 'Tops Woven', 'Trousers',
       'Outdoor/Blazers', 'Expressive Lingerie',
       'Bottoms', 'Socks', 'Jewellery',
       'Accessories', 'Woven bottoms',
       'Jewellery Extended', 'Tops Fancy Jersey',
       'Flats', 'Asia Assortment', 'Suit jacket',
       'Skirt', 'Denim Other Garments']

clothing = st.multiselect("Select your clothing items:", clothing_options)

# Filter the dataset based on the selected values
filtered_df = df[(df['department_name'].isin(clothing))]

ids = []
for index, row in filtered_df.iterrows():
    ids.append(row['article_id'])

articles_df = pd.read_csv('articles.csv')
articles_rcmnds = pd.read_csv('results/articles_rcmnds.csv')

for id in ids:
    rand_article = id
    article_data = articles_rcmnds[articles_rcmnds.article_id == rand_article]
    rand_article_desc = articles_df[articles_df.article_id == rand_article].detail_desc.iloc[0]
    image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds = get_rcmnds(article_data)
            
    rcmnds = (image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
    scores = get_rcmnds_scores(article_data)
    features = get_rcmnds_features(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
    images = get_rcmnds_images(image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
    detail_descs  = get_rcmnds_desc(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
    st.sidebar.image(get_item_image(str(rand_article), width=200, height=300))
    st.sidebar.write('Article description')
    st.sidebar.caption(rand_article_desc)

    with st.container():     
        for i, model, image_set, score_set, model_desc, detail_desc_set, features_set, rcmnds_set in zip(range(5), models, images, scores, model_descs, detail_descs, features, rcmnds):
            container = st.expander(model, expanded = model == 'Similar items based on image embeddings' or model == 'Similar items based on text embeddings')
            with container:
                    cols = st.columns(7)
                    cols[0].write('###### Similarity Score')
                    cols[0].caption(model_desc)
                    for img, col, score, detail_desc, rcmnd in zip(image_set[1:], cols[1:], score_set[1:], detail_desc_set[1:],  rcmnds_set[1:]):
                        with col:
                            st.caption('{}'.format(score))
                            st.image(img, use_column_width=True)
                            if model == 'Similar items based on text embeddings':
                                st.caption(detail_desc)


# Display the filtered dataframe
# st.dataframe(filtered_df)
