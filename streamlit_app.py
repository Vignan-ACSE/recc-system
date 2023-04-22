import streamlit as st
import pandas as pd
import numpy as np
from funcs import *
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

##################################################################################################
##################################################################################################

edata = pd.read_csv('./results/Electronics_data.csv')
edata.fillna(edata.mean(), inplace=True)

tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(edata['Title'])

tfidf_matrix = tfidf.transform(edata['Title'])
cosine_similarities = cosine_similarity(tfidf_matrix)

def get_similar_products(age, n=6):
    filtered_df = edata[edata['Age'] == age]
    indices = filtered_df.index
    mean_cosine_similarities = cosine_similarities[indices].mean(axis=0)
    top_indices = mean_cosine_similarities.argsort()[:-n-1:-1]
    top_titles = edata.iloc[top_indices]['Title']
    return top_titles

def get_image_url(title):
    row = edata[edata['Title'] == title]
    if row.empty:
        return None
    return row.iloc[0]['Image']


##################################################################################################
def main():

    st.set_page_config(layout="wide", initial_sidebar_state='expanded')
    
    sidebar_header = '''This is a recommender system that finds similar items to a given clothing article or recommend items for a customer using 2 different approaches:'''
    
    page_options = ["Reccomend from similar items",
                    "Recommendations based on customer purchase history",
                    "Profile Based",
                    "Product Captioning"]
    
#     st.sidebar.image('LOGO.jpg')
    st.sidebar.info(sidebar_header)
   
    
    page_selection = st.sidebar.radio("Try", page_options)
    articles_df = pd.read_csv('articles.csv')
    
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

#########################################################################################
#########################################################################################

    if page_selection == "Reccomend from similar items":

        articles_rcmnds = pd.read_csv('results/articles_rcmnds.csv')

        articles = articles_rcmnds.article_id.unique()
        get_item = st.sidebar.button('Get Random Item')
        
        if get_item:
            
            rand_article = np.random.choice(articles)
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
                                    
#########################################################################################
#########################################################################################

    if page_selection == "Product Captioning": 
        captions = pd.read_csv('caption_desc_embeds.csv', dtype={'id':str}).drop('Unnamed: 0', axis=1)
        
        
        get_item = st.sidebar.button('Get Random Item')      
        
        st.sidebar.warning('In this section you get try a transformer based model that generates product captions given its image')
        
            
        if get_item:
            
            
            
            rand_article = np.random.choice(captions.id)
            desc = captions[captions.id == rand_article].desc.iloc[0]
            caption = captions[captions.id == rand_article].caption.iloc[0].capitalize()
            
            cols = st.columns(2)
            with cols[0]:
                st.image(get_item_image(str(rand_article[1:]), resize=True, width=300, height=400))
            with cols[1]:
                with st.expander('Actual Product Description', expanded=True):
                    components.html(f"""
           <header>
            <h4 style="color: #253f4e;">{desc}</h4>
           </header>
            """)
                
                with st.expander('Generated Product Description', expanded=True):
                     components.html(f"""
           <header>
            <h4 style="color: #253f4e;">{caption}</h4>
           </header>
            """)
            
            
            
#########################################################################################
#########################################################################################
    if page_selection == "Recommendations based on customer purchase history":
        
        customers_rcmnds = pd.read_csv('results/customers_rcmnds.csv')
        customers = customers_rcmnds.customer.unique()        
        
        get_item = st.sidebar.button('Get Random Purchase History of Customer')
        if get_item:
            st.sidebar.write('#### Customer history')

            rand_customer = np.random.choice(customers)
            customer_data = customers_rcmnds[customers_rcmnds.customer == rand_customer]
            customer_history = np.array(eval(customer_data.history.iloc[0]))

            image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds = get_rcmnds(customer_data)
            
            scores = get_rcmnds_scores(customer_data)
            features = get_rcmnds_features(articles_df, combined_rcmnds, tfrs_rcmnds, image_rcmnds, text_rcmnds, feature_rcmnds)
            images = get_rcmnds_images(combined_rcmnds, tfrs_rcmnds, image_rcmnds, text_rcmnds, feature_rcmnds)
            detail_descs  = get_rcmnds_desc(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
            rcmnds = (image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)

            splits = [customer_history[i:i+3] for i in range(0, len(customer_history), 3)]
                            
            for split in splits:
                with st.sidebar.container():
                    cols = st.columns(3)
                    for item, col in zip(split, cols):
                        col.image(get_item_image(str(item), 100))
            
            with st.container():
                rand_customer_age = customers_rcmnds.loc[customers_rcmnds['customer'] == rand_customer, 'age'].values[0]
                rand_customer_status = customers_rcmnds.loc[customers_rcmnds['customer'] == rand_customer, 'club_status'].values[0]
                st.subheader("Customer Details")
                st.write(f"Customer age : {rand_customer_age}")
                st.write(f"Club Membership Status : {rand_customer_status}")
                st.write(f"---------------------------------------------------------------------------------------------------------")

            with st.container():          
                for i, model, image_set, score_set, model_desc, detail_desc_set, features_set, rcmnds_set in zip(range(5), models, images, scores, model_descs, detail_descs, features, rcmnds):
                    container = st.expander(model, expanded=True)
                    with container:
                        cols = st.columns(7)
                        cols[0].write('###### Similarity Score')
                        cols[0].caption(model_desc)
                        for img, col, score, detail_desc, rcmnd in zip(image_set[1:], cols[1:], score_set[1:], detail_desc_set[1:],  rcmnds_set[1:]):
                            with col:
                                st.caption('{}'.format(score))
                                st.image(img, use_column_width=True)
            
            results = get_similar_products(rand_customer_age)

            with st.container():
                container = st.expander("Electronics Reccomendation based on Age", expanded=True)
                with container:
                    cols = st.columns(7)
                    cols[0].write('###### Model Description')
                    cols[0].caption('Text description embeddings are calculated using "universal-sentence-encoder" from TensorFlow Hub')
                    for i,col in zip(results,cols[1:]):
                        with col:
                            image_url = get_image_url(i)
                            st.image(str(image_url))
                            st.caption(i)

######################################################################################################################  
######################################################################################################################
    if page_selection == 'Profile Based':
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
                    container = st.expander(model, expanded = True)
                    with container:
                        if model == 'Similar items based on image embeddings':
                            continue
                        cols = st.columns(7)
                        cols[0].write('###### Similarity Score')
                        cols[0].caption(model_desc)
                        for img, col, score, detail_desc, rcmnd in zip(image_set[1:], cols[1:], score_set[1:], detail_desc_set[1:],  rcmnds_set[1:]):
                            with col:
                                st.caption('{}'.format(score))
                                st.image(img, use_column_width=True)
                                if model == 'Similar items based on text embeddings':
                                    st.caption(detail_desc)






main()