from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pandas as pd
import numpy as np
from IPython.display import Image, display
import streamlit.components.v1 as components


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



customers_rcmnds = pd.read_csv('results/customers_rcmnds.csv')
customers = customers_rcmnds.customer.unique()        
        
get_item = st.sidebar.button('Get Random Purchase History of Customer')
if get_item:
    st.sidebar.write('#### Customer history')
    rand_customer = np.random.choice(customers)

rand_customer_age = customers_rcmnds.loc[customers_rcmnds['customer'] == rand_customer, 'age'].values[0]
results = get_similar_products(rand_customer_age)
with st.container():
                container = st.expander("Electronics", expanded=True)
                with container:
                    cols = st.columns(7)
                    cols[0].write('###### Model Description')
                    cols[0].caption('Text description embeddings are calculated using "universal-sentence-encoder" from TensorFlow Hub')
                    for i,col in zip(results,cols[1:]):
                        with col:
                            image_url = get_image_url(i)
                            st.image(str(image_url))
                            st.caption(i)