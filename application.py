import streamlit as st
import warnings
import numpy as np
from FunctionalModel import *
warnings.filterwarnings('ignore')
data = prepare_data()
beers_df = data['beers_df']
merge_df = data['merge_df']
a = st.text_input('username')
if st.button('Go'):
    st.write(run_recommender(a,beers_df,merge_df)['url'])
