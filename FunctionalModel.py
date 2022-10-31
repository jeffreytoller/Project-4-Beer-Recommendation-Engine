# %%
from surprise import Dataset, Reader
from surprise import SVD # implementation of Funk's SVD
from surprise import accuracy # metric
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV #train/test splits, etc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import warnings
warnings.filterwarnings('ignore')
# %%
def prepare_data():
    df=pd.read_csv('beer_reviews.csv')
    df.drop(columns=['beer_abv'],inplace=True)
    review_count_threshold = 2
    users = df.groupby('review_profilename').count()
    users = users.loc[users['beer_name'] >= review_count_threshold]
    users_to_keep = list(users.index)
    df = df.loc[df['review_profilename'].isin(users_to_keep)]
    beers = df.drop_duplicates(subset='beer_beerid')
    beers['complete_beer_name'] = beers.brewery_name + ' Brewery ' + beers.beer_name
    beers = beers[['beer_beerid','complete_beer_name','beer_style','brewery_id','beer_style']]
    df = df[['review_profilename','beer_beerid','review_overall']]
    df['review_overall'] = df.review_overall.astype('float16')
    df['beer_beerid'] = df.beer_beerid.astype('int32')
    df.drop_duplicates(subset = ['review_profilename','beer_beerid'],inplace=True)
    return({'beers_df':beers,'merge_df':df})

# %%
def extract_ratings(preceding_tr,user_url):
    r = requests.get('https://www.beeradvocate.com/user/beers/?ba='+user_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    beer_names = []
    beer_ratings = []
    beer_body_tr = soup.body.find_all('tr')
    beer_list = beer_body_tr[preceding_tr:]
    for i,v in enumerate(beer_list):
        beer_url = v.a.attrs['href']
        beer_id = beer_url.split('/')[4]
        beer_names.append(beer_id)
        beer_rating = beer_body_tr[i+preceding_tr].find_all('b')[-1].text
        beer_ratings.append(beer_rating)
        data = pd.DataFrame({'review_profilename':'user_prediction','beer_beerid': beer_names,'review_overall': beer_ratings})
        data['beer_beerid'] = data.beer_beerid.astype('int32')
        data['review_overall'] = data.review_overall.astype('float16')
    return(data)


# %%
def get_recommendations(merge_df,user_df,beers_df):
    items_to_predict = beers_df[~beers_df.beer_beerid.isin(user_df.beer_beerid)].beer_beerid.unique()
    reader = Reader()
    user_data = Dataset.load_from_df(user_df, reader=reader)
    full_df = pd.concat([merge_df.copy(),user_df.copy()],ignore_index=True)
    full_df.reset_index(inplace=True,drop=True)
    reader = Reader()#line_format='user rating item', sep=',')
    data = Dataset.load_from_df(full_df, reader=reader)
    trainset, testset = train_test_split(data, test_size=.2, random_state = 42)
    svd = SVD(reg_all = .05, lr_all = 0.008, n_factors = 11, n_epochs = 65)
    svd.fit(trainset)
    predictions = []
    for i,v in enumerate(items_to_predict):
        recommendation = svd.predict(uid='user_prediction',iid=v,verbose=False)
        predictions.append(recommendation.est)
    predictions
    pred_df = pd.DataFrame({'beer_beerid':items_to_predict,'predicted_rating':predictions})
    pred_df = pred_df.sort_values(by='predicted_rating',ascending=False)
    pred_df = pred_df.merge(beers_df)
    cols = ['brewery_id', 'beer_beerid']
    pred_df['url'] = pred_df[cols].apply(lambda row: '/'.join(row.values.astype(str)), axis=1)
    pred_df['url'] =  "https://www.beeradvocate.com/beer/profile/" + pred_df.url + "/"
    return(pred_df)

# %%
def run_recommender(username,beers_df,merge_df):
    ratings = extract_ratings(3,username)
    return(get_recommendations(merge_df,ratings,beers_df)[:50])


# a='Rug'
# data = prepare_data()
# beers_df = data['beers_df']
# merge_df = data['merge_df']
# run_recommender(a,beers_df,merge_df) 
# %%
