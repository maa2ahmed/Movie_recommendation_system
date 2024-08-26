import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import streamlit as st

# Dataset link --> https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata


movie_df = pd.read_csv('E:\\recommendation-system\\tmdb_5000_movies.csv')
credit_df = pd.read_csv('E:\\recommendation-system\\tmdb_5000_credits.csv')

# movie_df.head(1)

# movie_df.shape

# credit_df.head(1)

# credit_df.shape

movie = movie_df.merge(credit_df,on='title')

# movie.head(1)

# movie.shape

#dropping irrelevant columns
movie = movie[['genres','movie_id','keywords','overview','title','cast','crew']]

# movie.head(1)

# movie.shape

# movie.isnull().sum()

movie.dropna(inplace=True)

# movie.isnull().sum()

# movie.duplicated().sum()

# movie['genres'][0]

# type(movie['genres'][0])

#coverting into list
def converter(text):
  L  = []
  for i in ast.literal_eval(text):
    L.append(i['name'])
  return(L)

movie['genres'] = movie['genres'].apply(converter)

# movie.head(1)

# movie['keywords'][0]

movie['keywords'] = movie['keywords'].apply(converter)

# movie.head(1)

# movie['cast'][0]

# for i in ast.literal_eval(movie['cast'][0]):
#   print(i)
#   # break

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L

movie['cast'] = movie['cast'].apply(convert3)

# movie.head(1)

# movie['cast'][0]

# movie['crew'][0]

director_list = []
def director_name_extractor(text):
  director_list = []

  for i in ast.literal_eval(text):
    if i['job'] == 'Director':
      director_list.append(i['name'])
  return director_list

movie['crew'] = movie['crew'].apply(director_name_extractor)

# movie.head(2)

#removing space between words
def space_removal(lst):
  L = []
  for i in lst:
    L.append(i.replace(' ',''))
  return L

movie['cast'] = movie['cast'].apply(space_removal)
movie['genres'] = movie['genres'].apply(space_removal)
movie['crew'] = movie['crew'].apply(space_removal)
movie['keywords'] = movie['keywords'].apply(space_removal)

movie.head(1)

movie['overview'] = movie['overview'].apply(lambda x:x.split())

# movie.head(1)

movie['tags']= movie['overview']+movie['genres']+movie['keywords']+movie['cast']+movie['crew']

# movie.head(1)

df = movie.drop(columns=['cast','crew','keywords','overview','genres'])

df.head()

def join_words(tag_list):
    return " ".join(tag_list)

df['tags'] = df['tags'].apply(join_words)

df.head()

df['tags'] = df['tags'].apply(lambda x: x.lower())

df.head(3)

ps  = PorterStemmer()

def stem(text):
  stem_words = []
  for i in text.split():
    stem_words.append(ps.stem(i))

  return " ".join(stem_words)

df['tags'] = df['tags'].apply(stem)

cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(df['tags']).toarray()

feature_names = cv.get_feature_names_out()

# feature_names

# len(vector)

# vector.shape

# vector

similarity = cosine_similarity(vector)

# df[df['title'] == 'The Lego Movie'].index[0]

# df[df['title'] == 'The Lego Movie']

def recommend(movie):
    index = df[df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    recommended_movies = []
    for i in distances[1:6]:
        recommended_movies.append(df.iloc[i[0]].title)
    
    return recommended_movies
  
st.title('Movie Recommendation System')

user_input = st.text_input('Write movie name: ')

if user_input:
    response = recommend(user_input)
    for movie in response:
        st.write(movie)



