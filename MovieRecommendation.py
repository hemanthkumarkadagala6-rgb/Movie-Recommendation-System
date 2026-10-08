from flask import Flask,render_template,request
import numpy as np 
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
mov=Flask(__name__)
movies_dataset=pd.read_csv('C:/Users/heman/OneDrive/Desktop/Movie Recommendations/movies.csv')
selected_feature=['genres','keywords','tagline','cast','director']
for f in selected_feature:
   movies_dataset[f]=movies_dataset[f].fillna(' ')
    # print(movies_dataset.isnull().sum())
movies=movies_dataset['genres']+' '+movies_dataset['keywords']+' '+movies_dataset['tagline']+' '+movies_dataset['cast']+' '+movies_dataset['director']
tdfid=TfidfVectorizer()
vectorize_movies=tdfid.fit_transform(movies)
co_similar=cosine_similarity(vectorize_movies)
@mov.route('/')
def search():
    return render_template('web.html')
@mov.route('/recommend',methods=['POST','GET'])
def recommended():
    movie_name=request.form['cinema']
    recommendations = get_recommended(movie_name)
    if len(recommendations)!=0:
        return render_template('web.html', 
                           movie=movie_name, 
                           res=recommendations)
    else:
        return render_template('wrong.html',name=movie_name)
def get_recommended(x):

    titles=movies_dataset['title'].tolist()
    l=difflib.get_close_matches(x,titles)
    close_match=l[0]
    title_index=movies_dataset[movies_dataset['title']==close_match]['index'].values[0]
    similar_score=list(enumerate(co_similar[title_index]))
    sort_movies=sorted(similar_score,key=lambda x:x[1],reverse=True)
    i=1
    recommended_movies=[]
    for movie in sort_movies:
        index=movie[0]
        title_simliar=movies_dataset[movies_dataset['index']==index]['title'].values[0]
  
        if(i<=10):
            recommended_movies.append(title_simliar)
            i+=1
        else:
            break
    return recommended_movies

if __name__ == "__main__":
    mov.run(debug=True)

