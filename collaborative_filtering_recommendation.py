
# coding: utf-8

# In[16]:




# In[32]:

print ("Loading Dataset")
import pandas as pd

#Reading user file

user_att=['user_id','age','gender','occupation','zip_code']
users=pd.read_csv('C:\Users\Aashish Goyal\Documents\My ML Work\Recommendar\ml-100k\ml-100k\u.user', sep='|', names=user_att,encoding='latin-1')

#reading rating file
rate_att=['user_id','movie_id','rating','timestamp']
ratings=pd.read_csv('C:\Users\Aashish Goyal\Documents\My ML Work\Recommendar\ml-100k\ml-100k/u.data',sep='|',names=rate_att,encoding='latin-1')

#reading items file:

items_att=['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items=pd.read_csv('C:\Users\Aashish Goyal\Documents\My ML Work\Recommendar\ml-100k\ml-100k/u.item',sep='|',names=items_att,encoding='latin-1')

# see your data
print (users.shape)
print (users.head())


print (ratings.shape)
print (ratings.head())

print (items.shape)
print (items.head())


ratings_train=pd.read_csv('C:\Users\Aashish Goyal\Documents\My ML Work\Recommendar\ml-100k\ml-100k/ua.base', sep='\t', names=rate_att, encoding='latin-1')
ratings_test=pd.read_csv('C:\Users\Aashish Goyal\Documents\My ML Work\Recommendar\ml-100k\ml-100k/ua.test', sep='\t', names=rate_att, encoding='latin-1')

print(ratings_train.shape,ratings_test.shape)


# In[33]:

#coverting into Sframes
import graphlab
train_data = graphlab.SFrame(ratings_train)
test_data=graphlab.SFrame(ratings_test)


# In[34]:

#collaborative filtering model
col_fil_model=graphlab.item_similarity_recommender.create(train_data,user_id='user_id',item_id='movie_id',target='rating',similarity_type='jaccard')
    

# In[35]:

#print recommendations given by model for some users ,where k is for the top k recommendations given
col_fil_recomm=col_fil_model.recommend(users=range(1,5),k=5)
col_fil_recomm.print_rows(num_rows=30)

