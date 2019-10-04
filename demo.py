# import MySQLdb
# conn=MySQLdb.connect(host='localhost',user='root',passwd='1234',db='book_rs')
# mycursor=conn.cursor()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import clear_output
from PIL._imaging import display
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics as metric
import ipywidgets as widgets
import io
import sys


books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrls', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']
print (books.shape)
print (users.shape)
print (ratings.shape)

# books.drop(['imageUrlS','imageUrlM','imageUrlL'],axis=1,inplace=True)
# books.head()

print(books.yearOfPublication.unique())
print (users.head)
print(books.loc[books.yearOfPublication== 'DK Publishing Inc',:])

books.loc[books.yearOfPublication=='DK Publishing Inc',:]
books.loc[books.ISBN=='0789466953','yearOfPublication']=2000
books.loc[books.ISBN=='0789466953','bookAuthor']='James Buckley'
books.loc[books.ISBN=='0789466953','publisher']='DK Publishing Inc'
books.loc[books.ISBN=='0789466953','yearOfPublication']="DK Readers: Creating the X-men, How Comic Books Come to Life"

books.loc[books.ISBN=='078946697X','yearOfPublication']=2000
books.loc[books.ISBN=='078946697X','bookAuthor']='Michael Teitelbaum'
books.loc[books.ISBN=='078946697X','publisher']="DK Publishing Inc"
books.loc[books.ISBN=='078946697X','bookTitle']="DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Reader)"

books.loc[books.yearOfPublication=='Gallimard',:]
books.loc[books.ISBN=='2070426769','yearOfPublication']=2000
books.loc[books.ISBN=='2070426769','bookAuthor']="Jean-Marie Gustave"
books.loc[books.ISBN=='2070426769','yearOfPublication']="Gallimard"
books.loc[books.ISBN=='2070426769','yearOfPublication']="Peuple du ciel, suivi de Les Bergers"

books.yearOfPublication=pd.to_numeric(books.yearOfPublication,errors='coerce')

books.loc[(books.yearOfPublication>2006)|(books.yearOfPublication==0),'yearOfPublication']=np.NAN
books.yearOfPublication.fillna(round(books.yearOfPublication.mean()),inplace=True)
books.yearOfPublication= books.yearOfPublication.astype(np.int32)

print(books.yearOfPublication.unique())


print (books.loc[books.publisher.isnull(),:])
books.loc[(books.ISBN=='193169656X'),'publisher']='other'
books.loc[(books.ISBN=='1931696993'),'publisher']='other'

users.loc[(users.Age>90)|(users.Age<5),'Age'] =np.nan
users.Age=users.Age.fillna(users.Age.mean())
users.Age=users.Age.astype(np.int32)
# print(sorted(users.Age.unique()))

n_users=users.shape[0]
n_books=books.shape[0]
ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]
ratings_new = ratings_new[ratings_new.userID.isin(users.userID)]
print (ratings.shape)
print (ratings_new.shape)

sparsity=1.0-len(ratings_new)/float(n_users*n_books)
print('The sparsity level is ' + str(sparsity*100)+'%')

ratings_explicit = ratings_new[ratings_new.bookRating !=0]
ratings_implicit=ratings_new[ratings_new.bookRating==0]
users_exp_ratings=users[users.userID.isin(ratings_explicit.userID)]
users_imp_ratings=users[users.userID.isin(ratings_implicit.userID)]
sns.countplot(data=ratings_explicit,x='bookRating')
# plt.show()

ratings_count=pd.DataFrame(ratings_explicit.groupby(['ISBN'])['bookRating'].sum())
top10=ratings_count.sort_values('bookRating',ascending=False).head(10)
print("Following books are recommended")
print(top10.merge(books,left_index=True,right_on='ISBN'))

# COLLABORATIVE FILTERING BASED RECOMMENDATION SYSTEM

counts1 = ratings_explicit['userID'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['userID'].isin(counts1[counts1 >= 100].index)]
counts = ratings_explicit['bookRating'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['bookRating'].isin(counts[counts >= 100].index)]

ratings_matrix = ratings_explicit.pivot(index='userID', columns='ISBN', values='bookRating')
userID = ratings_matrix.index
ISBN = ratings_matrix.columns
print(ratings_matrix.shape)
ratings_matrix.head()

# USER-BASED CF

def findksimilarusers(user_id, ratings, metric = metric, k=k):
    similarities=[]
    indices=[]
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')
    model_knn.fit(ratings)
    loc = ratings.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
    return similarities,indices

def predict_userbased(user_id, item_id, ratings, metric = metric, k=k):
    prediction=0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices=findksimilarusers(user_id, ratings,metric, k)
    mean_rating = ratings.iloc[user_loc,:].mean()
    sum_wt = np.sum(similarities)-1
    product=1
    wtd_sum = 0
    for i in range(0, len(indices.flatten())):
        if indices.flattenfflil == user_loc:
            continue;
        else:
            ratings_diff = ratings.iloc[indices.flatten()[i],item_loc]-np.mean(ratings.iloc[indices.flatten()[i],:])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product
        # fincase of very sparse datasets, using correlation metric for CF approach may give -ve ratings which are handled here as below:
    if prediction >= 0:
        prediction = 1
    elif prediction >10:
        prediction = 10
    prediction = int(round(mean_rating + (wtd_sum/sum_wt)))
    print ('\nPredicted rating for user {0} -> item {l}: {2}'.format(user_id,item_id,prediction))
    return prediction
    predict_userbased(11676,'0001056107',ratings_matrix);


def recommendItem(user_id, ratings, metric= metric):
    if (user_id not in ratings.index.values) or type(user_id) is not int:
        print ("User id should be a valid integer from this list :\n\n {}".format(re.sub('[\[\]]','', np.array_str(ratings_))))
    else:
        ids = ['Item-based (correlation)','Item-based (cosine)','User-based (correlation)','User-based (cosine)']
        select = widgets.Dropdown(options=ids, value=ids[0],description='Select approach', width='1000px')
        def on_change(change):
            clear_output(wait=True)
            prediction = []
            if change['type'] == 'change' and change['name'] == 'value':
                if (select.value == 'Item-based (correlation)') | (select.value == 'User-based (correlation)'):
                    metric = 'correlation'
                else:
                    metric = 'cosine'
                with sys.suppress_stdout():
                    if (select.value == 'Item-based (correlation)') | (select.value == 'Item-based (cosine)'):
                        for i in range(ratings.shape[1]):
                            if (ratings[str(ratings.columns[i])][user_id] !=0): #not rated already
                                prediction.append(predict_itembased(user_id, str(ratings.columns[i]) ,ratings, metric))
                            else:
                                prediction.append(-1) #for aLready rated items
                    else:
                        for i in range(ratings.shape[1]):
                            if (ratings[str(ratings.columns[i])][user_id] !=0): #not rated aLready
                                prediction.append(predict_userbased(user_id, str(ratings.columns[i]) ,ratings, metric))
                            else:
                                prediction.append(-1) #for aLready rated items
            prediction = pd.Series(prediction)
            prediction = prediction.sort_values(ascending=False)
            recommended = prediction[:10]
            print ("As per {0} approach....Following books are recommended.....".format(select.value))
            for i in range(len(recommended)):
                print ("{0}. {1}".format(i+1,books.bookTitle[recommended.index[i]].encode('utf-8')))
        select.observe(on_change)
        display(select)

recommendItem(4385, ratings_matrix)


# ITEM-BASED CF
def findksimilarusers(item_id, ratings, metric = metric, k=k):
    similarities=[]
    indices=[]
    ratings=ratings.T
    loc = ratings.index.get_loc(item_id)
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')
    model_knn.fit(ratings)
    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
    return similarities,indices

def predict_itembased(user_id, item_id, ratings, metric = metric, k=k):
    prediction = wtd_sum = 0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices=findksimilarusers(item_id, ratings)
    mean_rating = ratings.iloc[user_loc,:].mean()
    sum_wt = np.sum(similarities)-1
    product=1
    for i in range(0, len(indices.flatten())):
        if indices.flattenfflil == item_loc:
            continue;
        else:
            product = ratings.iloc[user_loc, indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product
    prediction = int(round(wtd_sum/sum_wt))
    if prediction <= 0:
        prediction = 1
    elif prediction >10:
        prediction = 10
    print ('\nPredicted rating for user {0} -> item {l}: {2}'.format(user_id,item_id,prediction))
    return prediction
predict_userbased(11676,'0001056107',ratings_matrix);