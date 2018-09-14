import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from Source import SimpleNet

d = Path().resolve()
data_path = str(d) + "/Data/ml-latest-small/"
ratings_path = data_path + "ratings.csv"
utility_mat_path = data_path + "utility_mat.npz"
movie_list_path = data_path + "movies.csv"
weight_save_path = str(d) + "/weights/simple_net.ckpt"
weight_load_path = str(d) + "/weights/simple_net.ckpt"


df_movie_list = pd.read_csv(movie_list_path)
movie_list = df_movie_list['movieId'].values
movie_id_to_ind = dict()
for ind in range(movie_list.shape[0]):
    movie_id_to_ind[movie_list[ind]] = ind

df = pd.read_csv(ratings_path)
userId = df['userId'].values - 1
movieId = np.array([movie_id_to_ind[id] for id in df['movieId'].values])
rating = df['rating'].values / 5

n_user = np.max(userId) + 1
n_movie = np.max(movieId) + 1


rating_train_val, rating_test, movieId_train_val, movieId_test, userId_train_val, userId_test = train_test_split(
    rating, movieId, userId,
    test_size = 0.2
)

rating_train, rating_val, movieId_train, movieId_val, userId_train, userId_val = train_test_split(
    rating_train_val, movieId_train_val, userId_train_val,
    test_size = 0.1
)

model = SimpleNet(n_movie = n_movie, n_user = n_user)
model.fit(
    user = userId_train,
    movie = movieId_train,
    y = rating_train,
    user_val = userId_val,
    movie_val = movieId_val,
    y_val = rating_val,
    weight_save_path = weight_save_path,
    print_every = 1000,
    n_epoch = 100
)

model.load_weight(weight_load_path)
predictions = model.predict(userId_test, movieId_test)
print(predictions)
print(rating_test * 5)
print(model.evaluate(userId_test, movieId_test, rating_test))

## Create data:
# data = []
# for ind in range(rating.shape[0]):
#     X = np.zeros(shape = [inp_dim])
#     X[movieId[ind] - 1] = 1
#     X[n_movie + userId[ind] - 1] = 1
#     X[n]
#     data.append(X)
#
# data = np.array(data)
# print(data.shape)


