import pandas as pd
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

def data_prepare():
    tf.disable_v2_behavior()
    ratings_df = pd.read_csv('./ml-latest-small/ratings.csv')
    ratings_df.tail()
    movies_df = pd.read_csv('./ml-latest-small/movies.csv')
    movies_df.tail()
    movies_df['movieRow'] = movies_df.index
    movies_df.tail()
    movies_df = movies_df[['movieRow', 'movieId', 'title']]
    movies_df.to_csv('./ml-latest-small/moviesProcessed.csv', index=False, header=True, encoding='utf-8')
    movies_df.tail()
    ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
    ratings_df.head()

    ratings_df = ratings_df[['userId', 'movieRow', 'rating']]
    ratings_df.to_csv('./ml-latest-small/ratingsProcessed.csv', index=False, header=True, encoding='utf-8')
    ratings_df.head()

    userNo = ratings_df['userId'].max() + 1
    movieNo = ratings_df['movieRow'].max() + 1
    rating = np.zeros((movieNo, userNo))
    flag = 0
    ratings_df_length = np.shape(ratings_df)[0]
    for index, row in ratings_df.iterrows():
        rating[int(row['movieRow']), int(row['userId'])] = row['rating']
        flag += 1
    record = rating > 0
    # record
    record = np.array(record, dtype=int)
    # record
    rating_norm, rating_mean = toNormalizeRatings(rating, record)
    rating_norm = np.nan_to_num(rating_norm)
    # rating_norm
    rating_mean = np.nan_to_num(rating_mean)
    # rating_mean

    num_features = 10
    X_parameters = tf.Variable(tf.random.normal([movieNo, num_features], stddev=0.35))
    Theta_parameters = tf.Variable(tf.random.normal([userNo, num_features], stddev=0.35))
    loss = 1 / 2 * tf.reduce_sum(
        ((tf.matmul(X_parameters, Theta_parameters, transpose_b=True) - rating_norm) * record) ** 2) + 1 / 2 * (
                       tf.reduce_sum(X_parameters ** 2) + tf.reduce_sum(Theta_parameters ** 2))

    # optimizer = tf.train.AdamOptimizer(1e-4)
    # optimizer = tf.optimizers.Adam(1e-4)
    # optimizer = tf.train.GradientDescentOptimizer(1e-4)

    tf.compat.v1.disable_eager_execution()
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(1e-4)
    assert isinstance(optimizer.minimize, object)
    # train = optimizer.minimize(loss, var_list=None)
    w = tf.Variable(0, dtype=tf.float32)
    train = optimizer.minimize(loss, var_list=Theta_parameters)

    tf.summary.scalar('loss', loss)
    summaryMerged = tf.summary.merge_all()
    filename = './movie_tensorboard'
    writer = tf.summary.FileWriter(filename)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(5000):
        _, movie_summary = sess.run([train, summaryMerged])
        writer.add_summary(movie_summary, i)
    Current_X_parameters, Current_Theta_parameters = sess.run([X_parameters, Theta_parameters])
    predicts = np.dot(Current_X_parameters, Current_Theta_parameters.T) + rating_mean
    errors = np.sqrt(np.sum((predicts - rating)**2))
    # errors
    user_id = input('Please enter user number')
    sortedResult = predicts[:, int(user_id)].argsort()[::-1]
    idx = 0
    print('The top 20 movies recommended for this user are '.center(80, '='))
    for i in sortedResult:
        print('score: %.2f, Movie name: %s' % (predicts[i, int(user_id)], movies_df.iloc[i]['title']))
        idx += 1
        if idx == 20:
            break

def toNormalizeRatings(rating, record):
    m, n = rating.shape
    rating_mean = np.zeros((m, 1))
    rating_norm = np.zeros((m, n))
    for i in range(m):
        idx = record[i, :] != 0
        rating_mean[i] = np.mean(rating[i, idx])
        rating_norm[i, idx] -= rating_mean[i]
    return rating_norm, rating_mean


if __name__ == '__main__':
    data_prepare()
