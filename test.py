import learn
num_walks = 10
walk_length = 80
workers = 1 
num_epoch = 1
mobility_ratio = 0.2
K_neg = 10
win_size = 10
learning_rate = 0.001
dim_emb = 128

learn.apiFunction("temp/processed", learning_rate, K_neg, win_size, num_epoch, workers,
    mobility_ratio)