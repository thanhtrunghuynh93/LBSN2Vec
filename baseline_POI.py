import pdb
import numpy as np
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.io import loadmat
import random
import pdb
import math
import os
import multiprocessing
from evaluation import friendship_pred_persona, friendship_pred_ori, location_prediction
import argparse
import learn
from utils import save_info, sample_neg, read_embs, initialize_emb, random_walk, renumber_checkins
from link_pred_model import StructMLP
from sklearn.metrics import f1_score, accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(description="LBSN configurations")
    parser.add_argument('--num_walks', type=int, default=10)
    parser.add_argument('--walk_length', type=int, default=80)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--mobility_ratio', type=float, default=0.7)
    parser.add_argument('--K_neg', type=int, default=10)
    parser.add_argument('--win_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001) # 0.001 for code c
    parser.add_argument('--dim_emb', type=int, default=128)
    # often change parameters
    parser.add_argument('--dataset_name', type=str, default='NYC')
    parser.add_argument('--bias_randomwalk', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    return args


def mat_to_numpy_array(matt):
    return np.array([[int(matt[i, 0]), int(matt[i, 1])] for i in range(len(matt))])


def load_data(args):
    """
    this is for cleaned data
    """
    mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
    selected_checkins = mat['selected_checkins'] 
    friendship_old = mat["friendship_old"] # edge index from 0
    friendship_new = mat["friendship_new"] 
    selected_checkins, offset1, offset2, offset3, n_nodes_total, n_users = renumber_checkins(selected_checkins)
    
    ############## Train Test split for POI prediction ##################
    n_data = selected_checkins.shape[0]
    n_train = int(n_data * 0.8)
    
    sorted_checkins = selected_checkins[np.argsort(selected_checkins[:,1])]
    train_checkins = sorted_checkins[:n_train]
    val_checkins = sorted_checkins[n_train:]
    #####################################################################
    
    print("Build user checkins dictionary...")
    train_user_checkins = {}
    user_location = dict()
    for user_id in range(1, n_users+1): 
        inds_checkins = np.argwhere(train_checkins[:,0] == user_id).flatten()
        checkins = train_checkins[inds_checkins]
        train_user_checkins[user_id] = checkins
        user_location[user_id] = set(np.unique(checkins[:, 2]).tolist())
    
    # val_user_checkins = {}
    # for user_id in range(1, n_users+1): 
    #     inds_checkins = np.argwhere(val_checkins[:,0] == user_id).flatten()
    #     checkins = val_checkins[inds_checkins]
    #     val_user_checkins[user_id] = checkins
    # everything here is from 1

    offsets = [offset1, offset2, offset3]
    checkins = [train_checkins, val_checkins, train_user_checkins, user_location]
    count_nodes = [n_users, n_nodes_total]
    friendships = [friendship_old, friendship_new]
    return offsets, checkins, count_nodes, friendships, selected_checkins

if __name__ == "__main__":
    args = parse_args()
    print(args)

    ######################################### load data ##########################################
    offsets, checkins, count_nodes, friendships, selected_checkins= load_data(args)

    offset1, offset2, offset3 = offsets
    train_checkins, val_checkins, train_user_checkins, user_location = checkins
    n_users, n_nodes_total = count_nodes
    friendship_old, friendship_new = friendships
    ###############################################################################################

    if args.test:
        embs = np.load("Model1_{}.npy".format(args.dataset_name))

        embs_user = embs[:offset1]
        embs_time = embs[offset1:offset2]
        embs_venue = embs[offset2:offset3]

        val_checkins[:, 2] -= (offset2 + 1) # checkins to check in range (0 -- num_venues)
        val_checkins[:, 0] -= 1

        location_prediction(val_checkins, embs, embs_venue, k=10)
        exit()

    sentences = random_walk(friendship_old, n_users, args)
    neg_user_samples, neg_checkins_samples = sample_neg(friendship_old, selected_checkins)
    embs_ini = initialize_emb(args, n_nodes_total)
    save_info(args, sentences, embs_ini, neg_user_samples, neg_checkins_samples, train_user_checkins)

    learn.apiFunction("temp/processed/", args.learning_rate, args.K_neg, args.win_size, args.num_epochs, args.workers, args.mobility_ratio)
    embs_file = "temp/processed/embs.txt"
    embs = read_embs(embs_file)
    np.save("Model1_{}.npy".format(args.dataset_name), embs)
    embs_user = embs[:offset1]
    embs_time = embs[offset1:offset2]
    embs_venue = embs[offset2:offset3]

    val_checkins[:, 2] -= (offset2 + 1) # checkins to check in range (0 -- num_venues)
    val_checkins[:, 0] -= 1

    location_prediction(val_checkins, embs, embs_venue, k=10)

    # friendship_pred_ori(embs_user, friendship_old, friendship_new)

    """
    scripts:

    for data in hongzhi 
    do 
        python -u baseline_POI.py --dataset_name ${data} 
    done


    """

