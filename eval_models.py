import argparse
from scipy.io import loadmat
import os
import numpy as np
from evaluation import friendship_pred_persona, friendship_pred_ori, location_prediction
from utils import renumber_checkins

def parse_args2():
    parser = argparse.ArgumentParser(description="LBSN configurations")
    parser.add_argument('--emb_path', type=str, default="")
    parser.add_argument('--dataset_name', type=str, default="")
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--POI', action='store_true')
    args = parser.parse_args()
    return args

def read_emb(path, model):
    embs = None
    if model == "node2vec" or model == "deepwalk" or model == "line":
        try:
            file = open(path, 'r', encoding='utf-8')
        except:
            file = open(path[:-1], 'r', encoding='utf-8')
        count = 0
        embs = []
        for line in file:
            if count == 0:
                count += 1
                continue
            data_line = line.split()
            embs.append([float(ele) for ele in data_line])
        embs = np.array(embs)
        print(embs.shape)
        import pdb
        new_embs = np.zeros((int(np.max(embs[:, 0])), embs.shape[1] - 1))
        for i in range(len(embs)):
            new_embs[int(embs[i, 0]) - 1] = embs[i, 1:]

        for i in range(len(new_embs)):
            if (new_embs[i] ** 2).sum() == 0:
                new_embs[i] = new_embs[1]
            
        
        #embs = embs[np.argsort(embs[:, 0])][:, 1:]
        embs = new_embs
        #embs = np.random.rand(*embs.shape)
    elif model == "dhne":
        embs = np.load(path, allow_pickle=True)
        if not args.POI:
            embs = embs[0]
    return embs 


def read_input2(path):
    mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
    friendship_old = mat['friendship_old']
    friendship_new = mat['friendship_new']

    friendship_old = friendship_old[np.argsort(friendship_old[:, 0])]
    return friendship_old, friendship_new



def read_input(path):
    mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
    friendship_old = mat['friendship_old']
    selected_checkins = mat['selected_checkins']
    nodes = np.unique(friendship_old)
    print("Min: {}, Max: {}, Len: {}".format(np.min(nodes), np.max(nodes), len(nodes)))
    friendship_old = friendship_old[np.argsort(friendship_old[:, 0])]
    return friendship_old, selected_checkins


if __name__ == "__main__":
    args = parse_args2()
    print(args)
    model = args.model 
    embs = read_emb(args.emb_path, args.model)
    if args.POI:
        friendship, selected_checkins = read_input(args.dataset_name)
        friendship = friendship.astype(int)
        if model.lower() != "dhne":
            selected_checkins, o1, o2, o3, nt, nu = renumber_checkins(selected_checkins)
            if args.POI:
                n_trains = int(0.8 * len(selected_checkins))
                sorted_time = np.argsort(selected_checkins[:, 1])
                train_indices = sorted_time[:n_trains]
                test_indices = sorted_time[n_trains:]
                train_checkins = selected_checkins[train_indices]
                test_checkins = selected_checkins[test_indices]
                print(test_checkins)

            max_test_checkins = np.max(test_checkins)
            if max_test_checkins > embs.shape[0]:
                print("Max test checkins: {}, emb shape: {}".format(max_test_checkins, embs.shape))
                to_add = embs[0: max_test_checkins - embs.shape[0]].reshape(-1, embs.shape[1])
                embs = np.concatenate((embs, to_add), axis=0)

            embs_user = embs[:o1]
            embs_time = embs[o1:o2]
            embs_venue = embs[o2:o3]
            test_checkins[:, 2] -= (o2 + 1)
            test_checkins[:, 0] -= 1
            print(o1, o2, o3, nt, nu, embs.shape, embs_venue.shape)
            print(np.max(selected_checkins))
            print("x-"*50)
            print("Len embs_venue: {}".format(len(embs_venue)))
            print("x-"*50)
            location_prediction(test_checkins, embs, embs_venue, k=10)

        else:
            friendship, selected_checkins = read_input(args.dataset_name)
            friendship = friendship.astype(int)
            selected_checkins, o1, o2, o3, nt, nu = renumber_checkins(selected_checkins)
            max_node = np.max(selected_checkins)
            if args.POI:
                n_trains = int(0.8 * len(selected_checkins))
                sorted_time = np.argsort(selected_checkins[:, 1])
                train_indices = sorted_time[:n_trains]
                test_indices = sorted_time[n_trains:]
                train_checkins = selected_checkins[train_indices]
                test_checkins = selected_checkins[test_indices]
                print(test_checkins)

            train_checkins = np.delete(train_checkins, 3, 1)
            unique_users = np.unique(train_checkins[:, 0])
            unique_times = np.unique(train_checkins[:, 1])
            unique_locs = np.unique(train_checkins[:, 2])
            user_id2idx = {unique_users[i]: i for i in range(len(unique_users))}
            time_id2idx = {unique_times[i]: i for i in range(len(unique_times))}
            loc_id2idx = {unique_locs[i]: i for i in range(len(unique_locs))}
            for i in range(len(train_checkins)):
                train_checkins[i, 0] = user_id2idx[train_checkins[i, 0]]
                train_checkins[i, 1] = time_id2idx[train_checkins[i, 1]]
                train_checkins[i, 2] = loc_id2idx[train_checkins[i, 2]]

            new_embs = np.zeros((max_node, embs[0].shape[1]))
            for i in range(max_node):
                if i < o1:
                    try:
                        index = user_id2idx[i + 1]
                    except:
                        index = 1
                    new_embs[i] = embs[0][index]
                elif i < o2:
                    try:
                        index = time_id2idx[i + 1]
                    except:
                        index = 1
                    new_embs[i] = embs[1][index]
                else:
                    try:
                        index = loc_id2idx[i + 1]
                    except:
                        index = 1
                    new_embs[i] = embs[2][index]
                    
            embs = new_embs
            embs_user = embs[:o1]
            embs_time = embs[o1:o2]
            embs_venue = embs[o2:o3]
            test_checkins[:, 2] -= (o2 + 1)
            test_checkins[:, 0] -= 1
            location_prediction(test_checkins, embs, embs_venue, k=10)
    else:
        # train_checkins, test_checkins = read_input_POI(args.path)
        friendship_old, friendship_new = read_input2(args.dataset_name)
        n_users = max(np.max(friendship_old), np.max(friendship_new))
        embs = embs[:n_users]
        friendship_pred_ori(embs, friendship_old, friendship_new)

"""
####################### eval ###############################
for data in Istanbul
do
python eval_models.py --emb_path line_emb/${data}_M_POI.embeddings --dataset_name ${data} --model line --POI
done 


for data in hongzhi NYC TKY
do
python eval_models.py --emb_path line_emb/${data}.embeddings --dataset_name ${data} --model line
done 
#############################################################3

######################## gen embedding #########################


for data in NYC TKY hongzhi 
do     
deepwalk --format edgelist --input ../LBSN2Vec/edgelist_graph/${data}.edgelist     --max-memory-data-size 0 --number-walks 10 --representation-size 128 --walk-length 80 --window-size 10     --workers 16 --output ../LBSN2Vec/deepwalk_emb/${data}.embeddings
done

for data in NYC TKY hongzhi
do 
python run_node2vec --dataset_name ${data}
done

for data in NYC TKY hongzhi
do
python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}.embeddings 
done

for data in NYC hongzhi TKY
do
python src/hypergraph_embedding.py --data_path ../LBSN2Vec/dhne_graph/${data} --save_path ../LBSN2Vec/dhne_emb/${data} -s 16 16 16
done

#################################################################
"""
