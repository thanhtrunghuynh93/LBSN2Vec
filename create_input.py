
import argparse
from scipy.io import loadmat
import os
import numpy as np
from utils import renumber_checkins

def parse_args():
    parser = argparse.ArgumentParser(description="LBSN configurations")
    parser.add_argument('--dataset_name', type=str, default="")
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--POI', action='store_true')
    args = parser.parse_args()
    return args


def read_input(path):
    mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
    friendship_old = mat['friendship_old']
    selected_checkins = mat['selected_checkins']
    nodes = np.unique(friendship_old)
    print("Min: {}, Max: {}, Len: {}".format(np.min(nodes), np.max(nodes), len(nodes)))
    friendship_old = friendship_old[np.argsort(friendship_old[:, 0])]
    return friendship_old, selected_checkins


def save_deepwalk(edges, selected_checkins, dataset_name):
    """
    Edge is friendship
    """
    out_dir = "edgelist_graph"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    with open("{}/{}.edgelist".format(out_dir, dataset_name), 'w', encoding='utf-8') as file:
        for i in range(edges.shape[0]):
            file.write("{}\t{}\n".format(int(edges[i, 0]), int(edges[i, 1])))

    print("Creating Mobility Graph ...")
    M_name = "{}/{}_M.edgelist".format(out_dir, dataset_name)
    if args.POI:
        M_name = "{}/{}_M_POI.edgelist".format(out_dir, dataset_name)
    
    with open(M_name, 'w', encoding='utf-8') as file:
        for i in range(selected_checkins.shape[0]):
            for j in range(selected_checkins.shape[1] - 1):
                for k in range(j+1, selected_checkins.shape[1]):
                    file.write("{}\t{}\n".format(int(selected_checkins[i, j]), int(selected_checkins[i, k])))

    print("Creating Mobility and Friend Graph...")
    SM_name = "{}/{}_SM.edgelist".format(out_dir, dataset_name)
    if args.POI:
        SM_name = "{}/{}_SM_POI.edgelist".format(out_dir, dataset_name)
    with open(SM_name, 'w', encoding='utf-8') as file:
        for i in range(selected_checkins.shape[0]):
            for j in range(selected_checkins.shape[1] - 1):
                for k in range(j+1, selected_checkins.shape[1]):
                    file.write("{}\t{}\n".format(int(selected_checkins[i, j]), int(selected_checkins[i, k])))
        for i in range(edges.shape[0]):
            file.write("{}\t{}\n".format(int(edges[i, 0]), int(edges[i, 1])))
    print("Done!")


def save_line(edges, selected_checkins, dataset_name):
    out_dir = "line_graph"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open("{}/{}.edgelist".format(out_dir, dataset_name), 'w', encoding='utf-8') as file:
        for i in range(edges.shape[0]):
            file.write("{}\t{}\n".format(int(edges[i, 0]), int(edges[i, 1])))
            file.write("{}\t{}\n".format(int(edges[i, 1]), int(edges[i, 0])))

    print("Creating Mobility Graph ...")
    M_name = "{}/{}_M.edgelist".format(out_dir, dataset_name)
    if args.POI:
        M_name = "{}/{}_M_POI.edgelist".format(out_dir, dataset_name)
    
    with open(M_name, 'w', encoding='utf-8') as file:
        for i in range(selected_checkins.shape[0]):
            for j in range(selected_checkins.shape[1] - 1):
                for k in range(j+1, selected_checkins.shape[1]):
                    file.write("{}\t{}\n".format(int(selected_checkins[i, j]), int(selected_checkins[i, k])))
                    file.write("{}\t{}\n".format(int(selected_checkins[i, k]), int(selected_checkins[i, j])))
    file.close()

    SM_name = "{}/{}_SM.edgelist".format(out_dir, dataset_name)
    if args.POI:
        SM_name = "{}/{}_SM_POI.edgelist".format(out_dir, dataset_name)
    print("Creating Mobility and Friend Graph...")
    with open(SM_name, 'w', encoding='utf-8') as file:
        for i in range(selected_checkins.shape[0]):
            for j in range(selected_checkins.shape[1] - 1):
                for k in range(j+1, selected_checkins.shape[1]):
                    file.write("{}\t{}\n".format(int(selected_checkins[i, j]), int(selected_checkins[i, k])))
                    file.write("{}\t{}\n".format(int(selected_checkins[i, k]), int(selected_checkins[i, j])))
        for i in range(edges.shape[0]):
            file.write("{}\t{}\n".format(int(edges[i, 0]), int(edges[i, 1])))
            file.write("{}\t{}\n".format(int(edges[i, 1]), int(edges[i, 0])))
    file.close()
    print("Done!")


def save_hebe(edges, dataset_name):
    out_dir = "hebe_graph"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    pass


def save_dhne(selected_checkins, dataset_name):
    out_dir = "dhne_graph"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists("{}/{}".format(out_dir, dataset_name)):
        os.mkdir("{}/{}".format(out_dir, dataset_name))
    if not os.path.exists("{}_POI".format(out_dir)):
        os.mkdir("{}_POI".format(out_dir))
    if not os.path.exists("{}_POI/{}".format(out_dir, dataset_name)):
        os.mkdir("{}_POI/{}".format(out_dir, dataset_name))
    num_types = np.array([len(np.unique(selected_checkins[:, i])) for i in range(selected_checkins.shape[1])])
    
    if args.POI:
        np.savez('{}_POI/{}/train_data.npz'.format(out_dir, dataset_name), train_data=selected_checkins, nums_type=num_types)
    else:
        np.savez('{}/{}/train_data.npz'.format(out_dir, dataset_name), train_data=selected_checkins, nums_type=num_types)
    print("Done!")
    pass


def preprocess_selected_checkins2(selected_checkins):
    """
    What does this function do???
    1. sort selected checkins according to user ID
    2. renumberring phase 1
    """
    # selected_checkins = np.delete(selected_checkins, 1, 1)
    selected_checkins = selected_checkins[np.argsort(selected_checkins[:, 0])]
    unique_location = np.unique(selected_checkins[:, 2])
    unique_cate = np.unique(selected_checkins[:, 3])
    unique_time = np.unique(selected_checkins[:, 1])
    location_id2idx = {unique_location[i]: i for i in range(len(unique_location))}
    cate_id2idx = {unique_cate[i]: i for i in range(len(unique_cate))}
    time_id2idx = {unique_time[i]: i for i in range(len(unique_time))}
    for i in range(len(selected_checkins)):
        selected_checkins[i, 0] = selected_checkins[i, 0] - 1
        selected_checkins[i, 1] = time_id2idx[selected_checkins[i, 1]]
        selected_checkins[i, 2] = location_id2idx[selected_checkins[i, 2]]
        selected_checkins[i, 3] = cate_id2idx[selected_checkins[i, 3]]
    new_location = len(unique_location)
    new_cate = len(cate_id2idx)
    new_time = len(unique_time)
    num_user = np.max(selected_checkins[:, 0]) + 1
    unique_user = np.unique(selected_checkins[:, 0]).tolist()
    additional_checkins = []
    
    for i in range(num_user):
        if i not in unique_user:
            additional_checkins.append([i,new_time, new_location, new_cate])
            new_location += 1
            new_cate += 1
            new_time += 1
    additional_checkins = np.array(additional_checkins)
    print("Num add: {}".format(len(additional_checkins)))
    if len(additional_checkins) > 0:
        selected_checkins = np.concatenate((selected_checkins, additional_checkins), axis=0)
    return selected_checkins

if __name__ == "__main__":
    args = parse_args()
    print(args)
    model = args.model 
    friendship, selected_checkins = read_input(args.dataset_name)
    friendship = friendship.astype(int)
    
    if model.lower() != "dhne" or 1:
        selected_checkins, o1, o2, o3, nt, nu = renumber_checkins(selected_checkins) # from 1 to the end
        if args.POI:
            n_trains = int(0.8 * len(selected_checkins))
            sorted_time = np.argsort(selected_checkins[:, 1])
            train_indices = sorted_time[:n_trains]
            test_indices = sorted_time[n_trains:]
            test_checkins = selected_checkins[test_indices]
            train_checkins = selected_checkins[train_indices]
            """
            OK! Now we've got all the material we need
            """
        else:
            train_checkins = selected_checkins
        
    if model.lower() == "deepwalk":
        save_deepwalk(friendship, train_checkins, args.dataset_name)
    elif model.lower() == "node2vec":
        save_deepwalk(friendship, train_checkins, args.dataset_name)
    elif model.lower() == "line":
        save_line(friendship, train_checkins, args.dataset_name)
    elif model.lower() == "hebe":
        save_hebe(friendship, args.dataset_name)
    elif model.lower() == "dhne":
        # train_checkins, test_checkins = preprocess_selected_checkins(train_checkins)
        selected_checkins = np.delete(train_checkins, 3, 1)
        unique_users = np.unique(selected_checkins[:, 0])
        unique_times = np.unique(selected_checkins[:, 1])
        unique_locs = np.unique(selected_checkins[:, 2])
        user_id2idx = {unique_users[i]: i for i in range(len(unique_users))}
        time_id2idx = {unique_times[i]: i for i in range(len(unique_times))}
        loc_id2idx = {unique_locs[i]: i for i in range(len(unique_locs))}
        for i in range(len(selected_checkins)):
            selected_checkins[i, 0] = user_id2idx[selected_checkins[i, 0]]
            selected_checkins[i, 1] = time_id2idx[selected_checkins[i, 1]]
            selected_checkins[i, 2] = loc_id2idx[selected_checkins[i, 2]]
        save_dhne(selected_checkins, args.dataset_name)
    else:
        print("Have not implement yet...")

"""

for dataset in hongzhi TKY NYC 
do
    for model in dhne deepwalk line 
    do 
        python create_input.py --dataset_name ${dataset} --model ${model}
    done
done



for dataset in Istanbul Jakarta KualaLampur SaoPaulo
do
    for model in dhne deepwalk line 
    do 
        python create_input.py --dataset_name ${dataset} --model ${model}
    done
done



"""
