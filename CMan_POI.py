import pdb
import numpy as np
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.io import loadmat
import random
import math
import os
import multiprocessing
from evaluation import location_prediction, location_prediction_Persona2
import argparse
import learn
from utils import save_info, sample_neg, read_embs, initialize_emb, random_walk, renumber_checkins
from link_pred_model import StructMLP
from sklearn.metrics import f1_score, accuracy_score
np.random.seed(12345)


def parse_args():
    parser = argparse.ArgumentParser(description="LBSN configurations")
    parser.add_argument('--num_walks', type=int, default=10)
    parser.add_argument('--walk_length', type=int, default=80)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--mobility_ratio', type=float, default=0.7)
    parser.add_argument('--q_n2v', type=float, default=0.2)
    parser.add_argument('--p_n2v', type=float, default=0.2)
    parser.add_argument('--K_neg', type=int, default=10)
    parser.add_argument('--win_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001) # 0.001 for code c
    parser.add_argument('--add_flag', type=float, default=0.7)
    parser.add_argument('--dim_emb', type=int, default=128)
    # often change parameters
    parser.add_argument('--dataset_name', type=str, default='NYC')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--POI_level', type=str, default='3')
    parser.add_argument('--input_type', type=str, default="persona_ori", help="persona_ori or persona_POI") 
    parser.add_argument('--bias_randomwalk', action='store_true')
    parser.add_argument('--connect_center', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    return args


def load_ego_ori_dict(path2):
    """
    load file containing information about persona --> ori maps
    """
    maps_PtOri = dict() 
    maps_OritP = dict() 
    max_node = 0 
    with open(path2, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.strip().split(',')
            persona_node = int(data_line[0]) + 1
            ori_node = int(data_line[1])
            if ori_node not in maps_OritP:
                maps_OritP[ori_node] = set([persona_node])
            else:
                maps_OritP[ori_node].add(persona_node)
            maps_PtOri[persona_node] = ori_node
            if persona_node > max_node:
                max_node = persona_node
    return maps_PtOri, maps_OritP, max_node


def create_pseudo_edges(maps_OritP, maps_PtOri, max_node):
    """
    create pseudo_edges
    parameters: 
        maps_OritP: maps from ori node to persona node
        maps_PtOri: maps from persona node to ori node
        max_node: current max index
    """
    additional_edges = []
    center_ori_dict = dict()
    for key, value in maps_OritP.items():
        max_node += 1
        maps_PtOri[max_node] = key
        center_ori_dict[max_node] = key
        maps_OritP[key].add(max_node)
        for ele in value:
            additional_edges.append([max_node, ele])
    return additional_edges, center_ori_dict, maps_OritP, maps_PtOri


def load_persona_graph(path1):
    """
    load persona graph
    parameters:
        path1: path to persona graph
    return edges
    """
    edges = []
    with open(path1, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.strip().split()
            edges.append([int(ele) + 1 for ele in data_line[:2]])
    file.close()
    return edges


def read_poi_map(path4):
    """

    """
    POI_dict = dict() # POI_index ---> POI_id
    if path4 is not None:
        with open(path4, 'r', encoding='utf-8') as file:
            for line in file:
                data_line = line.split()
                POI_dict[int(data_line[0])] = int(data_line[1])
    return POI_dict


def friendship_to_center_friendship(friends, center_ori_maps, center_id2dix=None):
    if center_id2dix is None:
        center_id2dix = {center: center for center in center_ori_maps.keys()}
    ori_center_map = {v:k for k,v in center_ori_maps.items()}
    center_friends = []
    for i in range(friends.shape[0]):
        new_friend = [ori_center_map[friends[i,0]], ori_center_map[friends[i, 1]]]
        center_friends.append([center_id2dix[ele] for ele in new_friend])
    return np.array(center_friends)


def load_ego(path1, path2, path3=None, path4=None, friendship_old_ori=None):
    """
    load ego graph
    parameters:
        path1: edgeslist: Friendgraph after splitting
        path2: ego_net: ego_node --> ori_node
        path3: edgelist_POI: ego_node --> POI node
        path4: location_dict: to_ori_location
    """
    maps_PtOri, maps_OritP, max_node = load_ego_ori_dict(path2)
    additional_edges, center_ori_maps, maps_OritP, maps_PtOri = create_pseudo_edges(maps_OritP, maps_PtOri, max_node)
    if args.connect_center:
        center_friends = friendship_to_center_friendship(friendship_old_ori, center_ori_maps).tolist()
    else:
        center_friends = []

    persona_edges = load_persona_graph(path1)
    print("Number of edges before: {}".format(len(persona_edges)))
    persona_edges += additional_edges + center_friends
    print("Number of edges after: {}".format(len(persona_edges)))
    persona_edges = np.array(persona_edges)

    if path3 is not None:
        persona_POI = allocate_poi_to_user(path3, maps_PtOri, maps_OritP)

    if path4 is not None:
        POI_maps = read_poi_map(path4)

    if path3 is not None:
        return persona_edges, maps_PtOri, persona_POI, POI_maps, maps_OritP, center_ori_maps
    return persona_edges, maps_PtOri, maps_OritP, center_ori_maps

def allocate_poi_to_user(path3, maps_PtOri, maps_OritP):
    user_POI = dict()
    with open(path3, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.strip().split(',')
            user = int(data_line[0]) + 1
            location = int(data_line[1])
            if user not in user_POI:
                user_POI[user] = set([location])
            else:
                user_POI[user].add(location)
    file.close()
    return user_POI


def mat_to_numpy_array(matt):
    return np.array([[int(matt[i, 0]), int(matt[i, 1])] for i in range(len(matt))])


def create_persona_checkins(ori_checkins, maps_OritP, train_indices):
    persona_checkins = []
    count = 0
    new_train_indices = []
    new_test_indices = []
    for i in range(len(ori_checkins)):
        checkins_i = ori_checkins[i]
        user = checkins_i[0]
        for persona_user in maps_OritP[user]:
            persona_checkins.append([persona_user, checkins_i[1], checkins_i[2], checkins_i[3]])
            if i in train_indices:
                new_train_indices.append(count)
            else:
                new_test_indices.append(count)
            count += 1

    persona_checkins = np.array(persona_checkins)
    return persona_checkins, new_train_indices, new_test_indices


def create_personaPOI_checkins(old_checkins, maps_OritP, persona_POI, POI_maps, center_ori_dict, train_indices):
    """
    center_ori_dict: center_node --> ori_node (> 1)
    persona_POI: persona_node --> location_of_splitter (> 1)
    POI_maps: location_ori --> location_of_splitter
    maps_OritP: user_ori --> set_of_persona (not center)
    """
    ori_center_dict = {v:k for k,v in center_ori_dict.items()}
    personaPOI_checkins = []
    count = 0
    new_train_indices = []
    new_test_indices = []
    for i in tqdm(range(len(old_checkins))):
        old_checkini = old_checkins[i]
        user_ori = old_checkini[0]
        center_user = ori_center_dict[user_ori] # center user will have all checkins
        personaPOI_checkins.append([center_user, old_checkini[1], old_checkini[2], old_checkini[3]])
        if i in train_indices:
            new_train_indices.append(count)
        else:
            new_test_indices.append(count)
        count += 1
        location_ori = old_checkini[2]
        location_index = POI_maps[location_ori]
        add_flag = False 
        if np.random.rand() < args.add_flag:
            add_flag = True
        for persona_user in maps_OritP[user_ori]:
            if persona_user not in persona_POI:
                continue
            if (location_index in persona_POI[persona_user]) or add_flag:
                personaPOI_checkins.append([persona_user, old_checkini[1], old_checkini[2], old_checkini[3]])
                if i in train_indices:
                    new_train_indices.append(count)
                else:
                    new_test_indices.append(count)
                count += 1

    personaPOI_checkins = np.array(personaPOI_checkins)
    return personaPOI_checkins, new_train_indices, new_test_indices




def load_data(args):
    """
    this is for cleaned data

    There are two types of persona graph:
    1. persona_ori: original_persona
    2. persona_POI: persona with separated POIs

    use args.input_type to change between these types
    """
    mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
    friendship_old_ori = mat_to_numpy_array(mat['friendship_old'])
    friendship_new = mat_to_numpy_array(mat["friendship_new"]) 

    
    edgelist_path = 'Suhi_output/edgelist_{}_{}'.format(args.dataset_name, args.POI_level)
    persona_to_ori_path = 'Suhi_output/ego_net_{}_{}'.format(args.dataset_name, args.POI_level)
    edgelistPOI_path = 'Suhi_output/edgelistPOI_{}_{}'.format(args.dataset_name, args.POI_level)
    location_map_path = 'Suhi_output/location_dict_{}'.format(args.dataset_name)

    before_selected_checkins = mat['selected_checkins']
    n_train = int(len(before_selected_checkins) * 0.8)
    # before_selected_checkins = before_selected_checkins[np.argsort(before_selected_checkins[:, 1])]
    sorted_time = np.argsort(before_selected_checkins[:, 1])
    train_indices = sorted_time[:n_train]


    if args.input_type == "persona_ori":
        friendship_old_persona, maps_PtOri, maps_OritP, center_ori_maps  = load_ego(edgelist_path, persona_to_ori_path, friendship_old_ori=friendship_old_ori)
        persona_checkins, new_train_indices, new_test_indices = create_persona_checkins(mat['selected_checkins'], maps_OritP, train_indices)
    elif args.input_type == "persona_POI":
        friendship_old_persona, maps_PtOri, persona_POI, POI_maps, maps_OritP, center_ori_maps = load_ego(edgelist_path, persona_to_ori_path, edgelistPOI_path, location_map_path)
        persona_checkins, new_train_indices, new_test_indices = create_personaPOI_checkins(mat['selected_checkins'], maps_OritP, persona_POI, POI_maps, center_ori_maps, train_indices)

    persona_checkins, offset1, offset2, offset3, n_nodes_total, n_users = renumber_checkins(persona_checkins, maps_PtOri)
    
    train_checkins = persona_checkins[new_train_indices]
    val_checkins = persona_checkins[new_test_indices]
    new_val_checkins = []
    user_checkins = dict()
    for i in range(len(val_checkins)):
        checkin_i = val_checkins[i]
        user = checkin_i[0]
        time = checkin_i[1]
        location = checkin_i[2]
        ori_user = maps_PtOri[user]
        key = "{}_{}_{}".format(ori_user, time, location)
        if key not in user_checkins:
            user_checkins[key] = 0
            checkin_i[0] = ori_user
            new_val_checkins.append(checkin_i.tolist())
            
    
    print("Num val checkins before: {}".format(len(val_checkins)))
    val_checkins = np.array(new_val_checkins)
    print("Num val checkins after: {}".format(len(val_checkins)))

    ###############################################
    edgelist_path = 'Suhi_output/edgelist_{}_{}'.format(args.dataset_name, args.POI_level)
    persona_to_ori_path = 'Suhi_output/ego_net_{}_{}'.format(args.dataset_name, args.POI_level)
    edgelistPOI_path = 'Suhi_output/edgelistPOI_{}_{}'.format(args.dataset_name, args.POI_level)
    location_map_path = 'Suhi_output/location_dict_{}_{}'.format(args.dataset_name, args.POI_level)######################
    
    print("Build user checkins dictionary...")
    train_user_checkins = {}
    user_location = dict()
    for user_id in range(1, n_users+1): 
        inds_checkins = np.argwhere(train_checkins[:,0] == user_id).flatten()
        checkins = train_checkins[inds_checkins]
        train_user_checkins[user_id] = checkins
        user_location[user_id] = set(np.unique(checkins[:, 2]).tolist())

    offsets = [offset1, offset2, offset3]
    checkins = [train_checkins, val_checkins, train_user_checkins, user_location]
    count_nodes = [n_users, n_nodes_total]
    friendships = [friendship_old_ori, friendship_old_persona, friendship_new]
    maps = [maps_PtOri, maps_OritP]

    return offsets, checkins, count_nodes, friendships, maps, train_user_checkins, persona_checkins, center_ori_maps

if __name__ == "__main__":
    args = parse_args()
    print(args)

    ######################################### load data ##########################################
    offsets, checkins, count_nodes, friendships, maps, train_user_checkins, persona_checkins, center_ori_maps = load_data(args)

    offset1, offset2, offset3 = offsets
    train_checkins, val_checkins, train_user_checkins, user_location = checkins
    n_users, n_nodes_total = count_nodes
    friendship_old_ori, friendship_old_persona, friendship_new = friendships
    maps_PtOri, maps_OritP = maps
    embs = None
    if args.test:
        if args.bias_randomwalk:
            if args.input_type == "persona_POI":
                embs = np.load("Model4_{}.npy".format(args.dataset_name))
            if args.input_type == "persona_ori":
                embs = np.load("Model2_{}.npy".format(args.dataset_name))
        elif args.input_type == "persona_POI":
            embs = np.load("Model3_{}.npy".format(args.dataset_name))
        elif args.input_type == "persona_ori":
            embs = np.load("Model3_{}.npy".format(args.dataset_name))

        embs_user = embs[:offset1]
        embs_time = embs[offset1:offset2]
        embs_venue = embs[offset2:offset3]

        val_checkins[:, 2] -= (offset2 + 1) # checkins to check in range (0 -- num_venues)
        val_checkins[:, 0] -= 1

        location_prediction_Persona2(val_checkins, embs, embs_venue, k=10, user_persona_dict=maps_OritP)
        exit()

    ###############################################################################################
    sentences = random_walk(friendship_old_persona, n_users, args, user_location, center_ori_maps)
    neg_user_samples, neg_checkins_samples = sample_neg(friendship_old_persona, persona_checkins)
    embs_ini = initialize_emb(args, n_nodes_total)
    save_info(args, sentences, embs_ini, neg_user_samples, neg_checkins_samples, train_user_checkins)

    learn.apiFunction("temp/processed/", args.learning_rate, args.K_neg, args.win_size, args.num_epochs, args.workers, args.mobility_ratio)
    embs_file = "temp/processed/embs.txt"
    embs = read_embs(embs_file)
    
    if args.bias_randomwalk:
        if args.input_type == "persona_POI":
            np.save("Model4_{}".format(args.dataset_name), embs)
        if args.input_type == "persona_ori":
            np.save("Model2_{}".format(args.dataset_name), embs)
    elif args.input_type == "persona_POI":
        np.save("Model3_{}".format(args.dataset_name), embs)
    elif args.input_type == "persona_ori":
        np.save("Model2_{}".format(args.dataset_name), embs)
    
    embs_user = embs[:offset1]
    embs_time = embs[offset1:offset2]
    embs_venue = embs[offset2:offset3]

    val_checkins[:, 2] -= (offset2 + 1) # checkins to check in range (0 -- num_venues)
    val_checkins[:, 0] -= 1

    train_user_checkins


    friendship_old_persona -= 1
    location_prediction_Persona2(val_checkins, embs, embs_venue, k=10, user_persona_dict=maps_OritP)

    """
    scripts:

    try to use --connect_center

    for data in NYC hongzhi TKY
    do 
        for alpha in 0 0.000001 0.00001 0.0001 0.001 0.01 0.1
        do 
            python -u CMan.py --input_type persona_POI --dataset_name ${data} --bias_randomwalk --alpha ${alpha} > output/data${data}_alpha${alpha}
        done 
    done


for data in NYC 
do 
    python -u CMan_POI.py --input_type persona_POI --dataset_name ${data} --connect_center 
done

    """

