import numpy as np 
import os 
import networkx as nx
from scipy.io import loadmat
from collections import Counter
from tqdm import tqdm
from walkers import BasicWalker
from scipy.sparse import csr_matrix
from random_walks import RandomWalk


def load_ego(path1, path2):
    edges = []
    with open(path1, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.strip().split()
            edges.append([int(ele) + 1 for ele in data_line[:2]])
    edges = np.array(edges)

    maps = dict()
    with open(path2, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.strip().split(',')
            maps[int(data_line[0]) + 1] = int(data_line[1])
    return edges, maps


def random_walk(friendship_old, n_users, args, user_checkins=None, center_ori_maps=None):
    print("Performing random walks on hypergraph...")
    # graph: undirected, edges = friendship_old
    adj = csr_matrix((np.ones(len(friendship_old)), (friendship_old[:,0]-1, friendship_old[:,1]-1)), shape=(n_users, n_users), dtype=int)
    adj = adj + adj.T
    G = nx.from_scipy_sparse_matrix(adj)
    commons_x, nunis_x = [], []
    commons_y, nunis_y = [], []
    if args.bias_randomwalk:
        def add_edge_weights(G, user_poi_dict, center_ori_maps):
            for source, target in tqdm(G.edges()):
                this_score = 1
                source_poi = user_poi_dict[source + 1]
                target_poi = user_poi_dict[target + 1]
                common = len(source_poi.intersection(target_poi))
                uni = len(source_poi.union(target_poi))
                # commons.append(common)
                # nunis.append(uni)
                if source < min(list(center_ori_maps.keys())) and target < min(list(center_ori_maps.keys())):
                    # print(common, uni)
                    if uni == 0:
                        this_score = 1
                    else:
                        this_score = 1 + common / uni
                    commons_x.append(common)
                    nunis_x.append(uni)
                elif source > min(list(center_ori_maps.keys())) or target > min(list(center_ori_maps.keys())):
                    # print(common, uni)
                    commons_y.append(common)
                    nunis_y.append(uni)
                    this_score = 1
                print(this_score)

                G[source][target]['weight'] = this_score

            print("X")
            print("Common: Mean, Std: {}, {}".format(np.mean(commons_x), np.std(commons_x)))
            print("Nunis: Mean, Std: {}, {}".format(np.mean(nunis_x), np.std(nunis_x)))
            print("Center")
            print("Common: Mean, Std: {}, {}".format(np.mean(commons_y), np.std(commons_y)))
            print("Nunis: Mean, Std: {}, {}".format(np.mean(nunis_y), np.std(nunis_y)))
            return G
        
        G = add_edge_weights(G, user_poi_dict=user_checkins, center_ori_maps=center_ori_maps)

        # walker = BasicWalker(G, bias=True, user_poi_dict=user_checkins, center_ori_maps=center_ori_maps, alpha=args.alpha, beta=args.beta)
        print("using Node2vec walk...")
        random_walk = RandomWalk(G, walk_length=args.walk_length, num_walks=args.num_walks, p=args.p_n2v, q=args.q_n2v, workers=args.workers)
        sentences = random_walk.walks


    else:
        walker = BasicWalker(G)
        sentences = walker.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length, num_workers=args.workers)
    for i in range(len(sentences)):
        sentences[i] = [x+1 for x in sentences[i]]
    # sentences: args.walk_length of each walk may be different
    return sentences


def initialize_emb(args, n_nodes_total):
    # TODO: initialize here!
    embs_ini = (np.random.uniform(size=(n_nodes_total, args.dim_emb)) -0.5)/args.dim_emb
    embs_len = np.sqrt(np.sum(embs_ini**2, axis=1)).reshape(-1,1)
    embs_ini = embs_ini / embs_len
    return embs_ini


def read_embs(embs_file):
    embs = []
    with open(embs_file, "r") as fp:
        for line in fp.readlines()[1:]:
            embs.append([float(x) for x in line.strip().split()])
    embs = np.array(embs)
    return embs

def to_continuous(edges, maps):
    maps_real = dict() 
    maps2 = dict()
    count = 1
    for key, value in maps.items():
        maps_real[count] = value
        maps2[key] = count
        count += 1
    new_edges = []
    for i in range(len(edges)):
        edge_i = edges[i]
        new_edge_i = [maps2[ele] for ele in edge_i]
        new_edges.append(new_edge_i)
    new_edges = np.array(new_edges)
    return new_edges, maps_real

def load_data(args):
    maps = None
    new_maps = None
    if args.input_type == "mat":
        if args.clean:
            mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
        else:
            mat = loadmat('dataset/dataset_connected_{}.mat'.format(args.dataset_name))
        selected_checkins = mat['selected_checkins'] 
        friendship_old = mat["friendship_old"] # edge index from 0
        friendship_new = mat["friendship_new"] 
    elif args.input_type == "persona":
        if args.clean:
            mat = loadmat('dataset/cleaned_{}.mat'.format(args.dataset_name))
        else:
            mat = loadmat('dataset/dataset_connected_{}.mat'.format(args.dataset_name))
        edges, maps = load_ego('Suhi_output/edgelist_{}'.format(args.dataset_name), 'Suhi_output/ego_net_{}.txt'.format(args.dataset_name))
        edges, maps = to_continuous(edges, maps)
        friendship_old = edges
        friendship_n = mat["friendship_new"] 
        new_maps = dict()
        for key, value in maps.items():
            if value not in new_maps:
                new_maps[value] = set([key])
            else:
                new_maps[value].add(key)
        
        def create_new_checkins(old_checkins, new_maps):
            new_checkins = []
            for i in range(len(old_checkins)):
                checkins_i = old_checkins[i]
                user = old_checkins[i][0]
                for ele in new_maps[user]:
                    new_checkins.append([ele, checkins_i[1], checkins_i[2], checkins_i[3]])
            new_checkins = np.array(new_checkins)
            return new_checkins
                
        selected_checkins = create_new_checkins(mat['selected_checkins'], new_maps)
        friendship_new = friendship_n

    offset1 = max(selected_checkins[:,0])
    _, n = np.unique(selected_checkins[:,1], return_inverse=True) # 
    selected_checkins[:,1] = n + offset1 + 1
    offset2 = max(selected_checkins[:,1])
    _, n = np.unique(selected_checkins[:,2], return_inverse=True)
    selected_checkins[:,2] = n + offset2 + 1
    offset3 = max(selected_checkins[:,2])
    _, n = np.unique(selected_checkins[:,3], return_inverse=True)
    selected_checkins[:,3] = n + offset3 + 1
    n_nodes_total = np.max(selected_checkins)

    n_users = selected_checkins[:,0].max() # user
    print(f"""Number of users: {n_users}
        Number of nodes total: {n_nodes_total}""")

    n_data = selected_checkins.shape[0]
    if args.mode == "friend":
        n_train = n_data
    else:
        n_train = int(n_data * 0.8)

    sorted_checkins = selected_checkins[np.argsort(selected_checkins[:,1])]
    train_checkins = sorted_checkins[:n_train]
    val_checkins = sorted_checkins[n_train:]

    print("Build user checkins dictionary...")
    train_user_checkins = {}
    for user_id in range(1, n_users+1): 
        inds_checkins = np.argwhere(train_checkins[:,0] == user_id).flatten()
        checkins = train_checkins[inds_checkins]
        train_user_checkins[user_id] = checkins
    val_user_checkins = {}
    for user_id in range(1, n_users+1): 
        inds_checkins = np.argwhere(val_checkins[:,0] == user_id).flatten()
        checkins = val_checkins[inds_checkins]
        val_user_checkins[user_id] = checkins

    return train_checkins, val_checkins, n_users, n_nodes_total, train_user_checkins, val_user_checkins, friendship_old, friendship_new, selected_checkins, offset1, offset2, offset3, new_maps, maps


def sample_neg(friendship_old, selected_checkins):
    # for negative sampling
    user_ids, counts = np.unique(friendship_old.flatten(), return_counts=True)
    freq = (100*counts/counts.sum()) ** 0.75
    neg_user_samples = np.repeat(user_ids, np.round(1000000 * freq/sum(freq)).astype(np.int64)).astype(np.int64)
    neg_checkins_samples = {}
    for i in range(selected_checkins.shape[1]):
        values, counts = np.unique(selected_checkins[:,i], return_counts=True)
        freq = (100*counts/counts.sum()) ** 0.75
        neg_checkins_samples[i] = np.repeat(values, np.round(1000000 * freq/sum(freq)).astype(np.int64))
    return neg_user_samples, neg_checkins_samples


def save_info(args, sentences, embs_ini, neg_user_samples, neg_checkins_samples, train_user_checkins):
    input_dir = "temp/processed/"
    if not os.path.isdir(input_dir):
        os.makedirs(input_dir)
    print("Write walks")
    with open(f"{input_dir}/walk.txt", "w+") as fp:
        fp.write(f"{len(sentences)} {args.walk_length}\n")
        for sent in sentences:
            fp.write(" ".join(map(str, sent)) + "\n")

    print("Write user_checkins")
    with open(f"{input_dir}/user_checkins.txt", "w+") as fp:
        fp.write(f"{len(train_user_checkins)}\n") # num users
        for id in sorted(train_user_checkins.keys()):
            checkins = train_user_checkins[id]
            fp.write(f"{checkins.shape[0]}\n")
            for checkin in checkins:
                fp.write(" ".join(map(str, checkin)) + "\n")

    print("Write embs_ini")
    with open(f"{input_dir}/embs_ini.txt", "w+") as fp:
        fp.write(f"{embs_ini.shape[0]} {embs_ini.shape[1]}\n") # num users
        for emb in embs_ini:
            fp.write(" ".join([f"{x:.5f}" for x in emb]) + "\n")

    print("Write neg_user_samples")
    with open(f"{input_dir}/neg_user_samples.txt", "w+") as fp:
        fp.write(f"{neg_user_samples.shape[0]}\n") # num users
        for neg in neg_user_samples:
            fp.write(f"{neg}\n")

    print("Write neg_checkins_samples")
    with open(f"{input_dir}/neg_checkins_samples.txt", "w+") as fp:
        keys = sorted(neg_checkins_samples.keys())
        for key in keys:
            neg_table = neg_checkins_samples[key]
            fp.write(f"{neg_table.shape[0]}\n")
            fp.write("\n".join(map(str, neg_table)) + "\n")


def renumber_checkins(checkins_matrix, maps_PtOri=None):
    offset1 = max(checkins_matrix[:,0])
    if maps_PtOri is not None:
        offset1 = max(offset1, len(maps_PtOri))
    _, n = np.unique(checkins_matrix[:,1], return_inverse=True) # 
    checkins_matrix[:,1] = n + offset1 + 1
    offset2 = max(checkins_matrix[:,1])
    _, n = np.unique(checkins_matrix[:,2], return_inverse=True)
    checkins_matrix[:,2] = n + offset2 + 1
    offset3 = max(checkins_matrix[:,2])
    _, n = np.unique(checkins_matrix[:,3], return_inverse=True)
    checkins_matrix[:,3] = n + offset3 + 1
    n_nodes_total = np.max(checkins_matrix)
    n_users = offset1


    if min(checkins_matrix[:,0]) == 0:
        offset1 += 1
        offset2 += 1
        offset3 += 1
        n_users += 1
        n_nodes_total += 1
    print("N_users: {}, N_total: {}".format(n_users, n_nodes_total))
    
    return checkins_matrix, offset1, offset2, offset3, n_nodes_total, n_users

