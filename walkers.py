import numpy as np 
import networkx as nx 
import torch 
import multiprocessing

from random_walks import RandomWalk

# karate_g = nx.read_edgelist('../Random-Walk/graph/karate.edgelist')

# random_walk = RandomWalk(karate_g, walk_length=3, num_walks=10, p=1, q=1, workers=6)

# walklist = random_walk.walks



class BasicWalker:
    def __init__(self, G, start_nodes=None, user_poi_dict={}, bias=False, center_ori_maps=None, alpha=0.1, beta=0.1):
        self.G = G
        if hasattr(G, 'neibs'):
            self.neibs = G.neibs
        else:
            self.build_neibs_dict()
        if start_nodes is not None:
            self.start_nodes = start_nodes
        else:
            self.start_nodes = list(self.G.nodes())
        
        self.user_poi_dict = user_poi_dict
        self.bias = bias
        self.center_ori_maps = center_ori_maps
        self.alpha = alpha
        self.beta = beta


    def build_neibs_dict(self):
        self.neibs = {}
        for node in self.G.nodes():
            self.neibs[node] = list(self.G.neighbors(node))


    def simulate_walks(self, num_walks, walk_length, num_workers):
        pool = multiprocessing.Pool(processes=num_workers)
        walks = []
        print('Walk iteration:')
        nodes = self.start_nodes
        nodess = [np.random.shuffle(nodes)]
        for i in range(num_walks):
            _ns = nodes.copy()
            np.random.shuffle(_ns)
            nodess.append(_ns)
        params = list(map(lambda x: {'walk_length': walk_length, 'neibs': self.neibs, 'iter': x, \
                'nodes': nodess[x], 'bias': self.bias, 'user_poi_dict': self.user_poi_dict, \
                'center_ori_maps': self.center_ori_maps, 'alpha': self.alpha, 'beta': self.beta},
            list(range(1, num_walks+1))))
        
        walks = pool.map(deepwalk_walk, params)
        pool.close()
        pool.join()
        # walks = np.vstack(walks)
        while len(walks) > 1:
            walks[-2] = walks[-2] + walks[-1]
            walks = walks[:-1]
        walks = walks[0]
        return walks


def deepwalk_walk_wrapper(class_instance, walk_length, start_node):
    class_instance.deepwalk_walk(walk_length, start_node)


def deepwalk_walk(params):
    '''
    Simulate a random walk starting from start node.
    '''
    bias = params["bias"]
    user_poi_dict = params["user_poi_dict"]
    walk_length = params["walk_length"]
    neibs = params["neibs"]
    nodes = params["nodes"]
    center_ori_maps = params["center_ori_maps"]
    alpha = params["alpha"]
    beta = params["beta"]
    # if args["iter"] % 5 == 0:
    print("Iter:", params["iter"]) # keep printing, avoid moving process to swap

    walks = []
    for node in nodes:
        walk = [node]
        if len(neibs[node]) == 0:
            walks.append(walk)
            continue
        while len(walk) < walk_length:
            cur = int(walk[-1])
            cur_nbrs = neibs[cur]
            if len(cur_nbrs) == 0: break
            if not bias:
                walk.append(np.random.choice(cur_nbrs))
            else:
                walk.append(bias_walk(cur, cur_nbrs, user_poi_dict, center_ori_maps, alpha, beta))
        walks.append(walk)
    return walks


def bias_walk(cur, cur_nbrs, user_poi_dict, center_ori_maps,alpha=0.1, beta=0.1):
    """

    """

    thresh = min(list(center_ori_maps.keys()))
    this_poi = user_poi_dict[cur + 1]
    prob = []
    center = None

    prob_prev = []
    for i in range(len(cur_nbrs)):
        nb = cur_nbrs[i]
        nb_poi = user_poi_dict[nb + 1]
        if nb + 1 > thresh:
            continue 
        common = nb_poi.intersection(this_poi)
        union = nb_poi.union(this_poi)
        if len(union) == 0:
            prob_prev.append(0)
        else:
            prob_prev.append(len(common) / len(union))
    if len(prob_prev):
        mean_prob = np.mean(prob_prev)
    else:
        mean_prob = 0


    for i in range(len(cur_nbrs)):
        nb = cur_nbrs[i]
        nb_poi = user_poi_dict[nb + 1]
        if nb + 1 >= thresh:
            center = i
            prob.append(mean_prob / alpha)
            continue
        common = nb_poi.intersection(this_poi)
        union = nb_poi.union(this_poi)
        if len(union) == 0:
            prob.append(0)
        else:
            prob.append(len(common) / len(union))
    
    prob = np.array(prob, dtype=float)
    if np.max(prob) == 0:
        prob += 1
    if np.min(prob) == 0:
        if len(prob) == 1:
            prob += 1
        else:
            second_min = np.partition(prob, 1)[1]
            prob += second_min / 2
    
    prob /= prob.sum()
    return np.random.choice(cur_nbrs, p=prob)