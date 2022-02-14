"""Splitter Class."""

import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import trange
from walkers import DeepWalker
from ego_splitting import EgoNetSplitter
import networkx as nx
import traceback

class Splitter(torch.nn.Module):
    """
    An implementation of "Splitter: Learning Node Representations
    that Capture Multiple Social Contexts" (WWW 2019).
    Paper: http://epasto.org/papers/www2019splitter.pdf
    """
    def __init__(self, args, base_node_count, node_count):
        """
        Splitter set up.
        :param args: Arguments object.
        :param base_node_count: Number of nodes in the source graph.
        :param node_count: Number of nodes in the persona graph.
        """
        super(Splitter, self).__init__()
        self.args = args
        self.base_node_count = base_node_count
        self.node_count = node_count

    def create_weights(self):
        """
        Creating weights for embedding.
        """
        self.base_node_embedding = torch.nn.Embedding(self.base_node_count,
                                                      self.args.dimensions,
                                                      padding_idx=0)

        self.node_embedding = torch.nn.Embedding(self.node_count,
                                                 self.args.dimensions,
                                                 padding_idx=0)

        self.node_noise_embedding = torch.nn.Embedding(self.node_count,
                                                       self.args.dimensions,
                                                       padding_idx=0)

    def initialize_weights(self, base_node_embedding, mapping):
        """
        Using the base embedding and the persona mapping for initializing the embeddings.
        :param base_node_embedding: Node embedding of the source graph.
        :param mapping: Mapping of personas to nodes.
        """
        persona_embedding = np.array([base_node_embedding[n] for _, n in mapping.items()])
        self.node_embedding.weight.data = torch.nn.Parameter(torch.Tensor(persona_embedding))
        self.node_noise_embedding.weight.data = torch.nn.Parameter(torch.Tensor(persona_embedding))
        self.base_node_embedding.weight.data = torch.nn.Parameter(torch.Tensor(base_node_embedding),
                                                                  requires_grad=False)

    def calculate_main_loss(self, sources, contexts, targets):
        """
        Calculating the main embedding loss.
        :param sources: Source node vector.
        :param contexts: Context node vector.
        :param targets: Binary target vector.
        :return main_loss: Loss value.
        """
        node_f = self.node_embedding(sources)
        node_f = torch.nn.functional.normalize(node_f, p=2, dim=1)
        feature_f = self.node_noise_embedding(contexts)
        feature_f = torch.nn.functional.normalize(feature_f, p=2, dim=1)
        scores = torch.sum(node_f*feature_f, dim=1)
        scores = torch.sigmoid(scores)
        main_loss = targets*torch.log(scores)+(1-targets)*torch.log(1-scores)
        main_loss = -torch.mean(main_loss)
        return main_loss

    def calculate_regularization(self, pure_sources, personas):
        """
        Calculating the regularization loss.
        :param pure_sources: Source nodes in persona graph.
        :param personas: Context node vector.
        :return regularization_loss: Loss value.
        """
        source_f = self.node_embedding(pure_sources)
        original_f = self.base_node_embedding(personas)
        scores = torch.clamp(torch.sum(source_f*original_f, dim=1), -15, 15)
        scores = torch.sigmoid(scores)
        regularization_loss = -torch.mean(torch.log(scores))
        return regularization_loss

    def forward(self, sources, contexts, targets, personas, pure_sources):
        """
        Doing a forward pass.
        :param sources: Source node vector.
        :param contexts: Context node vector.
        :param targets: Binary target vector.
        :param pure_sources: Source nodes in persona graph.
        :param personas: Context node vector.
        :return loss: Loss value.
        """
        main_loss = self.calculate_main_loss(sources, contexts, targets)
        regularization_loss = self.calculate_regularization(pure_sources, personas)
        loss = main_loss + self.args.lambd*regularization_loss
        return loss

class SplitterTrainer(object):
    """
    Class for training a Splitter.
    """
    def __init__(self, graph,graph_friend,listPOI,mat,location_Dict, args):
        """
        :param graph: NetworkX graph object.
        :param args: Arguments object.
        """
        self.graph = graph
        self.graph_friend = graph_friend
        self.listPOI = listPOI
        self.selected_checkins = mat['selected_checkins']
        self.location_Dict = location_Dict
        self.args = args
        self.phase = args.phase
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.category_POI()

    def renumber_checkins(self,checkins_matrix):
        offset1 = max(checkins_matrix[:, 0])
        _, n = np.unique(checkins_matrix[:, 1], return_inverse=True)  #
        checkins_matrix[:, 1] = n + offset1 + 1
        offset2 = max(checkins_matrix[:, 1])
        _, n = np.unique(checkins_matrix[:, 2], return_inverse=True)
        checkins_matrix[:, 2] = n + offset2 + 1
        offset3 = max(checkins_matrix[:, 2])
        _, n = np.unique(checkins_matrix[:, 3], return_inverse=True)
        checkins_matrix[:, 3] = n + offset3 + 1
        n_nodes_total = np.max(checkins_matrix)
        n_users = checkins_matrix[:, 0].max()

        print(f"""Number of users: {n_users}
            Number of nodes total: {n_nodes_total}""")
        return checkins_matrix
    def category_POI(self): # tạo dict để phân lớp các POI
        self.POI_category_dict = dict()
        selected_checkins_sort = self.selected_checkins[self.selected_checkins[:, 1].argsort()]

        #print(self.selected_checkins)
        for i in self.selected_checkins:
            position = self.location_Dict[i[2]]
            category = i[3]
            if position in self.POI_category_dict:
                self.POI_category_dict[position].append(category)
            else:
                self.POI_category_dict[position] = [category]
        self.choose_20_per_del = []
        for i in range(int(0.8*len(selected_checkins_sort)), len(selected_checkins_sort)):
            friend = selected_checkins_sort[i][0]
            location = selected_checkins_sort[i][2]
            self.choose_20_per_del.append((friend,self.location_Dict[location]))
        #print(self.choose_20_per_del)

    def create_noises(self):
        """
        Creating node noise distribution for negative sampling.
        """
        self.downsampled_degrees = {}
        for n in self.egonet_splitter.persona_graph.nodes():
            self.downsampled_degrees[n] = int(1+self.egonet_splitter.persona_graph.degree(n)**0.75)
        self.noises = [k for k, v in self.downsampled_degrees.items() for i in range(v)]

    def base_model_fit(self):
        """
        Fitting DeepWalk on base model.
        """
        # print(self.graph.nodes)
        self.base_walker = DeepWalker(self.graph, self.args)
        print("\nDoing base random walks.\n")
        self.base_walker.create_features()
        print("\nLearning the base model.\n")
        self.base_node_embedding = self.base_walker.learn_base_embedding()
        print("\nDeleting the base walker.\n")
        del self.base_walker

    def create_split(self):
        """
        Creating an EgoNetSplitter.
        """
        self.egonet_splitter = EgoNetSplitter()
        print("Number node of origin graph  : ",len(self.graph.nodes))
        self.egonet_splitter.fit(self.graph_friend,self.listPOI)
        # print(self.egonet_splitter.persona_graph_edges)
        # print(self.egonet_splitter.persona_graph)
        # import pdb
        # pdb.set_trace()
        self.list_friend = self.graph_friend.nodes
        persona_map = self.egonet_splitter.personality_map

        # Graph persona của friend
        friend_subgraph = self.egonet_splitter.persona_graph
        # Xóa cạnh friend-friend đi
        # còn lại là graph giữa friend- Position
        friend_POI_graph = self.graph.copy()
        friend_POI_graph.remove_edges_from([i for i in self.graph_friend.edges])
        
        print("Số lượng  cạnh graph ban đầu  friend_POI_graph full: ", len(friend_POI_graph.edges))

        friend_POI_graph.remove_edges_from(self.choose_20_per_del)
        print("Số lượng  cạnh graph ban đầu  friend_POI_graph lay 80% : ", len(friend_POI_graph.edges))

        edges_list = friend_subgraph.edges
        nodes_list = friend_subgraph.nodes
        id = 0
        continue_map = {}
        for i in nodes_list:
            continue_map[i] = id
            id +=1
        edges_continue = [[continue_map[edge[0]],continue_map[edge[1]]] for edge in edges_list]
        persona_graph_continue = nx.from_edgelist(edges_continue)
        persona_map_continue = {continue_map[n]: persona_map[n] for n in nodes_list }
        persona_reverse_map_continue = {}
        edgelistPOI = []            # để lưu ra file, chứa cặp node (persona - POI)
        for i in persona_map_continue:
            if persona_map_continue[i] not in persona_reverse_map_continue:
                persona_reverse_map_continue[persona_map_continue[i]] = [i]
            else:
                persona_reverse_map_continue[persona_map_continue[i]].append(i)
        # while len(friend_POI_graph.edges) >0:
        if self.phase >= 2:
            friend_position_dict = dict()       # Taoj dict cho P2
            for e1,e2 in friend_POI_graph.edges:
                # print(e1,e2)
                if e1 in self.list_friend and e2 in self.listPOI:
                    e_friend = e1
                    e_pos = e2
                elif e2 in self.list_friend and e1 in self.listPOI:
                    e_friend = e2
                    e_pos = e1
                else:
                    print("cạnh này bị lỗi e1 : ",e1,"   , e2:  ",e2)
                    continue
                if e_friend not in friend_position_dict:
                    friend_position_dict[e_friend] = [e_pos]
                else:
                    friend_position_dict[e_friend].append(e_pos)
            print("checkin ban đầu  : ",len(friend_POI_graph.edges))
            #######################################
            # Tạo graph các POI ở chung với nhau trong graph ban đầu
            ## Du lieu tu graph ban dau
            #  For P2

            graph_POI_POI = nx.Graph()      # graph nối các POI ở cùng 1 persona
            for friend_node in friend_position_dict:
                e_poses = friend_position_dict[friend_node]
                for e1 in range(len(e_poses)):
                    for e2 in range(e1+1,len(e_poses)):
                        node1 = e_poses[e1]
                        node2 = e_poses[e2]
                        if graph_POI_POI.has_edge(node1,node2):
                            w = graph_POI_POI.get_edge_data(node1,node2)['weight']
                            # print(w)
                            graph_POI_POI.add_edge(node1,node2, weight=w+1)
                        else:
                            graph_POI_POI.add_edge(node1,node2,weight = 1)
            print("tạo xong graph POI-POI")
            del friend_position_dict

        graph_POI_persona = nx.Graph()      # graph chứa các cạnh nối từ POI -> persona
        if self.phase >= 1:
            print(" Bat dau P1 ")
            for e1,e2 in friend_POI_graph.edges:
                # print(e1,e2)
                if e1 in self.list_friend and e2 in self.listPOI:
                    e_friend = e1
                    e_pos = e2
                elif e2 in self.list_friend and e1 in self.listPOI:
                    e_friend = e2
                    e_pos = e1
                else:
                    print("cạnh này bị lỗi e1 : ",e1,"   , e2:  ",e2)
                    continue
                node_persona_respective = persona_reverse_map_continue[e_friend]
                have_edge = False
                # Tìm node perona có chung position
                ### Collocation
                for persona_node in node_persona_respective:
                    neighbor_nodes = persona_graph_continue.neighbors(persona_node)
                    for neighbor_node in neighbor_nodes:
                        if self.graph.has_edge(persona_map_continue[neighbor_node],e_pos) or self.graph.has_edge(e_pos,persona_map_continue[neighbor_node]):
                            have_edge_1 = False
                            have_edge_2 = False
                            if [persona_node,e_pos] in edgelistPOI:
                                have_edge_1 = True
                            if [neighbor_node,e_pos] in edgelistPOI:
                                have_edge_2 = True
                            if not have_edge_1:
                                edgelistPOI.append([persona_node,e_pos])
                                graph_POI_persona.add_edge(persona_node, e_pos)

                            if not have_edge_2:
                                edgelistPOI.append([neighbor_node,e_pos])
                                graph_POI_persona.add_edge(neighbor_node, e_pos)

        ## Xóa cạnh friend - pos đã được nối với nhau
        print(" Xoa  ",len(edgelistPOI),"  da ton tai")
        for del_edge in edgelistPOI:
            persona, e_pos = del_edge[0],del_edge[1]
            e_friend = persona_map_continue[persona]
            friend_POI_graph.remove_edges_from([(e_friend, e_pos)])
            friend_POI_graph.remove_edges_from([(e_pos, e_friend)])
        print(" Xong P1 ")
        print("Số lượng  cạnh graph ban đầu  friend_POI_graph sau P1 : ", len(friend_POI_graph.edges))
        print("edgelistPOI   : ",len(edgelistPOI))

        ########################################
        # P2
        # Ghép các position hay ở chung với nhau
        if self.phase >= 2 :
            print("Bat dau P2")
            ### P2 ghép các position đã từng ở chung với nhau
            for e1, e2 in friend_POI_graph.edges:
                if e1 in self.list_friend and e2 in self.listPOI:
                    e_friend = e1
                    e_pos = e2
                elif e2 in self.list_friend and e1 in self.listPOI:
                    e_friend = e2
                    e_pos = e1
                else:
                    print("cạnh này bị lỗi e1 : ",e1,"   , e2:  ",e2)
                    continue
                try:
                    # node_pos_friend = list(graph_POI_POI.neighbors(e_pos)).copy()
                    ## Tìm tất cả các position ở chung
                    ## Xếp theo các node có weight cao đến thấp
                    ### weight trọng số cao tức là 2 cái position liên quan đến nhau
                    node_pos_friend =  sorted(graph_POI_POI[e_pos].items(), key=lambda edge: edge[1]['weight'],reverse=True)

                except Exception as e:
                    # Random make persona-checkpoint
                    # Nếu không có position ở chung nào thì bỏ
                    # traceback.print_tb(e.__traceback__)
                    continue
                # Duyệt Tất cả node persona tương ứng với node user e_friend
                node_persona_respective = persona_reverse_map_continue[e_friend]
                have_edge = False
                for pos, weight in node_pos_friend:                                 # duyệt qua các position hay đi cùng
                    for persona_node in node_persona_respective:                    #duyệt qua các persona node
                        # print("weight :  ", graph_POI_POI.get_edge_data(e_pos, pos))
                        if graph_POI_persona.has_edge(pos,persona_node):
                            # Nếu có 1 persona nối với 1 position hay đi
                            # thì nối luôn e_pos kia vào
                            # print("có cạnh nè weight :  ",graph_POI_POI.get_edge_data(e_pos,pos))
                            edgelistPOI.append([persona_node, e_pos])
                            friend_POI_graph.remove_edges_from([(e1,e2)])          # Xóa pos đã được gán
                            have_edge = True
                            graph_POI_persona.add_edge(persona_node, e_pos)
                            # thêm cái POI mới vào danh sách
                            break
                        if have_edge:
                            continue
                    if have_edge:
                        continue

            print("xong P2")
            print("Số lượng  cạnh graph ban đầu  friend_POI_graph sau P2 : ", len(friend_POI_graph.edges))
            print("edgelistPOI   : ", len(edgelistPOI))
        #########################################
        ## Tạo graph các POI cùng lớp
        ## For P3
        if self.phase >= 3:
            graph_persona_class = nx.Graph()  # graph nối các POI ở cùng 1 class
            # print(len(self.category_POI_dict))
            for ed in edgelistPOI:
                persona_node, e_pos = ed[0],ed[1]
                clses = self.POI_category_dict[e_pos]
                # print(len(e_poses))
                # continue
                for cls in clses:
                    if graph_persona_class.has_edge(persona_node, cls):
                        w = graph_persona_class.get_edge_data(persona_node, cls)['weight'] ## Mức độ liên quan giữa 2 cái position
                        # print(w)
                        graph_persona_class.add_edge(persona_node, cls, weight=w + 1)
                    else:
                        graph_persona_class.add_edge(persona_node, cls, weight=1)
            print("tạo xong graph POI-class")

            ###############################
            # P3 Ghép các POI có cùng cls
            print("Bat dau P3")
            for e1, e2 in friend_POI_graph.edges:
                if e1 in self.list_friend and e2 in self.listPOI:
                    e_friend = e1
                    e_pos = e2
                elif e2 in self.list_friend and e1 in self.listPOI:
                    e_friend = e2
                    e_pos = e1
                else:
                    print("cạnh này bị lỗi e1 : ",e1,"   , e2:  ",e2)
                    continue
                try:
                    # node_pos_friend = list(graph_POI_POI.neighbors(e_pos)).copy()
                    cls_poses =  self.POI_category_dict[e_pos]
                except Exception as e:
                    traceback.print_tb(e.__traceback__)
                    continue
                # Tất cả node persona tương ứng với node user e_friend
                node_persona_respective = persona_reverse_map_continue[e_friend]
                persona_node_choose = None
                w_max = 0
                for persona_node in node_persona_respective:    #duyệt qua các persona node
                    for cls_pos in cls_poses:               # duyệt qua các class cua position do
                        if graph_persona_class.has_edge(persona_node,cls_pos):
                            # Nếu persona da tung noi voi class
                            w = graph_persona_class.get_edge_data(persona_node, cls_pos)['weight']
                            if w> w_max:
                                w_max = w
                                persona_node_choose = persona_node
                if persona_node_choose != None:
                    edgelistPOI.append([persona_node_choose, e_pos])
                    friend_POI_graph.remove_edges_from([(e_friend,e_pos)])
                    graph_POI_persona.add_edge(persona_node_choose, e_pos)
            print("xong P3")
            print("Số lượng  cạnh graph ban đầu  friend_POI_graph sau P3 : ", len(friend_POI_graph.edges))
            print("edgelistPOI   : ", len(edgelistPOI))
        ########################################################
        ### Khi vẫn còn dư  poi
        ### Nối bừa : random POI vào persona
        print(" Bat dau gan random")
        for e1, e2 in friend_POI_graph.edges:
            if e1 in self.list_friend and e2 in self.listPOI:
                e_friend = e1
                e_pos = e2
            elif e2 in self.list_friend and e1 in self.listPOI:
                e_friend = e2
                e_pos = e1
            else:
                print("cạnh này bị lỗi e1 : ", e1, "   , e2:  ", e2)
                continue
            persona_node = random.choice(persona_reverse_map_continue[e_friend])  # chọn bừa 1 persona node
            edgelistPOI.append([persona_node, e_pos])  # nối position-persona
            friend_POI_graph.remove_edges_from([(e_friend, e_pos)])
            graph_POI_persona.add_edge(persona_node, e_pos)

        print("Xong gán  random ")
        print("Số lượng  cạnh graph ban đầu  friend_POI_graph sau gán random  : ", len(friend_POI_graph.edges))
        print("edgelistPOI   : ", len(edgelistPOI))
        print("graph_POI_persona num edges   : ", len(graph_POI_persona.edges))
        # exit()
        print("Number node of persona graph  : ",len(persona_graph_continue.nodes))
        print("splitter number_connected_cmponents continue graph   :  ", nx.number_connected_components(persona_graph_continue))

        with open('Suhi_output/edgelistPOI_{}_{}'.format(self.args.lbsn,self.phase), 'w', encoding='utf-8') as file:
            for e1, e2 in edgelistPOI:
                # e1 persona node
                # e2 position node
                file.write('{},{}\n'.format(e1, e2))
        with open('Suhi_output/ego_net_{}_{}'.format(self.args.lbsn,self.phase), 'w', encoding='utf-8') as file:
            for key, value in persona_map_continue.items():
                file.write('{},{}\n'.format(key, value))

        # nx.write_edgelist(self.egonet_splitter.persona_graph, 'Suhi_output/edgelist_{}'.format(self.args.lbsn))
        nx.write_edgelist(persona_graph_continue, 'Suhi_output/edgelist_{}_{}'.format(self.args.lbsn,self.phase))

        print("DONE!, I'm in spliter.py, line 147")
        exit()
        # import pdb
        # pdb.set_trace()
        self.persona_walker = DeepWalker(self.egonet_splitter.persona_graph, self.args)
        print("\nDoing persona random walks.\n")
        self.persona_walker.create_features()
        self.create_noises()

    def setup_model(self):
        """
        Creating a model and doing a transfer to GPU.
        """
        base_node_count = self.graph.number_of_nodes()
        persona_node_count = self.egonet_splitter.persona_graph.number_of_nodes()
        self.model = Splitter(self.args, base_node_count, persona_node_count)
        self.model.create_weights()
        self.model.initialize_weights(self.base_node_embedding,
                                      self.egonet_splitter.personality_map)
        self.model = self.model.to(self.device)

    def transfer_batch(self, source_nodes, context_nodes, targets, persona_nodes, pure_source_nodes):
        """
        Transfering the batch to GPU.
        """
        self.sources = torch.LongTensor(source_nodes).to(self.device)
        self.contexts = torch.LongTensor(context_nodes).to(self.device)
        self.targets = torch.FloatTensor(targets).to(self.device)
        self.personas = torch.LongTensor(persona_nodes).to(self.device)
        self.pure_sources = torch.LongTensor(pure_source_nodes).to(self.device)

    def optimize(self):
        """
        Doing a weight update.
        """
        loss = self.model(self.sources, self.contexts,
                          self.targets, self.personas, self.pure_sources)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def process_walk(self, walk):
        """
        Process random walk (source, context) pairs.
        Sample negative instances and create persona node list.
        :param walk: Random walk sequence.
        """
        left_nodes = [walk[i] for i in range(len(walk)-self.args.window_size) for j in range(1, self.args.window_size+1)]
        right_nodes = [walk[i+j] for i in range(len(walk)-self.args.window_size) for j in range(1, self.args.window_size+1)]
        node_pair_count = len(left_nodes)
        source_nodes = left_nodes + right_nodes
        context_nodes = right_nodes + left_nodes
        persona_nodes = np.array([self.egonet_splitter.personality_map[source_node] for source_node in source_nodes])
        pure_source_nodes = np.array(source_nodes)
        source_nodes = np.array((self.args.negative_samples+1)*source_nodes)
        noises = np.random.choice(self.noises, node_pair_count*2*self.args.negative_samples)
        context_nodes = np.concatenate((np.array(context_nodes), noises))
        positives = [1.0 for node in range(node_pair_count*2)]
        negatives = [0.0 for node in range(node_pair_count*self.args.negative_samples*2)]
        targets = np.array(positives + negatives)
        self.transfer_batch(source_nodes, context_nodes, targets, persona_nodes, pure_source_nodes)

    def update_average_loss(self, loss_score):
        """
        Updating the average loss and the description of the time remains bar.
        :param loss_score: Loss on the sample.
        """
        self.cummulative_loss = self.cummulative_loss + loss_score
        self.steps = self.steps + 1
        average_loss = self.cummulative_loss/self.steps
        self.walk_steps.set_description("Splitter (Loss=%g)" % round(average_loss, 4))

    def reset_average_loss(self, step):
        """
        Doing a reset on the average loss.
        :param step: Current number of walks processed.
        """
        if step % 100 == 0:
            self.cummulative_loss = 0
            self.steps = 0

    def fit(self):
        """
        Fitting a model.
        """
        # self.base_model_fit()
        self.create_split()
        self.setup_model()
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer.zero_grad()
        print("\nLearning the joint model.\n")
        random.shuffle(self.persona_walker.paths)
        self.walk_steps = trange(len(self.persona_walker.paths), desc="Loss")
        for step in self.walk_steps:
            self.reset_average_loss(step)
            walk = self.persona_walker.paths[step]
            self.process_walk(walk)
            loss_score = self.optimize()
            self.update_average_loss(loss_score)

    def save_embedding(self):
        """
        Saving the node embedding.
        """
        print("\n\nSaving the model.\n")
        nodes = [node for node in self.egonet_splitter.persona_graph.nodes()]
        nodes.sort()
        nodes = torch.LongTensor(nodes).to(self.device)
        embedding = self.model.node_embedding(nodes).cpu().detach().numpy()
        embedding_header = ["id"] + ["x_" + str(x) for x in range(self.args.dimensions)]
        embedding = [np.array(range(embedding.shape[0])).reshape(-1, 1), embedding]
        embedding = np.concatenate(embedding, axis=1)
        embedding = pd.DataFrame(embedding, columns=embedding_header)
        embedding.to_csv(self.args.embedding_output_path, index=None)

    def save_persona_graph_mapping(self):
        """
        Saving the persona map.
        """
        with open(self.args.persona_output_path, "w") as f:
            json.dump(self.egonet_splitter.personality_map, f)                     
