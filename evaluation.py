import scipy.io
from scipy.io import loadmat
import numpy as np 
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def normalize_embedding(emb):
    normalize_factor = np.sqrt((emb ** 2).sum(axis=1))
    return emb / normalize_factor.reshape(-1, 1)


def friendship_pred_ori(embs, friendship_old, friendship_new, k=10):
    """
    Simplest Linkprediction Evaluation
    embs: user embeddings
    friendship_old: old friendship: node_id >= 1
    friendship_new: new friendship: node_id >= 1
    """

    friendship_old -= 1
    friendship_new -= 1
    ################# compute simi matrix #################
    num_users = embs.shape[0]
    normalize_embs = normalize_embedding(embs)
    simi_matrix = normalize_embs.dot(normalize_embs.T)
    #######################################################
    

    ################# preprocess simi matrix #########################
    for i in range(num_users):
        simi_matrix[i, i] = -2
    
    for i in range(friendship_old.shape[0]):
        simi_matrix[friendship_old[i, 0], friendship_old[i, 1]] = -2
    ################################################################
    
    # argsort
    arg_sorted_simi = simi_matrix.argsort(axis=1)
    
    ################# create friend_dict: node-> set of fiends #############
    friend_dict = dict()
    for i in range(friendship_new.shape[0]):
        source, target = friendship_new[i][0], friendship_new[i][1]
        if source not in friend_dict:
            friend_dict[source] = set([target])
        else:
            friend_dict[source].add(target)
    ########################################################################

    ###################### evaluate #########################
    print("ALL DCG:...")
    for kk in [10, 20, 50, 100, 200]:
        precision_k, recall_k, ndcg = compute_precision_recall(friend_dict, arg_sorted_simi, kk)
        f1_k = 2 * precision_k * recall_k / (precision_k + recall_k)

        # print(f"Precision@{kk}: {precision_k:.3f}")
        # print(f"Recall@{kk}: {recall_k:.3f}")
        # print("{:.4f}|{:.4f}".format(precision_k, recall_k))
        print("{:4f}".format(ndcg))
        # print("DCG{}: {:.4f}".format(kk, ndcg))
        # print(f"F1@{kk}: {f1_k:.3f}")
    #########################################################

def compute_idcg(num):
    ret = 0
    for i in range(num):
        ret += 1 / np.log2(i + 2)
    return ret


def compute_precision_recall(friend_dict, arg_sorted_simi, k):
    """
    DCG_k  = sum(1/log2(1 + rank))
    IDCG_k = 
    """
    precision = []
    recall = []
    rank_scores = []
    # rank_score_norm = []
    for key, value in friend_dict.items():
        n_relevants = 0
        this_rank_score = 0
        arg_simi_key = arg_sorted_simi[key][-k:]
        for target_node in value:
            for rk in range(1, len(arg_sorted_simi[key]) + 1):
                if arg_sorted_simi[key][-rk] == target_node:
                    rank = 1 / np.log2(rk + 1) 
                    break
            if target_node in arg_simi_key:
                n_relevants += 1
                this_rank_score += rank
        
        if n_relevants == 0:
            rank_scores.append(0)
        else:
            idcg = compute_idcg(n_relevants)
            rank_scores.append(this_rank_score / idcg)
        
        precision.append(n_relevants/k)
        recall.append(n_relevants/len(value))
    
    precision = np.mean(precision)
    recall = np.mean(recall)
    ndcg = np.mean(rank_scores)
    return precision, recall, ndcg


def friendship_pred_persona(embs_user, friendship_old_ori, friendship_new, k=10, maps_OritP=None, maps_PtOri=None):
    normalized_embs_user = normalize_embedding(embs_user)
    simi_matrix = normalized_embs_user.dot(normalized_embs_user.T)
    for i in range(len(simi_matrix)):
        simi_matrix[i, i] = -2

    # from sklearn.metrics import dcg_score
    for i in range(len(friendship_old_ori)):
        friendship_i = friendship_old_ori[i]
        source, target = friendship_i[0], friendship_i[1]
        group_source = maps_OritP[source]
        group_target = maps_OritP[target]
        for persona_s in group_source:
            for persona_t in group_target:
                simi_matrix[persona_s - 1, persona_t - 1] = -2
                simi_matrix[persona_t - 1, persona_s - 1] = -2
                
    arg_sorted_simi = simi_matrix.argsort(axis=1)

    # is a dictionary of source and list of neighbors of source
    friend_dict = dict()
    
    label_matrix = np.zeros(simi_matrix.shape)
    
    for i in range(len(friendship_new)):
        source, target = friendship_new[i][0], friendship_new[i][1]
        label_matrix[source, target] = 1
        if source not in friend_dict:
            friend_dict[source] = set([target])
        else:
            friend_dict[source].add(target)

    def is_match(ordered_candidates, target_gr, kk):
        """
        check wether there is a match between an item in top kk in ordered_candidates and target_gr
        """
        group = []
        count = 0
        for i in range(1, len(ordered_candidates)):
            target_index = ordered_candidates[-i] + 1
            group_target_index = maps_PtOri[target_index]
            if group_target_index not in group:
                group.append(group_target_index)
                count += 1
                if count == kk + 1:
                    # no candidate found!
                    break
            if target_index in target_gr:
                return 1, i
        return 0, None
    
    # def find_rank(ordered_candidates, target_gr):
    #     group = []
    #     count = 0
    #     for i in range(1, len(ordered_candidates)):
    #         target_index = ordered_candidates[-i] + 1
    #         group_target_index = maps_PtOri[target_index]
    #         if group_target_index not in group:
    #             group.append(group_target_index)
    #         if target_index in target_gr:
    #             return 1 / np.log2(i + 1)
    #     return 1 / np.log2(i + 1)


    def find_rank(ordered_candidates, target_friend, maps_PtOri, kk):
        rank = 0
        seen = set()
        for i in range(1, len(ordered_candidates) + 1):
            this_persona = ordered_candidates[-i] + 1
            this_ori = maps_PtOri[this_persona]
            if this_ori not in seen:
                rank += 1
                seen.add(this_ori)
            if rank > kk:
                return 0, 0
            if this_ori == target_friend:
                return rank, 1
        return 0, 0

    def cal_precision_recall_k(kk):
        precision = []
        recall = []
        dcg = []
        for user, friends_list in friend_dict.items():
            n_relevants = 0
            source_group = maps_OritP[user] # group of persona nodes of this user
            target_groups = [maps_OritP[fr] for fr in friends_list]
            
            local_ndcg = []
            for persona_s in source_group:
                t_dcg = 0
                this_n_relevants = 0
                ordered_candidates = arg_sorted_simi[persona_s - 1]
                for target_friend in friends_list:
                    rank, relevant = find_rank(ordered_candidates, target_friend, maps_PtOri, kk)
                    if relevant > 0:
                        this_n_relevants += 1
                        t_dcg += 1 / np.log2(rank + 1)

                if this_n_relevants == 0:
                    local_ndcg.append(0)
                else:
                    ti_dcg = compute_idcg(this_n_relevants)
                    this_ndcg = t_dcg / ti_dcg 
                    local_ndcg.append(this_ndcg)

                for j in range(len(target_groups)):
                    ttrue, rk = is_match(ordered_candidates, target_groups[j], kk)
                    if ttrue:
                        n_relevants += 1
                        target_groups[j] = []
            
            dcg.append(max(local_ndcg))

            precision.append(n_relevants/kk)
            recall.append(n_relevants/len(friends_list))
            # idcg = compute_idcg(n_relevants)
            # dcg.append(rank_score / idcg)
            # dcg.append(rank_score / rank_score_norm)
        precision = np.mean(precision)
        recall = np.mean(recall)
        dcg = np.mean(dcg)
        f1 = 2 * precision * recall / (precision + recall)

        # print("Precision@k | Recal@k {:.4f}|{:.4f}".format(precision, recall))
        # print("DCG@k: {:.4f}".format(dcg))
        print("{:.4f}".format(dcg))
    
    print("ALL DCGs")
    for kk in [10, 20, 50, 100, 200]:
        cal_precision_recall_k(kk)
        


def location_prediction(test_checkin, embs, poi_embs, k=10):
    """
    test_checkin: np array shape Nx3, containing a user, time slot and a POI
    """
    embs = normalize_embedding(embs) # N x d
    poi_embs = normalize_embedding(poi_embs) # Np x d
    user_time = test_checkin[:, :2] # user and time 
    user_time_emb = embs[user_time] # n x 2 x d
    user_time_with_poi = np.dot(user_time_emb, poi_embs.T) # nx2x(np)
    user_time_with_poi = np.sum(user_time_with_poi, axis=1) # nxnp
    # argptt = np.argpartition(user_time_with_poi, -k, axis=1)[:, -k:] # nx10
    argptt = np.argsort(-user_time_with_poi, axis=1)
    
    ranks = []
    hit10s = 0
    hit20s = 0
    hit30s = 0
    hit40s = 0
    hit50s = 0
    rank_scores = []
    # rank_score1 = 0
    # rank_score3 = 0
    # rank_score5 = 0
    # rank_score10 = 0
    # rank_score50 = 0
    # rank_score_norm = 0
    ranks1 = []
    ranks3 = []
    ranks5 = []
    ranks10 = []
    ranks50 = []
    # ranks1 = []

    for i in range(argptt.shape[0]):
        n_relevants1 = 0
        n_relevants3 = 0
        n_relevants5 = 0
        n_relevants10 = 0
        n_relevants50 = 0

        rank_score1 = 0
        rank_score3 = 0
        rank_score5 = 0
        rank_score10 = 0
        rank_score50 = 0
        rank = argptt.shape[1]
        for j in range(argptt.shape[1]):
            if test_checkin[i, 2] == argptt[i,j]:
                rank = j 
                rank_score_norm_this = 1 / np.log2(rank + 2)
                # rank_score_norm += rank_score_norm_this
                if rank < 1:
                    rank_score1 += rank_score_norm_this
                    rank_score3 += rank_score_norm_this
                    rank_score5 += rank_score_norm_this
                    rank_score10 += rank_score_norm_this
                    rank_score50 += rank_score_norm_this

                    hit10s += 1
                    hit20s += 1
                    hit30s += 1
                    hit40s += 1
                    hit50s += 1

                    n_relevants1 += 1
                    n_relevants3 += 1
                    n_relevants5 += 1
                    n_relevants10 += 1
                    n_relevants50 += 1
                elif rank < 3:
                    rank_score3 += rank_score_norm_this
                    rank_score5 += rank_score_norm_this
                    rank_score10 += rank_score_norm_this
                    rank_score50 += rank_score_norm_this
                    hit20s += 1
                    hit30s += 1
                    hit40s += 1
                    hit50s += 1

                    n_relevants3 += 1
                    n_relevants5 += 1
                    n_relevants10 += 1
                    n_relevants50 += 1
                elif rank < 5:
                    rank_score5 += rank_score_norm_this
                    rank_score10 += rank_score_norm_this
                    rank_score50 += rank_score_norm_this
                    hit30s += 1
                    hit40s += 1
                    hit50s += 1

                    n_relevants5 += 1
                    n_relevants10 += 1
                    n_relevants50 += 1
                elif rank < 10:
                    rank_score10 += rank_score_norm_this
                    rank_score50 += rank_score_norm_this
                    hit40s += 1
                    hit50s += 1

                    n_relevants10 += 1
                    n_relevants50 += 1
                elif rank < 50:
                    rank_score50 += rank_score_norm_this
                    hit50s += 1
                    n_relevants50 += 1
                break 

        idcg1 = compute_idcg(n_relevants1)
        idcg3 = compute_idcg(n_relevants3)
        idcg5 = compute_idcg(n_relevants5)
        idcg10 = compute_idcg(n_relevants10)
        idcg50 = compute_idcg(n_relevants50)

        ndcg1 = rank_score1 / idcg1
        ndcg3 = rank_score3 / idcg3
        ndcg5 = rank_score5 / idcg5
        ndcg10 = rank_score10 / idcg10
        ndcg50 = rank_score50 / idcg50

        ranks1.append(ndcg1)
        ranks3.append(ndcg3)
        ranks5.append(ndcg5)
        ranks10.append(ndcg10)
        ranks50.append(ndcg50)

        ranks.append(rank + 1)

    try:
        mean_rank = np.mean(ranks)
        # print(ranks)
        mrr = np.mean([1/ele for ele in ranks])
        hit10s /= len(test_checkin)
        hit20s /= len(test_checkin)
        hit30s /= len(test_checkin)
        hit40s /= len(test_checkin)
        hit50s /= len(test_checkin)

        rank_score1 = np.mean(ranks1)
        rank_score3 = np.mean(ranks3)
        rank_score5 = np.mean(ranks5)
        rank_score10 = np.mean(ranks10)
        rank_score50 = np.mean(ranks50)

        # print("Hit1: {:.4f}".format(hit10s))
        # print("Hit3: {:.4f}".format(hit20s))
        # print("Hit5: {:.4f}".format(hit30s))
        # print("Hit10: {:.4f}".format(hit40s))
        # print("Hit50: {:.4f}".format(hit50s))
        print("DCG 1 3 5 10 50")
        for ele in [rank_score1, rank_score3, rank_score5, rank_score10, rank_score50]:
            print(ele)

        # print("DCG1: {:.4f}".format(rank_score1))
        # print("DCG3: {:.4f}".format(rank_score3))
        # print("DCG5: {:.4f}".format(rank_score5))
        # print("DCG10: {:.4f}".format(rank_score10))
        # print("DCG50: {:.4f}".format(rank_score50))
        # print("MR: {:.4f}".format(mean_rank))
        # print("MRR: {:.4f}".format(mrr))
        # return acc
        return 1
    except Exception as err:
        print(err)
    


def rank_matrix(matrix, k=10):
    arg_max_index = []
    while len(arg_max_index) < 10:
        arg_max = np.argmax(matrix)
        column = arg_max % matrix.shape[1]
        arg_max_index.append(column)
        matrix[:, column] -= 2
    return arg_max_index

def location_prediction_Persona2(test_checkin, embs, poi_embs, k=10, user_persona_dict=None, persona_user_dict=None):
    """
    test_checkin: np array shape Nx3, containing a user, time slot and a POI
    """
    embs = normalize_embedding(embs) # N x d
    poi_embs = normalize_embedding(poi_embs) # Np x d
    users = test_checkin[:, 0]
    times = test_checkin[:, 1]
    
    ranks = []
    hit10s = 0
    hit20s = 0
    hit30s = 0
    hit40s = 0
    hit50s = 0

    rank_scores = []
    ranks1 = []
    ranks3 = []
    ranks5 = []
    ranks10 = []
    ranks50 = []
    # ranks1 = []

    for i, user in enumerate(users):
        n_relevants1 = 0
        n_relevants3 = 0
        n_relevants5 = 0
        n_relevants10 = 0
        n_relevants50 = 0

        rank_score1 = 0
        rank_score3 = 0
        rank_score5 = 0
        rank_score10 = 0
        rank_score50 = 0



        this_user_persona = user_persona_dict[user + 1]
        this_user_persona = [ele - 1 for ele in this_user_persona]
        this_user_time = times[i]
        time_emb = embs[this_user_time].reshape(1, -1)
        time_ranking = time_emb.dot(poi_embs.T).reshape(1, -1)
        this_user_persona_emb = embs[this_user_persona]
        this_user_persona_ranking = this_user_persona_emb.dot(poi_embs.T).reshape(len(this_user_persona), -1)
        final_ranking = time_ranking + this_user_persona_ranking
        # argptt = np.argpartition(final_ranking, -k, axis=1)[:, -k:] # nx10
        argptt = np.argsort(-final_ranking, axis=1)
        target = test_checkin[i, 2]
        rank = argptt.shape[1]
        for j in range(argptt.shape[1]):
            if target in argptt[:, j]:
                rank = j 
                rank_score_norm_this = 1 / np.log2(rank + 2)

                if rank < 1:
                    rank_score1 += rank_score_norm_this
                    rank_score3 += rank_score_norm_this
                    rank_score5 += rank_score_norm_this
                    rank_score10 += rank_score_norm_this
                    rank_score50 += rank_score_norm_this

                    hit10s += 1
                    hit20s += 1
                    hit30s += 1
                    hit40s += 1
                    hit50s += 1

                    n_relevants1 += 1
                    n_relevants3 += 1
                    n_relevants5 += 1
                    n_relevants10 += 1
                    n_relevants50 += 1
                elif rank < 3:
                    rank_score3 += rank_score_norm_this
                    rank_score5 += rank_score_norm_this
                    rank_score10 += rank_score_norm_this
                    rank_score50 += rank_score_norm_this
                    hit20s += 1
                    hit30s += 1
                    hit40s += 1
                    hit50s += 1

                    n_relevants3 += 1
                    n_relevants5 += 1
                    n_relevants10 += 1
                    n_relevants50 += 1
                elif rank < 5:
                    rank_score5 += rank_score_norm_this
                    rank_score10 += rank_score_norm_this
                    rank_score50 += rank_score_norm_this
                    hit30s += 1
                    hit40s += 1
                    hit50s += 1

                    n_relevants5 += 1
                    n_relevants10 += 1
                    n_relevants50 += 1
                elif rank < 10:
                    rank_score10 += rank_score_norm_this
                    rank_score50 += rank_score_norm_this
                    hit40s += 1
                    hit50s += 1

                    n_relevants10 += 1
                    n_relevants50 += 1
                elif rank < 50:
                    rank_score50 += rank_score_norm_this
                    hit50s += 1
                    n_relevants50 += 1
                break 

        idcg1 = compute_idcg(n_relevants1)
        idcg3 = compute_idcg(n_relevants3)
        idcg5 = compute_idcg(n_relevants5)
        idcg10 = compute_idcg(n_relevants10)
        idcg50 = compute_idcg(n_relevants50)

        ndcg1 = rank_score1 / idcg1
        ndcg3 = rank_score3 / idcg3
        ndcg5 = rank_score5 / idcg5
        ndcg10 = rank_score10 / idcg10
        ndcg50 = rank_score50 / idcg50

        ranks1.append(ndcg1)
        ranks3.append(ndcg3)
        ranks5.append(ndcg5)
        ranks10.append(ndcg10)
        ranks50.append(ndcg50)

        ranks.append(rank + 1)
        if rank == argptt.shape[1]:
            print("LOL")
        ranks.append(rank + 1)
    try:
        # acc = hit / len(test_checkin)
        # print(ranks)
        hit10s /= len(test_checkin)
        hit20s /= len(test_checkin)
        hit30s /= len(test_checkin)
        hit40s /= len(test_checkin)
        hit50s /= len(test_checkin)
        mean_rank = np.mean(ranks)
        mrr = np.mean([1/ele for ele in ranks])
        # print("Hit1: {:.4f}".format(hit10s))
        # print("Hit3: {:.4f}".format(hit20s))
        # print("Hit5: {:.4f}".format(hit30s))
        # print("Hit10: {:.4f}".format(hit40s))
        # print("Hit50: {:.4f}".format(hit50s))
        # print("MR: {:.4f}".format(mean_rank))
        # print("MRR: {:.4f}".format(mrr))

        rank_score1 = np.mean(ranks1)
        rank_score3 = np.mean(ranks3)
        rank_score5 = np.mean(ranks5)
        rank_score10 = np.mean(ranks10)
        rank_score50 = np.mean(ranks50)

        # print("Hit1: {:.4f}".format(hit10s))
        # print("Hit3: {:.4f}".format(hit20s))
        # print("Hit5: {:.4f}".format(hit30s))
        # print("Hit10: {:.4f}".format(hit40s))
        # print("Hit50: {:.4f}".format(hit50s))
        print("DCG 1 3 5 10 50")
        for ele in [rank_score1, rank_score3, rank_score5, rank_score10, rank_score50]:
            print(ele)
    except:
        print("lol")
    


def loadtxt(path, separator):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data_line = line.split(separator)
            data.append([float(ele) for ele in data_line])
    return np.array(data)


if __name__ == "__main__":
    embs_user = np.random.uniform(size=(4, 5))
    fo = np.array([[0,1], [2, 3]])
    fn = np.array([[0,2], [0,3], [1, 2]])
    # friendship_linkprediction(embs_user, fo, fn)

    # embs_cate = loadtxt('embs_cate.txt', ',').T 
    # embs_user = loadtxt('embs_user.txt', ',').T 
    # embs_time = loadtxt('embs_time.txt', ',').T 
    # embs_venue = loadtxt('embs_venue.txt', ',').T 

