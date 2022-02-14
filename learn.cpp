#include <pybind11/pybind11.h>
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "pthread.h"
#include "limits.h"
#include "string.h"
#include <fstream>
#include <sstream>
#include <iostream>

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define RAND_MULTIPLIER 25214903917
#define RAND_INCREMENT 11

double *expTable;

// input 1
long **walk;
long num_w;
long num_wl;
// input 2
std::vector<long**> user_checkins; // hyperedges
// input 3
long *user_checkins_count;
// input 4
double **emb_n; //node embedding
long num_n;
long dim_emb;
// input 5
double starting_alpha;
double alpha;
// input 6
long num_neg;
// input 7
long *neg_sam_table_social; // negative sampling table social network
long table_size_social;
// input 8
long win_size;
// input 9
long *neg_sam_table_mobility1;
long table_size_mobility1;
long *neg_sam_table_mobility2;
long table_size_mobility2;
long *neg_sam_table_mobility3;
long table_size_mobility3;
long *neg_sam_table_mobility4;
long table_size_mobility4;
// input 10
long num_epoch;
// input 11
long num_threads;
// input 12
double mobility_ratio;

void getNextRand(unsigned long *next_random){
    *next_random = (*next_random) * (unsigned long) RAND_MULTIPLIER + RAND_INCREMENT;
}

long get_a_neg_sample(unsigned long next_random, long *neg_sam_table, long table_size){
    long target_n;
    unsigned long ind;

    ind = (next_random >> 16) % table_size;
    target_n = neg_sam_table[ind];

    return target_n;
}

long get_a_checkin_sample(unsigned long next_random, long  table_size){
    return (next_random >> 16) % table_size;
}


double sigmoid(double f) {
    if (f >= MAX_EXP) return 1;
    else if (f <= -MAX_EXP) return 0;
    else return expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2 ))];
}

int get_a_neg_sample_Kless1(unsigned long next_random){
    double v_rand_uniform = (double) next_random/(double)(ULONG_MAX);
    if (v_rand_uniform<=num_neg){
        return 1;
    }else{
        return 0;
    }
}

int get_a_social_decision(unsigned long next_random){
    double v_rand_uniform = (double) next_random/(double)(ULONG_MAX);
    if (v_rand_uniform<=mobility_ratio){
        return 0;
    }else{
        return 1;
    }
}

int get_a_mobility_decision(unsigned long next_random){
    double v_rand_uniform = (double) next_random/(double)(ULONG_MAX);
    if (v_rand_uniform<=mobility_ratio){
        return 1;
    }else{
        return 0;
    }
}

double get_norm_l2_loc(long loc_node){
    double norm = 0;
    for (int d=0; d<dim_emb; d++) norm = norm + emb_n[loc_node][d] * emb_n[loc_node][d];
    return sqrt(norm);
}

double get_norm_l2_pr(double *vec){
    double norm = 0;
    for (int d=0; d<dim_emb; d++) norm = norm + vec[d] * vec[d];
    return sqrt(norm);
}

void learn_a_pair_loc_loc_cosine(int flag, long loc1, long loc2, double *loss)
{
    double f=0,tmp1,tmp2,c1,c2,c3; //f2=0,
    double norm1 = get_norm_l2_loc(loc1);
    double norm2 = get_norm_l2_loc(loc2);

    for (int d=0;d<dim_emb;d++)
        f += emb_n[loc1][d] * emb_n[loc2][d];

    c1 = 1/(norm1*norm2)*alpha;
    c2 = f/(norm1*norm1*norm1*norm2)*alpha;
    c3 = f/(norm1*norm2*norm2*norm2)*alpha;


    if (flag==1){
//         *loss += f;
        for (int d=0; d<dim_emb; d++){
            tmp1 = emb_n[loc1][d];
            tmp2 = emb_n[loc2][d];
            emb_n[loc2][d] += c1*tmp1 - c3*tmp2;
            emb_n[loc1][d] += c1*tmp2 - c2*tmp1;
        }
    }else{
//         *loss -= f/num_neg;
        for (int d=0; d<dim_emb; d++){
            tmp1 = emb_n[loc1][d];
            tmp2 = emb_n[loc2][d];
            emb_n[loc2][d] -= c1*tmp1 - c3*tmp2;
            emb_n[loc1][d] -= c1*tmp2 - c2*tmp1;
        }
    }

}

void learn_a_pair_loc_pr_cosine(int flag, long loc1, double *best_fit, double *loss)
{
    double f=0,a=0,c1,c2; //f2=0,
    double norm1 = get_norm_l2_loc(loc1);

    for (int d=0;d<dim_emb;d++)
        f += emb_n[loc1][d] * best_fit[d];
    a = alpha;
    c1 = 1/(norm1)*a;
    c2 = f/(norm1*norm1*norm1)*a;

    if (flag==1){
//         *loss += g;
        for (int d=0; d<dim_emb; d++)
            emb_n[loc1][d] += c1*best_fit[d] - c2*emb_n[loc1][d];
    }else{
//         *loss -= g/num_neg;
        for (int d=0; d<dim_emb; d++)
            emb_n[loc1][d] -= c1*best_fit[d] - c2*emb_n[loc1][d];
    }
}

void learn_an_edge(long word, long target_e, unsigned long *next_random, double* counter)
{
    long target_n, loc_neg;
    long loc_w = word-1;
    long loc_e = target_e-1;
    learn_a_pair_loc_loc_cosine(1, loc_w, loc_e, counter);

    if (num_neg<1){
        getNextRand(next_random);
        if (get_a_neg_sample_Kless1(*next_random)==1){
            getNextRand(next_random);
            target_n = get_a_neg_sample(*next_random, neg_sam_table_social, table_size_social);
            if ((target_n != target_e) && (target_n != word)){
                loc_neg = (target_n-1)*dim_emb;
                learn_a_pair_loc_loc_cosine(0, loc_w, loc_neg, counter);
            }
        }
    }else{
        for (int n=0;n<num_neg;n++){
            getNextRand(next_random);
            target_n = get_a_neg_sample(*next_random, neg_sam_table_social, table_size_social);
            if ((target_n != target_e) && (target_n != word)){
                loc_neg = (target_n-1)*dim_emb;
                learn_a_pair_loc_loc_cosine(0, loc_w, loc_neg, counter);
            }
        }
    }
}


void learn_an_edge_with_BFT(long word, long target_e, unsigned long *next_random, double *best_fit, double* counter)
{
    long target_n, loc_neg;
    double norm;
    long loc_w = word-1;
    long loc_e = target_e-1;

    // std::cout << "start learn bft - loc_w " << loc_w << " loc_e " << loc_e << "\n";
    for (int d=0; d<dim_emb; d++) best_fit[d] = emb_n[loc_w][d] + emb_n[loc_e][d];
    // std::cout << "get_norm_l2_pr\n";
    norm = get_norm_l2_pr(best_fit);
    for (int d=0; d<dim_emb; d++) best_fit[d] = best_fit[d]/norm;
    // std::cout << "learn_a_pair_loc_pr_cosine\n";
    learn_a_pair_loc_pr_cosine(1, loc_w, best_fit, counter);
    learn_a_pair_loc_pr_cosine(1, loc_e, best_fit, counter);

    if (num_neg<1){
        getNextRand(next_random);
        if (get_a_neg_sample_Kless1(*next_random)==1){
            getNextRand(next_random);
            target_n = get_a_neg_sample(*next_random, neg_sam_table_social, table_size_social);
            if ((target_n != target_e) && (target_n != word)){
                loc_neg = target_n-1;
                learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter);
            }
        }
    }else{
        for (int n=0;n<num_neg;n++){
            getNextRand(next_random);
            // std::cout << "get_a_neg_sample\n";
            target_n = get_a_neg_sample(*next_random, neg_sam_table_social, table_size_social);
            if ((target_n != target_e) && (target_n != word)){
                loc_neg = target_n-1;
                // std::cout << "learn_a_pair_loc_pr_cosine\n";
                learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter);
            }
        }
    }
    // std::cout << "end learn bft\n";
}



void learn_a_hyperedge(long *edge, long edge_len, unsigned long *next_random, double *best_fit, double* counter)
{
    long node, target_neg = -1;
    long loc_n, loc_neg;
    double norm;

//#################### get best-fit-line
    for (int d=0; d<dim_emb; d++) best_fit[d] = 0;
    for (int i=0; i<edge_len; i++) {
        loc_n = edge[i]-1;
        norm = get_norm_l2_pr(emb_n[loc_n]);
        for (int d=0; d<dim_emb; d++) best_fit[d] += emb_n[loc_n][d]/norm;
    }
//  normalize best fit line for fast computation
    norm = get_norm_l2_pr(best_fit);
    for (int d=0; d<dim_emb; d++) best_fit[d] = best_fit[d]/norm;

//#################### learn learn learn
    for (int i=0; i<edge_len; i++) {
        node = edge[i];
        loc_n = node-1;
        learn_a_pair_loc_pr_cosine(1, loc_n, best_fit, counter);
        if (num_neg<1){
            getNextRand(next_random);
            if (get_a_neg_sample_Kless1(*next_random)==1){
                getNextRand(next_random);
                if (i==0) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility1, table_size_mobility1);
                else if (i==1) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility2, table_size_mobility2);
                else if (i==2) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility3, table_size_mobility3);
                else if (i==3) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility4, table_size_mobility4);

                if (target_neg != node) {
                    loc_neg = target_neg-1;
                    learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter);
                }
            }
        }else{
            for (int n=0;n<num_neg;n++){
                getNextRand(next_random);
                if (i==0) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility1, table_size_mobility1);
                else if (i==1) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility2, table_size_mobility2);
                else if (i==2) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility3, table_size_mobility3);
                else if (i==3) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility4, table_size_mobility4);
                if (target_neg != node) {
                    loc_neg = target_neg-1;
                    learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter);
                }
            }
        }
    }
}


void merge_hyperedges(long *edge_merged, long* edge_merged_len, long *a_edge, long a_edge_len)
{
    memcpy(edge_merged+(*edge_merged_len), a_edge, a_edge_len * sizeof(long));
    *edge_merged_len += a_edge_len;
}



void normalize_embeddings(){
    long loc_node;
    double norm;
    int i,d;
    for (i=0;i<num_n;i++) {
        loc_node = i;
        norm=0;
        for (d=0; d<dim_emb; d++) norm = norm + emb_n[loc_node][d] * emb_n[loc_node][d];
        for (d=0; d<dim_emb; d++) emb_n[loc_node][d] = emb_n[loc_node][d]/sqrt(norm);
    }
}


void * learn(void *id)
{
    // std::cout << " === Learn " << (long)id << "\n";
    long word, target_e, a_checkin_ind;
    double *best_fit = (double *)malloc(dim_emb*sizeof(double)); //a node embedding

    double counter;
//     double norm;

    unsigned long next_random = (long) rand();
    long **a_user_checkins;
    long *edge;
    long edge_len = 4; // here 4 is a checkin node number user-time-POI-category

    

    long ind_start = num_w/num_threads * (long)id;
    long ind_end = num_w/num_threads * ((long)id+1);
    
    long ind_len = ind_end-ind_start;
    double progress=0,progress_old=0;
    alpha = starting_alpha;

    for (int pp=0; pp<num_epoch; pp++){
        counter = 0;
        // std::cout << " === Epoch " << pp << "\n";

        for (int w=ind_start; w<ind_end; w++) {
            progress = ((pp*ind_len)+(w-ind_start)) / (double) (ind_len*num_epoch);
            if (progress-progress_old > 0.001) {
                alpha = starting_alpha * (1 - progress);
                if (alpha < starting_alpha * 0.001) alpha = starting_alpha * 0.001;
                progress_old = progress;
            }

            for (int i=0; i<num_wl; i++) {
                word = walk[w][i];
                // std::cout << " === Word " << word << "  - w-i " << w << "-" << i << "\n";

                // std::cout << "learn bft\n";
                for (int j=1;j<=win_size;j++){
                    getNextRand(&next_random);
                    if (get_a_social_decision(next_random)==1){
                        // printf("social \n");
                        if (i-j>=0) {
                            target_e = walk[w][i-j];
                            if (word!=target_e)
                                learn_an_edge_with_BFT(word, target_e, &next_random, best_fit, &counter);
                        }
                        if (i+j<num_wl) {
                            target_e = walk[w][i+j];
                            if (word!=target_e)
                                learn_an_edge_with_BFT(word, target_e, &next_random, best_fit, &counter);
                        }
                    }
                }

                if ((user_checkins_count[word-1]>0) ){
                    for (int m=0; m < fmin(win_size*2,user_checkins_count[word-1]); m++){
                        getNextRand(&next_random);
                        if (get_a_mobility_decision(next_random)==1) {
                            a_user_checkins = user_checkins[word-1];
                            // std::cout << "get a user checkins\n";

                            getNextRand(&next_random);
                            a_checkin_ind = get_a_checkin_sample(next_random, user_checkins_count[word-1]);
                            edge = a_user_checkins[a_checkin_ind];
                            // std::cout << "start learn hyper\n";
                            learn_a_hyperedge(edge, edge_len, &next_random, best_fit, &counter);
                            // std::cout << "end learn hyper\n";
                        }
                    }
                }
            }

        }
//         printf("Thread %ld iteration %d loss: %f \n",(long)id, pp, counter);
    }
//     printf("counter (word=target_e) : %ld\n", counter);
    free(best_fit);
    pthread_exit(NULL);
}

void readWalks(std::string walkFile) {
    // read walk
    std::ifstream infile(walkFile);
    std::string line;
    std::getline(infile, line);
    std::istringstream iss(line);
    iss >> num_w;
    iss >> num_wl;
    printf("Num w %ld\n", num_w);
    printf("Walk length %ld\n", num_wl);
    walk = (long**)malloc(num_w * sizeof(long*));
    for(int i = 0; i < num_w; i++) {
        walk[i] = (long*) malloc(num_wl * sizeof(long));
        std::getline(infile, line);
        std::istringstream iss(line);
        for (int j = 0; j < num_wl; j++) {
            iss >> walk[i][j];
        }
    }
}

void readUserCheckins(std::string userCheckinsFile) {
    std::ifstream infile(userCheckinsFile);
    std::string line;
    std::getline(infile, line);
    std::istringstream iss(line);
    long num_u;
    iss >> num_u;
    printf("Num u %ld\n", num_u);
    user_checkins_count = (long*) malloc(num_u * sizeof(long));
    for (int i = 0; i < num_u; i++) {
        std::getline(infile, line);
        std::istringstream iss(line);
        int num_checkins;
        iss >>  num_checkins;
        user_checkins_count[i] = num_checkins;
        long **a_user_checkins = (long**) malloc(num_checkins*sizeof(long*));
        for (int j = 0; j < num_checkins; j++) {
            a_user_checkins[j] = (long*) malloc(4*sizeof(long));
            std::getline(infile, line);
            std::istringstream iss(line);
            for (int k = 0; k < 4; k++) {
                iss >> a_user_checkins[j][k];
            }
        }
        user_checkins.push_back(a_user_checkins);
    }
}

void readEmbN(std::string embFile) {
    // read walk
    std::ifstream infile(embFile);
    std::string line;
    std::getline(infile, line);
    std::istringstream iss(line);
    iss >> num_n;
    iss >> dim_emb;
    printf("Num n %ld\n", num_n);
    printf("Dim_emb %ld\n", dim_emb);
    emb_n = (double**)malloc(num_n * sizeof(double*));
    for(int i = 0; i < num_n; i++) {
        emb_n[i] = (double*) malloc(dim_emb * sizeof(double));
        std::getline(infile, line);
        std::istringstream iss(line);
        for (int j = 0; j < dim_emb; j++) {
            iss >> emb_n[i][j];
        }
    }
}

void readNegSamTableSocial(std::string filePath) {
    // read walk
    std::ifstream infile(filePath);
    std::string line;
    std::getline(infile, line);
    std::istringstream iss(line);
    iss >> table_size_social;
    printf("table_size_social %ld\n", table_size_social);
    neg_sam_table_social = (long*)malloc(table_size_social * sizeof(long));
    for(int i = 0; i < table_size_social; i++) {
        std::getline(infile, line);
        std::istringstream iss(line);
        iss >> neg_sam_table_social[i];
    }
}

void readNegCheckins(std::string filePath) {
    // read walk
    std::ifstream infile(filePath);
    std::string line;
    std::getline(infile, line);
    std::istringstream iss(line);
    iss >> table_size_mobility1;
    printf("table_size_mobility1 %ld\n", table_size_mobility1);
    neg_sam_table_mobility1 = (long*)malloc(table_size_mobility1 * sizeof(long));
    for(int i = 0; i < table_size_mobility1; i++) {
        std::getline(infile, line);
        std::istringstream iss(line);
        iss >> neg_sam_table_mobility1[i];
    }

    std::getline(infile, line);
    std::istringstream iss2(line);
    iss2 >> table_size_mobility2;
    printf("table_size_mobility2 %ld\n", table_size_mobility2);
    neg_sam_table_mobility2 = (long*)malloc(table_size_mobility2 * sizeof(long));
    for(int i = 0; i < table_size_mobility2; i++) {
        std::getline(infile, line);
        std::istringstream iss(line);
        iss >> neg_sam_table_mobility2[i];
    }

    std::getline(infile, line);
    std::istringstream iss3(line);
    iss3 >> table_size_mobility3;
    printf("table_size_mobility3 %ld\n", table_size_mobility3);
    neg_sam_table_mobility3 = (long*)malloc(table_size_mobility3 * sizeof(long));
    for(int i = 0; i < table_size_mobility3; i++) {
        std::getline(infile, line);
        std::istringstream iss(line);
        iss >> neg_sam_table_mobility3[i];
    }

    std::getline(infile, line);
    std::istringstream iss4(line);
    iss4 >> table_size_mobility4;
    printf("table_size_mobility4 %ld\n", table_size_mobility4);
    neg_sam_table_mobility4 = (long*)malloc(table_size_mobility4 * sizeof(long));
    for(int i = 0; i < table_size_mobility4; i++) {
        std::getline(infile, line);
        std::istringstream iss(line);
        iss >> neg_sam_table_mobility4[i];
    }
}

void apiFunction(std::string inputDir, double _starting_alpha, long _num_neg,
    long _win_size, long _num_epoch, long _num_threads, double _mobility_ratio)
{
    readUserCheckins(inputDir + "/user_checkins.txt");
    readWalks(inputDir + "/walk.txt");
    readEmbN(inputDir + "/embs_ini.txt");
    starting_alpha = _starting_alpha;
    num_neg = _num_neg;
    readNegSamTableSocial(inputDir + "/neg_user_samples.txt");
    win_size = _win_size;
    readNegCheckins(inputDir + "/neg_checkins_samples.txt");
    num_epoch = _num_epoch;
    num_threads = _num_threads;
    mobility_ratio = _mobility_ratio;

    std::cout << "starting_alpha : " << starting_alpha << "\n";
    std::cout << "num_neg : " << num_neg << "\n";
    std::cout << "win_size : " << win_size << "\n";
    std::cout << "num_epoch : " << num_epoch << "\n";
    std::cout << "num_threads : " << num_threads << "\n";
    std::cout << "mobility_ratio : " << mobility_ratio << "\n";
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, learn, (long *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

    // learn done, write emb to file
    std::ofstream embFile;
    embFile.open (inputDir + "/embs.txt");
    embFile << num_n << " " << dim_emb << "\n";
    for (int i = 0; i < num_n; i++) {
        for (int j = 0; j < dim_emb; j++) {
            embFile << emb_n[i][j] << " ";
        }
        embFile << "\n";
    }
    embFile.close();
}

PYBIND11_MODULE(learn, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("apiFunction", &apiFunction, "API Function");
}
