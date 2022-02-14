
######################## location prediction #####################
    # copy POI (WAIT)
        # normal randomwalk

mode="persona_ori"

for data in Jakarta 
do 
    python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} 
done

mode="persona_ori"

for data in NYC hongzhi TKY Istanbul Jakarta KualaLampur SaoPaulo
do 
    python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} > output/normalrw_sepPOI_${data}_POI
done



    # tach POI (wait)
        # normal random walk

mode="persona_POI"

for data in NYC 
do 
    python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} 
done

mode="persona_POI"

for data in NYC 
do 
    python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} --workers 32
done



        # new random walk 

mode="persona_POI"

for data in NYC 
do 
    python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} --bias_randomwalk --p_n2v 1 -q_n2v 1 --workers 32
done

mode="persona_POI"

for data in NYC 
do 
    for alpha in 0.1 
    do
        python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} 
    done
done




    # Original_version


for data in NYC 
do 
    python -u baseline_POI.py --dataset_name ${data} 
done


for data in NYC hongzhi TKY Istanbul Jakarta KualaLampur SaoPaulo
do 
    python -u baseline_POI.py --dataset_name ${data} > output/original_${data}_POI
done


# create input for those graphs


for dataset in NYC TKY hongzhi Istanbul Jakarta KualaLampur SaoPaulo
do
    for model in deepwalk line 
    do 
        python create_input.py --dataset_name ${dataset} --model ${model} --POI
    done
done


    # other baselines


for data in NYC TKY hongzhi Istanbul KualaLampur SaoPaulo
do     
# deepwalk --format edgelist --input ../LBSN2Vec/edgelist_graph/${data}.edgelist     --max-memory-data-size 0 --number-walks 10 --representation-size 128 --walk-length 80 --window-size 10     --workers 16 --output ../LBSN2Vec/deepwalk_emb/${data}.embeddings
deepwalk --format edgelist --input ../LBSN2Vec/edgelist_graph/${data}_M_POI.edgelist     --max-memory-data-size 0 --number-walks 10 --representation-size 128 --walk-length 80 --window-size 10     --workers 16 --output ../LBSN2Vec/deepwalk_emb/${data}_M_POI.embeddings
deepwalk --format edgelist --input ../LBSN2Vec/edgelist_graph/${data}_SM_POI.edgelist     --max-memory-data-size 0 --number-walks 10 --representation-size 128 --walk-length 80 --window-size 10     --workers 16 --output ../LBSN2Vec/deepwalk_emb/${data}_SM_POI.embeddings
done


for data in Jakarta
do
python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}.embeddings 
python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}_M_POI.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}_M_POI.embeddings 
python -m openne --method node2vec --input ../../LBSN2Vec/edgelist_graph/${data}_SM_POI.edgelist --graph-format edgelist --output ../../LBSN2Vec/node2vec_emb/${data}_SM_POI.embeddings --epochs 4
done



for data in Jakarta
do
python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}_SM_POI.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}_SM_POI.embeddings 
python -m openne --method node2vec --input ../../LBSN2Vec/edgelist_graph/${data}_M_POI.edgelist --graph-format edgelist --output ../../LBSN2Vec/node2vec_emb/${data}_M_POI.embeddings --epochs 4
done




for data in Istanbul KualaLampur SaoPaulo NYC
do
# python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}.embeddings 
# python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}_M.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}_M.embeddings 
# python -m openne --method line --input ../../LBSN2Vec/edgelist_graph/${data}_SM.edgelist --graph-format edgelist --output ../../LBSN2Vec/line_emb/${data}_SM.embeddings 
python -m openne --method node2vec --input ../../LBSN2Vec/edgelist_graph/${data}_M.edgelist --graph-format edgelist --output ../../LBSN2Vec/node2vec_emb/${data}_M.embeddings --epochs 4
python -m openne --method node2vec --input ../../LBSN2Vec/edgelist_graph/${data}_SM.edgelist --graph-format edgelist --output ../../LBSN2Vec/node2vec_emb/${data}_SM.embeddings --epochs 4
done





#TODO: how to eval ????
# create_data --> embedding --> eval

for data in NYC hongzhi TKY Istanbul Jakarta KualaLampur SaoPaulo
do
    for model in line
    do 
        # python -u eval_models.py --emb_path ${model}_emb/${data}.embeddings --dataset_name ${data} --model ${model} > ori_out/${model}_${data}
        python -u eval_models.py --emb_path ${model}_emb/${data}_M_POI.embeddings --dataset_name ${data} --model ${model} --POI > output/${model}_${data}_M_POI
        python -u eval_models.py --emb_path ${model}_emb/${data}_SM_POI.embeddings --dataset_name ${data} --model ${model} --POI > output/${model}_${data}_SM_POI
    done
    # model=dhne
    # python -u eval_models.py --emb_path ${model}_emb/${data}/model_16/embeddings.npy --dataset_name ${data} --model ${model} > ori_out/${model}_${data}_friend
done 


for data in NYC hongzhi TKY Istanbul KualaLampur SaoPaulo Jakarta
do
    for model in node2vec line   
    do 
        # python -u eval_models.py --emb_path ${model}_emb/${data}.embeddings --dataset_name ${data} --model ${model} > ori_out/${model}_${data}
        python -u eval_models.py --emb_path ${model}_emb/${data}_M_POI.embeddings --dataset_name ${data} --model ${model} --POI > ori_out/MMM${model}_${data}_POI
        python -u eval_models.py --emb_path ${model}_emb/${data}_SM_POI.embeddings --dataset_name ${data} --model ${model} --POI > ori_out/SSSMMM${model}_${data}_POI
    done
    # model=dhne
    # python -u eval_models.py --emb_path ${model}_emb/${data}/model_16/embeddings.npy --dataset_name ${data} --model ${model} --POI > ori_out/${model}_${data}_friend
done 



for data in NYC hongzhi TKY Istanbul KualaLampur SaoPaulo
do
    for model in line 
    do 
        # python -u eval_models.py --emb_path ${model}_emb/${data}.embeddings --dataset_name ${data} --model ${model} > ori_out/${model}_${data}
        python -u eval_models.py --emb_path ${model}_emb/${data}_M.embeddings --dataset_name ${data} --model ${model} > ori_out/${model}_${data}_M_Friend
        python -u eval_models.py --emb_path ${model}_emb/${data}_SM.embeddings --dataset_name ${data} --model ${model} > ori_out/${model}_${data}_SM_Friend
    done
    # model=dhne
    # python -u eval_models.py --emb_path ${model}_emb/${data}/model_16/embeddings.npy --dataset_name ${data} --model ${model} > ori_out/${model}_${data}_friend
done 




for dataset in NYC TKY hongzhi Istanbul Jakarta KualaLampur SaoPaulo
do
    for model in dhne deepwalk line 
    do 
        python create_input.py --dataset_name ${dataset} --model ${model} --POI
    done
done




mode="persona_POI"

for data in TKY hongzhi KualaLampur Istanbul SaoPaulo NYC
do
    for qq in 0.4 0.5 0.6
    do
        for pp in 0.6 0.8 1
        do 
            python -u CMan_POI.py --input_type ${mode} --dataset_name ${data} --bias_randomwalk --p_n2v ${pp} --q_n2v ${qq} --workers 32 --mobility_ratio 0.7 > output1/${data}_p${pp}_q${qq}_POI
        done 
    done
done


mode="persona_POI"

for p in 0.8
do
for q in 0.6
do
for lr in 0.0005
do
for dim in 300
do
for Kneg in 10
do
for data in Jakarta Istanbul
do
python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
--dataset_name ${data} --p_n2v ${p} --q_n2v ${q} \
--workers 12 --learning_rate ${lr} --K_neg ${Kneg} --mobility_ratio 0.7 --add_flag 0.9 --bias_randomwalk > location_pred2/model4_${data}_${dim}_${lr}_${Kneg}_p${p}_q${q}m0.7
done
done
done
done
done
done



mode="persona_POI"

for p in 0.8
do
for q in 0.6
do
for lr in 0.0005
do
for dim in 300 
do
for Kneg in 10
do
for data in hongzhi
do
python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
--dataset_name ${data} --p_n2v ${p} --q_n2v ${q} \
--workers 26 --learning_rate ${lr} --K_neg ${Kneg} --mobility_ratio 0.7 --add_flag 0.9  > location_pred2/model4_${data}_${dim}_${lr}_${Kneg}_p${p}_q${q}m0.7_tune
done
done
done
done
done
done




mode="persona_POI"

for data in SaoPaulo 
do
    for lr in 0.0005 
    do 
        for dim in 300
        do 
            for Kneg in 25
            do 
                python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
                --dataset_name ${data} --bias_randomwalk --p_n2v 0.8 --q_n2v 0.8 \
                --workers 32 --learning_rate ${lr} --K_neg ${Kneg} --workers 12 --test > sao_output/${dim}_${lr}_${Kneg}
            done
        done 
    done
done



mode="persona_ori"

for data in TKY 
do
    for lr in 0.0005 
    do 
        for dim in 300
        do 
            for Kneg in 25 
            do 
                python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
                --dataset_name ${data} --bias_randomwalk --p_n2v 0.8 --q_n2v 0.8 \
                --workers 32 --learning_rate ${lr} --K_neg ${Kneg} --workers 12 --test 
            done
        done
    done
done










###################### SCRIPTs

mode="persona_POI"

for data in hongzhi
do
for p in 0.8 0.2 0.4 0.6 1
do
for q in 0.6 0.2 0.4 0.8 1
do
for dim in 256 32 64 128 512  
do
for wl in 80 20 50 80 100 10
do 
for nw in 10 2 5 7 15 20 
do 
for mr in 0.7 0.3 0.5 0.1 0.9
do 
mkdir ${data}_ablation_study_POI
python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
--dataset_name ${data} --p_n2v ${p} --q_n2v ${q} \
--workers 14 --learning_rate 0.0005 --K_neg 10 --mobility_ratio ${mr} --add_flag 0.9 \
--bias_randomwalk --num_walks ${nw} --walk_length ${wl} > ${data}_ablation_study_POI
done
done
done
done
done
done
done 


mode="persona_POI"

for data in hongzhi SaoPaulo KualaLampur Jakarta Istanbul
do
    mkdir ${data}_ablation_study_POI
    for p in 0.8 0.2 0.4 0.6 1
    do
        for q in 0.6
        do
        for dim in 256
        do
        for wl in 80
        do 
        for nw in 10
        do 
        for mr in 0.7 
        do 
        python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
        --dataset_name ${data} --p_n2v ${p} --q_n2v ${q} \
        --workers 14 --learning_rate 0.0005 --K_neg 10 --mobility_ratio ${mr} --add_flag 0.9 \
        --bias_randomwalk --num_walks ${nw} --walk_length ${wl} > ${data}_ablation_study_POI/pn2v${p}
        done
        done
        done
        done
        done
    done
    for p in 0.8
    do
        for q in 0.6 0.2 0.4 0.8 1
        do
        for dim in 256
        do
        for wl in 80
        do 
        for nw in 10
        do 
        for mr in 0.7 
        do 
        python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
        --dataset_name ${data} --p_n2v ${p} --q_n2v ${q} \
        --workers 14 --learning_rate 0.0005 --K_neg 10 --mobility_ratio ${mr} --add_flag 0.9 \
        --bias_randomwalk --num_walks ${nw} --walk_length ${wl} > ${data}_ablation_study_POI/qn2v${q}
        done
        done
        done
        done
        done
    done
    for p in 0.8
    do
        for q in 0.6
        do
        for dim in 256 32 64 128 512
        do
        for wl in 80
        do 
        for nw in 10
        do 
        for mr in 0.7 
        do 
        python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
        --dataset_name ${data} --p_n2v ${p} --q_n2v ${q} \
        --workers 14 --learning_rate 0.0005 --K_neg 10 --mobility_ratio ${mr} --add_flag 0.9 \
        --bias_randomwalk --num_walks ${nw} --walk_length ${wl} > ${data}_ablation_study_POI/dim${dim}
        done
        done
        done
        done
        done
    done
    for p in 0.8
    do
        for q in 0.6 
        do
        for dim in 256
        do
        for wl in 80 20 50 80 100 10
        do 
        for nw in 10
        do 
        for mr in 0.7 
        do 
        python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
        --dataset_name ${data} --p_n2v ${p} --q_n2v ${q} \
        --workers 14 --learning_rate 0.0005 --K_neg 10 --mobility_ratio ${mr} --add_flag 0.9 \
        --bias_randomwalk --num_walks ${nw} --walk_length ${wl} > ${data}_ablation_study_POI/wl${wl}
        done
        done
        done
        done
        done
    done
    for p in 0.8
    do
        for q in 0.6 
        do
        for dim in 256
        do
        for wl in 80 
        do 
        for nw in 10 2 5 7 15 20 
        do 
        for mr in 0.7 
        do 
        python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
        --dataset_name ${data} --p_n2v ${p} --q_n2v ${q} \
        --workers 14 --learning_rate 0.0005 --K_neg 10 --mobility_ratio ${mr} --add_flag 0.9 \
        --bias_randomwalk --num_walks ${nw} --walk_length ${wl} > ${data}_ablation_study_POI/nw${wl}
        done
        done
        done
        done
        done
    done
    for p in 0.8
    do
        for q in 0.6 
        do
        for dim in 256
        do
        for wl in 80 
        do 
        for nw in 10 
        do 
        for mr in 0.7 0.1 0.3 0.5 0.9
        do 
        python -u CMan_POI.py --input_type ${mode} --dim_emb ${dim} \
        --dataset_name ${data} --p_n2v ${p} --q_n2v ${q} \
        --workers 14 --learning_rate 0.0005 --K_neg 10 --mobility_ratio ${mr} --add_flag 0.9 \
        --bias_randomwalk --num_walks ${nw} --walk_length ${wl} > ${data}_ablation_study_POI/${mr}
        done
        done
        done
        done
        done
    done
done 

