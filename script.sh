
################################## PERSONA ######################################

# persona: For one, we need to run splitter to create persona graph
for DATA in Istanbul Jakarta KualaLampur Saopaulo TKY NYC hongzhi
do
    python src/main.py --edge-path input/${DATA}_friends.csv --lbsn ${DATA}
done 

# persona: then we just need to run LBSN with persona graphs
for DATA in Istanbul Jakarta KualaLampur Saopaulo TKY NYC hongzhi
do
    python -u main.py --dataset_name ${DATA} --mode POI --input_type persona > results/POI_persona_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type persona > results/friend_persona_${DATA}
done
hongzhi NYC TKY


for DATA in TKY
do
    python src/main.py --edge-path input/${DATA}_friends.csv --lbsn ${DATA}
done

# persona: then we just need to run LBSN with persona graphs
for DATA in  TKY
do
    python -u main.py --dataset_name ${DATA} --mode POI --input_type persona
    python -u main.py --dataset_name ${DATA} --mode friend --input_type persona
done

########################### PESONA POI ################################################
for DATA in hongzhi NYC TKY Istanbul Jakarta KualaLampur Saopaulo
do
python src/main.py --edge-path input/${DATA}_friendPOI.csv --lbsn ${DATA} --listPOI input/location_${DATA}
done


python src/main.py --edge-path input/hongzhi_friends.csv --lbsn hongzhi --listPOI input/location_hongzhi
python src/main.py --edge-path input/NYC_friends.csv --lbsn NYC --listPOI input/location_NYC
python src/main.py --edge-path input/TKY_friends.csv --lbsn TKY --listPOI input/location_TKY
python src/main.py --edge-path input/Istanbul_friends.csv --lbsn Istanbul --listPOI input/location_Istanbul
python src/main.py --edge-path input/Jakarta_friends.csv --lbsn Jakarta --listPOI input/location_Jakarta
python src/main.py --edge-path input/KualaLampur_friends.csv --lbsn KualaLampur --listPOI input/location_KualaLampur
python src/main.py --edge-path input/Saopaulo_friends.csv --lbsn Saopaulo --listPOI input/location_Saopaulo

########################### PESONA FRIEND , SPLIT POI ################################################
python src/main.py --edge-path input/hongzhi_friendPOI.csv --edge-path-friend input/hongzhi_friends.csv --lbsn hongzhi --listPOI input/location_hongzhi

for DATA in hongzhi NYC TKY Istanbul Jakarta KualaLampur
do
python src/main.py --edge-path input/${DATA}_friendPOI.csv --location-dict Suhi_output/location_dict_${DATA} --edge-path-friend input/${DATA}_friends.csv --lbsn ${DATA} --listPOI input/location_${DATA}
done

################################## ORIGRAPH C++ ######################################

mkdir results/cpp

for DATA in KualaLampur 
do
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat  --clean --num_epochs 1 --walk_length 6
done

################################## FOR PYTHON #########################################

for DATA in NYC
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py --clean --num_epochs 10 > results/friend_ori_${DATA}
done

for DATA in Jakarta
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py --clean --num_epochs 10 > results/friend_ori_${DATA}
done


for DATA in Istanbul
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py --clean --num_epochs 10 > results/friend_ori_${DATA}
done


for DATA in SaoPaulo
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py --clean --num_epochs 10 > results/friend_ori_${DATA}
done


for DATA in KualaLampur
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py --clean --num_epochs 10 > results/friend_ori_${DATA}
done

for DATA in TKY
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py --clean --num_epochs 10 > results/friend_ori_${DATA}
done


python -u main.py --dataset_name KualaLampur --mode friend --input_type mat --clean --num_epochs 10




for DATA in hongzhi
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --clean --num_epochs 1 --lea
done

for DATA in NYC
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py > results/friend_ori_${DATA}
done

for DATA in Jakarta
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py > results/friend_ori_${DATA}
done


for DATA in Istanbul
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py > results/friend_ori_${DATA}
done


for DATA in SaoPaulo
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py > results/friend_ori_${DATA}
done


for DATA in KualaLampur
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py > results/friend_ori_${DATA}
done

for DATA in TKY
do
    # python -u main.py --dataset_name ${DATA} --mode POI --input_type mat --py > results/POI_ori_${DATA}
    python -u main.py --dataset_name ${DATA} --mode friend --input_type mat --py >
done
      
