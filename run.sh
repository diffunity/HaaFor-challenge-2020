python preprocess.py

nohup python main.py --epochs 5 > train_1_nohup.txt

-- albert
nohup python main.py \
    --batch_size 4 \
    --pretrainedPATH ./model_pretrained/albert-proper/
    --destination_folder ./result/albert-proper/ > ./result/albert-proper/train_albert_base_v2.txt

-- xlnet
nohup python main.py \
    --model xlnet-base-cased \
    --pretrainedPATH ./model_pretrained/xlnet/ \
    --batch_size 2 \
    --gradient_accumulation_steps 32
    --destination_folder ./result/xlnet-proper/ > ./result/xlnet-proper/train_based_cased.txt

-- albert with scheduler
nohup python main.py \
    --batch_size 4 \
    --gradient_accumulation_steps 32 \
    --pretrainedPATH ./model_pretrained/albert-scheduler/ 
    --destination_folder ./result/albert-scheduler/ > ./result/albert-scheduler/train_albert_base_v2.txt

-- xlnet scheduler
nohup python main.py \
    --model xlnet-base-cased \
    --pretrainedPATH ./model_pretrained/xlnet-scheduler/ \
    --batch_size 4 \
    --gradient_accumulation_steps 32 \
    --destination_folder ./result/xlnet-scheduler/ \
    --epochs 5 \
    --cls_pos -1 > ./result/xlnet-scheduler/train_xlnet_scheduler.txt