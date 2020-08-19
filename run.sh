python preprocess.py

nohup python main.py --epochs 5 > train_1_nohup.txt

nohup python main.py \
    --batch_size 4 \
    --pretrainedPATH ./model_pretrained/albert-proper/ > ./result/albert-proper/train_albert_base_v2.txt

nohup python main.py \
    --model xlnet-base-cased \
    --pretrainedPATH ./model_pretrained/xlnet/ \
    --batch_size 2 \
    --gradient_accumulation_steps 32 > ./result/xlnet-proper/train_based_cased.txt