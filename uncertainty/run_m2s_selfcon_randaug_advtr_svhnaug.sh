for i in {0..7}
do
    screen -dm bash -c "CUDA_VISIBLE_DEVICES=$i python3 main.py --exp_name summary_m2s_selfcon_randaug_advtr_svhnaug_$i --data.src MNIST --data.tar SVHN --data.aug_init svhnspec --data.aug randaug --training_type selfcon --train.init_advtr"
done
