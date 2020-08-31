for i in {0..4}
do
    screen -dm bash -c "CUDA_VISIBLE_DEVICES=$i python3 main.py --exp_name summary_m2s_selfcon_svhnaug_advtr_randaug_$i --data.src MNIST --data.tar SVHN --data.aug_init randaug --data.aug svhnspec --training_type selfcon --train.init_advtr"
done
