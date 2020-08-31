for i in {4..7}
do
    screen -dm bash -c "CUDA_VISIBLE_DEVICES=$i python3 main.py --exp_name summary_m2s_selfcon_advtrinit_noaug_$i --data.src MNIST --data.tar SVHN --data.aug_init '' --data.aug svhnspec --training_type selfcon --train.init_advtr"
done
