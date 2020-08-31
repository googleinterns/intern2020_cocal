for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=1 python3 main.py --exp_name summary_m2s_srconly_noaug_$i --data.src MNIST --data.tar SVHN --training_type srconly
done
