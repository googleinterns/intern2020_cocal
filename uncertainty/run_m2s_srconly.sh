for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=2 python3 main.py --exp_name summary_m2s_srconly_$i --data.src MNIST --data.tar SVHN --data.aug svhnspec --training_type srconly
done
