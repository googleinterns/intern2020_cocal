CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 plots/compute_stats.py --snapshot_prefix snapshots_final_presentation/summary_m2s_selfcon_svhnaug_advtr_svhnaug --data.src MNIST --data.tar SVHN
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 plots/compute_stats.py --snapshot_prefix snapshots_final_presentation/summary_m2s_selfcon_svhnaug_advtr_randaug --data.src MNIST --data.tar SVHN
