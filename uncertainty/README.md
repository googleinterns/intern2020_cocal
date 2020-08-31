# Self-training for UDA

## Shift: MNIST -> USPS

### Ablation study: Effect on consistency and augmantation (while using the same initialization approach)

Initialization: Advtr + svhn-aug <br/>
Self-training: consistency + svhn-aug
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp_name summary_m2s_selfcon_svhnaug_advtr_svhnaug_0 --data.src MNIST --data.tar SVHN --data.aug_init svhnspec --data.aug svhnspec --training_type selfcon --train.init_advtr
```
Also see `run_m2s_selfcon_svhnaug_advtr_svhnaug.sh`.
<br/><br/>

Initialization: Advtr + svhn-aug <br/>
Self-training: no-consistency + svhn-aug 
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp_name summary_m2s_self_svhnaug_advtr_svhnaug_0 --data.src MNIST --data.tar SVHN --data.aug_init svhnspec --data.aug svhnspec --training_type self --train.init_advtr
```
Also see `run_m2s_self_svhnaug_advtr_svhnaug.sh`.
<br/><br/>

Initialization: Advtr + svhn-aug <br/>
Self-training: consistency + rand-aug 
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp_name summary_m2s_selfcon_randaug_advtr_svhnaug_0 --data.src MNIST --data.tar SVHN --data.aug_init svhnspec --data.aug randaug --training_type selfcon --train.init_advtr
```
Also see `run_m2s_selfcon_randaug_advtr_svhnaug.sh`.
<br/><br/>


### Ablation study: Effect on initalization approaches (while using the same self-training approach)

Initialization: advtr + rand-aug <br/>
Self-training: consistency + svhn-aug
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp_name summary_m2s_selfcon_svhnaug_advtr_randaug_0 --data.src MNIST --data.tar SVHN --data.aug_init randaug --data.aug svhnspec --training_type selfcon --train.init_advtr
```
Also see `run_m2s_selfcon_svhnaug_advtr_randaug.sh`.
<br/><br/>

Initialization: advtr + no-aug <br/>
Self-training: consistency + svhn-aug
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp_name summary_m2s_selfcon_svhnaug_advtr_noaug_0 --data.src MNIST --data.tar SVHN --data.aug_init '' --data.aug svhnspec --training_type selfcon --train.init_advtr
```
Also see `run_m2s_selfcon_svhnaug_advtr_noaug.sh`.
<br/><br/>

Initialization: srconly + svhn-aug <br/>
Self-training: consistency + svhn-aug
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp_name summary_m2s_selfcon_svhnaug_srconly_svhnaug_0 --data.src MNIST --data.tar SVHN --data.aug_init svhnspec --data.aug svhnspec --training_type selfcon
```
Also see `run_m2s_selfcon_svhnaug_srconly_svhnaug.sh`.
<br/><br/>

Initialization: srconly + rand-aug <br/>
Self-training: consistency + svhn-aug
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp_name summary_m2s_selfcon_svhnaug_srconly_randaug_0 --data.src MNIST --data.tar SVHN --data.aug_init randaug --data.aug svhnspec --training_type selfcon
```
Also see `run_m2s_selfcon_svhnaug_srconly_randaug.sh`.
<br/><br/>

Initialization: srconly + no-aug <br/>
Self-training: consistency + svhn-aug
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp_name summary_m2s_selfcon_svhnaug_srconly_noaug_0 --data.src MNIST --data.tar SVHN --data.aug_init '' --data.aug svhnspec --training_type selfcon
```
Also see `run_m2s_selfcon_svhnaug_srconly_noaug.sh`.
<br/><br/>





