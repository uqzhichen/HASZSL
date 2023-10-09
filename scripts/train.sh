#python ../main.py --dataset CUB  --manualSeed 0 --xe 1 --attri 1e-2  --l_xe 1 --l_attri 1e-1 \
#--perturb_lr 3 --loops_adv 3 --entropy_cls 0.1 --gzsl \
#--entropy_attention -1 --latent_weight 3 --zsl_weight 1 \
#--perturb_start_epoch 0 --prob_perturb 0.5  --attention_sup 0.0001 --device 1 \
#--pretrain_lr 1e-4 --pretrain_epoch 5 --classifier_lr 1e-6 --size 448 --calibrated_stacking 0.4 --weight_perturb 5
# ZSL 76.3/20 76.9/46  | GZSL (67.6 75.5 71.3/21) (66.9 77.2 71.7/46) CUB 3131


#python ../main.py --dataset AWA2  --manualSeed 7048 --all --xe 1 --attri 1e-4 --l_xe 1 --l_attri 1e-2 --avg_pool \
# --perturb_lr 3 --loops_adv 3 --entropy_cls 1 \
#--entropy_attention -0.1 --latent_weight 0.1 --zsl_weight 1 \
#--perturb_start_epoch 0 --prob_perturb 0.5  --attention_sup 0.0001 --device 1 \
#--pretrain_lr 1e-4 --classifier_lr 1e-5 --size 448 --calibrated_stacking 0.9 --pretrain_epoch 5 --weight_perturb 4.0
## 73.8 64.0 87.1 12  70.7 6 AWA  #73.3

#python main.py --perturb_lr 3 --loops_adv 3 --entropy_cls 1 \
#--entropy_attention -0.1 --latent_weight 0.1 --zsl_weight 1 -gzsl\
#--perturb_start_epoch 0 --prob_perturb 0.5  --attention_sup -0.0001 --device 1 \
#--pretrain_lr 1e-4 --classifier_lr 1e-5 --size 448 --calibrated_stacking 0.9 --pretrain_epoch 5 --weight_perturb 4.0 # 73.3 62.9 87.9 12  72.1 6


python ../main.py --dataset SUN --manualSeed 2347 --xe 1 --attri 1e-4 --l_xe 1 --l_attri 5e-2 --avg_pool \
 --perturb_lr 3 --loops_adv 3 --entropy_cls 1 --gzsl \
--entropy_attention 0.1 --latent_weight 0.1 --zsl_weight 1 \
--perturb_start_epoch 0 --prob_perturb 0.5  --attention_sup 0.1 --device 1 \
--classifier_lr 1e-5 --size 448 --calibrated_stacking 0.4 --pretrain_epoch 5 \
--pretrain_lr 1e-3 --weight_perturb 2 #  41.2 17 SUN
