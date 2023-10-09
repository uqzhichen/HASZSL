#python ../main.py --dataset CUB  --manualSeed 0 --xe 1 --attri 1e-2  --l_xe 1 --l_attri 1e-1 \
#--perturb_lr 3 --loops_adv 3 --entropy_cls 0.1 \
#--entropy_attention -1 --latent_weight 3 --zsl_weight 1 \
#--perturb_start_epoch 0 --prob_perturb 0.5  --attention_sup 0.0001 --device 0 \
#--pretrain_lr 1e-4 --pretrain_epoch 5 --classifier_lr 1e-6 --size 448 --calibrated_stacking 0.4 --weight_perturb 5 \
#--only_evaluate  --resume '../out/CUB_GZSL_id_0.pth' #76.5 69.6 74.1 71.8

# AWA2 GZSL
#python ../main.py --dataset AWA2  --manualSeed 7048 --all --xe 1 --attri 1e-4 --l_xe 1 --l_attri 1e-2 --avg_pool \
# --perturb_lr 3 --loops_adv 3 --entropy_cls 1 \
#--entropy_attention -0.1 --latent_weight 0.1 --zsl_weight 1 \
#--perturb_start_epoch 0 --prob_perturb 0.5  --attention_sup 0.0001 --device 0 \
#--pretrain_lr 1e-4 --classifier_lr 1e-5 --size 448 --calibrated_stacking 0.9 --pretrain_epoch 5 --weight_perturb 4.0 \
#--only_evaluate  --resume '../out/AWA2_GZSL_id_1.pth' #68.3 63.1 87.3 73.3
#
# AWA2 ZSL
#python ../main.py --dataset AWA2  --manualSeed 7048 --all --xe 1 --attri 1e-4 --l_xe 1 --l_attri 1e-2 --avg_pool \
# --perturb_lr 3 --loops_adv 3 --entropy_cls 1 \
#--entropy_attention -0.1 --latent_weight 0.1 --zsl_weight 1 \
#--perturb_start_epoch 0 --prob_perturb 0.5  --attention_sup 0.0001 --device 1 \
#--pretrain_lr 1e-4 --classifier_lr 1e-5 --size 448 --calibrated_stacking 0.9 --pretrain_epoch 5 --weight_perturb 4.0 \
#--only_evaluate  --resume '../out/AWA2_ZSL_id_1.pth' # 71.4



# SUN GZSL
python ../main.py --dataset SUN --manualSeed 2347 --xe 1 --attri 1e-4 --l_xe 1 --l_attri 5e-2 --avg_pool \
 --perturb_lr 3 --loops_adv 3 --entropy_cls 1 \
--entropy_attention 0.1 --latent_weight 0.1 --zsl_weight 1 \
--perturb_start_epoch 0 --prob_perturb 0.5  --attention_sup 0.1 --device 0 \
--classifier_lr 1e-5 --size 448 --calibrated_stacking 0.4 --pretrain_epoch 5 \
--pretrain_lr 1e-3 --weight_perturb 2 \
--only_evaluate  --resume '../out/SUN_GZSL_id_2.pth' # 61.0 42.8 38.9 40.8

# SUN ZSL
python ../main.py --dataset SUN --manualSeed 2347 --xe 1 --attri 1e-4 --l_xe 1 --l_attri 5e-2 --avg_pool \
 --perturb_lr 3 --loops_adv 3 --entropy_cls 1 \
--entropy_attention 0.1 --latent_weight 0.1 --zsl_weight 1 \
--perturb_start_epoch 0 --prob_perturb 0.5  --attention_sup 0.1 --device 0 \
--classifier_lr 1e-5 --size 448 --calibrated_stacking 0.4 --pretrain_epoch 5 \
--pretrain_lr 1e-3 --weight_perturb 2 \
--only_evaluate  --resume '../out/SUN_ZSL_id_2.pth' # 63.2 45.8 35.8 40.2