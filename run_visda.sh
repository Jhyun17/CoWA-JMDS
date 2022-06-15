#################################### Basic Settings ####################################
GPU=0
SAVEPATH='./ckps'

# No learning rate decay
LRGAMMA=0.0

COEFF=JMDS

SEED=3

#################################### VISDA-C ####################################

MAXEPOCH=15
INTERVAL=15

LR=1e-2
ALPHA=2.0
WARM=1.0

for S in 0
do
    # source pre-training
    python image_source.py --trte val --output $SAVEPATH/source/ --da uda --gpu_id $GPU --net resnet101 --lr 1e-3 --dset VISDA-C --max_epoch 10 --s $S --seed $SEED
    
    # CoWA-JMDS training
    echo CLOSED-SET SETTING
    echo GPU: $GPU
    echo WARM: $WARM
    echo COEFF: $COEFF
    echo ALPHA: $ALPHA
    echo LR: $LR
    echo SEED: $SEED

    python image_target_CoWA.py --da uda --dset VISDA-C --gpu_id $GPU --net resnet101 --s $S --output_src $SAVEPATH/source/ --output $SAVEPATH/target/ --max_epoch $MAXEPOCH --interval $INTERVAL --alpha $ALPHA --layer wn --smooth 0.1 --lr $LR --batch_size 64 --lr_gamma $LRGAMMA --seed $SEED --coeff $COEFF --warm $WARM
done 
