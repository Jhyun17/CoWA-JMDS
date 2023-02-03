#################################### Basic Settings ####################################
GPU=0
SAVEPATH='./ckps'

# No learning rate decay
LRGAMMA=0.0

COEFF=JMDS

SEED=3

#################################### VISDA-C ####################################

MAXEPOCH=30
INTERVAL=30

LR=1e-2
ALPHA=0.2
WARM=0.0

for S in 0 1 2 3
do
    # source pre-training
    python image_source.py --trte val --output $SAVEPATH/source/ --da uda --gpu_id $GPU --net resnet50 --lr 1e-2 --dset office-home --max_epoch 50 --s $S --seed $SEED
    
    # CoWA-JMDS training
    echo CLOSED-SET SETTING
    echo GPU: $GPU
    echo WARM: $WARM
    echo COEFF: $COEFF
    echo ALPHA: $ALPHA
    echo LR: $LR
    echo SEED: $SEED

    python image_target_CoWA.py --da uda --dset office-home --gpu_id $GPU --net resnet50 --s $S --output_src $SAVEPATH/source/ --output $SAVEPATH/target/ --max_epoch $MAXEPOCH --interval $INTERVAL --alpha $ALPHA --layer wn --smooth 0.1 --lr $LR --batch_size 64 --lr_gamma $LRGAMMA --seed $SEED --coeff $COEFF --warm $WARM
done 
