timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# exp_name="${exp_name}_${timestamp}"
exp_name="AGRestormerv3_RDScityscapes_rainstreak"

python train.py -exp_name "${exp_name}" -train_batch_size 8 -learning_rate 0.0005 -num_epochs 100 \
    -lambda_loss 0.02 -theta_loss 0. -seed 3969 -debug