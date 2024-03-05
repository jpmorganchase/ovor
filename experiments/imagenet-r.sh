# bash experiments/imagenet-r.sh
# experiment settings
GPUID='0 1 2 3'
CONFIG=configs/imnet-r_prompt.yaml
REPEAT=5

###############################################################

# OnePrompt
#
# prompt parameter args:
#    arg 1 = # tasks
#    arg 2 = e-prompt length (k and v combined)
#    arg 3 = g-prompt length (k and v combined)
python -u run.py --config $CONFIG --gpus $GPUID --repeat $REPEAT \
    --learner_type prompt --learner_name OnePrompt \
    --prompt_param 10 40 10

# OnePrompt with Virtual Outlier Regularization (40 epochs regular training and 10 epochs with regularization)
python -u run.py --config $CONFIG --gpus $GPUID --repeat $REPEAT \
    --learner_type prompt --learner_name OnePrompt \
    --prompt_param 10 40 10 --epochs 0 40 10

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
python -u run.py --config $CONFIG --gpus $GPUID --repeat $REPEAT \
    --learner_type prompt --learner_name CODAPrompt \
    --prompt_param 100 8 0.0

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
python -u run.py --config $CONFIG --gpus $GPUID --repeat $REPEAT \
    --learner_type prompt --learner_name DualPrompt \
    --prompt_param 10 40 10

# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
python -u run.py --config $CONFIG --gpus $GPUID --repeat $REPEAT \
    --learner_type prompt --learner_name L2P \
    --prompt_param 30 20 -1