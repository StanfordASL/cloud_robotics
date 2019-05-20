# what do we call the run of interest? [change this as we change simulation params]
PREFIX=facenet_4action_run3_newcost

RESULTS_DIR=$CLOUD_ROOT_DIR/scratch_results/train_RL_${PREFIX}/
rm -rf ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}

RL_TRAINER_DIR=$CLOUD_ROOT_DIR/simulate_RL/rl_trainer

# list of trace seeds to evaluate the RL agent on periodically during training
TEST_SEEDS_LIST="10,20,30,40,50,60,70,80"

# config parameters for training RL
CONFIG=$CLOUD_ROOT_DIR/simulate_RL/rl_configs/FourAction_RL_configs.ini

# train the RL agent, can view progress using tensorflow
# the environment name is FourAction
cd $RL_TRAINER_DIR
python3 $RL_TRAINER_DIR/main.py --config-path $CONFIG --test-seeds ${TEST_SEEDS_LIST} --env-name "FourAction" --mode 'train' --base-dir $RESULTS_DIR
