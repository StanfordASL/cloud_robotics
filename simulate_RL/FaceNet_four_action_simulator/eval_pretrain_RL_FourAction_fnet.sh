# run a pre-trained RL agent against benchmarks, plot the boxplots of all results

# where code to train RL agent is
RL_TRAINER_DIR=$CLOUD_ROOT_DIR/simulate_RL/rl_trainer/

# where simulator for offloading is 
BASE_4ACTION_DIR=$CLOUD_ROOT_DIR/simulate_RL/FaceNet_four_action_simulator/

# what do we call the run of interest? [change this as we change simulation params]
PREFIX=facenet_4action_run3_newcost

CONFIG=$CLOUD_ROOT_DIR/simulate_RL/rl_configs/FourAction_RL_configs.ini

# where did we save the model checkpoint and parameters?
###############
MODEL_SAVE_PATH=$CLOUD_ROOT_DIR/DNN_models/RL_checkpoints/train_RL_${PREFIX}/model/
#MODEL_SAVE_PATH=$CLOUD_ROOT_DIR/scratch_results/train_RL_${PREFIX}/model/

# where all logs and outputs go
LOG_PATH=$CLOUD_ROOT_DIR/scratch_results/RL_data_${PREFIX}/
rm -rf $LOG_PATH
mkdir -p $LOG_PATH

BASE_RESULTS_DIR=$CLOUD_ROOT_DIR/scratch_results/

cd ..

ENV_NAME="FourAction"

# what network conditions did we train on? each is a list of N_BUDGET/T
# this is the fraction of cloud queries an agent gets at the START of its interval
TRAIN_QUERY_LIST="0.10,0.20,0.50,0.70,1.0" 

# this is for the test traces
QUERY_LIST="0.05,0.15,0.30,0.45,0.80,0.9,0.95" 
#QUERY_LIST="0.10" 

# seeds for the stochastic traces to test on
TEST_SEEDS="10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200"

# uncomment for smaller tests
#TEST_SEEDS="10,20,30,40,50"
#TEST_SEEDS="10"

# 1. EVALUATE A PRE-TRAINED RL AGENT on the new test traces and log the results
python3 $RL_TRAINER_DIR/evaluate_RL_offload_utils.py --config-path $CONFIG --test-seeds $TEST_SEEDS --env-name $ENV_NAME --log-path $LOG_PATH --model-save-path $MODEL_SAVE_PATH --query-budget-fraction-list $QUERY_LIST

# plot the RL agent
#cd $BASE_4ACTION_DIR
#python $BASE_4ACTION_DIR/timeseries_plot_FourAction.py --RL_present 'RL' --prefix $PREFIX 

# run the baselines
python3 $BASE_4ACTION_DIR/FourAction_policy_rollouts.py --prefix $PREFIX --test-seeds $TEST_SEEDS --query-budget-fraction-list $QUERY_LIST --base-results-dir $LOG_PATH

# plot a boxplot of all different controllers
python3 $BASE_4ACTION_DIR/pubQuality_boxplot_FourAction_env.py --prefix $PREFIX --RL_present 'both' --base-results-dir $LOG_PATH

# plot a pareto optimal covariance plot shown in paper
python3 $BASE_4ACTION_DIR/loss_cost_pareto_plot_ellipsoid.py --prefix $PREFIX --RL_present 'both' --base-results-dir $LOG_PATH
