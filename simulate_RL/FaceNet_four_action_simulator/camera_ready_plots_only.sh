# just plot for camera ready

# where simulator for offloading is 
BASE_4ACTION_DIR=$CLOUD_ROOT_DIR/simulate_RL/FaceNet_four_action_simulator/

# what do we call the run of interest? [change this as we change simulation params]
PREFIX=facenet_4action_run3_newcost

# where did we save the model checkpoint and parameters?
###############
# where all logs and outputs go
LOG_PATH=$CLOUD_ROOT_DIR/scratch_results/RL_data_${PREFIX}/

cd ..

# what network conditions did we train on? each is a list of N_BUDGET/T
# this is the fraction of cloud queries an agent gets at the START of its interval
# plot a boxplot of all different controllers
python3 $BASE_4ACTION_DIR/pubQuality_boxplot_FourAction_env.py --prefix $PREFIX --RL_present 'both' --base-results-dir $LOG_PATH

# plot a pareto optimal covariance plot shown in paper
python3 $BASE_4ACTION_DIR/loss_cost_pareto_plot_ellipsoid.py --prefix $PREFIX --RL_present 'both' --base-results-dir $LOG_PATH

