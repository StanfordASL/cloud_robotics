overall pipeline to train and evaluate an RL agent for offloading


EVALUATE A TRAINED RL AGENT

./eval_pretrain_RL_FourAction_fnet.sh

	1. call RL_TRAINER_DIR/evaluate_RL_offload_utils.py to run RL agent on TEST traces and save results


	2. call FourAction_policy_rollouts.py
		to run the baselines on the SAME test traces

	3. call boxplot_FourAction_env.py to plot all the results over traces to see RL's gains
	 

HOW TO VIEW TENSORBOARD ON A REMOTE MACHINE
ssh -L 16006:127.0.0.1:6006 username@remote_ip

then go to localhost:16006 on local machine, maps port 6006 to 16006 on localhost

