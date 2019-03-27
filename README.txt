# cloud_robotics
A deep reinforcement learning approach to cloud robotics.

NOTE: Before using this repo in python3, make sure to export the base code directory
on your machine to as CLOUD_ROBOTICS_DIR in your bashrc, like:
    export CLOUD_ROOT_DIR='/Users/csandeep/Documents/cloud_robotics' 

Key Directories:

utils: generic utils to plot and load jsons etc.

simulate_RL: train an RL agent in tensorflow for cloud offloading

hardware_experiments: run an RL agent on a Jetson TX2 embedded GPU with IP camera

data: where to put pre-recorded videos or faces of people to train FaceNet DNN models

DNN_models: 
    - pretrained, base vision DNNs to detect faces and the OpenFace models

scratch_results:
    - where outputs of runs go, such as logfiles and models

Citations:
We modified publicly available code and used vision DNNs from:

1. pyimagesearch: https://www.pyimagesearch.com/

2. OpenFace project: https://cmusatyalab.github.io/openface/

3. jsonsockets library


