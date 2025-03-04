# 

Usage: python -m train.train [locomotion] --eval_frecuency m --n_envs n --seed i

python test [locomotion]

install.sh and upload_files.sh are scrpits for training in cloud, only tested with Azure

# Imporatant variables

## Access to information in mujoco

Information of the principal body: 

data.qvel = \[v_{x}, v_{y}, v_{z}, 
			\omega_{x}, \omega{y}, \omega{z},
			\dot{\theta_{1}}, \dot{\theta_{2}}, ...
			\dot{\theta_{n}} \]

data.qpos = \[x, y, z, 
			w, x, y, z,
			\theta_{1}, \theta_{2}, ...
			\theta_{n} \]

data.ncon = contacts number

mujoco._functions.mj_contactForce() this function get the force of a contact according to the order of appearance

Project structure 

1. debug_scene contains scripts that help to build the kinematic tree of the robot
2. environments contains a custom environments for each locomotion (maybe OOP inherence could help), this is the same for reward calculation.
3. wireless_communication is the viewer of python biding with a server udp and a client. For the moment it's only half-duplex communication
4. models are the neural networks trained for the specific locomotion.


├── best_model
│   └── best_model.zip
├── debug_scene
│   ├── Debug.py
│   ├── get_rotations.py
│   └── show_scene.py
├── environments
│   ├── jump_environment.py
│   ├── landing_environment.py
│   └── walk_environment.py
├── install.sh
├── logs
├── models
│   ├── jump
│   └── walk
├── our_robot
├── ppo_robot_tensorboard
├── README.md
├── rewards
│   ├── jump_environment_reward_calc.py
│   ├── landing_environment_reward_calc.py
│   └── walk_environment_reward_calc.py
├── test.py
├── train
│   ├── CurstomNetwork.py
│   ├── RobotUtilities.py
│   ├── train.py
│   └── VideoRecorder.py
├── unitree_go1/
├── upload_files.sh
├── videos
│   ├── jump
│   └── walk
├── viewer.py
└── wireless_comunication
    ├── client.py
    └── server.py
