[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

The goal of this project is to train an agent to navigate and collect bananas in a large, square world. The agent receives a reward of +1 for collecting a yellow banana and a reward of -1 for collecting a blue banana. Hence, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. The agent is able to perform the following four discrete actions:
- **`0`** - move forward;
- **`1`** - move backward;
- **`2`** - turn left;
- **`3`** - turn right.

The task is episodic and the agent must get an average score of +13 over 100 consecutive episodes in order to solve the environment.

### Setting up the project

0. Clone this repository.

1. Download the environment from one of the links below. Select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Put the file in the repository folder and unzip (or decompress) the file. 

### Training and testing the agent

#### Training the agent

Run the `agent_train.py` script as follows:

```bash
$ python agent_train.py --episodes 1000 --weights weights.pth --plot plot.png
```

The `--episodes` parameter specifies the maximum number of episodes, the `--weights` parameter specifies the name of the file to which the weights of the trained network need to be saved, and the `--plot` parameter specifies the name of the file to which the plot of the obtained scores needs to be saved.

#### Testing the agent

Run the `agent_test.py` script as follows:

```bash
$ python agent_test.py --weights weights.pth
```

The `--weights` parameter specifies the name of the file from which the weights of the network need to be loaded.
