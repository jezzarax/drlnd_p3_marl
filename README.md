Solution to the first project of the deep reinforcement learning nanodegree at Udacity.

## Problem definition

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

## Usage

### Preparation

Please ensure you have [Pipenv](https://pipenv.readthedocs.io/en/latest/) installed. Clone the repository and use `pipenv --three install` to create yourself an environment to run the code in. Otherwise just install the packages mentioned in Pipfile.

Due to the transitive dependency to tensorflow that comes from unity ml-agents and the [bug](https://github.com/pypa/pipenv/issues/1716) causing incompatibility to jupyter you might want to either drop the jupyter from the list of dependencies or run `pipenv --three install --skip-lock` to overcome it.

To activate a virtual environment with pipenv issue `pipenv shell` while in the root directory of the repository.

For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* [Headless linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

After unzipping the virtual environment binary you need to set a `DRLUD_P3_ENV` shell environment which must point to the binaries of the Unity environment. Example of for Mac OS version of binaries it might be 
```
DRLUD_P3_ENV=../deep-reinforcement-learning/p3_navigation/Tennis.app; export DRLUD_P3_ENV
```

### Training (easy way)

Just follow the [training notebook](Training.ipynb).

### Training (proper way)

The executable part of code is built as a three-stage pipeline comprising of
* training pipeline
* analysis notebook
* demo notebook

The training pipeline was created with the idea of helping the researcher to keep track of his experimentation process as well as keeping the running results. The training process is spawned by executing the `trainer.py` script and is expected to be idempotent to the training results, i.e. if the result for a specific set of hyperparameters already exists and persisted, the trainer will skip launching a training session for this set of hyperparameters.

The sets of hyperparameters for training are defined inside of `trainer.py` in the `simulation_hyperparameter_reference` dictionary which is supposed to be append-only in order to keep consistency of the training result data. Each of the hyperparameters sets will produce a file with a scores of number of runs of an agent which will be stored inside of `./hp_search_results` directory with an id referring to the key from the `simulation_hyperparameter_reference` dictionary. The neural networks weights for every agent training run will be stored in the same directory with the relevant hyperparameters key as well as random seed used.

To train an agent with a new set of hyperparameters just add an item into `simulation_hyperparameter_reference` object. Here's an example of adding an item with id `25` after existing hyperparameter set with id `24`:

```
simulation_hyperparameter_reference = {
###
###  Skipped some items here
###

    25:   ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-3, 0,    1, True,  False, False, 0,  1),
    26:   ac_parm(-1, -1, int(1e5), "", 256,  0.99, 1e-3, 1e-4, 1e-4, 0,    1, False, False, True, 700, 1),
}
```

The set of hyperparameters is represented as an instance of a namedtuple `ac_parm` which has the following set of fields in respective order:

* Technical value, leave `-1`
* Technical value, leave `-1` (yup, second time)
* memory_size - number of experiences to persist
* Technical value, leave `""`
* batch_size - numer of experiences to use during a learning session
* gamma
* tau
* lr_actor
* lr_critic
* weight_decay
* number of times to rerun the simulation
* noise_enabled
* gradient_clipping
* enable boostrapping that will create several episodes with random actions
* number of episodes for bootstrapping
* learn_every - number of steps between learning sessions

The meaning and effects of other values for these field are discussed in the [hyperparameter search notebook](Training_hyperparameter_search_analysis.ipynb). 

## Implementation details

Two neural network architectures are defined in the `models.py` file. 
* Actor class implement a three-layer neural network that learns to project the state on to an action.
* Critic class implements a three-layer neural network that learns to map state and an action into a Q-value.

For further notes regarding the actor/critic model and the neural network architecture please check the [report notebook](Report.ipynb).

## Results 

Please check the [following notebook](Report.ipynb) for the best set of hyperparameters I managed to identify.
