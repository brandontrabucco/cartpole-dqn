# cartpole-dqn
In this project, we train an agent to balance a cartpole using Tensorflow Eager Execution and OpenAI Gym. A neural network is trained to approximate the Q-function, and the action that maximizes the expected discounted future reward is chosen at each timestep.

## Setup

Clone the project and enter the project folder.

```
git clone github.com/brandontrabucco/cartpole-dqn
cd cartpole-dqn
```

Install the project dependencies.

```
pip install -r requirements.txt
```

## Usage

Run a training session using train.py

```
python train.py
```

Run a visual testing session using test.py

```
python test.py
```
