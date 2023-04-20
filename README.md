# ME5406-Project-2
Chaser RL Pybullet
```
conda create -n me5406_2_env python=3.8
```
```
conda activate me5406_2_env
```
```
pip install -r requirements.txt
```
```
conda install libffi==3.3
```
```
# If you want to train models
python3 train_DQN.py
python3 train_PPO.py
```
```
# If you want to test models
python3 test_DQN.py
python3 test_PPO.py