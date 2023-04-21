# ME5406-Project-2 Group 4
Robot Chaser Reinforcement Learning in Pybullet
Author: 
Liu Chenchen (DQN)
Ravi Girish  (PPO)
Zheng Jiezhi (DDPG)

#####1. Please create the environment by conda
```
conda create -n me5406_2_env python=3.8
conda activate me5406_2_env
conda install libffi==3.3
```

Note: Do this for libcudnn.8
```
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
```
LD_LIBRARY_PATH=/content/conda-env/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
```

#####2. Install requirements by pip
```
pip install -r requirements.txt
```

#####3. Test the model if you want to see the results directly
```
python3 test_DQN.py
python3 test_PPO.py
```

#####4. Train the model manually if you want
```
python3 train_DQN.py
python3 train_PPO.py
```
