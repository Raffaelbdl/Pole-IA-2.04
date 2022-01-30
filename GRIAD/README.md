# How to set up your environment

To start you have to configure your virtual environment in Python 3.7 (This is a requirements from CARLA), then run these commands :

```bash
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html # To install JAX CUDA
pip install -r ./requirements.txt
```

# How to create a dataset

We don't share the dataset because of its size

First open CARLA by typing in CARLA folder :

```bash
./CarlaUE4.sh --carla-world-port=3000 -opengl
```

Then launch dataset_maker.py

```bash
python ./fetch_dataset/dataset_maker.py
```

# How to run RL agent

First open CARLA by typing in CARLA folder :

```bash
./CarlaUE4.sh --carla-world-port=3000 -opengl
```

Then launch rl_main.py configured as you wanted

```bash
python ./rl/rl_main.py
```
