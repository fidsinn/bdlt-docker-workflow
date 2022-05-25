
# Deep Learning on SLURM - Deployment

Log into the ssh server and start a screen session.

## Running a command on the server

```
hostname
srun hostname
```

## Interactive shell

```
srun --pty bash -i
hostname
exit
```

Why can this become problematic?

## Executing containers

For Github Packages, go to [https://github.com/settings/tokens](https://github.com/settings/tokens) and generate an access token that can read packages.

For Docker Hub, go to [https://hub.docker.com/settings/security](https://hub.docker.com/settings/security) and generate an access token.

Upload the credentials to the server:

```
mkdir -p $HOME/.config/enroot/
touch $HOME/.config/enroot/.credentials
chmod 600 $HOME/.config/enroot/.credentials
echo "machine auth.docker.io login DOCKERHUB_USERNAME password DOCKERHUB_ACCESS_TOKEN_WITH_PERMISSION_READ_PACKAGES" >> $HOME/.config/enroot/.credentials
echo "machine ghcr.io login GITHUB_USERNAME password GITHUB_ACCESS_TOKEN_WITH_PERMISSION_READ_PACKAGES" >> $HOME/.config/enroot/.credentials
```

## Running a container

```
srun --mem=5g --container-image="python:3.8" --pty bash -i
python3
```

What does `--mem` do?

To exit:

```
exit()
exit
```

## Running and resuming a container instance with `--container-name`

```
srun --mem=5g --container-image="python:3.8" --container-name="python_test" --container-writable --pty bash -i
python3

import numpy
exit()

pip3 install numpy

exit

srun --mem=5g --container-name="python_test" --container-writable --pty bash -i
python3
import numpy
exit()

exit
```

Removing container instances:

```
srun enroot list
srun enroot remove -f pyxis_python_test
```

## Importing containers from a registry

This could be done directly from the `srun` command, but we will download a sqsh file as an intermediate step:

```
srun --mem=32g enroot import --output imagename.sqsh docker://REGISTRY#USERNAME/REPO:BRANCH
```

Here:

```
srun --mem=32g enroot import --output 20ng.sqsh docker://ghcr.io#USERNAME/REPO:BRANCH
```

We can now instantiate the container from the sqsh file:

```
srun --mem=5g --container-image="./20ng.sqsh" --container-name="20ng" --container-writable --pty bash -i
```

Let's stay in here to...

## Download the dataset

Because we will not finetune DistilBert itself, we the dataset from last week with precomputed embeddings.

```
apt-get install unzip
cd $HOME
curl https://files.webis.de/bdlt-ss22/20ng_bert_data.zip --output 20ng_bert_data.zip
unzip 20ng_bert_data.zip -d "20ng_bert_data"

exit
```

## Using GPU

Observe the difference:

```
srun --mem=5g --container-image="./20ng.sqsh" --container-name="20ng" python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"

srun --mem=5g --gres=gpu:1g.5gb --container-image="./20ng.sqsh" --container-name="20ng" python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

## Running the training

For the training, we will use the `training_unparametrized.py`. It is a modified version of last week's script.

```
srun --mem=5g --gres=gpu:1g.5gb --container-image="./20ng.sqsh" --container-name="20ng" python3 training_unparametrized.py

squeue

scancel JOB_ID
```

## Hyperparameter Sweeping with Weights and Biases

### Logging in

```
srun --mem=5g --container-image="./20ng.sqsh" --container-name="20ng" --pty bash -i
wandb login
exit
```

This will put your credentials into `~/.netrc`.

### Set up the following files:

- `training.py` (a parametrized version of our training script)
- `sweep_agent.py`
- `sweep_server.py`
- `sweep_server.sh`
- `sweep_agent.sh`
- `sweep_loop.sh`
- `run_sweep.sh` (remember to adjust your username here)

```
chmod u+x *.sh
```

### Run the sweep

```
./run_sweep.sh
```

### Have a look at the results on wandb.ai