# Hypermodel for Self Supervised Domain Adaptation
Stanford CS 329D (Fall 2021) Course Project


## Installation

### Install the required packages
Use pip to install WILDS as recommended:
```bash
pip install wilds
```
To run `fmow_wilds_evaluate.py`, requires the [manual installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-binaries) of the `torch-scatter` and `torch-geometric` packages.

### Clone the relevant repositories
Clone the official [WILDS repository](https://github.com/p-lambda/wilds) alongside this repository

```bash
git clone https://github.com/p-lambda/wilds.git
git clone https://github.com/SConsul/multi-head-training-da.git
```
## Usage

### Obtaining FMoW-mini

### Multi-Headed Training
To run the multi-headed training for domain adaptation on the fmow_mini dataset, use the following command:
```bash
python main.py --target_domain <test/val> --num_pseudo_heads <num_pseudo_heads>
```
**Required arguments:**
| Parameter                 | Choices       | Description   |	
| :------------------------ |:-------------:| :-------------|
|`--target_domain`|test / val| choice of dataset to do domain adaptation on
|`--num_pseudo_heads`| 0 | can be kept to any number greater than 1 for multi-headed training or 0 for vanilla ERM

**Optional arguments:**
Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
|`--batch_size`| 64| batch size used, can be varied depending on the available GPU memory
|`--threshold`| 0.9| threshold on confidence used while pseudo-labelling
|`--bootstrap`|False| can be set to True to use a variant of bootstrapping for source training of pHeads
|`--num_epochs`| 30| no. of epochs of source training. Set to 50 for ERM
|`--num_pseudo_steps`|10| no. of domain adaptation steps are done
|`--num_adapt_epochs`| 2| no. of epochs of training during each domain adaptation step
|`--orig_frac`| 1.0| if less than 1, a randomly sampled portion of the train and target data is used
|`--saved_model_path`|None| path of saved model weights
|`--epoch_offset`|0| can be set to a positive number to start source training from a particular epoch
|`--da_step_offset`|0| an be set to a positive number to start domain adaptation from a particular step

### Evaluation
While the training scripts ensure periodic evaluation of the model on the target dataset, it is possible to evaluate the performance of a saved model by running:
```bash
python evaluate.py --target_domain <test/val> --num_pseudo_heads <num_pseudo_heads> --model_path <path_to_saved_weights>
```
**Required arguments:**
| Parameter                 | Choices       | Description   |	
| :------------------------ |:-------------:| :-------------|
|`--target_domain`|test / val| choice of dataset to do evaluate on
|`--num_pseudo_heads`| 0 | can be kept to any number greater than 1 for multi-headed training or 0 for vanilla ERM
|`--model_path`| None| path of saved model weights

**Optional arguments:**
| Parameter                 | Choices       | Description   |	
| :------------------------ |:-------------:| :-------------|
|`--batch_size`|64| batch size used, can be varied depending on the available GPU memory
|`--frac`|1.0| if less than 1, the model is evaluated on a randomly sampled portion of target data

## Credits
The repository is part of a course project for Stanford CS 329D (Fall 2021), taught by [Prof. Tatsunori Hashimoto](https://thashim.github.io/).

The base code has been taken from the official [WILDS repository](https://github.com/p-lambda/wilds)

Team Members: Anmol Kagrecha ([**@akagrecha**](https://github.com/akagrecha)), Sarthak Consul ([**@SConsul**](https://github.com/SConsul))
