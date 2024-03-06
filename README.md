# SparseTensorULM
This is the joint repository for the paper : [[arXiv]](https://arxiv.org/abs/2402.09359)

# Citation
If you use the code, please cite the corresponding papers:

For SparseTensor ULM
```
@misc{raubyPruningSparseTensor2024,
  title = {Pruning {{Sparse Tensor Neural Networks Enables Deep Learning}} for {{3D Ultrasound Localization Microscopy}}},
  author = {Rauby, Brice and Xing, Paul and Por{\'e}e, Jonathan and Gasse, Maxime and Provost, Jean},
  year = {2024},
  month = feb,
  number = {arXiv:2402.09359},
  eprint = {2402.09359},
  primaryclass = {cs, eess},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2402.09359},
  archiveprefix = {arxiv},
  keywords = {Computer Science - Computer Vision and Pattern Recognition,Electrical Engineering and Systems Science - Image and Video Processing,I.4.9}
}
```
For DeepST-ULM
```
@article{mileckiDeepLearningFramework2021b,
  title = {A {{Deep Learning Framework}} for {{Spatiotemporal Ultrasound Localization Microscopy}}},
  author = {Milecki, L{\'e}o and Por{\'e}e, Jonathan and Belgharbi, Hatim and Bourquin, Chlo{\'e} and Damseh, Rafat and {Delafontaine-Martel}, Patrick and Lesage, Fr{\'e}d{\'e}ric and Gasse, Maxime and Provost, Jean},
  year = {2021},
  month = may,
  journal = {IEEE Transactions on Medical Imaging},
  volume = {40},
  number = {5},
  pages = {1428--1437},
  issn = {1558-254X},
  doi = {10.1109/TMI.2021.3056951}
}
```

# Installation 
## Requirements 
Since we are using [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), you will need to match the requirements described on the github page.

If you can't install MinkowskiEngine, you still can use the code for the dense networks.

## Dependencies
First, install the dependencies for the dense networks 
``` 
pip -r requirements_dense.txt
```
Then, you can also install the dependencies for the sparse networks
``` 
pip -r requirements_dense.txt
```

## COMET-ML 
In order to monitor the training you can set up comet-ml: 
* create an account
* put your api key and put it in a .comet.config
* create a project 'sparsetensorulm' 
* update the workspace paramter in the script you are using 

## Dataset 

You can download the 2D dataset: [Zenodo](https://zenodo.org/records/10711657)

The 3D training set is available on demand. The weights for the network and the test set are available : 

Once you have downloaded the datasets, put it in a folder which path would be the `data_prefix`

# Reproducing results
We provide the script to train and evaluate the models. 

## Deep-stULM
To reproduce the results using Deep-stULM methods you can launch the training script and it will automatically the test evaluation for every concentration. 
```
python train_deep_stulm.py --config_path <config_path> --data_prefix <data_prefix> --save_path_test  <save_path_test>  --seed <seed> --tags <comet-ml_tag>

```

* `config_path`: path to the training config (e.g. `config/deep-stULM/deep-stULM.json` or `config/deep-stULM/deep-stULM_topk.json`).

* `data_prefix`: path where you put your dataset (e.g. for `folder1/folder2/dataset2` the prefix path is `folder1/folder2`).

* `save_path_test`: path where you want to store the inference results on the test sets.

* `seed`: seed to control the stochasticity. This doesn't work with MinkowskiEngine.

* `comet-ml_tag` : tag for the comet-ml project, this can be useful to filter the monitored trainings.

## Sparse 2D
To reproduce the results using sparse methods in 2D you can launch the training script and it will automatically the test evaluation for every concentration. 

```
export OMP_NUM_THREADS=2; python train_sparse_deep_stulm.py --config_path <config_path> --data_prefix <data_prefix> --save_path_test  <save_path_test>  --seed <seed> --tags <comet-ml_tag>

```

* `config_path`: path to the training config (each one of the json in `config/sparse_deepstULM_2D` corresponds to a config in the results section of the paper). 

* `data_prefix`: path where you put your dataset (e.g. for `folder1/folder2/dataset2` the prefix path is `folder1/folder2`).

* `save_path_test`: path where you want to store the inference results on the test sets.

* `seed`: seed to control the stochasticity. This doesn't work with MinkowskiEngine.

* `comet-ml_tag` : tag for the comet-ml project, this can be useful to filter the monitored trainings.

### configuration description
- `config/sparse_deepstULM_2D/sparse2d_deepSTULM_mask.json`: config that can be used to train a sparse model with a dense-to-sparse operation based on the cnn prediction 
- `config/sparse_deepstULM_2D/sparse2d_deepSTULM_threshold_010_cascaded.json`:  config using a threshold set to 0.1 for dense-to-sparse aswell as cascaded learning and pruning
- `config/sparse_deepstULM_2D/sparse2d_deepSTULM_threshold_010_pruning.json`: config using a threshold set to 0.1 for dense-to-sparse aswell as pruning
- `config/sparse_deepstULM_2D/sparse2d_deepSTULM_threshold_010_ds.json`: config using a threshold set to 0.1 for dense-to-sparse aswell as deep-supervision (intermediate loss) but no pruning
- `config/sparse_deepstULM_2D/sparse2d_deepSTULM_threshold_010.json`: config using a threshold set to 0.1 for dense-to-sparse 
- `config/sparse_deepstULM_2D/sparse2d_deepSTULM_topk.json`: config using a top k operation for dense-to-sparse 


## Sparse 3D
To reproduce the results using sparse methods in 3D you can launch the training script and it will automatically the test evaluation for every concentration. You can use the same training script as in 2D. You just need to change the configuration path to `config/sparse_deepstULM_3D/sparse3d_deepSTULM.json`

# TEST 

```
python test_deep_stulm.py --run_dir run_deepST_26_09_2023/13130191_amaranth_lungfish_4886 --num_workers 6 --batch_size 8 --save_path /home/raubyb/scratch/data/results_clean
```


```
python test_sparse_deep_stulm.py --run_dir runs_3D_sparse_28_09/13346549_inquisitive_hedgehog_5889 --num_workers 6 --batch_size 8 --save_path /home/raubyb/scratch/data/results_clean_debug
```

## Generating mask from dense-to-sparse CNN 

``` 
python generate_cnn_mask.py --run_dir dense2sparse_model_weights/13148035_dynamic_ladybug_8857 --mask_id dynamic_ladybug_8857
```
