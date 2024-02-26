import os
import argparse
import torch
import torch.utils.data as data
import numpy as np
import json
import tqdm
import MinkowskiEngine as ME
from datasets.SparseDataset import SparseDataset, sparse2sparse_collation_fn
    
def count_one_dataset(concentration, config):
    filename = 'test_set_{}.h5'.format(
        concentration)
    dataset = SparseDataset(filename, **config["data"])
    val_loader = data.DataLoader(dataset,             
                                 batch_size=1, 
                                 shuffle=False,
                                 collate_fn = sparse2sparse_collation_fn,
                                 drop_last=True)

    n_pixels = 0

    with torch.no_grad():
        for batch_idx, batch in tqdm.tqdm(enumerate(val_loader)):
            coords_x, feats_x, coords_y, feats_y = batch
            n_pixels += feats_x.shape[0]
        print(concentration, ': ', n_pixels/len(dataset))
    return(n_pixels/len(dataset))
            
def compute_sparsity(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_prefix', required=True,
                        type=str, help='path to the config')
    parser.add_argument('--config_dir', required=True,
                        type=str, help='path to the directory containing the config')
    parser.add_argument('--save_path', required=True,
                        type=str, help='path to save directory')
    args = parser.parse_args(raw_args)

    file_list = os.listdir(args.config_dir)
    
    results = {"1MB":{},"5MB":{},"10MB":{},"20MB":{}}

# Filtering out only JSON files
    config_list = [file for file in file_list if file.endswith('.json')]
    for config_filename in config_list:
        print(config_filename)
        with open(os.path.join(args.config_dir,config_filename)) as json_file:
            config = json.load(json_file)
        config['data']['data_path'] = os.path.join(args.data_prefix,
                                                config['data']['data_path'] )
        if 'thresholdSparsifier' in config['data'].keys():
            method_key="threshold"
            method_value=config['data']['thresholdSparsifier']
        elif 'topk' in config['data'].keys(): 
            method_key="topk"
            method_value=config['data']['topk']

        elif 'mask_id' in config['data'].keys(): 
            method_key="mask"
            method_value=config['data']['mask_id']
        else :
            raise NotImplementedError
        print(method_key+'_'+str(method_value))
        for concentration in results.keys():  
            n_pixels = count_one_dataset(concentration,config)
            results[concentration][method_key+'_'+str(method_value)]={'method':method_key,'value':method_value, "n_pixels":n_pixels}
            print(results)
    
    # Convert and write JSON object to file
    os.makedirs(args.save_path, exist_ok=True)

    with open(os.path.join(args.save_path,
    "sparsity_results.json"), "w") as outfile: 
        json.dump(results, outfile) 
if __name__=='__main__':
    compute_sparsity()
    