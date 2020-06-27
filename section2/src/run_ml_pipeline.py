"""
This file contains code that will kick off training and testing processes
"""
import os
import json

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

import numpy as np

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"/home/workspace/TrainingSet"
        self.n_epochs = 10 #10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "out/RESULTS"

if __name__ == "__main__":
    # Get configuration

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)
    print(f'data.shape={data.shape}')

    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()
    N = len(data)

    print(f'keys={keys}')
    np.random.seed(42)
    indexes = np.random.choice(N, N, False)
    print(f'indexes={indexes}')
    print(f'np.sum(indexes)={np.sum(indexes)}, np.sum(keys)={np.sum(keys)}')

    n_train = int(N * 0.70)
    n_val = int(N * 0.20)
    n_test = N - n_train - n_val

    print(f'n_train={n_train}, n_val={n_val}, n_test={n_test}, total={sum([n_train, n_val, n_test])}')

    split = dict()

    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    # <YOUR CODE GOES HERE>

    split['train'] = indexes[:n_train]
    split['val'] = indexes[n_train:n_train + n_val]
    split['test'] = indexes[n_train + n_val:]

    print('final verification={}'.format(sum([np.sum(split[k]) for k in split.keys()])))
    
    # Set up and run experiment
    
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    # del dataset 

    # run training
    exp.run()

    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

