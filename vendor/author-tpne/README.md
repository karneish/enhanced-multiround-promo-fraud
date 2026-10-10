## Requirements
The program was developed and tested using the following library versions:

```
dgl==2.0.0+cu121
networkx==3.1
numpy==1.24.3
pandas==2.2.1
scikit-learn==1.3.0
scipy==1.12.0
seaborn==0.12.2
torch==2.2.2
torch_geometric==2.5.3
xgboost==2.0.3
```

## Installation
```
python setup.py install
```

## Run
### Datasets
All datasets should be placed directly in `\dataset` in a DGL readable format as shown for the `tolokers` dataset. 
At the moment only the `tolokers` dataset is hosted on Github. In the meantime, please directly contact the corresponding author  at hafizh(@)net.comp.isct.ac.jp for the other datasets.
### Run the Code
#### Using the notebook example
Simply run the notebook `notebook\example_experiment.ipynb` using a kernel that satisfies the requirements listed above.
#### Using the python script
```
cd scripts
python main.py -c [config_filename_without_json]
```
### Outputs
All outputs will be generated under the `\result` folder. Each experiment will generate its own folder named after the timestamp of the execution. The generated files include:
- `meta.txt`: Contains metadata of the experiment including the actual values of all changeable config.
- `[dataset_name]-[exp_dict_item_1]-...-[exp_dict_item_n]-E.csv`: Results from a single run of from the search spanned by the `EXP_DICT` item in the config file
- `combined_result.csv`: Combined results of all the individual `.csv` files above
The notebook `\notebook\example_data_process.csv` contains example codes to interpret and process the experiment results.
### Configurations
All configuration to the parameter should be specified using a `.json` file placed in the `\scripts` folder. An example of the config file is shown in `scripts/config_example.json`. For a list of all parameters that are available to place under the `EXP_DICT` item, refer to the source file `\src\utils\utils_const.py`

## Citations
```
@article{prasetya2025multi,
  title={A multi-rounded adversarial scenario for graph-based promo fraud detection},
  author={Prasetya, Hafizh Adi and Liu, Xin and Murata, Tsuyoshi and Matono, Akiyoshi},
  journal={Social Network Analysis and Mining},
  year={2025},
  publisher={Springer}
}
```

## Copyright
© 2025 National Institute of Advanced Industrial Science and Technology (AIST)
