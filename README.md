# mlptraining

Library for training MLP neural networks from arbitrary data

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

before installing, you should make sure you have a working installtion of keras built on top of tensorflow by running the following

```
pip install keras
pip install tensorflow-gpu
```

## Examples
As of july 2018, the library only provides a way to train a possibly deep MLP network on an arbitrary dataset located in:
```
\mlptraining\data\
```
You can modify or create new parameters json files located in 
```
\mlptraining\src\cfg
```
and pass them as options to the train.py script. The default config file is named params.json

Once your config file is created, you can train the specified model, on the specified dataset by running:
```
python train.py -i your_params_file.json
```

## Authors

**Frederick Auger-Morin** 

