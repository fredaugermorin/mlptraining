# mlptraining

Library for training MLP neural networks for american options pricing.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

before installing, you should make sure you have a working installtion of keras built on top of tensorflow by running the following

```
pip install keras
pip install tensorflow-gpu
```

## Examples
As of july 2018, the library only provides a way to traing an arbitrary set of data located in:
```
\american_options\data\
```
You can modify or create new parameters json files located in 
```
\american_options\src\cfg
```
and pass them a options. The default config file is name params.json
Once your config file is created, you can train the specified model, on the specified dataset by running:
```
python train.py -i your_params_file.json
```

## Authors

**Frederick Auger-Morin** 

