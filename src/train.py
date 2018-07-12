import utils
import data
import mlp 
import dbconfig

import sys, getopt

def train_algo(params):
    """
        Fetch the total data to train the model on
    """
    dataset = data.DataReader(params['data']['type'])
    
    if params['data']['type'] == 'sql':
        dataset.fetch_data(con=dbconfig.connections_[params['data']['source']](), query=params['data']['detail'])
    elif params['data']['type'] == 'file':
        dataset.fetch_data(file=params['data']['source'])
    
    data_ = data.TrainingData(dataset, target='Put', data_split=params['data']['data_split'])

    """
        Construct the MLP and optimizer arguments
    """

    mlp_args = {'data':data_, **params['algo']}

    """
        Define and train the network
    """
    mdl = mlp.MLP(**mlp_args)

    # train the network
    train_history = mdl.sgd_train(**params['train'], optim_options=params['optim'])
    
    # save the model
    mdl.save(params['results']['model_path'])

    # test the network
    test_results = mdl.test()

    # show results
    mdl.test_scatter(test_results, save_fig= params['results']['test_scatter'])
    mdl.plot_train_valid_curve(train_history, save_fig= params['results']['train_curve'])

def train(argv):
    # retreive command line options
    opts,_ = getopt.getopt(argv, "i:") 

    found = False

    for opt , arg in opts:
        if opt == '-i':
            params = utils.read_params(arg)
            found = True
    
    if not found:
        params = utils.read_params()

    train_algo(params)

if __name__ == "__main__":
    train(sys.argv[1:])