import tensorflow as tf
import keras as k
import matplotlib.pyplot as plt
from settings import libPath

class MLP():
    def __init__(self, data, dim_in, dim_out, hidden_layers, n_units, activation, fully_connected=True, optimizer='sgd'):
        # private attributes
        self.__data_ = data # a Data object instance containing the appropriate data to fit the model on
        self.__dim_in_ = dim_in
        self.__dim_out_ = dim_out
        self.__fully_connected_ = fully_connected
        self.__optimizer_ = optimizer
        
        # public attributes
        self.model_ =  k.models.Sequential()

        # Initialiszation
        if hidden_layers > 1:
            self.__n_units_ = list()
            self.__activation_ = list()

            if len(n_units) != hidden_layers and len(n_units) != len(activation):
                raise ValueError('Length of activation does not match number of hidden units.')
            else:
                self.__h_layers_ = hidden_layers
                
                self.__n_units_ = n_units
                self.__activation_ = activation

        else:
            self.__h_layers_ = hidden_layers
            self.__n_units_ = n_units

        self.__construct()

    def __construct(self):
        # create the first layer
        self.model_.add(k.layers.Dense( self.__n_units_[0], input_dim=self.__dim_in_, activation=self.__activation_[0]))

        # create all subsequent layers
        for n, act in zip(self.__n_units_[1:], self.__activation_[1:]):
            self.model_.add(k.layers.Dense( n, activation=act ) )
    
    def sgd_train(self, max_epochs, batch_size, loss=k.metrics.mean_squared_error, optim_options={}):
        """
        This method trains the model. 
            Input:
                max_epochs: the maximum number of epochs for training
                batch_size: size of batches for gradient dscent style network optimization
                loss: a callable. Is used as the loss function to minimize during training
                optim_options: a dict containg all metrics to instatiate the SGD optimizer
            Output:
                train_loss: the loss on the training after each training epoch
                valid_loss: the loss on validation set after each training epoch
        """
        sgd = k.optimizers.SGD(**optim_options)

        self.model_.compile(loss=loss, optimizer=sgd, metrics=[loss])


        history = self.model_.fit(self.__data_.train_x.values, self.__data_.train_y.values, validation_data = (self.__data_.valid_x.values, self.__data_.valid_y.values), \
                        epochs=max_epochs, 
                        batch_size=batch_size, 
                        callbacks=[k.callbacks.ModelCheckpoint(libPath + '\\src\\results\\weights.{epoch:02d}-{val_loss:.2f}.hdf5',period=5)])
        
        return history

    def plot_train_valid_curve(self, history, save_fig=None):
        """
        Plots the history of the training
            Input:
                history: the keras training history returned by _train method
                save_fig: a string indicating the path to use to save the plot, if None, figure not saved.
        """
        epochs = len(history.history['loss'])

        plt.figure()
        plt.suptitle('Learning curve for MLP model')
        plt.title(' TBD ')
        plt.plot(range(1, epochs+1), history.history['loss'])
        plt.plot(range(1, epochs+1), history.history['val_loss'])
        plt.legend(['Train', 'Valid'])
        plt.xlabel('Number of Epochs')
        plt.ylabel(' Loss ')
        plt.draw()

        if save_fig is not None:
            print('Saving figure to: %s' % save_fig)
            plt.savefig(libPath + '\\src\\results\\' + save_fig)

        plt.show()

    def test(self):
        """
        method that tests the the model on its test set
            Output:
                the value of the loss function on the test set and
                the predicted and realized values as numpy arrays
        """
        return {'loss': self.model_.test_on_batch(self.__data_.test_x.values, self.__data_.test_y.values),
                'y_pred': self.model_.predict(self.__data_.test_x.values),
                'y_true': self.__data_.test_y.values
                }
    
    def test_scatter(self, test_results=None, save_fig=None):
        if test_results is None:
            test_results = self.test()
        else:
            plt.figure()
            plt.suptitle(' Scatterplot Test results ')
            plt.title('prediction vs realized')
            plt.scatter(test_results['y_true'], test_results['y_pred'])
            plt.xlabel('y_true')
            plt.ylabel('y_pred')
            plt.draw()

            if save_fig is not None:
                print('Saving figure to: %s' % save_fig)
                plt.savefig(libPath + '\\src\\results\\' + save_fig)
                
            plt.show()

    def save(self,path='model.h5'):
        self.model_.save(libPath + '\\src\\results\\' + path)
