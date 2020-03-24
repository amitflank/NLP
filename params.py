import tensorflow as tf
from random import randint
from abc import ABC, abstractmethod

#this class is a holder class for restrictions placed on hyper-parameters
class net_restrictions():

	#max_conoloutional: maximum number of convuloution layers in net, max_lin_layer: max number of FC layers in net
	#max_filter_size: width and height of our convoloutional kernels, max_pixel_drop: maximum percentage of elements we drop on each layer
	#max_pool: max width and height of pooling kernels, max_filters: maximum number of kernels in each convoloutional layer
	#max_neurons: max number of neurons in each FC layer
        def __init__(self, max_convolutions, max_lin_layers, max_filter_size, max_pixel_drop, max_pool, max_filters, max_neurons, max_stride, max_lr, max_decay, max_runs, max_w_frac, max_h_frac, max_ws):
                self.max_convolutions = max_convolutions
                self.max_lin_layers = max_lin_layers
                self.max_filter_size = max_filter_size
                self.max_pixel_drop = max_pixel_drop
                self.max_pool = max_pool
                self.max_filters = max_filters
                self.max_neurons = max_neurons
                self.max_stride = max_stride
                self.max_lr = max_lr
                self.max_decay = max_decay
                self.max_runs = max_runs
                self.max_w_frac = max_w_frac
                self.max_h_frac = max_h_frac
                self.max_ws = max_ws

class meta_restrict():

        def __init__(self, max_pixel_drop, max_neurons, max_lr, max_decay, max_runs, max_batch_size):
                self.max_pixel_drop = max_pixel_drop
                self.max_neurons = max_neurons
                self.max_lr = max_lr
                self.max_decay = max_decay
                self.max_batch_size = max_batch_size
                self.max_runs = max_runs


#A single location where we can place what nodes are legally teminal.
class Legal_Term_Nodes():
    
    def __init__(self):
        self.legal_term_nodes = ['LSTM', 'FC', 'Embedding', 'Attention']

class activation_fxns():

        def __init__(self):
                self.functions = [tf.nn.elu, tf.nn.relu, tf.nn.relu6, tf.nn.selu, tf.nn.softmax, tf.nn.softsign, tf.nn.sigmoid, tf.tanh, None]
                self.labels = ['elu', 'relu', 'relu6', 'selu', 'softmax', 'softsign', 'sigmoid', 'tanh', 'None']

#restrictions variables that should generically exist across nets
class Gen_Restrict(ABC):
    
    def __init__(self, max_runs = 6, learning_rate = .3, decay = .9):
        self.max_runs = max_runs
        self.learning_rate = learning_rate
        self.decay = decay

#note variable names may not be super accurate since we are trying to keeps names consitenet for convience of just indexing keys. The parameter pass value names are fairly accurate though.
class FC_restrict(Gen_Restrict):

    def __init__(self, max_neurons = 2000, max_pixel_drop = .9):
        self.params = {'neurons': max_neurons, 'keep_prob': max_pixel_drop}
        super().__init__()

class LSTM_restrict(Gen_Restrict):

    def __init__(self,  neurons = 2000, activation= 'tanh', recurrent_activation= 'sigmoid',  dropout = .9,  recurrent_dropout = .9):
        self.params = {'neurons': neurons, 'activation': activation, 'recurrent_activation': recurrent_activation, 'dropout': dropout, 'recurrent_dropout': recurrent_dropout}
        super().__init__()

class Emb_restrict():
    
    def __init__(self, max_seq_len = 150, max_output_dim = 2000):
        self.params = {'seq_len': max_seq_len  , 'output_dim': max_output_dim}


class Valid_Node_Inputs():

    node_dict = {'FC': ['FC', 'LSTM', 'merge', 'Attention'], 'LSTM': ['LSTM'], 'merge': ['FC', 'LSTM', 'merge', 'Attention'], 'Attention': ['LSTM']}

    @staticmethod
    def legal_input(node, input_node):
        return input_node.get_node_type() in Valid_Node_Inputs.node_dict[node.get_node_type()]


#This function checks if the passed global dictionary matchs the required one for the passed net
#additional gloabal key lists should be added to the body of this function as new networks are introduced
def valid_globals(net_type, global_dict):

    LSTM_global_keys = ['seq_len', 'learning_rate', 'decay', 'runs', 'batch_size']

    if net_type is "LSTM":

        for val in LSTM_global_keys:
            
            if val not in global_dict:
                return False

        return True

