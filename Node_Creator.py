import tensorflow as tf
from LSTM_Node import LSTM_Node, FC_Node, Network_Tree, LSTM_Tree
from abc import ABC, abstractmethod
from keras.layers import Activation, Dense, Dropout, LSTM, Embedding, Input
from keras.models import Sequential


#Probably want to make some generic tree creator at some point and have all the net possibilties have some type of inheritance system.
#For now I'm going to just imp. LSTM since thats our only real tree type

#This class takes a LSTM Node tree and creates the tensorflow implementation of that tree
class LSTM_Node_Creator():

    #LSTM_Net. LSTM_Tree that contains the nodes we will use to generate our net. Currently assumes input_vals are padded to same len. 
    #I have to decide where i wish to imp that feature.
    def __init__(self, LSTM_Net, input_vals):
        self.LSTM_Net = LSTM_Net
        self.feed_input_data(input_vals)
        self.node_fxn_map = {'LSTM': self.create_LSTM_layer, 'FC': self.create_FC_Layer, 'Attention': self.create_Attention_Layer, 'merge': self.create_merge_layer}

    #feed input values to terminal tensors in our lstm net.
    def feed_input_data(self, input_vals):
        term_tensors = []

        for idx, inp_val in enumerate(input_vals):
           term_tensors.append(Input(shape = (self.LSTM_Net.terminal_nodes[idx], name = 'term{0}'.format(idx)))

        self.LSTM_Net.assign_terminal_tensors(term_tensors)

    def create_FC_Layer(self, node):
        neurons = node.param_dict['neurons']
        drop_rate = 1 - node.param_dict['keep_prob']
        act = node.param_dict['activation']

        tf_drop = Dropout(drop_rate)(node.get_tensor_input())
        layer = Dense(neurons, activation= act)(tf_drop)
    
        return layer

    #This function is identical to FC since activiation is forced on creation so nothing diffrent needs to be done.
    #Function is here so I don't question why attention layers don't have a init when I invevitably forget.
    def create_Attention_Layer(self, node):
        return self.create_FC_Layer(node)

    def create_LSTM_layer(self, node):
        dropout = node.get_param_dict['dropout']
        activation = node.get_param_dict['activation']
        recurrent_activation = node.get_param_dict['recurrent_activation']
        recurrent_dropout = node.get_param_dict['recurrent_dropout']
        return_sequences = node.get_param_dict['return_sequences']
        neurons = node.param_dict['neurons']


        layer = LSTM(neurons, activation = activation, recurrent_activation = recurrent_activation, recurrent_dropout = recurrent_dropout, dropout = dropout, return_sequences = return_sequences)
        return layer

    #not sure if correct axis atm please test
    def create_merge_layer(self, node):
        nodes = node.get_tensor_input()
        return tf.concat(nodes, axis = 1)


    #Create a tf implementaion of a LSTM_Node object. Return a tf layer. Takes LSTM Node. 
    def create_tf_layer(self, node):
        return self.node_fxn_map[node.node_type](node)

    #main funtion that should aggregate other class function to build the tf imp
    def build_net(self):

        #We will loop until we reach the final node of our net
        while not self.LSTM_Net.cur_node.is_final_node():
            
            node = self.LSTM_Net.get_next_node()

            output = self.create_tf_layer(node)

            self.LSTM_Net.push_results_to_outputs(node, output)

        self.model = Model(self.LSTM_Net.get_terminal_tensors(), outputs= output)
