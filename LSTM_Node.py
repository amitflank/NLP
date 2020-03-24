import tensorflow as tf
from keras.layers import Activation, Dense, Dropout, LSTM, Embedding, Flatten
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate, Reshape
from abc import ABC, abstractmethod
from params import Valid_Node_Inputs
import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

#general note for all layers we will be sticking with defualt intializers for all elements. We can do some research/have intializers by an optimizer
#component in the future. However, for now lets just get it functioning.


#Base class for all our nerual network Nodes. Defines genric methods all nodes should have.
class Network_Node(ABC):
    
    def __init__(self, input_node = None, output_nodes = None, node_type = 'Network'):
        self.input_node = input_node
        self.node_type = node_type

        if output_nodes is None:
            out_nodes = []

        #make sure output is valid
        self.valid_output(out_nodes)
        self.output_nodes = [node for node in out_nodes] 

    @abstractmethod
    def num_inputs(self):
        pass

    #Create a tf layer for given node type 
    @abstractmethod
    def create_layer(self):
        pass

    #should return bool indication wether this node is an operational node
    @abstractmethod
    def is_op_node(self):
        pass

    #determine if this node is a terminal node
    @abstractmethod
    def is_terminal_node(self):
        pass

    #return input node(s) in array 
    @abstractmethod
    def get_input_nodes(self):
        pass

    #return the current input node (for classes with only one input this wont change) 
    def get_cur_input_node(self):
        pass

    #assign an input node
    @abstractmethod
    def assign_input_node(self, input_node, link = True):
        pass

    #replace node w/input_node without overiding existing node
    @abstractmethod
    def replace_empty_input(self, input_node, error):
        pass

    #Checks if node has requisite inputs to generate tensor layer atm
    @abstractmethod
    def is_layer_legal(self):
        pass

    #return dictionary representation of nodes parameters
    @abstractmethod
    def get_param_dict(self):
        pass

    #add tensor input to node. Overide: determins if we allow existing tensors to be overidden.
    @abstractmethod
    def add_tensor_input(self, tensor_input, overide = False):
        pass

    #clear tensor input(s)
    @abstractmethod
    def clear_tensor_inputs(self):
        pass
  
    #clear input nodes.
    @abstractmethod
    def clear_input(self):
        pass
   
    #check if passed var is a list. Since this is the format output nodes expects.
    #Throws error if not legal type.
    def valid_output(self, var):

        valid_out = isinstance(var, list)

        if not valid_out:
            raise ValueError('Expected var to be of type list but got type {0} instead'.format(type(var)))

    #copy input/output of node to this nodes input/output
    def copy_links(self, node):
        pass

    #empty output list indicates this is a final node
    def is_final_node(self):
        return not self.get_output_nodes()
    
    #check if passed node would be a legal input for this node
    def legal_input_node(self, input_node):
        if Valid_Node_Inputs.legal_input(self, input_node):
            return True
        else:
            return False

    #delete an ouptut node
    def del_output(self, node):
        if self.has_output(node):
            self.get_output_nodes().remove(node)
        else: 
            raise ValueError('Cannot delete passed node as this node is not a recognized output node!')

    #return this nodes type
    def get_node_type(self):
        return self.node_type

    #clear output nodes.
    def clear_output(self):
        self.output_nodes = []

    #reset all the nodes connections
    def reset_node(self):
        self.clear_input()
        self.clear_output()
        self.clear_tensor_inputs()

    #return this nodes output node. Returns as array of node(s -> since im still pretneding we are doing multi-out at some point
    def get_output_nodes(self):
        return self.output_nodes

    #add an output node to this node object. So ahh theres no legal output checks here which seems kinda fish. 
    def add_output_node(self, node):
        self.get_output_nodes().append(node)

    #check if out node is an element of this nodes output list
    def has_output(self, out_node):
        return out_node in self.get_output_nodes()

    #swap output node for new output node. Note this won't check for final nodes (will break this fxn) so you MUST handle that somewhere else.
    def swap_out(self, new_out, old_out):
        if self.has_output(old_out):
            idx = self.get_output_nodes().index(old_out)
            self.get_output_nodes()[idx] = new_out
        else:
            raise ValueError('The node you are trying to replace does not exist!')
    
    #add all the output node of c_node to this node.
    def copy_outputs(self, c_node):
        for node in c_node.output_nodes:
            self.add_output_node(node)

    #get number of output nodes 
    def num_outputs(self):
        return len(self.get_output_nodes())
   

#class representing our operational nodes. Thats is nodes that perform computation on some set of given nodes but have no parameters themselves.
class Operational_Node(Network_Node):
  
    def __init__(self, input_node, output_nodes, node_type = 'Op'):
        super().__init__(input_node, output_nodes, node_type)

    #All classes that inherit from this class should be op nodes.
    def is_op_node(self):
        return True

    #Operational nodes cannot be terminal nodes
    def is_terminal_node(self):
        return False

    #indicatassign_input_nodee op node has no params
    def get_param_dict(self):
        return -1


#This class represents an merge operational node which takes two process nodes and combines them.
class Merge_Node(Operational_Node):

    def __init__(self, input_node = None, output_nodes = None):
        self.input_toggle = 0
        self.tensor_input = [None, None]

        #python wierd mutable defualt rules so we can't just defualy [None, None]
        if input_node is None:
            input_node = [None, None]

        super().__init__(input_node, output_nodes, node_type = 'merge')

    #return max number of input nodes this node can store
    def num_inputs(self):
        return 2

    #assign input toggle to val
    def assign_toggle(self, val):
        if val != 0 and val != 1:
            raise ValueError('Illegal toggle value {0} passed please pass only 0 or 1'.format(val))

        if self.input_toggle != val:
            self.toggle_input()

    #copy input/ouput of node to this nodes input/output
    def copy_links(self, node):

        if self.num_inputs() == node.num_inputs():

            self.assign_toggle(node.input_toggle) 
            self.assign_input_node(node.get_cur_input_node(), link = False)
            self.toggle_input()
            node.toggle_input()
            self.assign_input_node(node.get_cur_input_node(), link =  False)

            self.copy_outputs(node)
        else:
            raise ValueError('When copying links nodes must have same number of inputs. This node has {0} inputs while node being copied from has {1} inputs'.format(self.num_inputs(), node.num_inputs()))

    #replace input_node old node with new_node
    def swap_input_node(self, new_node, old_node):
        if self.get_cur_input_node() is old_node:
            self.assign_input_node(new_node, link = False)
        else:
            self.toggle_input()

            if self.get_cur_input_node() is old_node:
                self.assign_input_node(new_node, link = False)
            
            else:
                raise ValueError('The node you are trying to replace does not exist!')


    #return all input node locations
    def get_input_nodes(self):
        return self.input_node

    #get the current input node being pointed to
    def get_cur_input_node(self):
        return self.input_node[self.input_toggle]

    #toggle the input node being pointed to 
    def toggle_input(self):
        if self.input_toggle == 0:
            self.input_toggle = 1

        elif self.input_toggle == 1: 
            self.input_toggle = 0

        else:
            raise ValueError('Found Illegal input toggle value {0}'.format(self.input_toggle))

    #check if input node loaction currently toggled to is empty
    def cur_input_empty(self):
        if self.get_cur_input_node() is None:
            return True
        else:
            return False

    #clear all tensor inputs
    def clear_tensor_inputs(self):
        for idx, _ in enumerate(self.tensor_input):
            self.tensor_input[idx] = None
 
    #clear inputs
    def clear_input(self):
        self.input_node = [None, None]

    #add input tensor to current toggled input then toggle input. 
    def add_tensor_input(self, tensor_input, overide = False):
        if overide:
            self.tensor_input[self.input_toggle] = tensor_input
        else:
            self.empty_tensor_toggle() #try and toggle to any empty tensor

            if self.tensor_input[self.input_toggle] is None:
                self.tensor_input[self.input_toggle] = tensor_input
            else:
                raise ValueError('You tried overiding a tensor input with overide flag set to False, if you intended to do this please use overide = True')

        self.toggle_input()


    #Add input to current toggled input element. Warning will overide existing values. Use replace empty input if you don't want to overide an element.
    def assign_input_node(self, i_node, link = True):
        if self.legal_input_node(i_node): 
            self.input_node[self.input_toggle] = i_node

            if link:
                i_node.add_output_node(self)
        else:
           raise ValueError('{0} node does not accept input nodes of type {1}'.format(self.get_node_type(), i_node.get_node_type()))

    #replace empty input location if available otherwise take no action. If error is set to True throws exception instead of warning.
    def replace_empty_input(self, input_node, error = False, link = True):

        if input_node is None:
            raise ValueError('Cannnot replace empty input with None!')

        if self.cur_input_empty():
            self.assign_input_node(input_node, link)
        else:
            self.toggle_input()
            if self.cur_input_empty():
                self.assign_input_node(input_node, link)
            else:
                print('Found existing input nodes {0}'.format(self.get_input_nodes())) 
                if error:
                    raise ValueError('Warning Tried to replace empty input but found no empty space')
                else:
                    print('Warning Tried to replace empty input but found no empty space so no action was taken')
    
    #try toggling to empty tensor_input
    def empty_tensor_toggle(self):
        if self.tensor_input[self.input_toggle] is not None:
            self.toggle_input()
            
            if self.tensor_input[self.input_toggle] is not None:
                print('Warning there appears to be no empty tensors, reseting to intial input_toggle')
                self.toggle_input()
            

    #check if we legally create a tf layer
    def is_layer_legal(self):
        if None in self.tensor_input:
            return False
        else:
            return True

    def get_tensor_input(self):
        return self.tensor_input

    #create tf layer from nodes inputs
    def create_layer(self):
        if self.is_layer_legal():
            for idx, inp in enumerate(self.tensor_input):

                #flatten input if it's not already flat. Tf errors will catch if we are under 2 so we check for exactly.
                if len(inp.get_shape()) == 3:
           
                    #resize tensor to [batch_size, all_other_dim_merged] -> this is a Flatten op
                    r_layer = Reshape((inp.get_shape()[1] * inp.get_shape()[2],), input_shape = (inp.get_shape()[1], inp.get_shape()[2],))
                    self.tensor_input[idx] = r_layer(self.tensor_input[idx])

            return concatenate([self.tensor_input[0], self.tensor_input[1]], axis = 1)

        else:
            raise ValueError('Merge operation needs exactly 2 inputs!')

#Class representing a normal node. I.E: some conventional neural network operation: LSTM, Convoloution, FC ect.
class Process_Node(Network_Node):
    
    def __init__(self, input_node, output_nodes, node_type = 'Proc'):
        super().__init__(input_node, output_nodes, node_type)
        self.tensor_input = None

    @abstractmethod
    def get_param_dict(self):
        pass

    def get_tensor_input(self):
        return self.tensor_input

    #All classes that inherit from this class should be non-op nodes.
    def is_op_node(self):
        return False
   
    #all process nodes should only have 1 input node
    def num_inputs(self):
        return 1

    #assign nodes input/output nodes to this nodes input/output node
    def copy_links(self, node):
        if self.num_inputs() == node.num_inputs():
            self.assign_input_node(node.get_cur_input_node(), link = False)
            self.copy_outputs(node)
        else:
            raise ValueError('When copying links nodes must have same number of inputs. This node has {0} inputs while node being copied from has {1} inputs'.format(self.num_inputs(), node.num_inputs()))

    #if our input node is None they this node must be a terminal node
    def is_terminal_node(self):
        if self.input_node is None:
            return True
        else:
            return False
    
    #assign an input node. Warning -> This will overide any existing node
    def assign_input_node(self, input_node, link = True):
        if self.legal_input_node(input_node): 
            self.input_node = input_node
        
            if not self.is_terminal_node() and link: 
                input_node.add_output_node(self)
        else:
           raise ValueError('{0} node does not accept input nodes of type {1}'.format(self.get_node_type(), input_node.get_node_type()))

   
    #swap input node old_node for new_node
    def swap_input_node(self, new_node, old_node):
        if self.get_cur_input_node() is old_node:
                self.assign_input_node(new_node, link = False)
        else:
            raise ValueError('The input node you are trying to replace does not exist!')
    
    #return input node in array 
    def get_input_nodes(self):
        return [self.input_node]

    #Since we can only have 1 input in process nodes we will just return that
    def get_cur_input_node(self):
        return self.input_node

    #replace empty input location if available otherwise take no action
    def replace_empty_input(self, input_node, error = False, link = False):
        if self.get_cur_input_node() is None:
            self.assign_input_node(input_node, link)
        else:
            if error:
                raise ValueError('Warning Tried to replace empty input but found no empty space')
            else:
                print('Warning Tried to replace empty input but found no empty space so no action was taken')

    #clear tensor input
    def clear_tensor_inputs(self):
        self.tensor_input = None

    #clear input node
    def clear_input(self):
        self.input_node = None

    #add tensor_input as new tensor input.
    def add_tensor_input(self, tensor_input, overide = False):
        if overide:
            self.tensor_input = tensor_input
        else:
            if self.tensor_input is None:
                self.tensor_input = tensor_input
            else:
                raise ValueError('You tried overiding a tensor input with overide flag set to False, if you intended to do this please use overide = True')

    def is_layer_legal(self):
        if self.tensor_input != None:
            return True
        else:
            return False

#class that defines a LSTM layer 
class LSTM_Node(Process_Node):

    def __init__(self, neurons, input_node = None, activation = 'tanh', recurrent_activation ='sigmoid', dropout = 0.0,  recurrent_dropout=0.0, output_nodes = None):
        self.param_dict = {"neurons": neurons, "activation": activation, "recurrent_activation": recurrent_activation, "dropout": dropout, "recurrent_dropout": recurrent_dropout, "return_sequences":True}
        super().__init__(input_node, output_nodes, node_type = 'LSTM')

    #create a LSTM tf layer from input
    def create_layer(self):
        if self.is_layer_legal():
            test = LSTM(units = self.param_dict['neurons'], activation = self.param_dict['activation'], recurrent_activation = self.param_dict['recurrent_activation'],
                    dropout = self.param_dict['dropout'], recurrent_dropout = self.param_dict['recurrent_dropout'], return_sequences = self.param_dict['return_sequences'])(self.tensor_input)
            return test
        else:
            raise ValueError('Cannot create layer with no valid Tensor inputs!')
    
    def get_param_dict(self):
        return self.param_dict


    #mod a paramter in parameter dictionary. Key: param to mod. val: new value of param.
    def mod_param(self, key, val):
        self.param_dict[key] =  val

#Not sure I need train_placeholder since we use the evaluate/fit terminology with keras which seems to auto do that so i'm going to get rid of it for now
#and just make note in case I want to add in back in the future
#Class that defines fully connected layer
class FC_Node(Process_Node):

    def __init__(self, keep_prob, neurons, activation, input_node = None, output_nodes = None):
        self.param_dict = {'keep_prob': keep_prob, 'neurons': neurons, 'activation': activation}
        super().__init__(input_node, output_nodes, node_type = 'FC')
    
    
    #FC layers include a dropout, creae tf dropout layer.
    def drop_layer(self, input_vals):
        drop_rate = 1.0 - self.param_dict['keep_prob']
        return Dropout(rate = drop_rate)(input_vals)

    #create tf imlementation of this nodes variables
    def create_layer(self):
        if self.is_layer_legal():

            #flatten input if it's not already flat. Tf errors will catch if we are under 2 so we check for exactly.
            if len(self.tensor_input.get_shape()) == 3:
           
                #resize tensor to [batch_size, all_other_dim_merged] -> this is a Flatten op
                r_layer = Reshape((-1,))
                self.tensor_input = r_layer(self.tensor_input)


            drop = self.drop_layer(self.tensor_input)
            FC= Dense(self.param_dict['neurons'], activation = self.param_dict['activation'])(drop)
            FC = BatchNormalization()(FC)
            return FC

        else:
            raise ValueError('Cannot create layer with no valid Tensor inputs!')

    def get_param_dict(self):
        return self.param_dict 

    #mod a paramter in parameter dictionary. Key: param to mod. val: new value of param.
    def mod_param(self, key, val):
        self.param_dict[key] =  val

class Embedding_Node(FC_Node):

    def __init__(self, seq_len, num_emb, output_dim, weights, input_node = None, output_nodes = None):
        self.param_dict = {'seq_len': seq_len, 'num_emb': num_emb, 'output_dim': output_dim, 'weights': weights}
        Process_Node.__init__(self, input_node, output_nodes, node_type = 'Embedding')

    #need to Overwrite imp. Question is do we want to pass data here or at end of tf net creation. 
    #my guess is we probably just end up sticking with genric imp from process node and have pass at end thing.
    #Toturial does it on layer  creation but that seems silly tbh.
    def is_layer_legal(self):
        pass
    
    def create_layer(self):
        emb = Embedding(num_emb,
            self.param_dict['emb_len'],  # Embedding size
            weights=[self.param_dict['weights']],
            input_length=self.param_dict['max_seq_len'],
            trainable=False)

        return emb

#So the goal of this class is to create a single attention layer I guess.
class Attention_build(Layer):
        def __init__(self, step_dim,
                    W_regularizer=None, b_regularizer=None, #weight and bisa regulizers pretty self explanatory.
                    W_constraint=None, b_constraint=None, #weight and bias constraints also pretty obv.
                    bias=True, **kwargs):

            self.supports_masking = True #Optimized padding sorta where we skip computation of 0 elements by copying previous cell state. 
            self.init = initializers.get('glorot_uniform') #get some random intializer i guess
            self.W_regularizer = regularizers.get(W_regularizer) #pass the regulizer to some api with regulizers do the same for the other regulizer and constraints.
            self.b_regularizer = regularizers.get(b_regularizer)
            self.W_constraint = constraints.get(W_constraint)
            self.b_constraint = constraints.get(b_constraint)
            self.bias = bias #boolean value determining if we want a bias value.
            self.step_dim = step_dim #step dimensions ?
            self.features_dim = 0 #feature dimensions
            super(Attention_build, self).__init__(**kwargs) #initializing some parameter we inherited from layer class. What they are I have no idea.

        def build(self, input_shape):
            assert len(input_shape) == 3 #Make sure our input is a 3-D vector
            self.W = self.add_weight((input_shape[-1],),
                                    initializer=self.init,
                                    name='{}_W'.format(self.name),
                                    regularizer=self.W_regularizer,
                                    constraint=self.W_constraint)
            self.features_dim = input_shape[-1] #looks like we assign feature dimensions to less elment of imput shape (which has 3 elemnts 3-D vector representation)
            if self.bias: #if our boolean bias term is true we generate a bias value
                self.b = self.add_weight((input_shape[1],),
                                        initializer='zero',
                                        name='{}_b'.format(self.name),
                                        regularizer=self.b_regularizer,
                                        constraint=self.b_constraint)
            else:
                self.b = None #Otherwise we assign no bias
            self.built = True #Indicate that the network has been built

        def compute_mask(self, input, input_mask=None):
            return None #apparently we have no mask? why is this even a thing?


        def call(self, x, mask=None):
            features_dim = self.features_dim #get local feature dimensions
            step_dim = self.step_dim #get local step dimensions
            #note K represent keras backend.

            #We take the product of reshaped input and reshaped weight matrices. Note -1 means we are asking the comp. to determine the size.
            #Then we reshaped the resulting matrix to some undetermined size by step dimensions.
            #Bassicaly wheight mutiplying the inputs and then adding bias term. Normal nueral net step.
            eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

            #add the bias term if it exists
            if self.bias:
                eij += self.b

            #Apply activation function
            eij = K.tanh(eij)
                        #Element wise exponetial -> take e^veactor elemnt Calculate exponetial for all elements in input array. Why do we do this????
            #Like we still have the softmax function so it probably isn't a scalling operation.
            a = K.exp(eij)

            #looks like we apply masking operation if mask exists?
            if mask is not None:
                a *= K.cast(mask, K.floatx()) #a * a mask array caster to float but like why?? are there diffrent types of masking opps?


            #K.episolon is the fuzz factor. What is a fuzz factor you ask? Well it's the cozziness cofficient of a blanket divided by the flufiness value.
            #Or if you're boring it is the small floating point value used by Keras to aviod divided by zero errors and other related problems. 

            #We will divide a by the sum of itself across the one axis (pluss fuzz factor to aviod issues)
            a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
            a = K.expand_dims(a) #add 1 sized dimension at index axis (defualt if not provided is -1)
            weighted_input = x * a #out wieghted input is now our input x * a -> so all those steps do our weighting op i guess
            return K.sum(weighted_input, axis=1) #return the sum of our wieghted input

        #method that computes output shape for our attention model
        def compute_output_shape(self, input_shape):
            return input_shape[0],  self.features_dim


class Attention(FC_Node):
    
    def __init__(self, keep_prob, neurons,  input_node = None, output_nodes = None):
        super().__init__(keep_prob, neurons, activation = 'tanh', input_node = input_node, output_nodes = output_nodes)
        self.node_type = 'Attention'

    def create_layer(self):
        tmp = Attention_build(self.param_dict['neurons'])(self.tensor_input)
        print(tmp)
        return tmp

#This class should represent a generic tree that stores Network nodes and can be used to construct and modify some network.
class Network_Tree(ABC):

    #learning_rate: the intial update rate for each run. decay: how quickly learning rate decays after each run. runs: number of training runs.    
    #R: Restriction object used to help create new nodes. step_frac: clustering variable for new node creation. root_node: the root node of this tree.     
    def __init__(self, learning_rate, decay, runs, step_frac, root_node, node_types, loss = None, min_inputs = 1):
        self.root_node = root_node
        self.phase = 'search' #we always defualt to search phase on intilization (as we start at root)
        self.step_frac = step_frac
        self.cur_node = root_node
        self.min_inputs = min_inputs
        self.node_list = []
        self.loss = loss
        self.type_list = node_types 
        self.create_global_param_dict(learning_rate, decay, runs)
        self.reset_node_elements(node_types)
        self.create_node_list(self.root_node) #create various lists that store nodes in various categories
   
    def create_global_param_dict(self, learning_rate, decay, runs):
        self.global_param = {'learning_rate': learning_rate, 'decay': decay, 'runs': runs}

    #create a dictionary to stores our nodes sorted by their type
    def create_type_dict(self, node_types):
        self.nodes_by_type = {}

        for key in node_types:
           self.nodes_by_type[key] = []

        
    #Check if the parameter values for node1 and node2 are the same
    def compare_node_parameters(self, node1, node2):

        #We cannot compare operational nodes so we will raise an error if either one of our passed nodes is an op node
        if node1.is_op_node():
            raise ValueError('Cannot compare opertaional nodes')

        if node2.is_op_node():
            raise ValueError('Cannot compare opertaional nodes')

        #Get a dictionary of our 2 nodes parameters
        node1_params = node1.get_param_dict()
        node2_params = node2.get_param_dict()

        #if nodes are not of same type they can't have same parameters
        if node1.node_type != node2.node_type:
            return False

        #If any parameter varies return false if we get through loop return true
        for key in node1_params:

            if node1_params[key] != node2_params[key]:
                return False

        return True

    #assign loss accosiated with this net
    def assign_loss(self, loss):
        self.loss = loss

    #reset tensor_inputs for all nodes      
    def reset_all_node_tensor_inputs(self):
        nodes = self.get_node_list()
         
        for node in nodes:
            node.clear_tensor_inputs()

        self.terminal_tensors = None


    #this method should be called when generating Eval Net so that intial terminal inputs can be read in.   
    def assign_terminal_tensors(self, terminal_tensors):
        self.terminal_tensors = terminal_tensors
        self.terminal_num = 0 #index of current terminal input
                                                    
    #check if node exists
    def valid_node(self, node):
        if node is not None:
            return True
        else:
            return False

    #reset the dict that stores our nodes by type
    def reset_node_dict(self, type_list):
        self.create_type_dict(type_list)

    #clear list of terminal nodes
    def clear_terminal_nodes(self):
        self.terminal_nodes = []

    #assign node to its relevant type list
    def assign_node(self, node):
    
        #if node is terminal node we will add it to list of terminal nodes
        if node.is_terminal_node():
            self.terminal_nodes.append(node)

        #now add to node_type list in our dictionary    
        self.nodes_by_type[node.node_type].append(node)

        #final add node to global list
        self.node_list.append(node)

    #append nodes to node_lists recursivly from left bottom most node
    def create_node_list(self, node):
        if self.valid_node(node):

            for n_node in node.get_input_nodes():
                self.create_node_list(n_node)

            #assign node to it's relevant lists
            self.assign_node(node)

    #get the tensors belonging to the terminal nodes in this net.
    #Note: You need to have mannually assigned them or this fxn. will fail since assignment is not done on init.
    def get_terminal_tensors(self):
        return self.terminal_tensors

    #get the terminal nodes for this tree
    def get_terminal_nodes(self):
        return self.terminal_nodes

    #clear various list and dicts that stores nodes for new build
    def reset_node_elements(self, type_list):
        self.node_list = []
        self.clear_terminal_nodes()
        self.reset_node_dict(type_list)

    #get list of nodes from tree
    def get_node_list(self):

        #we'll create our node list if it is empty
        if not self.node_list:
            self.create_node_list(self.root_node)

        return self.node_list

    #get number of nodes in tree
    def num_nodes(self):
        return len(self.get_node_list())

    #call this version of get node list if you have made changes to tree since node list does not automatically update
    def get_updated_node_list(self):
                                   
        #make sure node list is cleared
        self.reset_node_elements(self.type_list)

        #get on empty node list will update
        return self.get_node_list()

    #assign the tensor input for the current terminal node
    def assign_terminal_tensor(self):
        try:
            self.cur_node.add_tensor_input(self.terminal_tensors[self.terminal_num])
        except IndexError:
            raise IndexError('Tried assinging a terminal input out of range, attempted to get idx {0}, term_input values {1}'.format(self.terminal_num, self.terminal_tensors))

        self.terminal_num += 1

    #assign node as new root node if legal
    def assign_root(self, node):
        if node.is_final_node():
            self.root_node = node
        else:
            raise ValueError('Cannot assign root node with outputs!')

    #check if passed node is root_node
    def is_root(self, node):
        if node == self.root_node:
            return True
        else:
            return False


    #find the next terminal node in our tree        
    def get_next_terminal_node(self, node):
        #assign current input node as the terminal node we found        
        self.cur_node = self.terminal_nodes[self.terminal_num]

        #assign node tensor
        self.assign_terminal_tensor()

        #switch phase to read
        self.phase = 'read'

    #check if passed object is defined
    def obj_exists(self, obj):
        if obj is None:
            return False
        else:
            return True

    #add the result passed to first available tesnor_input in node. Raises exception if no spots available for result placement. 
    def add_input_tensor(self, node, result):

        if not self.obj_exists(result):
            raise ValueError('The result you passed does not exist, you must pass a real result')

        node.add_tensor_input(result)

    #perform a search operation on our tree
    def search_op(self):
        self.get_next_terminal_node(self.cur_node)
        return self.cur_node

    #perform a read operation on our tree
    def read_op(self):
        if self.cur_node.is_layer_legal():
            return self.cur_node               
        else:
            if self.cur_node.num_inputs() == 1:
                raise ValueError('You cannot switch inputs for a {0} node as it only has 1 input'.format(self.cur_node.node_type))

            return self.search_op()

    #get next node to create in our tree
    def get_next_node(self):
        if self.phase == 'search':
            n_node = self.search_op()
        else:
            n_node = self.read_op()

        if n_node == None:
            raise ValueError('Trying to return NoneType node something went wrong')

        return n_node

    #read current node out and increment to next node. Warning: This implementation likely won't work if we allow nodes to feed into multiple outputs.
    def increment_node(self):
        #assign current node to next node and return saved node
        self.cur_node = self.cur_node.output_nodes[0] #Warning this imp does not work for mult. outputs

    #push results of prevous layer to all its output nodes
    def push_results_to_outputs(self, node, result):
        for out_node in node.output_nodes:
            self.add_input_tensor(out_node, result)

        self.increment_node()

    #returns this trees root node
    def get_root(self):
        return self.root_node

    #get number of inputs this tree takes
    def num_inputs(self):
        return len(self.terminal_nodes)

    #minimum number of inputs this tree can have
    def get_min_inputs(self):
        return self.min_inputs

   
    #Build the tf implementation of this network from existing nodes. Retruns the tensor output of root node
    def build_net(self, term_tensors):
        
        #feed in the inputs needed for terminal nodes in tree
        self.assign_terminal_tensors(term_tensors)

        for idx, val in enumerate(self.node_list):

            #get the next node
            node = self.get_next_node()

            #we will exit loop when current node has no outputs (indication it is final node)
            if node.is_final_node():
                break
            else:
                #push the results of this layer to all output nodes 
                self.push_results_to_outputs(node, node.create_layer())
            
        return node.create_layer()

#This class represents an LSTM specefic node tree. It inherits from Network Tree.
class LSTM_Tree(Network_Tree):
   


    #learning_rate: the intial update rate for each run. decay: how quickly learning rate decays after each run. runs: number of training runs.    
    #R: Restriction object used to help create new nodes. step_frac: clustering variable for new node creation. root_node: the root node of this tree.     
    def __init__(self, learning_rate, decay, runs, step_frac, root_node, loss = None):
        node_types = ['FC', 'LSTM', 'merge', 'Attention', 'Embedding'] #should add a concat node here but not sure how I want to fromat that class yet 
        super().__init__(learning_rate, decay, runs, step_frac, root_node, node_types, loss)
        self.net_type = 'LSTM'
        #self.init_all_toggles()


    #toggle all merge inputs to 0
    def init_all_toggles(self):
        for node in self.nodes_by_type['merge']:
            if node.input_toggle == 1:
                nodete_layer.toggle_input()
