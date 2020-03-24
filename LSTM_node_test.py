import tensorflow as tf
from param_helper import Opt_Params
from param_generator import Hyp_Gen, LSTM_Gen
from LSTM_Node import LSTM_Node, FC_Node, Network_Tree, LSTM_Tree, Merge_Node, Operational_Node, Embedding_Node
from analysis import Merge_Anylizer, Node_Anylizer, FC_Anylizer, LSTM_Anylizer, Tree_Anylizer, create_node_anylizer, blockPrint, enablePrint
from keras.layers import Activation, Dense, Dropout, LSTM, Embedding
from keras.layers import GlobalAveragePooling1D, BatchNormalization, concatenate
from abc import ABC, abstractmethod
from random import randint
from copy import deepcopy
from tot import LSTM_Eval
import random
from tmp_net import pre_file_test_bootstrap

#basically a warper class for activation functions and corresponding labels 
class activation_fxns():

    #maybe store as dict? prob better but would require going through code and modifying lots of things.
    def __init__(self):
        self.functions = [tf.nn.elu, tf.nn.relu, tf.nn.relu6, tf.nn.selu, tf.nn.softmax, tf.nn.softsign, tf.nn.sigmoid, tf.tanh, None]
        self.labels = ['elu', 'relu', 'relu6', 'selu', 'softmax', 'softsign', 'sigmoid', 'tanh', 'None']


    #get the name of a random activation function. Typically passed to a keras like fxn.
    def get_rand_act_name(self):
        return self.labels[randint(0, len(self.labels) - 1)]

#Create LSTM layer with random number of neurons in specified range. Can overide other defualts if desired but not required.
def create_rand_LSTM_node(min_neurons, max_neurons, input_node = None, activation = 'tanh', recurrent_activation ='sigmoid', dropout = 0.0,  recurrent_dropout=0.0):
   
    neurons = randint(min_neurons, max_neurons)
    return LSTM_Node(neurons, input_node, activation, recurrent_activation, dropout, recurrent_dropout)

#Create FC layer with random number of neurons in specified range. Can overide other defualts if desired but not required.
def create_rand_FC_node(min_neurons, max_neurons, keep_prob = 1.0, activation = 'relu', input_node = None):
    neurons = randint(min_neurons, max_neurons)
    return FC_Node(keep_prob, neurons, activation, input_node)

#Create embedding layer with values of seq_len and output_dim being b/w min and max values. 
def create_random_emb_layer(min_seq_len, max_seq_len, min_output_dim, max_output_dim, weights, num_emp):
    seq_len = randint(min_seq_len, max_seq_len) 
    output_dim = randint(min_output_dim, max_output_dim) 
    return Embedding_Node(seq_len, num_emp, output_dim, weights)

#just creates an empty merge node
def create_empty_merge_node():
    return Merge_Node()

#make sure input and output nodes are properly assigned for a process node
def test_process_input_output_assignment(input_node, output_node):
    input_node.add_output_node(output_node)
    output_node.assign_input_node(input_node)

    if(output_node.get_cur_input_node() is not input_node):
        raise ValueError('Failed to assign input node expected node: {0} but got: {1}'.format(input_node, output_node.get_cur_input_node()))

    if(not input_node.has_output(output_node)):
        raise ValueError('Failed to assign input node expected node: {0} in list but output list is: {1}'.format(output_node, input_node.output_nodes))

    print("Sucessfully assigned inputs and outputs")

class LSTM_tree_gen():
   
    #num_terminal_inputs: int of intial branchs in tree. LSTM_nodes: int of how many LSTM nodes this tree will contain. FC_nodes: same as LSTM nodes for FC.
    #FC_min_neurons: min FC neurons in layer. FC_max_neurons: max neurons in FC layer. LSTM min/max same as FC for LSTM layers. FC_rand_drop: drop percentage for FC layers.
    #LSTM_rand_drop: drop percentage for LSTM input. LSTM_rand_rec_drop: drop percentage for recurrent layer. bool variables indicate if we use corresponding parameters.
    def __init__(self, num_terminal_inputs, LSTM_nodes, FC_nodes, FC_min_neurons, FC_max_neurons, 
                LSTM_min_neurons, LSTM_max_neurons, FC_rand_drop = False, LSTM_rand_drop = False, 
                LSTM_rand_rec_drop = False, LSTM_rand_act = False, LSTM_rand_rec_act = False, FC_rand_act = False):

       self.num_terminal_inputs = num_terminal_inputs
       self.LSTM_nodes = LSTM_nodes
       self.FC_nodes = FC_nodes
       self.FC_min = FC_min_neurons
       self.FC_max = FC_max_neurons
       self.LSTM_min = LSTM_min_neurons
       self.LSTM_max = LSTM_max_neurons
       self.FC_rand_drop = FC_rand_drop
       self.LSTM_rand_drop = LSTM_rand_drop
       self.LSTM_rand_rec_drop = LSTM_rand_rec_drop
       self.LSTM_rand_act = LSTM_rand_act
       self.LSTM_rand_rec_act = LSTM_rand_rec_act
       self.FC_rand_act = FC_rand_act
       self.poss_act = activation_fxns()
       self.create_LSTM_node_list()
       self.create_FC_node_list()

    #generate an LSTM node using class params
    def LSTM_node_gen(self):
        LSTM_param_dict = {}

        if self.LSTM_rand_drop:
            LSTM_param_dict['dropout'] = random.random()
        else:
            LSTM_param_dict['dropout'] = 0.0 
        
        if self.LSTM_rand_rec_drop:
            LSTM_param_dict['recurrent_dropout'] = random.random()
        else:
            LSTM_param_dict['recurrent_dropout'] = 0.0 

        if self.LSTM_rand_act:
            LSTM_param_dict['activation']  = self.poss_act.get_rand_act_name()
        else:
            LSTM_param_dict['activation']  = 'tanh' 
        
        if self.LSTM_rand_act:
            LSTM_param_dict['recurrent_activation']  = self.poss_act.get_rand_act_name()
        else:
            LSTM_param_dict['recurrent_activation']  = 'sigmoid' 
        
        LSTM_param_dict['min_neurons'] = self.LSTM_min
        LSTM_param_dict['max_neurons'] = self.LSTM_max

        return create_rand_LSTM_node(**LSTM_param_dict) 

    #generate FC node using class params
    def FC_node_gen(self):
        FC_param_dict = {}

        if self.FC_rand_drop:
            FC_param_dict['keep_prob'] = random.random()
        else:
            FC_param_dict['keep_prob'] = 1.0 

        if self.FC_rand_act:
            FC_param_dict['activation']  = self.poss_act.get_rand_act_name()
        else:
            FC_param_dict['activation']  = 'relu' 
        
        FC_param_dict['min_neurons'] = self.FC_min
        FC_param_dict['max_neurons'] = self.FC_max

        return create_rand_FC_node(**FC_param_dict) 

    #create list of FC nodes that we can use to construct net
    def create_FC_node_list(self):
        self.FC_node_list = []
        for idx in range(self.FC_nodes):
            self.FC_node_list.append(self.FC_node_gen())

    #create list of LSTM nodes we can use to construct net
    def create_LSTM_node_list(self):
        self.LSTM_node_list = []
        for idx in range(self.LSTM_nodes):
            self.LSTM_node_list.append(self.LSTM_node_gen())


    #return 0: LSTM, 1; FC, 2: merge
    def flip_node(self):
        num = randint(0, self.LSTM_nodes + self.FC_nodes + self.num_terminal_inputs)
        
        if num < self.LSTM_nodes:
            return 0
        elif num > self.LSTM_nodes and num < self.LSTM_nodes + self.FC_nodes:
            return 1
        else:
            return 2
    
    #pick a branch
    def get_branch(self, node_list):
        return randint(0, len(node_list) - 1)

    #add an LSTM node as output linked to passed node. 
    def add_LSTM_node(self, input_node):
        if input_node is not None:
            input_node.add_output_node(self.LSTM_node_list[self.cur_lstm_node])
        
        node = self.LSTM_node_list[self.cur_lstm_node]
        node.replace_empty_input(input_node, error = True)
        self.cur_lstm_node += 1
        return node

    #add an FC node as output linked to passed node. 
    def add_FC_node(self, input_node):
        if input_node is not None:
            input_node.add_output_node(self.FC_node_list[self.cur_FC_node])
        
        node = self.FC_node_list[self.cur_FC_node]
        node.replace_empty_input(input_node, error = True)
        self.cur_FC_node += 1
        return node

    #check if we have LSTM nodes remaining to add
    def remaining_LSTM_nodes(self):
        if self.cur_lstm_node < len(self.LSTM_node_list):
            return True
        else:
            return False

    #check if we have FC nodes remaining in our node list
    def remaining_FC_nodes(self):
            if self.cur_FC_node < len(self.FC_node_list):
                return True
            else:
                return False

    #check if node list has valid LSTM loc. not at cur loc
    def has_LSTM_loc(self, node_list, cur_node):
        for node in node_list:

            #check if we have empty or LSTM node open
            if node is None or node.get_node_type() is 'LSTM':

                #check to make sure open node is not current node
                if node is not cur_node:
                    return True
       
        #if we don't meet prev. cond then we don't have a valid loc and we return false
        return False

    #check if we can add FC at cur_node
    def can_add_FC(self, node_list, cur_node):
        
        #we can always add FC if we have no remaing LSTM nodes
        if self.remaining_LSTM_nodes():

            if self.has_LSTM_loc(node_list, cur_node):
                return True
            else:
                return False

        return True

    #check if we can add LSTM to passed node
    def can_add_LSTM(self, node):
        if node is None or node.get_node_type() is 'LSTM':
            return True
        else:
            return False

    #check if we can legally merge cur_node and merge node
    def can_add_merge(self, node_list, cur_node, merge_node):
        if len(node_list) > 1:

            #if no remaing LSTM nodes we good
            if not self.remaining_LSTM_nodes():
                return True

            #otherwise we need to ensure that we have at least 1 open LSTM location not being merged
            for node in node_list:

                if self.can_add_FC(node_list, node) and (node is not cur_node) and (node is not merge_node):
                    return True
    
        return False

    #get node with which m_node will merge (and fufill the prophecy)
    def merge_partner(self, m_node, node_list):
        idx = node_list.index(m_node)
        flip = randint(0, 1)

        #if we flip 0 we try and get the left idx node if legal, otherwise we return right idx node
        if flip == 0:

            if idx == 0:
                return node_list[idx + 1]
            else:
                return node_list[idx - 1]

        #we assum node_list is of at least size 2 so we don't check for idx error. Error handling should take place somewhere else.
        else:

            if idx == len(node_list) - 1:
                return node_list[idx - 1]
            else:
                return node_list[idx + 1]

    #add create merge layer node
    def add_merge_node(self, node, node_list):

        #so ugly the way we end up doing this. barf -> can't get >1 check before since need new node but new node needs to check > 1 lol.
        if len(node_list) > 1:
            new_node = self.merge_partner(node, node_list)
       
            if self.can_add_merge(node_list, new_node, node):
                node_list.remove(node)
                node_list.remove(new_node)
                m_node = Merge_Node()  
                m_node.replace_empty_input(node, error = True)
                m_node.replace_empty_input(new_node, error = True)
                node_list.append(m_node)
                return True

        return False

    #add non-terminal node to tree
    def add_node(self, node_list):
        node_idx = self.get_branch(node_list)
        added_node = False

        while not added_node:
            
            flip = self.flip_node()
            
            if flip == 0 and self.remaining_LSTM_nodes():

                node_list[node_idx] = self.add_LSTM_node(node_list[node_idx])
                added_node = True

            elif flip == 1:
                    
                if self.can_add_FC(node_list, node_list[node_idx]) and self.remaining_FC_nodes():
                    node_list[node_idx] = self.add_FC_node(node_list[node_idx])
                    added_node = True
            else:
                added_node = self.add_merge_node(node_list[node_idx], node_list)

    #rules for adding terminal nodes to tree differ from generic add rules so we have a special function to deal with it.
    def add_terminal_nodes(self, node_list):
        
        for idx, _ in enumerate(node_list):    
        
            added_node = False

            while not added_node:

                flip = 2
                while flip == 2:
                    flip = self.flip_node()

                if flip == 0 and self.remaining_LSTM_nodes():
                    node_list[idx] = self.add_LSTM_node(node_list[idx])
                    added_node = True
                else:
                    
                    if idx == self.num_terminal_inputs - 1: #if we are at last index we will check since during terminal assign we are garunteed to have open if not on last
                        
                        if self.can_add_FC(node_list, node_list[idx]) and self.remaining_FC_nodes():
                            node_list[idx] = self.add_FC_node(node_list[idx])
                            added_node = True

                    else:
                        node_list[idx] = self.add_FC_node(node_list[idx])
                        added_node = True

    #We need to cleanup at end of tree build if we have more then one remaining branch and we have exhuasted all our nodes.
    #So we merge everything down to 1 branch.
    def cleanup(self, node_list):

        while len(node_list) >= 2:
            self.add_merge_node(node_list[0], node_list) 

    #build our LSTM network tree from class parameters
    def build_tree(self):

        #define/reset node tracker parameters
        self.cur_lstm_node = 0
        self.cur_FC_node = 0

        #create our terminal nodes sepratly since there rules differ a bit
        node_list = [None for i in range(self.num_terminal_inputs)]
        self.add_terminal_nodes(node_list)

        #loop until we run out of nodes
        while self.remaining_FC_nodes() or self.remaining_LSTM_nodes():
            self.add_node(node_list)

        #clean up remaining branchs
        self.cleanup(node_list)

        params = self.gen_rand_globals()
        params['root_node'] = node_list[0] #post cleanup we should have only 1 node in out list.

        return LSTM_Tree(**params)

    #generate random gloabl params for LSTM tree
    def gen_rand_globals(self):

        #step_frac value is place holder for now should replace this with obj. step frac latter
        return {'learning_rate': random.random() / 1000, 'decay': random.random(), 'runs': randint(1, 8), 'step_frac': 12}

#really really need a good name. More of Specefic LSTM tester at this poin tbh. At some point we will node to go back and break this into a parent class with genric tree
#tester method and an LSTM class to maximize code reuse. Will probabably have to break down some functions a bit.
class tree_tester_thing(ABC):

    def __init__(self, min_FC_min_neurons = 50, max_FC_min_neurons = 500, min_FC_max_neurons = 500, max_FC_max_neurons = 2000, min_LSTM_min_neurons = 50, max_LSTM_min_neurons = 500,
                    min_LSTM_max_neurons= 500, max_LSTM_max_neurons = 2000, min_terminals = 1, max_terminals = 4):
        self.params = locals() #not sure if this is sketch or not but it saves me a bunch of time.

    def create_tree_gen_param(self):
        tree_gen_param = {}
        tree_gen_param['num_terminal_inputs'] = randint(self.params['min_terminals'], self.params['max_terminals'])
        tree_gen_param['LSTM_nodes'] = randint(2, 20)
        tree_gen_param['FC_nodes'] = randint(2, 20)
        tree_gen_param['FC_min_neurons'] = randint(self.params['min_FC_min_neurons'], self.params['max_FC_min_neurons'])
        tree_gen_param['FC_max_neurons'] = randint(self.params['min_FC_max_neurons'], self.params['max_FC_max_neurons'])
        tree_gen_param['LSTM_min_neurons'] = randint(self.params['min_LSTM_min_neurons'], self.params['max_LSTM_min_neurons'])
        tree_gen_param['LSTM_max_neurons'] = randint(self.params['min_LSTM_max_neurons'], self.params['max_LSTM_max_neurons'])
        return tree_gen_param

    #When you just need to gen a random lstm node and don't really care about whats in it
    def build_lstm_node(self, node_type):
        if node_type is 'merge':
            return create_empty_merge_node()
        elif node_type is 'FC':
            return create_rand_FC_node(100, 1000)
        elif node_type is 'LSTM':
            return create_rand_LSTM_node(100, 100)

    #Build an LSTM tree with random parameters
    def build_LSTM_tree(self):
        tree_gen_param = self.create_tree_gen_param()
        gen = LSTM_tree_gen(**tree_gen_param)
        tree = gen.build_tree()

        self.tree_gen_param = tree_gen_param #this is horrible fix later plz
        return tree
    
    #need better name
    def test_n_trees_meta(self, n, supress = False):

        keyword_node_map = {'FC_nodes': 'FC', 'LSTM_nodes': 'LSTM'}

        for _ in range(n):
            tree = self.build_LSTM_tree()

            test = Tree_Anylizer(tree, supress)
            num_branchs, num_nodes, num_node_type = test.print_tree_sum_info()

            for key, val in keyword_node_map.items():
                if self.tree_gen_param[key] != num_node_type[val]:
                    raise ValueError('Actual param did not match expected param. Expected {0} nodes of type {1} but got {2} instead'.format(self.tree_gen_param[key], val, num_node_type[val]))

            if num_branchs != self.tree_gen_param['num_terminal_inputs']:
                raise ValueError('Expected {0} intial branchs but got {1} branchs instead'.format(self.tree_gen_param['num_terminal_inputs'], num_branchs))

        print('\nNo errors found in tree meta info. Ran {0} random nets'.format(n))

    #test if node iteration return the nodes in proper order
    def test_node_iter(self, tree = None):
        
        if tree is None:
            tree = self.build_LSTM_tree()

        placeholder = tf.compat.v1.placeholder(tf.float32, shape=(1024, 1024)) 
        term_place = [placeholder for _ in range(tree.num_inputs())]

        #feed in the inputs needed for terminal nodes in tree
        tree.assign_terminal_tensors(term_place)

        tree_nodes = tree.get_node_list()

        for idx, val in enumerate(tree_nodes):
            
            #get the next node
            node = tree.get_next_node()
          
            if val is not node:
                g_node_any = create_node_anylizer(node)
                e_node_any = create_node_anylizer(val)

                print('Expected node: ')
                e_node_any.print_node_params()
                print('Found node: ')
                g_node_any.print_node_params()
                raise ValueError('Node found was not the next node in the list. Expected node {0} found node {1}'.format(val.get_node_type(), node.get_node_type()))

            #we will exit loop when current node has no outputs (indication it is final node)
            if node.is_final_node():
                break
            
            else:
                #push the results of this layer to all output nodes 
                tree.push_results_to_outputs(node, placeholder)

    #test link copy fxn in Node classes
    def test_link_copy(self, n):
        
        #Build tree and tools to help anylize it
        tree = self.build_LSTM_tree()

        for i in range(n):
            helper = Opt_Params()
            node, _ = helper.pick_random_node(tree)

            new_node = self.build_lstm_node(node.get_node_type()) 
            new_node.copy_links(node)

            if new_node.get_output_nodes() != node.get_output_nodes():
                raise ValueError('Output nodes failed to copy properly')

            if new_node.get_input_nodes() != node.get_input_nodes():
                raise ValueError('Input nodes failed to copy properly')
    
        print('\nSuccesfully copied node links {0} times'.format(n)) 

    #test random node addition
    def test_add_node(self, supress = False):

        #Build tree and tools to help anylize it
        tree = self.build_LSTM_tree()
        gen = LSTM_Gen([tree]) 
        anyl = Tree_Anylizer(tree, supress = True)

        #get information about tree pre node addition
        num_branchs, num_nodes, num_node_by_type = anyl.print_tree_sum_info()
        terminal_nodes = tree.get_terminal_nodes()
        root = tree.get_root()
        
        #get info about tree post addition
        is_term, is_final, c_node, idx, node = gen.add_node(tree, debug = True)
        tree.get_updated_node_list()
        node_type = c_node.get_node_type() 
        new_terminal = tree.get_terminal_nodes()
        new_root = tree.get_root()

        #A Few test to make sure meta info lists update properly post addition
        if num_nodes + 1 != tree.num_nodes():
            add_anyl =  create_node_anylizer(node)
            c_anyl = create_node_anylizer(c_node)
          
            print('\nAdded node info')
            add_anyl.print_node_params()
            print('\ncopied node info')
            c_anyl.print_node_params()

            raise ValueError('Expected {0} nodes post add but got {1} instead.'.format(num_nodes + 1, tree.num_nodes()))

        if num_node_by_type[node_type] + 1 != len(tree.nodes_by_type[node_type]):
            raise ValueError('Expected {0} {1} nodes but got {2} nodes instead'.format(num_node_by_type[node_type] + 1, node_type, tree.nodes_by_type[node_type]))

        if num_branchs != tree.num_inputs():
            raise ValueError('expected {0} input nodes but got {1} instead'.format(num_branchs, tree.num_inputs()))

        #Make sure node locations are updated properly if terminal add was run and that terminal list tracker was properly updated
        if is_term: 

            if idx != tree.get_node_list().index(node):
                raise ValueError('On terminal node addition expected index of new node to be {0} but got {1} instead'.format(idx, tree.get_node_list().index(node)))
            
            if idx + 1 != tree.get_node_list().index(c_node):
                raise ValueError('On terminal node addition expected index of old node to be {0} but got {1} instead'.format(idx + 1, tree.get_node_list().index(c_node)))

            if c_node in new_terminal:
                raise ValueError('After add terminal operation old node {0} was found in terminals'.format(c_node))

            if node not in new_terminal:
                raise ValueError('Failed to find added node in terminal nodes post add terminal operation!')
        
        #Make sure node locations are updated properly if terminal add was not run
        else:

            if idx + 1  != tree.get_node_list().index(node):
                raise ValueError('On node addition expected index of new node to be {0} but got {1} instead'.format(idx + 1, tree.get_node_list().index(node)))

            if idx != tree.get_node_list().index(c_node):
                raise ValueError('On node addition expected index of old node to be {0} but got {1} instead'.format(idx, tree.get_node_list().index(c_node)))

        #Make sure root node is properly updated post root node addition
        if is_final:

            if new_root is not node:
                raise ValueError('After root addition op expected new root to be {0} but found blank instead {1}'.format(node, root))

        #Make sure root node was not updated for all other cases
        else:
            if new_root is not root:
                raise ValueError('After normal add op expected same root node {0} but found new root node {1}'.format(root, new_root))

        #check if node connections match those expected given the trees new node list
        self.test_node_iter(tree)

         
        #don't display msg
        if supress:
            blockPrint()
           
        print('No Errors post node addition')
        enablePrint()

        return is_term, is_final

    #run add node tester n times and record types of addition operations used
    def add_node_n_times(self, n, supress = False):
        normal_count = 0
        term_count = 0
        final_count = 0

        for _ in range(n):
            is_term, is_final = self.test_add_node(supress)

            if is_term:
                term_count += 1
            elif is_final:
                final_count +=1
            else:
                normal_count += 1

        print('\nNo Errors found after {0} normal adds, {1} terminal adds and {2} final adds'.format(normal_count, term_count, final_count))

    #gen n random LSTM trees node iteration
    def test_n_node_iter(self, n):
        for _ in range(n):
            self.test_node_iter()

        print('\nNodes were iterated properly {0} times'.format(n))

    #Incredible Naming. HOW DOES HE DO IT????
    def exp_node_pos_change_post_del(self, node_idx, del_idx, merge_del = None):
        
        if node_idx < del_idx:
            return node_idx
        
        if merge_del is None:
            return node_idx + 1

        else:

            if node_idx >= del_idx and node_idx < del_idx:
                return node_idx + 1
            else:
                return node_idx + 2

    #check proper number of nodes post branch deletion
    def check_idxs_post_del(self, tree, init_node_list, del_idx, merge_del):

        for idx, n_node in enumerate(tree.get_node_list()):

            exp_idx = self.exp_node_pos_change_post_del(idx, del_idx, merge_del)

            #should include node params print out at some point myb. since we can get wrong nodes with same types
            if init_node_list[exp_idx] != n_node:
                raise ValueError('Expected node with type {0} at idx {1}, but found diffrent node of type {2} intead'.format(init_node_list[exp_idx].get_node_type(), idx, tree.get_node_list()[idx].get_node_type()))
        
    def test_del_node(self, supress = False, tree = None):
    
        #Build tree and tools to help anylize it
        if tree is None:
            tree = self.build_LSTM_tree()

        gen = LSTM_Gen([tree]) 
        anyl = Tree_Anylizer(tree, supress = True)

        #get information about tree pre node deletion
        num_branchs, num_nodes, num_nodes_by_type = anyl.print_tree_sum_info()
        
        init_node_list = tree.get_node_list()
        root = tree.get_root()

        #get info about tree post deletion
        deleted, node, node_idx, is_term, is_final, branch_del, m_node_idx = gen.del_node(tree, debug = True)

        node_type = node.get_node_type() 
        tree.get_updated_node_list()
        new_terminal = tree.get_terminal_nodes()
        new_root = tree.get_root()

        if not deleted:
            return self.test_del_node(supress)
        
        if node in tree.get_node_list():
            print('Was term: {0}, was final {1}, branch_del: {2}'.format(is_term, is_final, branch_del))
            raise ValueError('Deleted node found in node_list')

        #check if sorted nodes update properly for deleted node
        if num_nodes_by_type[node_type] - 1 != len(tree.nodes_by_type[node_type]):
            raise ValueError('Expected {0} nodes of type {1} but got {2} instead'.format(num_nodes_by_type[node_type] - 1, node_type, len(tree.nodes_by_type[node_type])))


        if node in tree.get_terminal_nodes():
            raise ValueError('Found deleted node in terminal list')

        if node in tree.nodes_by_type[node.get_node_type()]:
            raise ValueError('Found deleted node in type list')

        #check proper number of nodes post branch deletion
        if branch_del:
            
            if num_nodes - 2 != tree.num_nodes():
                raise ValueError('Expected {0} nodes post add but got {1} instead.'.format(num_nodes - 2, tree.num_nodes()))

            #should always have 1 less merge node post branch deletion
            if num_nodes_by_type['merge'] - 1 != len(tree.nodes_by_type['merge']):
                raise ValueError('Expected {0} nodes of type merge but got {1} instead'.format(num_nodes_by_type['merge'] - 1, len(tree.nodes_by_type['merge'])))
            
            if num_branchs - 1 != tree.num_inputs():
                raise ValueError('After branch deletion Expected: {0} inputs, but got {1} instead'.format(num_branchs - 1, tree.num_inputs()))

            if not is_term:
                raise ValueError('deletion in non-term node resulted in branch deletion, something has gone horribly wrong. Hide the Women and children!!!')

        else:

            if num_nodes - 1 != tree.num_nodes():
                raise ValueError('Expected {0} nodes post add but got {1} instead.'.format(num_nodes - 1, tree.num_nodes()))
        
            if num_branchs != tree.num_inputs():
                raise ValueError('After normal deletion Expected: {0} inputs, but got {1} instead'.format(num_branchs, tree.num_inputs()))

            
        if is_term:
            
            if node in new_terminal:
                raise ValueError('After terminal del op found old node {0} in terminals'.format(node))

            if not branch_del:
                    
                if tree.get_node_list()[node_idx] not in new_terminal:
                    raise ValueError('On non-branch terminal deletion could not find expected new node {0} in terminals'.format( tree.get_node_list().index[node_idx]))

        if is_final:

            if node is tree.get_root():
                raise ValueError('Found deleted node as root node')


        #check if node connections match those expected given the trees new node list
        self.test_node_iter(tree)

        if not supress:
            print('Succesful node deletion')

        return is_term, is_final, branch_del

    #test to see if deletion op allows illegal deletions
    def test_illegal_del(self):
        tree = self.build_LSTM_tree()
        gen = LSTM_Gen([tree]) 

        num_nodes = tree.num_nodes()

        for i in range(num_nodes):
            deleted, node, node_idx, is_term, is_final, branch_del, m_node_idx = gen.del_node(tree, debug = True)
            tree.get_updated_node_list()

            if tree.num_nodes() < 2:
                print (tree.num_inputs())
                raise ValueError('Deletion op allowed creation of sub 2 node tree')

            if tree.get_min_inputs() > tree.num_inputs():
                raise ValueError('Deletion op allowed branch deletion below min inputs')

    #test n trees for illegal deletion ops
    def test_n_illegal_del(self, n):
        for _ in range(n):
            self.test_illegal_del()

        print('\nNo illegal deletion ops perfomed on {0} trees'.format(n))

    def test_del_node_n_times(self, n, supress = False):
        num_term_del = 0
        num_branch_del = 0
        num_final_del = 0

        for _ in range(n):
            is_term, is_final, branch_del = self.test_del_node(supress)

            if is_term:
                num_term_del += 1
                 
                if branch_del:
                    num_branch_del += 1

            if is_final:
                num_final_del += 1

        normal_del = n - num_term_del - num_final_del

        print('\nSuccesfully completed {0} deletion with {1} normal del, {2} branch del, {3} final del and {4} term del'.format(n, normal_del, num_branch_del, num_final_del, num_term_del))

    def legal_prob(self, prob):
        if prob > 1 or prob < 0:
            return False
    
        return True

    def legal_int(self, val, restrict = None):
        
        if val < 0:
            return False

        if restrict is not None:
           
            if val > restrict:
                return False
        return True

    #Test for legal FC node values post mod
    def test_mod_FC(self, node):
        neurons = node.get_param_dict()['neurons']
        keep_prob = node.get_param_dict()['keep_prob']

        if not self.legal_int(neurons):
            raise ValueError('Found illegal neuron value {0}'.format(neurons))
        
        if not self.legal_prob(keep_prob):
            raise ValueError('Found illegal keep_prob value {0}'.format(keep_prob))

    #test for legal LSTM node vals post mod
    def test_mod_LSTM(self, node):
        neurons = node.get_param_dict()['neurons']
        dropout = node.get_param_dict()['dropout']
        rec_dropout = node.get_param_dict()['recurrent_dropout']
        
        if not self.legal_int(neurons):
            raise ValueError('Found illegal neuron value {0}'.format(neurons))
        
        if not self.legal_prob(dropout):
            raise ValueError('Found illegal dropout value {0}'.format(dropout))
        
        if not self.legal_prob(rec_dropout):
            raise ValueError('Found illegal dropout value {0}'.format(rec_dropout))

    
    def LSTM_node_mod(self, prob = 1000):
        tree = self.build_LSTM_tree()
        gen = LSTM_Gen([tree]) 
        init_node_list = deepcopy(tree.get_node_list())

        helper = Opt_Params()
        p_node, node_idx = helper.pick_random_node(tree)
        gen.mod_node(p_node)
        new_list = tree.get_node_list()

        for idx, node in enumerate(init_node_list):
            
            if idx != node_idx and not isinstance(node, Operational_Node):

                if not tree.compare_node_parameters(node, new_list[idx]):
                    print('Original node params {0}'.format(node.get_param_dict()))
                    print('New node params {0}'.format(new_list[idx].get_param_dict()))
                    raise ValueError('node not selcted was modified post mod node.')

            elif idx == node_idx and not isinstance(node, Operational_Node):
              
                if tree.compare_node_parameters(node, p_node):
                    raise ValueError('Found same parameters post node copy')

                else:

                    if node.get_node_type() is 'FC':
                        self.test_mod_FC(p_node)
                    elif node.get_node_type() is 'LSTM':
                        self.test_mod_LSTM(p_node)
                    else:
                        raise ValueError('Not sure how we got here so this kinda sucks. Got a non-LSTM non-FC node where it should not be')

    #test randomly modifying n nodes in lstm network 
    def test_n_LSTM_node_mods(self, n):

        for i in range(n):
            self.LSTM_node_mod()

        print('\nSuccefully modded {0} nodes in LSTM net'.format(n))

    #run all tests in this class defualts to 100 runs for each test
    def run_all_tests(self, link_tests = 100, add_tests = 100, del_tests = 100, iter_tests = 100, meta_tests = 100, illegal_del_tests = 100, mod_nodes = 100):
        self.test_link_copy(link_tests)
        self.add_node_n_times(add_tests, supress = True)
        self.test_del_node_n_times(del_tests, supress = True)
        self.test_n_node_iter(iter_tests)
        self.test_n_LSTM_node_mods(mod_nodes)
        self.test_n_trees_meta(meta_tests, supress = True)
        #self.test_n_illegal_del(illegal_del_tests)
        print('\nRan all tests with no issues detected')

class Cluster_Tests():


    def __init__(self, root_net = None, k = 4, step_frac = 1):

        self.k = k
        self.helper = tree_tester_thing()
        self.step_frac = step_frac
        self.root_net = root_net


    #Tries to create k trees from root_net and complains if it doesn't find that number
    def gen_root_trees(self, root_net, k, step_frac = 1,):
        
        if root_net is None:
            self.root_net = self.helper.build_LSTM_tree()
            self.root_net.loss = random.randint(0, 10)
            self.gen =  LSTM_Gen([self.root_net], step_frac)

        else:
            self.root_net.loss = random.randint(0, 10)
            self.gen = LSTM_Gen([self.root_net])
       
        #create clustered trees
        self.gen.root_trees = self.gen.gen_k_clustered_trees(self.root_net, k)
        
        #make sure generated trees where built properly
        for tree in self.gen.root_trees:
            self.helper.test_node_iter(tree)

        assert len(self.gen.root_trees) == k, "After root tree gen expected {0} trees but found {1} instead".format(k, len(self.gen.root_trees))

    def test_clusters_per_root(self):
        assert len(self.gen.root_trees) == self.k, "Please make sure gen has k root elements before using this test"

        new_nets = self.gen.gen_k_trees_per_root(self.k)
        assert len(new_nets) == self.k *self.k,"After root tree gen expected {0} trees but found {1} instead".format(self.k * self.k, len(new_nets))

    #run all tests in this class
    def run_all_tests(self):
        self.gen_root_trees(self.root_net, self.k, self.step_frac)
        self.test_clusters_per_root()

        print('Succesfully run cluster tests with no issues')

#number of data parameter we prune to -> should be set to max when network is finished but mostly used as a way to speed up testing/training when still playing with network
#bassically this prevent you from waiting an hour for results when you are testing something other then network perfomance/debugging.
NUMBER_ELEMENTS = 5000

#output size of first layer
EMBEDDINGS_LEN = 300

#Compute the max lenght of a text
MAX_SEQ_LENGHT=60

epochs = 8
batch_size = 128
seed = 40

#A walking abomination of a function that generates a pre-defined LSTM_Tree. 
tree = pre_file_test_bootstrap()

net = LSTM_Eval(NUMBER_ELEMENTS, EMBEDDINGS_LEN, seed, MAX_SEQ_LENGHT, batch_size, epochs, tree)
net.call()
#emb = create_random_emb_layer(20, 100, 300, 2000, net.embeddings_index, len(net.vectorizer.get_feature_names()) + 1)

#tester = tree_tester_thing()
#tester.run_all_tests()

#clus_tester = Cluster_Tests()
#clus_tester.run_all_tests()
