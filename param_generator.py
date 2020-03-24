import tensorflow as tf
import numpy as np
import random
from random import randint
from param_helper import Opt_Params, Prob_Adjuster
from copy import deepcopy
from abc import ABC, abstractmethod
from params import Legal_Term_Nodes

class Hyp_Gen(ABC):

    #trees a list of tree obj. step_frac: clustering parameter(lower = more clustered). Defualt 1 doesn't really make sense atm should pass value in general
    def __init__(self, root_trees, step_frac = 1):
        self.root_trees = root_trees
        self.step_frac = step_frac
        self.p_helper = Opt_Params()
        self.prob_adj = Prob_Adjuster()

    #check if passed varaible is an instance
    def is_instance(self, obj, inst_type):
        if not isinstance(obj, inst_type):
            raise ValueError('You tried passing an non-instance where an instance was required')
    
    #replace existing prob adj with a new one
    def sub_prob_adj(self, new_prob_adj):
        self.prob_adj = new_prob_adj

    #each tree should implement it's own variant of mod_node based on it's node types
    @abstractmethod
    def mod_node(self, node):
        pass


    #modify all the nodes in passed tree    
    def prob_mod_nodes(self, tree):
        for node in tree.get_node_list():
            self.mod_node(node)


    #create a slightly modified copy of passed node. (sort of a evolutionary copy)
    def get_moded_node_copy(self, node):
           
        #create a copied node
        c_node = deepcopy(node)

        #clear all links to other nodes
        c_node.reset_node()

        #modify node and return
        self.mod_node(c_node)
        return c_node

    #generate num_trees new hyp_trees each of which are a copy of passed tree with the exception of having the node at node_idx randomly modified.
    def gen_x_trees_w_single_node_mod(self, node_idx, tree, num_trees):
    
        new_trees = [None for i in range(num_trees)]

        for i in range(num_trees):

            modded_tree = deepcopy(tree)
            node_to_mod = modded_tree.node_list[node_idx]
       
            self.mod_node(node_to_mod)
            new_trees[i] = modded_tree
        
        return new_trees

    #generate num_trees new trees with each node being modified (big change).
    def gen_x_trees_w_all_node_mod(self, node_idx, tree, num_trees):
    
        new_trees = [None for i in range(num_trees)]

        for i in range(num_trees):

            modded_tree = deepcopy(tree)
            self.prob_mod_nodes(modded_tree)
            new_trees[i] = modded_tree
        
        return new_trees

    #bad name. modify probablisticly mod single node, gloabal params, add node or delete node. 
    def mod_tree_and_gen(self, node_helper, tree, num_trees):

            init_num_nodes = tree.num_nodes()

            #get clean version of tree so we don't effect passed tree               
            modded_tree = deepcopy(tree)

            #first modify non-node based parameters w/some random prob
            self.p_helper.mod_global_params(modded_tree, self.step_frac, self.prob_adj.global_param_prob)

            #b/c of how the net is structures if we delete below 2 the net doesn't exits(min 1 conv, flat node means we delete only non-op node)
            #if init_num_nodes > 2:

                #roll and possible delete a random node from our new tree
                #self.prob_del_node(chance = self.prob_adj.del_node_prob, tree = modded_tree)

            #roll and possible add a new node to our new tree
            #self.prob_add_node(chance = self.prob_adj.add_node_prob, tree = modded_tree)

            #replace our tree w/new one
            node_helper.replace_tree(modded_tree)

            #get the diffrence in nodes post possible additions/deletions
            num_nodes_diff =  len(modded_tree.get_node_list()) - init_num_nodes

            #if jumping idx to account for new nodes doesn't put us below 0 idx jump the cur_idx in node helper
            if not node_helper.get_cur_idx() + num_nodes_diff < 0:
                node_helper.jump_cur_idx(num_nodes_diff)

            #Use the modded tree we generated to create num_trees new trees. 
            return self.gen_x_trees_w_single_node_mod(node_helper.get_cur_idx(), modded_tree, num_trees)


    #geneate new hyp_trees clustered around root_tree       
    @abstractmethod
    def gen_tree(self, root_tree):
        pass


    #check if this node has mode than 1 input
    def multiple_input_node(self, node):
        if node.num_inputs() > 1:
            return True
        else: 
            return False

    #toggle our node until it's input is input_node 
    def toggle_input_to_node(self, node, input_node):
            if self.multiple_input_node(node):
                    
                for i in range(node.num_inputs()):

                    if node.get_cur_input_node() == input_node:
                        return True
                        
                    node.toggle_input()

                raise ValueError('Could not find input node')

    #Std node deletion op with no special case
    def normal_node_del(self, node):
            for o_node in node.get_output_nodes():

                #for outputs with multiple input nodes we want to garuntee we are replacing the correct input
                self.toggle_input_to_node(o_node, node)

                #now we replace the output nodes input with nodes input
                o_node.assign_input_node(node.get_cur_input_node())

                #delete node from it's inputs, output nodes
                node.get_cur_input_node().del_output(node)
                
            #finally unlink node from everything
            node.reset_node()
              
    #delete terminal node
    def terminal_del(self, node, tree, debug = False):
            branch_del = False
            m_node_idx = None

            for o_node in node.get_output_nodes():
                o_node.swap_input_node(node.get_cur_input_node(), node)
                   
                #If terminal node if the only node in branch then delete branch.
                if self.multiple_input_node(o_node):
                    o_node.toggle_input()
                    m_node_idx = tree.get_node_list().index(o_node)
                    self.normal_node_del(o_node)
                    branch_del = True
                    

            #finally unlink node from everything
            node.reset_node()
            
            if debug:
                return branch_del, m_node_idx
        
    #delete a final node
    def final_del(self, node, tree):

            if self.multiple_input_node(node):
                raise ValueError('Cannot delete final node with multiple inputs')

            if node.get_cur_input_node().num_outputs() > 1:
                raise ValueError('Cannot delete final node if it has an input node with multiple outputs')

            node.get_cur_input_node().del_output(node)
            tree.assign_root(node.get_cur_input_node()) #need to update root for updated nodes list fxn to work properly post modification
            node.reset_node()


    #delete random non-operation node from tree. Also clean up branchs if deletion requires it.
    def del_node(self, tree, debug = False):
            node, node_idx = self.p_helper.randomly_pick_noop_node(tree) #for del ops we only select from no operational nodes

            is_term = False
            is_final = False
            branch_del = False 
            m_node_idx = None

            if self.p_helper.legal_del(node, tree):
                
                if node.is_terminal_node():
                    branch_del, m_node_idx = self.terminal_del(node, tree, debug = debug) #This is legal since we don't care about branch del in cases where debug = False.
                    is_term = True

                elif node.is_final_node():
                    self.final_del(node, tree)
                    is_final = True

                else:
                    self.normal_node_del(node)
                    
                deleted = True
           
            else:
                deleted = False
            
            #return some useful debugging info
            if debug:
                return deleted, node, node_idx, is_term, is_final, branch_del, m_node_idx

    #flip a coin
    def coin_filp(self):
            return bool(randint(0,1))

    #noraml add op when node has valid input and output
    def normal_add_node(self, node, output_node, input_node):
            node.replace_empty_input(input_node)
            node.add_output_node(output_node)
            input_node.del_output(output_node)
            input_node.add_output_node(node)
            output_node.swap_input_node(node, input_node)

    #add new terminal node
    def add_terminal_node(self, node, output_node):
            output_node.replace_empty_input(node)

    #add a new final node
    def add_final_node(self, node, input_node, tree):
            node.replace_empty_input(input_node)
            tree.assign_root(node)

    #just put the last line here for now. Eventually this fxn should select op being implemented and pick proper fxn to perform op.
    def op_selector(self, tree):
            tree.get_updated_node_list() #update our tree with changes

    #should add some restriction params. Note as we add more operaional nodes we will need to add a condition on when op nodes can be added.
    def add_node(self, tree, debug = False):

            c_node, node_idx = self.p_helper.randomly_pick_noop_node(tree)
            node = deepcopy(c_node)
            node.reset_node()

            #get info useful for debugging
            if debug:
               is_term = c_node.is_terminal_node()
               is_final = c_node.is_final_node()

            #Since we defual to picked node as input we randomly flip a coin in terminal case to determine new terminals
            if c_node.is_terminal_node():
              
                if self.coin_filp():
                    self.add_terminal_node(node, c_node)
                else:
                    self.normal_add_node(node, c_node.get_output_nodes()[0], c_node) #note this does NOT work for multiple outputs
                    
                    if debug:
                        is_term = False #so ugly 

            elif c_node.is_final_node():
                self.add_final_node(node, c_node, tree)

            else:
                self.normal_add_node(node, c_node.get_output_nodes()[0], c_node)  #note this does NOT work for multiple outputs

            #return info useful for debugging 
            if debug:
                return is_term, is_final, c_node, node_idx, node

    #get k new hyp_tree around passed hyp_tree. root_tree: tree to cluster around
    def gen_k_clustered_trees(self, root_tree, k):
        tree_list = [None for i in range(k)]

        for i in range(k):
            tree_list[i] = self.gen_tree(root_tree)

        return tree_list

    #generate k trees clustered around each root_tree.     
    def gen_k_trees_per_root(self, k):
                        
        num_roots = len(self.root_trees)
        tree_list = [None for i in range(k * num_roots)]

        for i, tree in enumerate(self.root_trees):
            
            tmp_list = self.gen_k_clustered_trees(tree, k)
                
            for j, tree in enumerate(tmp_list):
                idx = i * k + j #i: array idx, k: array size, j: current element in arr
                tree_list[idx] = tree
            
        return tree_list

    #insert a hyper_param into sorter hyper-parameter list
    def insert_to_param_list(self, n_tree, k):

        for idx, tree in enumerate(self.root_trees):
        
            #insert param if loss is less then loss of current param
            if tree.loss > n_tree.loss:
                self.root_trees.insert(idx, n_tree)
                return True
                
            if idx > k + 1:                                                                                                                                                                                                 return False
        
        #if we reached this point append to end 
        self.root_trees.append(n_tree)

    #insert all our new parameters into param list and prune list to length k
    def insert_and_prune(self, new_trees, k):

        for tree in new_trees:
            self.insert_to_param_list(tree, k)

        #set root trees to temp list
        self.root_trees = self.root_trees[:k]

    #reset the tensor input for each of our root trees              
    def reset_root_tensor_inputs(self):
        for tree in self.root_trees:
            tree.reset_all_node_tensor_inputs()

class LSTM_Gen(Hyp_Gen):
    
    #trees a list of tree obj. step_frac: clustering parameter(lower = more clustered). Defualt 1 doesn't really make sense atm should pass value in general
    def __init__(self, root_trees, step_frac = 1):
        super().__init__(root_trees, step_frac)
        self.legal_term_nodes = ['LSTM', 'FC', 'Embedding']

    #mod a node if it is non-op otherwise do nothing        
    def mod_node(self, node):
        if node.is_op_node():
            placeholder = -1 #we do nothing here
        else:
            if node.get_node_type() == 'LSTM':
                self.p_helper.mod_LSTM_node(self.step_frac, self.prob_adj.element_change_prob, node)
            elif node.node_type == 'FC':
                self.p_helper.mod_FC_node(self.step_frac, self.prob_adj.element_change_prob, node)
            elif node.node_type == 'Embedding':
                self.p_helper.mod_Embedding_node(node, step_frac, self.prob_adj.element_change_prob)
            else:
                raise ValueError('You tried modifying a unrecognized node {0}, valid non-operational nodes are FC, LSTM and Embedding'.format(node.node_type))

    #geneate new hyp_trees clustered around root_tree       
    def gen_tree(self, root_tree):

        #deepcopy does not work on tf var so we must get rid of them pre copy
        root_tree.reset_all_node_tensor_inputs()

        #create copy of our root tree which we will use as seed
        hyp_tree = deepcopy(root_tree)
            
        #make sure our loss is None as we have not evaluated this net
        hyp_tree.assign_loss(None)
            
        #first modify non-node based parameters
        self.p_helper.mod_global_params_dict(hyp_tree, self.step_frac, self.prob_adj.global_param_prob)

        #get a list of nodes in the tree
        node_list = hyp_tree.get_node_list()
            
        #conditionally modify each element in each node in our new tree
        self.prob_mod_nodes(hyp_tree)

        #roll and possible delete a random node from our new tree
        #self.prob_del_node(chance = self.prob_adj.del_node_prob, tree = hyp_tree)

        #roll and possible add a new node to our new tree
        #self.prob_add_node(chance = self.prob_adj.add_node_prob, tree = hyp_tree)

        #roll and possible pair swap for each node type in tree
        #self.prob_swap_all_node_types(chance = self.prob_adj.swap_node_prob, tree = hyp_tree)
            
        #return our new modified hyp_tree       
        return hyp_tree
