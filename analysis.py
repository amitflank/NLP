import tensorflow as tf
from LSTM_Node import LSTM_Node, FC_Node, Network_Tree, LSTM_Tree
from abc import ABC, abstractmethod
import sys, os

#This file should contain various classes that can be used to anylize and debug our code.


#Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

#Restore
def enablePrint():
    sys.stdout = sys.__stdout__


#Abstract class that should be a parent to specefic node anlyizers for each of our node types
class Node_Anylizer(ABC):

    def __init__(self, node):
        self.node = node

    #checks if passed node is valid
    @abstractmethod
    def valid_node(self, node):
        pass

    #returns node params of a node
    def print_node_params(self):
        
        out = [node.get_node_type() for node in self.node.get_output_nodes()]
        term_node = False

        print('Node type is {0} with following parameters {1}'.format(self.node.get_node_type(), self.node.get_param_dict()))
        print('Node has {0} output nodes with following types {1}'.format(len(out), out))
        
        for inp_node in self.node.get_input_nodes():
            if inp_node is None:
                term_node = True

        if term_node:
            print(self.node.get_input_nodes())
            print('Node is a terminal node')

        else:
            inp = [node.get_node_type() for node in self.node.get_input_nodes()]
            print('Node has {0} input nodes with following types {1}'.format(len(inp), inp))

    #assign an input node. Warning -> This will overide any existing node
    def assign_input_node(self, input_node):
        if self.input_node.get_node_type() is 'FC' or self.input_node.get_node_type() is 'merge':
            raise ValueError('LSTM node does not accpet inputs of type {0}'.format(self.input_node.get_node_type()))

        self.input_node = input_node
        return node.get_param_dict()

#anlylizer class for our merge node
class Merge_Anylizer(Node_Anylizer):

    def __init__(self, node):
        self.valid_node(node)
        super().__init__(node)

    #Make sure we are using correct node type
    def valid_node(self, node):
        if node.get_node_type() is not 'merge':
            raise ValueError('Merge anylizer only accepts nodes of type merge. You passed a node of type: {0}!'.format(node.get_node_type()))

#anlylizer class for our FC node
class FC_Anylizer(Node_Anylizer):

    def __init__(self, node):
        self.valid_node(node)
        super().__init__(node)

    #Make sure we are using correct node type
    def valid_node(self, node):
        if node.get_node_type() is not 'FC':
            raise ValueError('FC anylizer only accepts nodes of type FC. You passed a node of type: {0}!'.format(node.get_node_type()))

#anlylizer class for our FC node
class LSTM_Anylizer(Node_Anylizer):

    def __init__(self, node):
        self.valid_node(node)
        super().__init__(node)

    #Make sure we are using correct node type
    def valid_node(self, node):
        if node.get_node_type() is not 'LSTM':
            raise ValueError('LSTM anylizer only accepts nodes of type LSTM. You passed a node of type: {0}!'.format(node.get_node_type()))

#create a the node anylizer corresponding to the passed node
def create_node_anylizer(node):
    key_dict = {'FC': FC_Anylizer, 'LSTM': LSTM_Anylizer, 'merge': Merge_Anylizer}
    
    return key_dict[node.get_node_type()](node)

class Tree_Anylizer():

    def __init__(self, tree, supress = False):
        self.tree = tree
        self.supress = supress

    #get number of branchs in our tree
    def get_num_branchs(self):
        return len(self.tree.terminal_nodes)

    #print sumary info about this tree
    def print_tree_sum_info(self):

        if self.supress:
            blockPrint()

        num_branchs = len(self.tree.terminal_nodes)
        num_nodes = len(self.tree.node_list)
        num_node_type = {}

        for key, val in self.tree.nodes_by_type.items():
            print('This tree has: {0} {1} nodes'.format(len(val), key))
            num_node_type[key] = len(val)

        print('Total number of nodes: {0}, in {1} branchs'.format(num_nodes, num_branchs))
        print('Tree hyp meta info: runs: {0}, lr: {1}, loss {2}, decay factor: {3}'.format(self.tree.global_param['runs'], 
                                    self.tree.global_param['learning_rate'], self.tree.loss, self.tree.global_param['decay']))

        enablePrint()

        return num_branchs, num_nodes, num_node_type

    #get a list of nodes from passed node to next merge node
    def get_remaining_branch(self, node):
        cur_node = self.tree.terminal_nodes[idx]
        node_list = []


    #Get info about terminal branch in idx
    def get_term_branch_info(self, branch_idx):
        if branch_idx >= self.get_num_branchs():
            print('you picked invalid branch idx: {0}, options are from 0 to {1}'.format(branch_idx, self.get_num_branchs()))
        
        else:
            node_list = self.get_remaining_branch(self.tree.terminal_nodes[idx])
            print('Number nodes in terminal branch with index: {0} is {1}'.format(branch_idx, len(node_list)))
            print('nodes in branch {0}'.format(node_list))
         
            return node_list

    #function that lets us anylizing remaining elements in some branch from passed node 
    def get_remaining_branch_info(self, node):
        node_list = self.get_remaining_branch(node)
        print('Nodes remaining in current branch: {0}'.format(len(node_list)))
        print('nodes by type remaining in branch {0}'.format(node_list))
       
        return node_list
