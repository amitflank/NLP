import tensorflow as tf
import numpy as np
from param_helper import Opt_Params
from params import net_restrictions

#this class represent a convoloutional layer
class Conv_Node():

	#intialize conv node. keep_prob: probabilty an element if kept. pool_w: width of pooling kernel. pool_h: height of pooling kernel. filt: number of filters output.
	#filter_width: width of convoloutional kernel. filter_height: height of convoloutional kernel. act_fxn: activation function used in this layer.
	#w_stride: number of steps taken horizontal on stride. h_stride: number of steps take vertically on stride. ws: seed for intial weights of layer.
	#w_frac: width fraction. h_frac: height fraction. output_nodes: output layer that recieve input from this layer	
	def __init__(self, keep_prob, pool_w, pool_h, filt, filter_width, filter_height, act_fxn,
                         w_stride, h_stride, ws, input_node = None, output_nodes = []):

		self.node_type = 'conv'
		self.keep_prob = keep_prob
		self.pool_w = pool_w
		self.pool_h = pool_h
		self.filt = filt
		self.filter_width = filter_width
		self.filter_height = filter_height
		self.act_fxn = act_fxn
		self.w_stride = w_stride
		self.h_stride = h_stride
		self.ws = ws
		self.input_node = input_node
		self.output_nodes = [node for node in output_nodes]	
		self.num_inputs = 1
		self.tensor_input = [None]

	#get dictionary containing parameter which can be modified
	def get_param_dict(self):
		return {'keep_prob': self.keep_prob, 'pool_w': self.pool_w, 'pool_h': self.pool_h, 'filt': self.filt, 'filter_width': self.filter_width, 
			'filter_height': self.filter_height, 'act_fxn': self.act_fxn, 'w_stride': self.w_stride, 'h_stride': self.h_stride, 'ws': self.ws}

	#assign an input node to this instant (warning will overwrite existing input)	
	def assign_input_node(self, node):
		self.input_node = node

	#add an output node to this node obj.
	def add_output_node(self, node):
		self.output_nodes.append(node)

	#return input node in array 
	def get_input_nodes(self):
		return [self.input_node]

	#return the current input node (for classes with only one input this wont change) 
	def get_cur_input_node(self):
		return self.input_node

		
#this class represents a linear layer
class Lin_Node():

	#keep_prob: probabilty an element is not zeroed in this layer. neurons: neurons in layer. act_fxn: which activation function is used.
	#ws: seed for intializing weights. input_node: which layer feeds into this one. output_node: which layers this one feeds.	
	def __init__(self, keep_prob, neurons, act_fxn, ws, input_node = None , output_nodes = []):
		self.node_type = 'lin'
		self.keep_prob = keep_prob
		self.neurons = neurons
		self.act_fxn = act_fxn
		self.ws = ws
		self.input_node = input_node
		self.output_nodes = [node for node in output_nodes]
		self.num_inputs = 1
		self.tensor_input = [None]
	
	#get dictionary containing parameter which can be modified
	def get_param_dict(self):
		return {'keep_prob': self.keep_prob, 'neurons': self.neurons, 'act_fxn': self.act_fxn, 'ws': self.ws}


	#assign an input node to this instant (warning will overwrite existing input)	
	def assign_input_node(self, node):
		self.input_node = node

	#add a node to this instances outputs
	def add_output_node(self, node):
		self.output_nodes.append(node)

	#return input node(s) as an array
	def get_input_nodes(self):
		return [self.input_node]

	#return the current input node (for classes with only one input this wont change) 
	def get_cur_input_node(self):
		return self.input_node

#This class represents an operation
class Op_Node():

	#op_name: name of operation. num_inputs: how many inputs this operation requires. input_node: input(s) that feed into this operation.
	#output_nodes: layer(s) this operation feeds.
	def __init__(self, op_name, num_inputs, input_node, output_nodes = []):
		self.node_type = op_name
		self.num_inputs = num_inputs
		self.input_node = input_node
		self.output_nodes = [node for node in output_nodes]
		self.cur_input = self.get_input_nodes()[0]
		self.tensor_input = [None for i in range(num_inputs)]


	#add an output into which this object feeds
	def add_output_node(self, node):
		self.output_nodes.append(node)
	

	#get array of input nodes to this object
	def get_input_nodes(self):
		if self.num_inputs == 1:
			return [self.input_node]
		else:
			return [elm for elm in self.input_node]


	#return the current input node (for classes with only one input this wont change) 
	def get_cur_input_node(self):
		return self.cur_input

	#switch the current input to second input
	def switch_current_input(self):
		if self.num_inputs == 1:
			raise ValueError('You tried switching inputs, but there is only one input for this Op_Node')
		else:
			self.cur_input = self.get_input_nodes()[1]

class Hyp_Tree():

	#learning_rate: the intial update rate for each run. decay: how quickly learning rate decays after each run. runs: number of training runs.
	# R: Restriction object used to help create new nodes. step_frac: clustering variable for new node creation. root_node: the root node of this tree.	
	def __init__(self, learning_rate, decay, runs, R, step_frac, root_node, loss = None, w_frac = None, h_frac = None):
		self.root_node = root_node
		self.learning_rate = learning_rate
		self.decay = decay
		self.runs = runs
		self.step_frac = step_frac
	#	self.R = R
		self.cur_node = root_node
		self.phase = 'search' #we always defualt to search phase on intilization (as we start at root)
		self.node_list = []
		self.loss = loss 
		self.terminal_nodes = [] #array to store terminal nodes
		self.nodes_by_type = {'LSTM': [], 'FC': [], 'merge': []} #store nodes by type
		self.create_node_list(self.root_node) #create various lists that store nodes in various categories

	#Check if the parameter values for node1 and node2 are the same
	def compare_node_parameters(self, node1, node2):

		#We cannot compare operational nodes so we will raise an error if either one of our passed nodes is an op node
		if node1.node_type is  'merge':
			raise ValueError('Cannot compare opertaional nodes')

		if node2.node_type is 'merge':
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

	#reset tensor_inputs of passed node
	def reset_node_tensor_inputs(self, node):
		node.tensor_input = [None for i in range(node.num_inputs)]

	#reset tensor_inputs for all nodes	
	def reset_all_node_tensor_inputs(self):
		nodes = self.get_node_list()
		for node in nodes:
			self.reset_node_tensor_inputs(node)

		self.terminal_inputs = None

	#substitute node output o_node for n_node. if n_node is None will clear all outputs, this currently prevents final nodes from having multiple outputs not sure if that leads to bugs or just
	#flexebity issue. would need to go through and change the terminal node notation to None from an empty list to allow final nodes to have multiple outputs if thats a thing
	#we decided to ever do.
	def sub_output_node(self, node, o_node, n_node):

		#loop through nodes outputs
		for idx, val in enumerate(node.output_nodes):

			#make sure o_node is a real output otherwise we will raise an error
			if val is o_node:

				#here we check and deal with empty outputs 
				if isinstance(n_node, list) and len(n_node) == 0:
					node.output_nodes = []
				else:
					node.output_nodes[idx] = n_node

				return True

		raise ValueError('The output node you are searching for does not exist!!!')

	#substitute input node for nodes w/1 input
	def sub_single_input_node(self, node, o_input, n_input, sub_out = True):
			if node.input_node == o_input:
				node.input_node = n_input
				if n_input is not None and sub_out:
					self.sub_output_node(n_input, o_input, node)
				return True
			else:
				return False
	
	#substitue an input node for nodes w/multiple inputs
	def sub_mult_input_node(self, node, o_input, n_input, sub_out = True):
		for idx, i_node in enumerate(node.get_input_nodes()):
			if i_node == o_input:
				node.input_node[idx] = n_input
				if n_input is not None and sub_out:
					self.sub_output_node(n_input, o_input, node) #might be a bug here do some testing, node used to be n_input -> that doesn't seem to make sense so im assuming
				return True					# it was a typo but maybe I meant to do that and i acctually messed something up
		return False


	#extract valid element if it exists from possible list obj.
	def get_valid_input(self, val):
		if isinstance(val, list):
			for elm in val:
				if elm is not None:
					return elm
			raise ValueError('you passed a list with no valid inputs')

		else:
			return val


	#substitute an existing input node with a new input node in passed node.	
	def substitute_input_node(self, node, old_input_node, new_input_node, sub_out = True):
		if node.num_inputs == 1:
			result = self.sub_single_input_node(node, old_input_node, new_input_node, sub_out)
		else:
			result = self.sub_mult_input_node(node, old_input_node, self.get_valid_input(new_input_node), sub_out)


		if result is False:
			raise ValueError('The node you are trying to substitute does not exist')
		 

	#assign loss for this tree
	def assign_loss(self, loss):
		self.loss = loss

	#this method should be called when generating Eval Net so that intial terminal inputs can be read in.	
	def assign_terminal_inputs(self, terminal_inputs):
		self.terminal_inputs = terminal_inputs
		self.terminal_num = 0



	#link pair of nodes. Note this function only works on nodes which take a maximum of 1 input
	def link_node_pair(self, node1, node2):
		node2.assign_input_node(node1)
		node1.add_output_node(node2)
	
	
	#generate flatten node with input node. 
	def flatten_conv_node(self, node):
		if node.node_type == 'conv':
			flat_op = Op_Node('flatten', 1, node)
			node.add_output_node(flat_op)
		else:
			raise ValueError('You tried passing a {0} node to flatten op'.format(node.node_type))

	#create a merge_op node composed of 2 passed nodes.
	def merge_nodes(self, node1, node2):
		if node1.node_type == 'lin' or node1.node_type == 'flatten' and node2.node_type == 'lin' or node2.node_type == 'flatten':
			merge_op = Op_Node('merge', 2, [node1, node2])
			node1.add_output_node(merge_op)
			node2.add_output_node(merge_op)

		else:
			raise ValueError('You tried passing {0} and {1} nodes to merge op, you must pass 2 lin nodes or op equivalents'.format(node1.node_type, node2.node_type))
	
	#print the node type of passed node	
	def print_node(self, node):
		print node.node_type

	#check if node exists
	def valid_node(self, node):
		if node is not None:
			return True
		else:
			return False

	#clear terminal nodes
	def clear_terminal_nodes(self):
		self.terminal_nodes = []

	#reset the dict that stores our nodes by type
	def reset_node_dict(self):
		self.nodes_by_type = {'conv': [], 'lin': [], 'merge': [], 'flatten': []}

	#assign node to its relevant type list
	def assign_node(self, node):
		#if node is terminal node we will add it to list of terminal nodes
		if self.is_terminal_node(node):
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

	#substiture the terminal node old_term node with new_term_node
	def sub_terminal_node(self, new_term_node, old_term_node):
		for idx, node in enumerate(self.terminal_nodes):
			if node is old_term_node:
				self.terminal_nodes[idx] = new_term_node
				return True

		raise ValueError('You tried to substitute a terminal node that does not exist')
	
	#delete passed terminal noed and it's accosiated input dimensions	
	def del_terminal_node(self, old_term_node):
		for idx, node in enumerate(self.terminal_nodes):
			if node is old_term_node:
				self.terminal_nodes.remove(node)	
	
				#remove dimensions from correponding lists
				del self.w_frac[idx]
				del self.h_frac[idx]

				return True
		
		raise ValueError('You tried to delete a terminal node that does not exist')


	#clear node list
	def reset_node_list(self):
		self.node_list = []
		self.terminal_nodes = []
		self.nodes_by_type = {'conv': [], 'lin': [], 'merge': [], 'flatten': []}

	#get list of nodes from tree
	def get_node_list(self):
		
		#we'll create our node list if it is empty
		if not self.node_list:
			self.create_node_list(self.root_node)
		 
		return self.node_list		

	#call this version of get node list if you have made changes to tree since node list does not automatically update
	def get_updated_node_list(self):
		#make sure node list is cleared
		self.reset_node_list()

		self.create_node_list(self.root_node)
		return self.node_list

	#check if passed node is a terminal node	
	def is_terminal_node(self, node):
		chk = node.get_input_nodes()[0]
		if chk == None:
			return True
		else:
			return False


	#assign the tensor input for the current terminal node
	def assign_terminal_tensor(self):
		try:
			self.cur_node.tensor_input = [self.terminal_inputs[self.terminal_num]]

		except IndexError:
			raise IndexError('Tried assinging a terminal input out of range, attempted to get idx {0}, term_input values {1}'.format(self.terminal_num, self.terminal_inputs))

		self.terminal_num += 1

	#find the next terminal node in our tree	
	def get_next_terminal_node(self, node):
		#assign passed node as current node
		#cur_node = node

		#if node is not a termonal node assign nodes current input as next node
		#while not self.is_terminal_node(cur_node):
		#	cur_node =  cur_node.get_cur_input_node()

			
		#assign current input node as the terminal node we found	
		self.cur_node = self.terminal_nodes[self.terminal_num]
			
		#assign node tensor
		self.assign_terminal_tensor()

		#switch phase to read
		self.phase = 'read'

	#read current node out and increment to next node
	def increment_node(self):

		#assign current node to next node and return saved node
		self.cur_node = self.cur_node.output_nodes[0]


	#check if nodes inputs are valid
	def has_tensor_inputs(self, node):
		for val in node.tensor_input:
			if val == None:
				return False
		return True

	#switch nodes input branch
	def switch_node_inputs(self, node):
		node.switch_current_input()	
		
	#perform a read operation on our tree
	def read_op(self):
		if self.has_tensor_inputs(self.cur_node):
			return self.cur_node
		else:
			if self.cur_node.num_inputs == 1:
				raise ValueError('You cannot switch inputs for a {0} node as it only has 1 input'.format(self.cur_node.node_type))

			self.switch_node_inputs(self.cur_node)
			return self.search_op()
			

	#perform a search operation on our tree
	def search_op(self):
		self.get_next_terminal_node(self.cur_node)
		return self.cur_node

	#get next node to create in our tree
	def get_next_node(self):
		n_node = None
		if self.phase == 'search':
			n_node = self.search_op()
		else:
			n_node = self.read_op()
		
		if n_node == None:
			raise ValueError('Trying to return NoneType node something went wrong')

		return n_node

	#check if passed node if root_node
	def is_root(self, node):
		if node == self.root_node:
			return True
		else:
			return False

	#check if passed node has any output values
	def has_output(self, node):
		if len(node.output_nodes) > 0:
			return True
		else:
			return False

	#checl if passed object is defined
	def obj_exists(self, obj):
		if obj is None:
			return False
		else:
			return True

	#add the result passed to first available tesnor_input in node. Raises exception if no spots available for result placement. 
	def add_input_tensor(self, node, result):
	
		if not self.obj_exists(result):
			raise ValueError('The result you passed does not exist, you must pass a real result')
	
		for idx, tensor in enumerate(node.tensor_input):
			if tensor == None:
				node.tensor_input[idx] = result
				return True
		raise ValueError('You tried adding an input tensor node_type {0},  but there is no valid location to place it in node tensor_input {1}'.format(node.node_type, node.tensor_input))

	#push results of prevous layer to all its output nodes
	def push_results_to_outputs(self, node, result):
		for out_node in node.output_nodes:
			self.add_input_tensor(out_node, result)	

		self.increment_node()

