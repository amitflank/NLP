import tensorflow as tf
import numpy as np
import time
from copy import deepcopy
from hyp_tree import Op_Node
from hyp_tree import Hyp_Tree
from hyp_gen import Hyp_Gen


#ONLY PARTIALY TESTED!!!
#class used to represent executable node for optimization network
class Opt_Exec_Node():  

	#hyp_params an array representing our hyper parameters. lstm_size: int, representing number hidden_state values. loss: if layer has been evaled pass value otherwise defulat to none.  
	def __init__(self, node_type, hyp_params, list_size, loss = None):
		self.org_num_hyps = len(hyp_params)
		self.opt_node_type = node_type
		self.list_size = list_size
		self.hyp_params = self.pad_list_w_zeros(hyp_params, list_size)
		self.num_hyps = len(self.hyp_params)
		self.loss = loss

	#adj hyp_params value for term node to help our net identify diffrences more easily
	def term_adj(self):
		adj_hyps = [0 for i in range(self.num_hyps)]

		for idx, elm in enumerate(self.hyp_params):
			if idx == 7:
				adj_hyps[idx] = elm * 100
			elif idx == 10:
				adj_hyps[idx] = elm * 100
			elif idx == 12:
				adj_hyps[idx] = elm * 1000000
			else:
				adj_hyps[idx] = elm * 10
			
		return adj_hyps
		
	#adj hyp_params value for conv node to help our net identify diffrences more easily
	def conv_adj(self): 
		adj_hyps = [0 for i in range(self.num_hyps)]
		
		for idx, elm in enumerate(self.hyp_params):
			if idx == 7:
				adj_hyps[idx] = elm * 100
			else:
				adj_hyps[idx] = elm * 10
		return adj_hyps

	#adj hyp_params value for lin node to help our net identify diffrences more easily
	def lin_adj(self):
		adj_hyps = [0 for i in range(self.num_hyps)]
		
		for idx, elm in enumerate(self.hyp_params):
			if idx == 0:
				adj_hyps[idx] = elm * 100
			if idx == 1:
				adj_hyps[idx] = elm * .01
			else:
				adj_hyps[idx] = elm * 10

		return adj_hyps
		

	#make adj to our hyp_params so they operate on same order of magnitude to make it easier for our prediction net 
	def sensitivy_adj(self):
		if self.opt_node_type is 'term':
			return self.term_adj()

		elif self.opt_node_type is 'conv':
			return self.conv_adj()

		else:
			return self.lin_adj()

		

	#pad the passed list with zeros to a desired length
	def pad_list_w_zeros(self, list_, desired_length):

		len_list = len(list_)
		
		#raise expection if illegal parameters were passed
		if len_list > desired_length:
			raise ValueError('You tried padding list of size {0} to size {1}, list must be smalled  desired list size!!!'.format(len_list, desired_length))

		#calculate how many zeros we need to add to our list
		num_zeros = desired_length - len_list 

		#pad our list
		new_list = np.pad(list_, (0, num_zeros), 'constant', constant_values=(0, 0))
			
		#return our new padded list
		return new_list
	
	#remove any padding on hyp_params
	def remove_padding(self):
		self.hyp_params = [self.hyp_params[i] for i in range(self.org_num_hyps)]

		
#This class represents a merge node for optimization network.
class Opt_Merge_Node():

	#hyp1, hyp2: set of 2 hyper_parmeter arrays representing Lin nodes (or Lin node equivalents). loss: if layer has been evaled pass value otherwise defulat to none.  
	def __init__(self, lstm_size, loss = None):
		self.opt_node_type = 'merge'
		self.lstm_size = lstm_size
		self.loss = loss
	
	#Do nothing
	def remove_padding(self):
		a = 0

#This class represent a Lin Node for optimization network. 
class Opt_Flatten_Node():
	
	#hyp_param: hyper parameters of conv node being flattened. loss: if layer has been evaled pass value otherwise defulat to none.
	def __init__(self, hyp_params, list_size, loss = None):
		self.org_num_hyps = len(hyp_params)
		self.opt_node_type = 'flatten'
		self.list_size = list_size
		self.hyp_params = self.pad_list_w_zeros(hyp_params, list_size)
		self.loss = loss
		self.num_hyps = len(self.hyp_params)

	#make adj to our hyp_params so they operate on same order of magnitude to make it easier for our prediction net	
	def sensitivy_adj(self):
		adj_hyps = [0 for i in range(self.num_hyps)]
		
		for idx, elm in enumerate(self.hyp_params):
			if idx == 0:
				adj_hyps[idx] = elm * 100
			if idx == 1:
				adj_hyps[idx] = elm * .01
			else:
				adj_hyps[idx] = elm * 10

		return adj_hyps
	
	#pad the passed list with zeros to a desired length
	def pad_list_w_zeros(self, list_, desired_length):

		len_list = len(list_)
		
		#raise expection if illegal parameters were passed
		if len_list > desired_length:
			raise ValueError('You tried padding list of size {0} to size {1}, list must be smalled  desired list size!!!'.format(len_list, desired_length))

		#calculate how many zeros we need to add to our list
		num_zeros = desired_length - len_list 

		#pad our list
		new_list = np.pad(list_, (0, num_zeros), 'constant', constant_values=(0, 0))
			
		#return our new padded list
		return new_list
		
	#remove any padding on hyp_params
	def remove_padding(self):
		self.hyp_params = [self.hyp_params[i] for i in range(self.org_num_hyps)]

#dirty hacked together holder class atm should write this in some cleaner better abstracted format later, also has a terrible name so theres that as well
class Node_Input_Lengths():
	
	def __init__(self):
		#variable storing the input lengths of each node type
		self.length_dict = {'conv': 10, 'lin': 4, 'term': 13, 'merge': 8, 'flatten': 10} #flatten is kinda made up atm since its input is some abitrary filler

		#should be used to dictate how to place varaibles from an unordered dictionary into an ordered list 
		self.list_orders = {'term': ['h_stride', 'pool_w', 'ws', 'filter_height', 'act_fxn', 'filt', 'w_stride', 'keep_prob', 'pool_h', 'filter_width', 'decay', 'runs', 'lr'], 
					'conv': ['h_stride', 'pool_w', 'ws', 'filter_height', 'act_fxn', 'filt', 'w_stride', 'keep_prob', 'pool_h', 'filter_width'],
					'lin': ['keep_prob', 'neurons', 'act_fxn', 'ws']} 
		
#This class builds the node tree used to construct our optimization net
class Opt_Tree_Builder():

	def __init__(self, hyp_tree, losses, lstm_size):

		#number of hidden units in each of our optimization net nodes
		self.lstm_size = lstm_size

		#create a list of losses if current list is invalid 
		self.losses = self.gen_valid_loss_list(hyp_tree, losses)

		#check valid nodes and losses and init
		self.check_equal_nodes_and_losses(hyp_tree, self.losses)
		self.hyp_tree = hyp_tree
		
		#hacked together atm fix
		self.meta_info = Node_Input_Lengths()

		#Create a optimization net representation of each node in passed hyp_tree	
		self.prop_node_list(hyp_tree, self.losses)

	#create an empty loss list if none was passed othewise return existing list
	def gen_valid_loss_list(self, hyp_tree, losses):
		if losses == None:
			
			num_nodes = len(hyp_tree.node_list)
			new_losses = []

			for i in range(num_nodes):
				new_losses.append(None)	
	
			return new_losses		

		else:
			return losses

	#check if node is operational node
	def is_op_node(self, node):
		if node.node_type is 'conv' or node.node_type is 'lin':
			return False
		elif node.node_type is 'merge' or node.node_type is 'flatten':
			return True
		else:
			raise ValueError('Node is not a recognized type')

	#make sure we have the same number of nodes and losses
	def check_equal_nodes_and_losses(self, hyp_tree, losses):
		num_nodes = len(hyp_tree.node_list)
		num_losses = len(losses)

		if num_nodes != num_losses:
			raise ValueError('You have {0} nodes and {1} losses these values must be equal!'.format(num_nodes, num_losses))

		#create our node_list and losses vars	
		self.node_list = [None for i in range(num_nodes)]

	#add the gloabl params to our params list
	def add_global_params(self, node_params):
		node_params['decay'] = self.hyp_tree.decay
		node_params['runs'] = self.hyp_tree.runs
		node_params['lr'] = self.hyp_tree.learning_rate

	#convert a hyper parameter dictionary into a ordered list
	def hyp_dict_to_list(self, hyp_dict, node_type):

		#get number of params in node so we can easily init lists
		num_params = self.meta_info.length_dict[node_type]

		#init our list w/0's	
		hyp_list = np.zeros(num_params)
 
		#get a list that tells us what order we should place our variables
		ordered_var_list = self.meta_info.list_orders[node_type]

		for idx, key in enumerate(ordered_var_list):
			hyp_list[idx] = hyp_dict[key]	

		return hyp_list


	#create a representation of an operational node for opt_net	
	def gen_exec_node(self, node_type, hyp_node, loss): 
		
		#get local hyper params
		hyps = hyp_node.get_param_dict()

		#if we are dealing with a teminal node we will add the global params as additional inputs
		if self.hyp_tree.is_terminal_node(hyp_node):
			self.add_global_params(hyps)
			node_type = 'term' #we wil re-assign the node_type as the optimization network distingushes between terminal and non-teminal conv nodes
	
		#convert into a ordered list
		hyp_list = self.hyp_dict_to_list(hyps, node_type)

		#initialize our optimization node and return it	
		node = Opt_Exec_Node(node_type, hyp_list, self.lstm_size, loss) 
		return node


	#create a representation of an non-operational node for opt_net	
	def gen_op_node(self, hyp_node, loss):
			
		#get number of inputs we will feed to this node 
		node_len = self.meta_info.length_dict[hyp_node.node_type]

		if hyp_node.node_type is 'flatten':
			hyp_dict  = hyp_node.get_input_nodes()[0].get_param_dict() #get a dict of hyper params of input node
			hyp_list =  self.hyp_dict_to_list(hyp_dict, 'conv') #Input node must be conv so we declare it as such
			return Opt_Flatten_Node(hyp_list, self.lstm_size, loss)
			
		
		#merge input nodes will be one of the hidden_state params from it's input nodes
		elif hyp_node.node_type is 'merge':
			return Opt_Merge_Node(self.lstm_size, loss)	

		else:
			raise ValueError('You attempted to initialize optimization node with node of unkown type {0}, options are {1} or {2}'.format(node_type, 'flatten', 'merge'))
 
	#create a node repsentation for optimization net. hyp_node: the node from eval_net, loss: evaluated loss of eval_net at this node if it exists.
	def gen_optimization_node(self, hyp_node, loss):

		#check if the node we have is an operational node
		if self.is_op_node(hyp_node):
			return self.gen_op_node(hyp_node, loss)

		#node is non-opertaional
		else:
			#get out node type
			node_type = hyp_node.node_type
				
			return self.gen_exec_node(node_type, hyp_node, loss)

	#create opt_node list w/empty losses for passed hyp_tree
	def gen_opt_node_list(self, hyp_tree):
		node_list = [None for _ in hyp_tree.node_list]
	
		for idx, hyp_node in enumerate(hyp_tree.node_list):
			node_list[idx] = self.gen_optimization_node(hyp_node, None)

		return node_list

	#propgate a list of node representations that can be evaluated by optimzation network into our empty node list
	def prop_node_list(self, hyp_tree, losses):
		
		#loop through each node in our hyp_tree
		for idx, hyp_node in enumerate(hyp_tree.node_list):
			
			loss = losses[idx]
			self.node_list[idx] = self.gen_optimization_node(hyp_node, loss)  

	#create a hyp_tree using passed node as root. 
	def gen_subtree(self, node, learning_rate, decay, runs, R, step_frac, w_frac, h_frac):

		#make a copy of this node
		c_node = deepcopy(node)

		#we'll clear it's old outputs since we don't need them
		c_node.output_nodes = []

		#we defualt to the current node being our root
		r_node = c_node

		#if our current node is conv we'll need to flatten it before we can eval it's loss so we'll add an additional node
		if c_node.node_type is 'conv':

			#create a flat node
			r_node = Op_Node(op_name = "flatten", num_inputs = 1, input_node = c_node)

			c_node.output_nodes = [r_node]
	
		tree = Hyp_Tree(learning_rate = learning_rate, decay = decay, runs = runs, R = R, 
			step_frac = step_frac, root_node = r_node, w_frac = w_frac, h_frac = h_frac)

		return tree


	#create a new tree for each node in node_list using meta_params from base tree
	def gen_n_trees_from_base(self, base_tree, node_list):
		
		#get some meta data from our tree that we can't extract from each node  
		learning_rate = base_tree.learning_rate
		runs = base_tree.runs
		decay = base_tree.decay
		R = base_tree.R
		step_frac = base_tree.step_frac
		w_frac = base_tree.w_frac
		h_frac = base_tree.h_frac

		#vairable to store our trees
		trees = []

		#for each node if our tree make a copy and use at as the root for a new tree
		for idx, node in enumerate(node_list):

			tree = self.gen_subtree(node, learning_rate, decay, runs, R, step_frac, w_frac, h_frac)

			#append the subtree to our list
			trees.append(tree)

		#return list of subtrees
		return trees

	#raise exception if loss is not a legal value	
	def check_valid_loss(self, loss):
		
		if loss is None:
			raise ValueError('You cannot sort losses without a valid loss list!!!')
		

					
	#get the n nodes w/lowest loss values in the class hyp_tree 
	def get_n_hyp_nodes_w_best_losses(self, n):

		#This array store a list of tuples accosiating a node and its equiv loss
		value_pairs = []

		#loop over each node in hyp_tree get it's accosiated loss and add it to our value pairs
		for idx, node in enumerate(self.hyp_tree.node_list):

			current_loss = self.losses[idx]
			self.check_valid_loss(current_loss) #here we check to make sure we initialized our losses properly

			value_pairs.append((current_loss, node))

		#create the data type defining our tuple pairs so we can sort
		dtype = [('loss', float), ('tree', type(node))]

		#get our tuples sorted by lowest loss value
		tmp = np.array(value_pairs, dtype = dtype)	
		sorted_losses = np.sort(tmp, order = 'loss')

		#get the n first tuples of our sorted list
		best_results = sorted_losses[:n]
		
		#unpack the nodes from our tuple list and return only them
		best_nodes = [node for _, node in best_results]
		best_losses = [loss for loss, _ in best_results]

		return best_nodes, best_losses

	#adjust the fractions dimensions for a subtree
	def adjust_frac_dims(self, base_tree, new_tree):
	
		new_idxs = []
	
		for term_node in new_tree.terminal_nodes:
		
			for idx, old_term in enumerate(base_tree.terminal_nodes):
		
				if base_tree.compare_node_parameters(term_node, old_term):
					new_idxs.append(idx)
					break	
		new_w_frac = []
		new_h_frac = []

		for idx in new_idxs:
			w_frac = base_tree.w_frac[idx]
			h_frac = base_tree.h_frac[idx]

			new_w_frac.append(w_frac)
			new_h_frac.append(h_frac)
		
		new_tree.w_frac = new_w_frac
		new_tree.h_frac = new_h_frac



	#create subtrees for the nodes with the num_trees smallest loss values from class hyp_tree
	def gen_x_best_subtrees(self, num_trees):
		
		#get some meta data from our tree that we can't extract from each node 	
		learning_rate = self.hyp_tree.learning_rate
		runs = self.hyp_tree.runs
		decay = self.hyp_tree.decay
		R = self.hyp_tree.R
		step_frac = self.hyp_tree.step_frac
		w_frac = self.hyp_tree.w_frac
		h_frac = self.hyp_tree.h_frac

		#get num_tree nodes w/lowest loss vals
		best_nodes, best_losses =  self.get_n_hyp_nodes_w_best_losses(num_trees)

		#var to store our hyp trees we are about to generate
		trees = []

		#for each node create a hyp_tree and add it to our list of trees
		for node in best_nodes:
			tree = self.gen_subtree(node, learning_rate, decay, runs, R, step_frac, w_frac, h_frac)
			self.adjust_frac_dims(self.hyp_tree, tree)
			trees.append(tree)

		return trees, best_losses 

	#create a hyp_tree for each sub_tree of main hyp_tree
	def gen_sub_trees(self):

		#get some meta data from our tree that we can't extract from each node 	
		learning_rate = self.hyp_tree.learning_rate
		runs = self.hyp_tree.runs
		decay = self.hyp_tree.decay
		R = self.hyp_tree.R
		step_frac = self.hyp_tree.step_frac
		w_frac = self.hyp_tree.w_frac
		h_frac = self.hyp_tree.h_frac

		#list of subtree nets
		trees = []

		#for each node if our tree make a copy and use at as the root for a new tree
		for idx, node in enumerate(self.hyp_tree.node_list):

			tree = self.gen_subtree(node, learning_rate, decay, runs, R, step_frac, w_frac, h_frac)

			#append the subtree to our list
			trees.append(tree)

		#return list of subtrees
		return trees

	#propegate list of losses to our nodes
	def prop_losses(self, losses):

		for idx, node in enumerate(self.node_list):
			node.loss = losses[idx]
  
#This class constructs and evaluate optimization nets
class Opt_Net_Eval():

	def __init__(self, lstm_size, weights = None):
		self.lstm_size = lstm_size #equiv to number of neurons
		self.weights = weights #weight values of opt_net
		self.num_batchs = 0
		self.cum_batch_error = 0
		self.results = None



	#Check and assign if needed the next input value for upcoming merge node. idx: what idx in node_list we are at. node_list: a list of nodes comprising our optimization net.
	#state: The state accosiated with the current node in our node_list.
	def check_and_assign_merge_input(self, idx, node_list, state):
		
		if idx != len(node_list) - 1:
			
			if node_list[idx + 1].opt_node_type is 'term':
				last_state_idx = len(state) - 1
				self.next_merge_input = state[last_state_idx]

  
	#Construct the optimization net in tensorflows computaional graphs -> somewhat misleading comment? can't really think of better description atm.
	def contruct_opt_net(self, node_list, lstm_cell, batch_size):
		
		#init variables to track net outputs, states 
		outputs = []
		states = []

		self.input_nodes = []

		for idx, opt_node in enumerate(node_list):

			#terminal node states should be zeroed out as they have no inputs
			if opt_node.opt_node_type is 'term':
				new_state = lstm_cell.zero_state(batch_size, tf.float32)

			#create the variable that will be used as input feed 					
			if opt_node.opt_node_type is 'merge':
				node_input, _ = self.next_merge_input
			else:
				node_input = tf.placeholder(tf.float32, [batch_size, opt_node.num_hyps])

			#Here we make a call to our lstm cell and retrieve an output and state tuple value
			#output is the hidden_state and state is a tuple of current state and hidden state
			output, new_state = self.lstm(node_input, new_state)

			#store results from each layer
			outputs.append(output)
			states.append(new_state)
			self.input_nodes.append(node_input)

			#check and assign if needed the next input value for upcoming merge node
			self.check_and_assign_merge_input(idx, node_list, new_state)
	
		return outputs, states

	#This function nodes in the node_list whose losses are defined and returns them and their idxs
	def get_evaled_nodes(self, node_list):

		loss_idxs = []
		cur_labels = []
		
		for idx, node in enumerate(node_list):
			
			if node.loss is not None:
				loss_idxs.append(idx)
				cur_labels.append(node.loss)

		return loss_idxs, cur_labels
				

	#create training variables
	def create_training_var(self, node_lists):
		
		losses = [] #track weighted losses for each node_list
		labels = [] #track labels for each node_list
		results = [] #track actual results
		prediction_list  = []
	
		for node_list in node_lists:
	
			#get outputs whose nodes have been evaluated and accosiated idxs
			loss_idxs, cur_results = self.get_evaled_nodes(node_list)

			#we'll only make predictions on nodes we can check against for trainng
			predictions = [self.logits_series[0][loss_idx] for loss_idx in loss_idxs]
			prediction_list.append(predictions)

			#create placeholder for actual results
			cur_labels = tf.placeholder(tf.float32, shape = (len(loss_idxs)))
		
			#add tensor rep and actual evaled values of losses to accosiated variables		
			labels.append(cur_labels)		
			results.append(cur_results)		

			#and current loss to loss list
			losses.append(tf.losses.huber_loss(labels = cur_labels, predictions = predictions))		

		#save list of exp losses for print out purposes
		self.predictions = prediction_list

		#define lables as class var so we can feed it on run time 
		self.labels = labels

		#save acutal results for feed
		self.results = results

		#let's get the cumlulative loss as this is the value we want to minimize during training
		self.total_loss = tf.reduce_mean(losses)

		#A placeholder to pass learning rate. We need this b/c we want to apply a decay factor as training progresses
		self.lr_placeholder = tf.placeholder(tf.float32, shape = ())

		#Create the optimizer for our lstm
		optimizer = tf.train.AdamOptimizer(learning_rate = self.lr_placeholder) 
		self.train_op = optimizer.minimize(self.total_loss)

	#create a lstm cell with size of class variable lstm_size
	def lstm_cell(self, num_cells):
		return tf.contrib.rnn.BasicLSTMCell(num_cells, state_is_tuple = True, reuse = tf.AUTO_REUSE)


	#ensure values in cell are legal
	def legal_cell(self, cell):
		
		for layer in cell:
			if  layer < 1:
				raise ValueError('Layer must contain at least 1 unit')

	#ugly imp. can definitly be made cleaner with dicts in future. This function adds the proper input and output layers for a lstm module based on the opt_node_type.
	def construct_cell_list(self, opt_node, core_cells):

		#make sure are core cells are valid values
		self.legal_cell(core_cells)

		#should have all core cells plus input and output 
		new_cells = np.zeros(len(core_cells) + 2)
		
		for idx, val in enumerate(core_cells):
			new_cells[idx + 1] = val

		if opt_node.opt_node_type is 'term':
			new_cells[0] = 13 
			new_cells[len(new_cells) - 1] = 10 

		elif opt_node.opt_node_type is 'conv':
			new_cells[0] = 10 
			new_cells[len(new_cells) - 1] = 10 

		elif opt_node.opt_node_type is 'flatten':
			new_cells[0] = 10 
			new_cells[len(new_cells) - 1] = 4 

		elif opt_node.opt_node_type is 'merge':
			new_cells[0] = 4 
			new_cells[len(new_cells) - 1] = 4 

		elif opt_node.opt_node_type is 'lin':
			new_cells[0] = 4
			new_cells[len(new_cells) - 1] = 4 

		else:
			raise ValueError('Passed unrecognized opt_node_type {0}'.format(opt_node.opt_node_type))


		return new_cells
	
	#Build an lstm module. cell_list: a list of integers representing number of cells in each layer of our module
	def build_lstm_module(self, cell_list):
		lstm_layers = []
		for num_cells in cell_list:
			lstm_layers.append(self.lstm_cell(num_cells)) 
				
		return tf.contrib.rnn.MultiRNNCell(lstm_layers)


	#create the various variables need to run our optimization network. node_list: of a list of Opt_nodes and non_Opt_nodes. training: wether we will be training this network
	def gen_opt_net_var(self, node_lists, training, batch_size = 1, num_lstm_layers = 5):
		
		#create layered lstm network
		self.lstm = tf.contrib.rnn.MultiRNNCell([self.lstm_cell(self.lstm_size) for _ in range(num_lstm_layers)])

		#outputs: list of cell outputs, states: list of cell states
		self.outputs, self.states = self.contruct_opt_net(node_lists[0], self.lstm, batch_size = batch_size)

		#this is temporary until we replace it with discrete  array representation of loss using CE.
		W2 = tf.Variable(np.random.rand(self.lstm_size, 1), dtype=tf.float32)
		b2 = tf.Variable(np.zeros((batch_size, 1)), dtype=tf.float32)
			
		#create placeholder for prediction outputs
		tmp_series  = [tf.transpose((tf.matmul(state, W2) + b2))[0] for state in self.outputs] #shape (num_nodes, batch_size)
		self.logits_series = tf.transpose(tmp_series) #shape (batch_size, num_nodes)

		#Create relevant training variables	
		if training:
			self.create_training_var(node_lists)

		#We'll save the weights of our net to be used in future runs
		self.saver = tf.train.Saver(self.lstm.weights)
		
		#create our session
		self.sess = tf.Session()

		#create variable that intializes all tensorflow variables in the computational graph
		self.tf_init = tf.global_variables_initializer()
	
	#print the results of optimization network for each individual node. losses: list of actual eval_net results, exp_losses: list of opt nets predicted results	
	def print_idv_node_results(self, losses, exp_losses):

		for idx, loss in enumerate(losses):
			
			exp_loss = exp_losses[idx]	
			diff = abs(loss - exp_loss)

			print 'In node: {0} loss was = {1}, expected loss was {2}, prediction error was {3}'.format(idx, loss, exp_loss, diff) 		


	#raise exception of nodes inputs were not assigned. node: a Opt_Merge_Node
	def check_valid_merge_assignment(self, node):
		if node.inputs == None:
			raise ValueError('the merge node inputs were not assigned in this optimization net, something wen wrong!!!')


	#Check if None is not an element of the list
	def none_list(self, new_list):

		for val in new_list:		
			if val is None:
				return False

		return True	

	#create input node feed value for each node in batch at passed idx
	def create_in_node(self, node_lists, idx):

		num_node_lists = len(node_lists)
		cur_in_node = [None] * num_node_lists

			
		for j, node_list in enumerate(node_lists):

			cur_node = node_list[idx]

			#we only assign values for non-merge nodes as merge inputs are automaticaly caculated
			if cur_node.opt_node_type is not 'merge':
				cur_in_node[j] = cur_node.sensitivy_adj()

		return cur_in_node
	
	#create the input dictionary we need to feed to evaluate and train our optimization network	
	def create_opt_input_dict(self, node_lists, train = False):
	
		#init the dict we'll used to feed values during run-time
		input_dict = {}

		if train:	

			if self.results is None:
				raise ValueError('You must initialize results before training')

			for idx, result in enumerate(self.results):

				#get label at current idx
				label = self.labels[idx]

				#assign feed value to label
				input_dict[label] = result


		#get number of nodes in each list
		num_nodes = len(node_lists[0])

		#assign the inputs for each node that needs them assigned
		for i in range(num_nodes):

			#get input node name
			in_node_name = self.input_nodes[i]

			#get feed value for current input node
			cur_in_node = self.create_in_node(node_lists, i)

			#if node has legal assignment assign
			if self.none_list(cur_in_node):
				input_dict[in_node_name] = cur_in_node 


		return input_dict
	

	#Get predicted results of a hyp_tree representation at each node 
	def make_prediction(self, node_lists, batch_size):
		
		#create the variable we need to run our optimization net
		self.gen_opt_net_var(node_lists, training = False, batch_size = batch_size)
		
		#init variables in computational graph
		self.sess.run(self.tf_init)

		#get are dictionary feed values
		input_dict = self.create_opt_input_dict(node_lists)		
		
		#re-init weights if cleared
		self.saver.restore(self.sess, "/tmp/model.ckpt")

		#get expected results
		exp_losses = self.sess.run(self.logits_series, feed_dict = input_dict)
		
		#free up session recources 
		self.sess.close()
		tf.reset_default_graph()

		return exp_losses

	#get predicted results for each hyp_tree representaion in node_lists. We will used batch operations here so opt_net sizes extracted from node_list must be the same. 
	def make_N_predictions(self, node_lists):

		#Determine batch size
		batch_size = len(node_lists)

		#get expected losses and return
		exp_losses = self.make_prediction(node_lists, batch_size)
		return exp_losses
	

	#update some training run tracking varaibles
	def update_training_trackers(self, batch_error):
		self.num_batchs += 1
		self.cum_batch_error += batch_error

	#reset training tracking variables
	def reset_training_trackers(self):
		self.num_batchs = 0
		self.cum_batch_error = 0

	#print some useful information about current training run
	def print_train_run_info(self, run):
		avg_error = self.cum_batch_error / self.num_batchs
		print 'The average batch error across all batchs on run: {0} was: {1}'.format(run, avg_error)
		self.reset_training_trackers()

	#only on list of lists 
	def get_avg_list_value(self, values):
	
		#get sum exp loss for batch			
		sum_values = 0
		num_elm = 0 

		for val in values:
			for elm in val:
				sum_values += elm
				num_elm += 1 

		#get avg exp loss for batch
		avg_val = sum_values / num_elm

		return avg_val

	#create and train our optimization net
	def train_run(self, node_lists, lr, decay_factor):
	
		#get size of batch
		batch_size = len(node_lists)
	
		#create the variable we need to run our optimization net
		self.gen_opt_net_var(node_lists, training = True, batch_size = batch_size)

		#get the feed for this opt net
		input_dict = self.create_opt_input_dict(node_lists, train = True)
			
		#init variables in computational graph
		self.sess.run(self.tf_init)

		#compute adj learning rate and it to input dictionary
		adj_lr = lr * decay_factor
		input_dict[self.lr_placeholder] = adj_lr

		#If weights overwrite from file -> can prob get rid of this cond and all accosiated fxns.
		if self.get_weights() != None:
			self.saver.restore(self.sess, "/tmp/model.ckpt")

		#now we'll train our opt net and make not of the prediction error between actual and predicted loss
		exp_losses, error,  _ =  self.sess.run([self.predictions, self.total_loss, self.train_op], feed_dict = input_dict)
			
		#update tracking info for current training run	
		self.update_training_trackers(error)

		avg_exp_loss = self.get_avg_list_value(exp_losses)
	
		print 'avg_expected loss for current batch was = {0}, prediction error in current batch was = {1}'.format(avg_exp_loss, error) 
			
		#Can't use set weights b/c it doesn't work for some reason?
		weights = self.sess.run(self.lstm.weights)
		self.set_weights(weights)

		#Save weights for future use
		self.saver.save(self.sess, "/tmp/model.ckpt")

		#free up session recources 
		self.sess.close()
		tf.reset_default_graph()


	#set weights accosiated w/lstm
	def set_weights(self, weights):
		self.weights = weights

	#get fxn for lstm weights. 
	def get_weights(self):
		return self.weights

	#train the opt_net over each node_list in node_lists. node_lists: a list of opt_net node lists.
	def train_op_net(self, opt_helper, lr, decay_factor):	

		while(opt_helper.has_next_batch()): #tmp comment this line out for testing speed up purposes

			#get next batch
			batch = opt_helper.get_next_batch()
		
			#we'll get start time needed to record time needed to train opt_net     
			start = time.time()
	
			#get batch length for info to be used to help w/optimizing peformance
			len_batch = len(batch)

			#train on batch
			self.train_run(batch, lr, decay_factor)

			#get time of batch run
			diff = time.time() - start
			
			#get average training time per net
			time_per_net = diff / len_batch

			print('training time = {0}, batch size = {1}, train time per net = {2}'.format(diff, len_batch, time_per_net))

	#run over all batchs num_iter times when training. 
	def train_opt_x_times(self, opt_helper, num_iter, lr = .01, decay_rate = .4):

		for i in range(num_iter):

			#compute decay factor based on run for lr adjustment
			decay_factor = pow(decay_rate, i)

			#train net and print training results 
			self.train_op_net(opt_helper, lr, decay_factor)
			self.print_train_run_info(run = i + 1)

#This class has some useful ops for helping genrate new hyp_trees and interacting as a bridge class b/w opt_net runs and hyp_gen.
class Node_Gen_Helper():

	def __init__(self, hyp_tree):
		self.tree = hyp_tree
		self.cur_idx = self.get_root_idx(hyp_tree)
		self.upstream_nodes = []

	#return the idx of trees root node
	def get_root_idx(self, tree):
		root_idx = len(tree.node_list) - 1
		return root_idx

	
	#decrement the cur_idx if valid -> search right to left 
	def decrement_node_idx(self): 
		if self.cur_idx == 0:
			return False
		else:
			self.cur_idx -= 1
			return True

	#jump current idx jump units (jump may be any integer value). Raises IndexError if new idx not in trees node_list range.
	def jump_cur_idx(self, jump):
		new_idx = self.cur_idx + jump
		node_list_size = len(self.tree.get_node_list())
		if new_idx >= node_list_size:
			raise IndexError('You tried assigning illegal idx value = {0}, for node_list of size {1}'.format(new_idx, node_list_size))
	
		elif new_idx < 0:
			raise IndexError('You tried assigning illegal idx value = {0}'.format(new_idx))

		self.cur_idx = new_idx	

	#Get current node idx
	def get_cur_idx(self):
		return self.cur_idx

	#return node at cur_idx of tree
	def get_cur_node(self):
		return self.tree.node_list[self.cur_idx]

	#replace the tree
	def replace_tree(self, new_tree):
		self.tree = new_tree

	#bassivally a re_init op? can i just call init? should I just be creating a new obj. whenever im tempted to use this? (i.e is thier an effecieny benefit?)
	def update_w_new_tree_and_reset(self, tree):
		self.tree = tree
		self.cur_idx = self.get_root_idx(hyp_tree)
		self.upstream_nodes = []

	#warnings this function cannot deal w/multiple outputs
	def return_upstream_nodes(self, cur_node):
		self.upstream_nodes.append(cur_node)

		if self.tree.is_root(cur_node):
			return self.upstream_nodes

		#make sure node has legal number of output values
		num_out_nodes = len(cur_node.output_nodes)
		if num_out_nodes != 1:
			raise ValueError('This Function only accepts nodes with a single output you passed a node with {0} outputs'.format(num_out_nodes))

		out_node = cur_node.output_nodes[0] #since we only accept single outputs we'll grab the first and only out node
		return self.return_upstream_nodes(out_node)	
		


	
