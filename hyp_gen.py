import tensorflow as tf
import numpy as np
import random
from random import randint
from param_helper import Opt_Params
from hyp_tree import Hyp_Tree
from hyp_tree import Op_Node
from copy import deepcopy

#class for generating new hyperparameter trees
class Hyp_Gen():

	#hyp_tree: the tree around which we cluster new trees, chance: probailty of modifying any given parameter. step_frac: clustering parameter(lower = more clustered)
	def __init__(self, hyp_trees, prob_adj, step_frac, R):

		#ensure hyp_trees are valid and assign them as a property to this instance
		self.valid_hyp_trees(hyp_trees)
		self.p_helper = Opt_Params(R)
		self.prob_adj = prob_adj 
		self.step_frac = step_frac
	
	#replace existing prob adj with a new one
	def sub_prob_adj(self, new_prob_adj):
		self.prob_adj = new_prob_adj	

	#check if passed varaible is an instance
	def is_instance(self, obj, inst_type):
		if not isinstance(obj, inst_type):	
			raise ValueError('You tried passing an non-instance where an instance was required')

	#make sure passed hyp_trees are valid Hyp_Tree instances
	def valid_hyp_trees(self, trees):
		#make sure trees is an instance
		self.is_instance(trees, list)

		#make sure each element in treees in a Hyp_Tree instance
		for tree in trees:
			if tree.__class__.__name__ is not 'Hyp_Tree':
				raise ValueError('You tried passing a {0} object, you must pass Hyp_Tree objects'.format(tree.__class__.__name__))

		#now that we know trees is a valid Hyp_Tree array we will assign it
		self.root_trees = trees 



	#mod a node if it is non-op otherwise do nothing	
	def mod_node(self, node):
		if self.p_helper.is_Op_Node(node):
			placeholder = -1 #we do nothing here
		else:
			if node.node_type == 'lin':
				self.p_helper.mod_lin_node(self.step_frac, self.prob_adj.element_change_prob, node)
			elif node.node_type == 'conv':
				self.p_helper.mod_conv_node(self.step_frac, self.prob_adj.element_change_prob, node)
			else:
				raise ValueError('You tried modifying a unrecognized node {0}, valid non-operational nodes are lin and conv'.format(node.node_type))
		
	#modify all the nodes in passed tree	
	def prob_mod_nodes(self, tree):
		for node in tree.get_node_list():	
			self.mod_node(node)		

			
	#create a slightly modified copy of passed node. (sort of a evolutionary copy)
	def get_moded_node_copy(self, node):
		#create a copied node
		c_node = deepcopy(node)

		#clear input output             
		c_node.input_node = None
		c_node.output_nodes = []
		c_node.tensor_input = [None]

		#modify node and return
		self.mod_node(c_node)
		return c_node

	#mod dimension inputs for passed hyperparameter tree
	def mod_input_dimensions(self, hyp_tree):
		hyp_tree.w_frac = self.p_helper.mod_fraction_dims(hyp_tree.w_frac, self.step_frac, self.prob_adj.element_change_prob)
		hyp_tree.h_frac = self.p_helper.mod_fraction_dims(hyp_tree.h_frac, self.step_frac, self.prob_adj.element_change_prob)

	
	#generate num_trees new hyp_trees each of which are a copy of passed tree with the exception of having the node at node_idx randomly modified.
	def gen_x_trees_w_single_node_mod(self, node_idx, tree, num_trees):
		new_trees = [None for i in range(num_trees)]

		for i in range(num_trees):

			modded_tree = deepcopy(tree)
			self.prob_mod_nodes(tree) #this looks like a bug

			#node_to_mod = modded_tree.node_list[node_idx]
			#self.mod_node(node_to_mod)

			new_trees[i] = modded_tree

		return new_trees

	#bad name. modify probablisticly mod single node, gloabal params, add node or delete node. 
	def mod_tree_and_gen(self, node_helper, tree, num_trees):

		init_num_nodes = len(tree.get_node_list())	

		#get clean version of tree so we don't effect passed tree		
		modded_tree = deepcopy(tree)
		
		#first modify non-node based parameters w/some random prob
		self.p_helper.mod_global_params(modded_tree, self.step_frac, self.prob_adj.global_param_prob)
		self.mod_input_dimensions(modded_tree)
	
		#b/c of how the net is structures if we delete below 2 the net doesn't exits(min 1 conv, flat node means we delete only non-op node)
		if init_num_nodes > 2:
  	
			#roll and possible delete a random node from our new tree
			self.prob_del_node(chance = self.prob_adj.del_node_prob, tree = modded_tree)
		
		#roll and possible add a new node to our new tree
		self.prob_add_node(chance = self.prob_adj.add_node_prob, tree = modded_tree)
		
		#roll and possible add new branch to our tree
		#self.prob_add_new_branch(chance = self.prob_adj.add_branch_prob, tree = modded_tree)

		#replace our tree w/new one
		node_helper.replace_tree(modded_tree)

		#get the diffrence in nodes post possible additions/deletions
		num_nodes_diff =  len(modded_tree.get_node_list()) - init_num_nodes

		#if jumping idx to account for new nodes doesn't put us below 0 idx jump the cur_idx in node helper
		if not node_helper.get_cur_idx() + num_nodes_diff < 0:
			node_helper.jump_cur_idx(num_nodes_diff)

		#Use the modded tree we generated to crete num_trees new trees.	
		return self.gen_x_trees_w_single_node_mod(node_helper.get_cur_idx(), modded_tree, num_trees)	

		
	#geneate new hyp_trees clustered around root_tree	
	def gen_tree(self, root_tree):

		#create copy of our root tree which we will use as seed
		hyp_tree = deepcopy(root_tree)
		
		#make sure our loss is None as we have not evaluated this net
		hyp_tree.assign_loss(None)

		#first modify non-node based parameters
		self.p_helper.mod_global_params(hyp_tree, self.step_frac, self.prob_adj.global_param_prob)
		self.mod_input_dimensions(hyp_tree)

		#get a list of nodes in the tree
		node_list = hyp_tree.get_node_list()

		#conditionally modify each element in each node in our new tree
		self.prob_mod_nodes(hyp_tree)

		#roll and possible delete a random node from our new tree
		self.prob_del_node(chance = self.prob_adj.del_node_prob, tree = hyp_tree)
		
		#roll and possible add a new node to our new tree
		self.prob_add_node(chance = self.prob_adj.add_node_prob, tree = hyp_tree)

		#roll and possible add new branch to our tree
		self.prob_add_new_branch(chance = self.prob_adj.add_branch_prob, tree = hyp_tree)

		#roll and possible pair swap for each node type in tree
		self.prob_swap_all_node_types(chance = self.prob_adj.swap_node_prob, tree = hyp_tree)
	
		#return our new modified hyp_tree	
		return hyp_tree  
	
	#check if this conv node is legal in current format
	def legal_conv(self, node):
		if node.input_node.node_type == 'conv':
			return True
		else:
			raise ValueError('Conv node should not be able to recieve {0} node something went wrong'.format(node.input_node.node_type))
	
	#check if this lin node is legal in current format
	def legal_lin(self, node):	
		if node.input_node.node_type == 'conv':
			raise ValueError('Lin should not be able to recieve a conv node something went wrong')
		else:
			return True

	#check if this flatten node is legal in current format
	def legal_flatten(self, node):
		if node.input_node.node_type == 'conv':
			return True
		else:
			raise ValueError('flatten node cannot accept node type {0} as input node'.format(node.node_type))
 
	#check if node with single input is legal in current format
	def legal_single_input(self, node):
		if node.input_node is None:
			if node.node_type == 'conv':
				return True
			else:
				return False

		elif node.node_type is 'conv':
			return self.legal_conv(node)

		elif node.node_type is 'lin':
			return self.legal_lin(node)

		elif node.node_type is 'flatten':
			return self.legal_flatten(node)

		else:
			raise ValueError('Recived unrecognized node type: {0}, valid node types are lin and conv'.format(node.node_type))

	
	#delete passed merge node. does not check to make sure node is valid.	
	def del_merge_node(self, node, tree):
		for o_node in node.output_nodes:
			
			for i_node in node.input_node:
				if i_node is not None:
					tree.substitute_input_node(o_node, node, i_node)

	#check if all inputs are none
	def has_inputs(self, node):
		for input_node in node.input_node:
			if input_node == None:
				return False

		return True

	#check if this merge node is legal in current format
	def legal_merge(self, node, tree):
		if self.has_inputs(node):
			for i_node in node.input_node:
				if i_node.node_type == 'conv':
					raise ValueError('Merge node cannot accept a conv input')
			return True
		else:
			return False 

	#check if node with multiple inputs is legal in current format
	def legal_mult_input(self, node, tree):
		if node.node_type == 'merge':
			return self.legal_merge(node, tree)
		else:
			raise ValueError('Recived illegal node type: {0}, valid node type(s) are merge'.format(node.node_type))
			
	#check if passed node is legal	
	def is_legal_node(self, node, tree):
		if node.num_inputs == 1:
			return self.legal_single_input(node)
		else:
			return self.legal_mult_input(node, tree)

	#check if passed node is a recognized node type	
	def valid_node(self, node):
		if node.node_type == 'lin' or node.node_type == 'conv' or node.node_type == 'merge' or node.node_type == 'flatten':
			return True
		else:
			return False
	
	#delete passed node and any nodes made illegal by set nodes deletion. (i.e: nodes whose inputs will now be invalid)
	def delete_node(self, node, tree):
		if self.valid_node(node):

			self.num_del_calls += 1

			#since final nodes won't trigger loop we deal with them here
			if self.is_final_node(node):

				#really need to break this up into multiple function super messy, i'm just really at my end today so...
				if node.node_type == 'merge':
					for elm in node.input_node:
						if elm is not None:
							in_node = elm

					tree.sub_output_node(in_node, node, [])
					tree.root_node = in_node
				else:
					tree.sub_output_node(node.input_node, node, [])
					tree.root_node = node.input_node

			for o_node in node.output_nodes:
				tree.substitute_input_node(o_node, node, node.input_node)
				
				if not self.is_legal_node(o_node, tree):
					self.delete_node(o_node, tree)	
		else:
			raise ValueError('Recived unrecognized node type: {0}, valid node types are lin and conv'.format(node.node_type))


	#delete a node and update tree information post deletion
	def delete_node_and_update(self, node, tree):
		#check if node being deleted is terminal node for ops below
		is_term = tree.is_terminal_node(node)

		#delete the node
		self.delete_node(node, tree)

		#probably put this in it's own function at some point, bassically if we deleted a terminal node by itself then substitue in a new terminal node,
		#if we deleted the entire branch then del accosiated terminal node and dimensions, fxn name update_post_node_deletions maybe?
		if is_term:
			if self.num_del_calls == 1:
				tree.sub_terminal_node(node.output_nodes[0], node)  #this seems kinda sketch i'll be honest
			else:		
				tree.del_terminal_node(node)
					
		#update our node list	
		tree.get_updated_node_list()


	#delete a random node w/probabilty chance	
	def prob_del_node(self, chance, tree):
		if self.p_helper.binary_roll(chance): 

			#get a node in tree 
			node, node_idx = self.p_helper.randomly_pick_noop_node(tree)

			self.num_del_calls = 0

			self.delete_node_and_update(node, tree)

			#return node for testing reasons
			return node


	#assign node input(s) to node
	def assign_node_inputs(self, node, input_nodes):
		#assign nodes input nodes -> this is a really stupid system atm need to simplify it.
		if len(input_nodes) == 1:
			node.input_node = input_node[0]
		else:
			node.input_node = input_nodes

	#assign output to terminal node
	def add_terminal_output(self, node, output_nodes):
		if output_nodes == None:
			raise ValueError('You cannnot have a terminal node with no outputs')

		#assign passed outputs
		node.output_nodes = output_nodes
	
		#assign node as output_nodes new input	
		for o_node in output_nodes:
			o_node.input_node = node	
	
				
	#assign node output(s) to node
	def assign_noop_node_outputs(self, node, input_node, tree):		
				
		#get output node(s)
		out_nodes = input_node.output_nodes
	
			
		if self.is_final_node(input_node):
			tree.root_node = node

		#assign node as new_input to input_nodes outputs
		for o_node in out_nodes:
			try:
				tree.substitute_input_node(o_node, input_node,  node, sub_out = False)
			except AttributeError:
				raise AttributeError('while trying to substitute inputs was passed a list {0} instead of a node, all outputs {1}'.format(o_node, out_nodes))
		

		#node gets outputs of input node and input_nodes new output is node
		node.output_nodes = out_nodes
		input_node.output_nodes = [node] #we won't try and selectivly insert node on some outputs as we can deal with this on link modifier
	


	#output nodes should be passed if node will be a terminal node. node: node to be added. input_nodes: A list of nodes. output_idxs: a list of integers. output_nodes: list of nodes if passed.
	def add_noop_node(self, node, input_node, tree, output_nodes = None):
		#change this conditional to checking op?
		if self.valid_node(node):
	
			#assign input to our new node	
			node.input_node = input_node
			
			#this conditional checks if new node will be a terminal node	
			if input_node is None:
				self.add_terminal_output(node, output_nodes)
			else:	
				self.assign_noop_node_outputs(node, input_node, tree)

	#add random node to tree w/probabilty chance, this function will never add a terminal node	
	def prob_add_node(self, chance, tree):
		if self.p_helper.binary_roll(chance):
		
			type_roll = np.random.randint(2, size = 1)
			elm = type_roll[0]
			if elm == 0:
				get_type = 'conv'
			else:
				get_type = 'lin'
				if len(tree.nodes_by_type[get_type]) == 0:
					get_type = 'conv'
	

			print 'type picked {0}'.format(get_type)
			#pick nodes until we get node of desired type
			while True:
 
				#get a node in tree and its idx
                		node, node_idx = self.p_helper.randomly_pick_noop_node(tree)
	
				if node.node_type is get_type:
					break

			print node

			#get new number of nodes in tree
			new_node = self.get_moded_node_copy(node)

			#add the nodes  
 			self.add_noop_node(new_node, node, tree)
		
			#update tree w/changes
			tree.get_updated_node_list()

			#return some values for testing purposes -> mayber we can add a testing arguement that deterimnes return at some point
			return new_node, node_idx
		

	#Check if passed nodes are of the same type.
	def same_node_type(self, node1, node2):
		if node1.node_type is node2.node_type:
			return True
		else:
			return False

	#check if node1 and node2 share out_node as one of their output nodes	
	def shared_output(self, out_node, node1, node2):
		if out_node.node_type == 'merge':
			return (out_node.input_node[0] == node1 and out_node.input_node[1] == node2) or (out_node.input_node[0] == node2 and out_node.input_node[1] == node1)
				
		return False	
		
	#for nodes with 2 inputs switch the locations of those inputs, note this fxn does not check if node has 2 inputs
	def switch_node_inputs(self, node):
		tmp_node = node.input_node[0]
		node.input_node[0] = node.input_node[1]
		node.input_node[1] = tmp_node	
				

	#swap outputs of node1 for those in node2
	def assign_node_output_swap(self, node1, node2, output_nodes, tree, swap_if_shared = False):
		for idx, out_node in enumerate(output_nodes):
			if self.shared_output(out_node, node1, node2):
				if swap_if_shared:
					self.switch_node_inputs(out_node)
				continue

			if out_node is node1:
				node1.output_nodes[idx] = node2
			else:
				node1.output_nodes[idx] = out_node	
				tree.substitute_input_node(out_node, node2, node1, sub_out = False)
				
	
	#swap the outputs of node1 and node2
	def assign_output_swap(self, node1, node2, tree):

		#temporary storage for node1 outputs
		node1_out_nodes  = [elm for elm in node1.output_nodes] 

		self.assign_node_output_swap(node1, node2, node2.output_nodes, tree)
		self.assign_node_output_swap(node2, node1, node1_out_nodes, tree, swap_if_shared = True)

	#assign node input, raise exception if illegal idx passed
	def assign_node_input(self, node, new_input, idx):
		if node.num_inputs == 1:
			if idx > 0:
				raise ValueError('Passed invalid input idx {0} to node w/1 idx'.format(idx))
			node.input_node = new_input
		else:
			if idx > 1:
				raise ValueError('Passed invalid input idx {0} to node w/2 idxs'.format(idx))
			node.input_node[idx] = new_input		

	#swap node1 inputs with those of node2
	def assign_node_input_swap(self, node1, node2, input_nodes, tree):
		
		for idx, in_node in enumerate(input_nodes):
			if in_node is node1:
				self.assign_node_input(node1, node2, idx)
			else:
				self.assign_node_input(node1, in_node, idx)
				tree.sub_output_node(in_node, node2, node1)

	#swap inputs of passed two nodes	
	def assign_input_swap(self, node1, node2, tree):
		#temporary storage for node1 inputs
		node1_inputs = node1.get_input_nodes()

		self.assign_node_input_swap(node1, node2, node2.get_input_nodes(), tree)
		self.assign_node_input_swap(node2, node1, node1_inputs, tree)
		

	#called in special case when node swap involves a final node
	def swap_final_node(self, f_node, non_f_node, tree):

			#create tmp varaible to store non_f_output nodes so we can perform output swap
			tmp_out = [node for node in non_f_node.output_nodes]

			#assign non_f_node f_nodes outputs, since we are garunteed it's below f_node on tree
			non_f_node.output_nodes = []

			#now assign f_node non_f_node outputs	
			for idx, out_node in enumerate(tmp_out):
				if out_node is f_node:
					f_node.add_output_node(non_f_node)
				else:
					f_node.add_output_node(out_node)
					tree.substitute_input_node(out_node, non_f_node, f_node, sub_out = False)
					
 
			#since there is nothing unique on input swap for final node we can just use a regulat input swap fxn call
			self.assign_input_swap(f_node, non_f_node, tree)
	
			#since final node acts as root of tree we need to update our root node	
			tree.root_node = non_f_node
	
	#swap nodes in special case where 1 of the nodes is a terminal node
	def swap_single_terminal_node(self, term_node, non_term_node, tree):
		self.assign_output_swap(term_node, non_term_node, tree)

		tmp_in = non_term_node.input_node
		non_term_node.input_node = None

		if tmp_in is term_node:
			term_node.input_node = non_term_node
		else:
			term_node.input_node = tmp_in
			tree.sub_output_node(tmp_in, non_term_node, term_node)


	#swap nodes in special case w/2 terminal nodes
	def swap_multiple_terminal_nodes(self, node1, node2, tree):
		self.assign_output_swap(node1, node2, tree)	
	
		#kinda confused about what i did here probably should test this 	
		tree.sub_terminal_node(node1, node2)
		tree.sub_terminal_node(node2, node1)

	
				
	#make sure passed node is a valid final node 
	def is_final_node(self, node):
		if not node.output_nodes:
			return True
		else:
			return False

	#swap node1 and node2 for non-special cases
	def normal_swap(self, node1, node2, tree):
		self.assign_output_swap(node1, node2, tree)
		self.assign_input_swap(node1, node2, tree)
					
	#swap locations of passed nodes
	def swap_nodes(self, node1, node2, tree):
		if self.same_node_type(node1, node2):
			if self.is_final_node(node1):
				self.swap_final_node(node1, node2, tree)
			elif self.is_final_node(node2):
				self.swap_final_node(node2, node1, tree)
			elif tree.is_terminal_node(node1) and tree.is_terminal_node(node2):
				self.swap_multiple_terminal_nodes(node1, node2, tree)
			elif tree.is_terminal_node(node1):
				self.swap_single_terminal_node(node1, node2, tree)
			elif tree.is_terminal_node(node2):
				self.swap_single_terminal_node(node2, node1, tree)
			else:
				self.normal_swap(node1, node2, tree)
		else:
			raise ValueError('You cannot swap nodes if they are not of the same type')
	

	#return a list of num_samples, samples w/out replacement b/w min_val(inclusive) and max_val(exclusive)
	def sample_w_out_replacement(self, num_samples, min_val, max_val): #this should probably ge thrown in param_helper maybe?
		return random.sample(range(min_val, max_val), num_samples)

	#roll to swap a pair of nodes from each node type
	def prob_swap_all_node_types(self, tree, chance):
		for node_type in tree.nodes_by_type:

			#we roll here
			if self.p_helper.binary_roll(chance):

				#get list of nodes of node_type
				node_list = tree.nodes_by_type[node_type]
                        	nodes_in_list = len(node_list) #get size of list

				#get two unique nodes in list
				node_idxs = self.sample_w_out_replacement(2, 0, nodes_in_list)
				node1 = node_list[node_idxs[0]]
				node2 = node_list[node_idxs[1]]
				self.swap_nodes(node1, node2, tree) #swap nodes
		
				#update tree w/changes
				tree.get_updated_node_list()
			

	#add a new chain to tree
	def add_new_branch(self, tree):
		#get a random conv_node from our tree 
		num_conv_nodes = len(tree.nodes_by_type['conv'])
		rand_conv_idx = randint(0, num_conv_nodes - 1)
		conv_node = tree.nodes_by_type['conv'][rand_conv_idx]

		#Create a slightly moded version of this conv_node, we will use this moded node as the teminal input for our new branch
		new_term_node = self.get_moded_node_copy(conv_node)

		#create a flatten node, assign our terminal conv node as it's input and add it to the term_nodes outputs
		flat_node = Op_Node(op_name = 'flatten', num_inputs = 1, input_node = new_term_node)
		new_term_node.add_output_node(flat_node)

		#we'll create a new merge node to allow us to link into the tree
		new_merge_node = Op_Node(op_name = 'merge', num_inputs = 2, input_node = [flat_node, None])
		flat_node.add_output_node(new_merge_node)

		#get a random merge node from our tree
		num_merge_nodes = len(tree.nodes_by_type['merge'])
		rand_merge_idx =  randint(0, num_merge_nodes - 1)
		merge_node = tree.nodes_by_type['merge'][rand_merge_idx]

		#randomly pick one of the inputs of this merge node to be the other half of our new link
		rand_merge_input_idx =  randint(0, 1)
		merge_input = merge_node.input_node[rand_merge_input_idx]

		#get the output idx of the merge node in our selected input
		for idx, o_node in enumerate(merge_input.output_nodes):
			if o_node is merge_node:
				old_merge_output_idx = idx

		#now we will add the selected merge nodes input as the second input for our new merge node and set the new merge node as the replacement input
		new_merge_node.input_node[1] = merge_input	
		merge_input.output_nodes[old_merge_output_idx] = new_merge_node
		merge_node.input_node[rand_merge_input_idx] = new_merge_node	

		#we'll use the same input dimensions of our copied branch for our new branch
		new_w_frac = tree.w_frac[rand_merge_idx]
		new_h_frac = tree.h_frac[rand_merge_idx]

		#now insert the new dimensions into list, we won't bother determining if we placed our new merge node before or after our copied values b/c they are the same
		tree.w_frac.insert(rand_merge_idx, new_w_frac)
		tree.h_frac.insert(rand_merge_idx, new_h_frac)
				
		#update tree w/changes
		tree.get_updated_node_list()

	#add a new branch to passed tree w/probabilty chance
	def prob_add_new_branch(self, chance, tree):
		if self.p_helper.binary_roll(chance):
			self.add_new_branch(tree)			

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

			if idx > k + 1:
				return False
	
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

