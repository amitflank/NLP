import tensorflow as tf
import numpy as np
from params import activation_fxns, FC_restrict, LSTM_restrict, Gen_Restrict, Emb_restrict
from random import randint

#this class will contain various optimization algorithims for our hyper_params
class Opt_Params():

	#param: hyper_param object, R: A net_restriction object
	#this class can be intialized by either passing a hyper param as its base
	#Or you can just pass a net_restriction and use it to randomly generate the base
	#hyper_param obj.
        def __init__(self, param = None):
                self.param = param

		#this where we store our best k hyper_param objects for various algorithims, this list is sorted
                self.top_k_params = [self.param]

		#get object representing possible activation functions
                self.act_fxns = activation_fxns()

                #modification ops for our non-op node types
                self.FC_ops = {'neurons': self.perform_step, 'keep_prob': self.fraction_mod, 'activation': self.mod_act_fxn}
                self.LSTM_ops = {'neurons': self.perform_step, "activation": self.mod_act_fxn, "recurrent_activation": self.mod_act_fxn, "dropout": self.fraction_mod, "recurrent_dropout": self.fraction_mod}
                self.global_ops = {'learning_rate': self.global_frac_mod, 'decay': self.fraction_mod, 'runs': self.perform_step} 
                
                        
        #parameter adjuster for gloabl fraction variables
        def global_frac_mod(self, param, chance):
                    
            if self.binary_roll(chance):
                                
                frac = randint(1, 50)
                is_neg = randint(0, 1)
                step = (frac / 100.0) * param
                                                                    
                if is_neg == 0:
                    step = step * - 1.0
                                                                                                    
                new_val = self.valid_value(param, param + step)
                return new_val

            return param

        #param: parameter whose validity we must check certain parameters cannot be b/w -1 and 1 
	#so we adjust them to our minimum if they are not valid. this function
	#also casts int for convienence. returns adjusted param
        def check_valid_int(self, param):
                if param > 0 and param < 1:
                    return 1
                elif param < 0 and param > -1:
                    return -1
                else:
                    return int(param)

	#take a max step and returns a valid step	
        def get_step(self, max_step):

                step = max_step
		
                #give step 50% chance of being negative
                make_neg = randint(1,2) 
                if make_neg == 1:
                    step = -step

                if step > 0:
                    return randint(1, max_step)
                else:
                    return randint(step, -1)

	#generate a acivation function from list of functions
        def gen_activation_fxn(self):

                #get number of possible activation functions
                num_fxns = len(self.act_fxns.labels)

                #get index of chosen funtion and return
                func_idx = randint(0, num_fxns - 1)
                return self.act_fxns.labels[func_idx]

	#modify activation function with probabilty chance
        def mod_act_fxn(self, chance, act):
                if self.binary_roll(chance):
                    return self.gen_activation_fxn()
                else:
                    return act


        """
        #WARNING: Changes to mod_act_fxns have broken this method
	#generate list of n convoloutional acivation functions
        def gen_n_act_fxns(self, hyper, chance):	

                layers = len(hyper.act_fxns)

                functions = self.zeros(layers)
	    
                for i in range(layers):

                    functions[i] = self.mod_act_fxn(chance, hyper.act_fxns[i])
	
                return functions

        #WARNING: Changes to mod_act_fxns have broken this method
	#generate a list of n linear activation functions
        def gen_n_lin_act_fxns(self, hyper, chance):
                layers = len(hyper.lin_act_fxns)

                functions = self.zeros(layers)

                for i in range(layers):
                    functions[i] = self.mod_act_fxn(chance, hyper.lin_act_fxns[i])

                return functions
        """
	
	#roll with prob chance return True if rolled below chance False if above	
        def binary_roll(self, chance):
                roll = randint(0, 1000)
		
                if roll < chance:
                    return True
                else:
                    return False

	#get an integer step based on max_val. step_frac: the maximum step fraction. max_val: the value to use as step_frac base multiplier.
        def gen_int_step(self, step_frac, max_val):

                #get max valid step
                max_step = self.check_valid_int(step_frac * max_val)

                #return actual step
                return self.get_step(max_step)

	#get and integer step w/prob chance otherwise return original value
        def conditional_int_step_w_addition(self, val, step_frac, max_val, chance):
                if self.binary_roll(chance):
                    return self.gen_int_step(step_frac, max_val) + val
                else:
                    return val 

	#return new_val if positive else return val	
        def valid_value(self, val, new_val):
                if new_val > 0:
                    return  new_val
                else:
                    return val

	#perform a step op
        def perform_step(self, val, step_frac, max_val, chance, mod = 1):
                new_val = self.conditional_int_step_w_addition(val, step_frac, max_val, chance)
                new_val = self.valid_value(val, new_val) / mod 

                return new_val

	#create zeros array of length size with dtype. defualts to int.
        def zeros(self, size, dtype = np.int16):
                return [dtype(0) for i in range(size)]
		
	#create an array of step values. size: number of steps to geneate, step_frac: clustering factor, base: scale of variable	
        def gen_step_arr(self, size, step_frac, base, dtype = int,  mod = 1):
                step_arr = self.zeros(size, dtype)

                for i in range(size):
                    step_arr[i] = self.gen_int_step(step_frac, base) / mod  

                return step_arr	


	#parameter adjuster for gloabl fraction variables
        def global_frac_mod(self, param, chance):
		
            if self.binary_roll(chance):
			
                frac = randint(1, 50)
                is_neg = randint(0, 1)
                step = (frac / 100.0) * param
			
                if is_neg == 0:
                    step = step * - 1.0
				
                new_val = self.valid_value(param, param + step)
                return new_val

            return param

        #get new keep value 
	#not sure how i feel about this fxn right now maybe trash it
        def fraction_mod(self, keep, step_frac, max_val, chance):

                if self.binary_roll(chance):
                    step = self.gen_int_step(step_frac, max_val) / 100.0
                    new_val = self.valid_value(keep, keep + step)
                    return self.valid_drop(new_val)
                else:
                    return keep
        #conditionally modify gloabal paremeters: lr: learning_rate, decay: decay rate. runs: training runs. step_frac: clustering variable. 
        #chance: probabilty variable is modified
        def mod_global_params(self, hyp_tree, step_frac, chance, R = Gen_Restrict()):
            hyp_tree.learning_rate = self.global_frac_mod(hyp_tree.learning_rate, chance)
            hyp_tree.decay = self.global_frac_mod(hyp_tree.decay, chance)
            hyp_tree.decay = self.valid_drop(hyp_tree.decay)
            hyp_tree.runs = self.perform_step(hyp_tree.runs, step_frac, R.max_runs, chance) 

        #same as mod global params but with dictionary input
        def mod_global_params_dict(self, tree, step_frac, chance, R = Gen_Restrict()):
            tree.global_param['learning_rate'] = self.global_frac_mod(tree.global_param['learning_rate'], chance)
            tree.global_param['decay'] = self.global_frac_mod(tree.global_param['decay'], chance)
            tree.global_param['decay'] = self.valid_drop(tree.global_param['decay'])
            tree.global_param['runs'] = self.perform_step(tree.global_param['runs'], step_frac, R.max_runs, chance) 

	#generate a new conv_node clustered about passed conv_node. step_frac and chance parameters dictate degree of clustering (lower = more clustered)
        def mod_LSTM_node(self, step_frac, chance, node, R = LSTM_restrict()):

                for key, val in self.LSTM_ops.items():

                    if node.get_param_dict()[key] in self.act_fxns.labels:
                        node.mod_param(key, self.LSTM_ops[key](chance, node.get_param_dict()[key]))
                    else:
                        node.mod_param(key, self.LSTM_ops[key](node.get_param_dict()[key], step_frac, R.params[key], chance))

        #mod an Embedding node. Note since all values that can be moded in embeding node are integer values we don't need a map to check
        #other function ops.
        def mod_Embedding_node(self, node, step_frac, chance, R = Emb_restrict()):
            for key, val in self.node.param_dict.items():
                node.mod_param(key, self.perform_step(val, step_frac, R.params[key], chance))


	#generate a new lin_node clustered about passed lin_node. step_frac and chance parameters dictate degree of clustering (lower = more clustered)
        def mod_FC_node(self, step_frac, chance, node, R = FC_restrict()):

                for key, val in self.FC_ops.items():

                    if node.get_param_dict()[key] in self.act_fxns.labels:
                        node.mod_param(key, self.FC_ops[key](chance, node.get_param_dict()[key]))
                    else:
                        node.mod_param(key, self.FC_ops[key](node.get_param_dict()[key], step_frac, R.params[key], chance))

	#make sure our drop rate doesn't return stupid or illegal values
        def valid_drop(self, keep_prob):
            if keep_prob < 0.0:
                return 0.0
            elif keep_prob > .99:
                return .99
            else:
                return keep_prob


        #check if deleting this node is a legal operation in passed tree
        def legal_del(self, node, tree):
            if tree.num_nodes() <= 2:
                return False

            if node.is_terminal_node() and tree.num_inputs() == tree.get_min_inputs():

                for o_node in node.get_output_nodes():

                    if o_node.num_inputs() > 1:
                        return False

            return True

        #pick random node in passed tree
        def pick_random_node(self, tree):
			
            #get number of nodes in tree
            num_nodes = tree.num_nodes()

            #subtract 1 b/c endpoints` are inclusive in randint    
            node_idx = randint(0, num_nodes - 1)

            #get node
            node = tree.get_node_list()[node_idx]

            return node, node_idx

	#pick a random non-op node from passed tree
        def randomly_pick_noop_node(self, tree):

                while True:
                        
                    #get node
                    node, node_idx = self.pick_random_node(tree)

                    #exit loop if node is a non-op node otherwise select new node
                    if not node.is_op_node():
                        break

                return node, node_idx


#Class that generates some random value chance values for our diffrent adjustment options 
class Prob_Adjuster():
	
	#note all probabilties are out of a thousand so a 5 roll would be equiavelent to a .5% probabilty.
	#Also at some point i would love to have some meta neural net dynamically select these values based on how good our results tend to be with diffrent probs.  
	def __init__(self):
		self.global_param_prob = randint(100, 250)
		self.element_change_prob = randint(1000, 5000) #probabilty an element in a node is changed 
		self.del_node_prob = randint(10, 80) #probabilty a node gets deleted
		self.add_node_prob = randint(10, 80)  #probability a node is added a given location
		self.add_branch_prob = randint(10, 80) #probailty we add a branch to a tree
		self.swap_node_prob = randint(10, 80) #probabilty we swap some pair from each type of node

	#set global param prob 
	def set_element_change_prob(self, prob):
		self.element_change_prob = prob		

	#should I have funtions to random init each one of these variable b/c right we would have to gen a new prob_adjuster each time we wanted diffrent probs.
	#though i kinda of feel like I just want one Prob_Adjusted per set. So not particulary costly  to just gen each set and reduces need for many additional 
	#ops during generation i guess.



