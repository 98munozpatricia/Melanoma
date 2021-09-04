#!/usr/bin/env python

"""
   The following module prunes a pretrained model
   using the PyTorch API for Pruning and 
   Torch-Pruning API from VainF.
"""
import torch.nn.utils.prune as prune
import torch_pruning as tp

def print_size_of_model(model):
	torch.save(model.state_dict(), "temp.p")
	print('Size (MB):', os.path.getsize("temp.p")/1e6)
	os.remove('temp.p')

def count_parameters(model):
	params = 0
	for param in model.parameters():
		params += torch.count_nonzero(param)
	return params
		
def unstructuredPruning (model_path):
	model = torch.load(model_path)
	print("Number of parameters before pruning: ", count_parameters(model))
	parameters_to_prune = (
		(model.arch._blocks[0]._depthwise_conv, 'weight'),
		(model.arch._blocks[1]._depthwise_conv, 'weight'),
		(model.arch._blocks[2]._depthwise_conv, 'weight'),
		(model.arch._blocks[3]._depthwise_conv, 'weight'),
		(model.arch._blocks[4]._depthwise_conv, 'weight'),
		(model.arch._blocks[5]._depthwise_conv, 'weight'),
		(model.arch._blocks[6]._depthwise_conv, 'weight'),
		(model.arch._blocks[7]._depthwise_conv, 'weight'),
		(model.arch._blocks[8]._depthwise_conv, 'weight'),
		(model.arch._blocks[9]._depthwise_conv, 'weight'),
		(model.arch._blocks[10]._depthwise_conv, 'weight'),
		(model.arch._blocks[11]._depthwise_conv, 'weight'),
		(model.arch._blocks[12]._depthwise_conv, 'weight'),
		(model.arch._blocks[13]._depthwise_conv, 'weight'),
		(model.arch._blocks[14]._depthwise_conv, 'weight'),
		(model.arch._blocks[15]._depthwise_conv, 'weight'),
		
		(model.arch._blocks[0]._project_conv, 'weight'),
		(model.arch._blocks[1]._project_conv, 'weight'),
		(model.arch._blocks[2]._project_conv, 'weight'),
		(model.arch._blocks[3]._project_conv, 'weight'),
		(model.arch._blocks[4]._project_conv, 'weight'),
		(model.arch._blocks[5]._project_conv, 'weight'),
		(model.arch._blocks[6]._project_conv, 'weight'),
		(model.arch._blocks[7]._project_conv, 'weight'),
		(model.arch._blocks[8]._project_conv, 'weight'),
		(model.arch._blocks[9]._project_conv, 'weight'),
		(model.arch._blocks[10]._project_conv, 'weight'),
		(model.arch._blocks[11]._project_conv, 'weight'),
		(model.arch._blocks[12]._project_conv, 'weight'),
		(model.arch._blocks[13]._project_conv, 'weight'),
		(model.arch._blocks[14]._project_conv, 'weight'),
		(model.arch._blocks[15]._project_conv, 'weight'),
		
		(model.arch._blocks[2]._expand_conv, 'weight'),
		(model.arch._blocks[3]._expand_conv, 'weight'),
		(model.arch._blocks[4]._expand_conv, 'weight'),
		(model.arch._blocks[5]._expand_conv, 'weight'),
		(model.arch._blocks[6]._expand_conv, 'weight'),
		(model.arch._blocks[7]._expand_conv, 'weight'),
		(model.arch._blocks[8]._expand_conv, 'weight'),
		(model.arch._blocks[9]._expand_conv, 'weight'),
		(model.arch._blocks[10]._expand_conv, 'weight'),
		(model.arch._blocks[11]._expand_conv, 'weight'),
		(model.arch._blocks[12]._expand_conv, 'weight'),
		(model.arch._blocks[13]._expand_conv, 'weight'),
		(model.arch._blocks[14]._expand_conv, 'weight'),
		(model.arch._blocks[15]._expand_conv, 'weight'),
		
		(model.arch._conv_head, 'weight'),
		(model.arch._conv_stem, 'weight'),
	)
	for module, param in parameters_to_prune:
		prune.ln_structured(module, name = param, amount = 0.01, n = 2, dim = 0)
		prune.global_structured(
		   	 parameters_to_prune,
		   	 pruning_method=prune.L1Unstructured,
		   	 amount=0.1,
		)
	print(model.parameters())
	
	for module, param in parameters_to_prune:
		prune.remove(module,'weight')
	
	validation(model)
	print("Number of parameters after pruning: ", count_parameters(model))


def unstructuredPruning (model_path):
	model = torch.load(model_path)
	print("Number of parameters before pruning: ", count_parameters(model))
	
	parameters_to_prune = (
		(model.arch._blocks[0]._depthwise_conv, 'weight'),
		(model.arch._blocks[1]._depthwise_conv, 'weight'),
		(model.arch._blocks[2]._depthwise_conv, 'weight'),
					[...]
		(model.arch._blocks[14]._expand_conv, 'weight'),
		(model.arch._blocks[15]._expand_conv, 'weight'),
		
		(model.arch._conv_head, 'weight'),
		(model.arch._conv_stem, 'weight'),
	)
	
	prune.global_structured(
	   	 parameters_to_prune,
	   	 pruning_method=prune.L1Unstructured,
	   	 amount=0.1,
	)
	
	for module, param in parameters_to_prune:
		prune.remove(module,'weight')
		
	print("Number of parameters after pruning: ", count_parameters(model))


def torchPruningExec (model_path):
	model = torch.load(model_path)
	parameters_to_prune = (
		(model.arch._blocks[0]._depthwise_conv, 'weight'),
		(model.arch._blocks[1]._depthwise_conv, 'weight'),
		(model.arch._blocks[2]._depthwise_conv, 'weight'),
		[...]
		(model.arch._blocks[14]._expand_conv, 'weight'),
		(model.arch._blocks[15]._expand_conv, 'weight'),
		
		(model.arch._conv_head, 'weight'),
		(model.arch._conv_stem, 'weight'),
	)
	strategy = tp.strategy.L1Strategy()
	DG = tp.DependencyGraph()
	DG.build_dependency(model, example_inputs=torch.randn(1,3,255,255))
	for module, param in parameters_to_prune:
		pruning_idxs = strategy(module.weight, amount = 0.05)
		pruning_plan = DG.get_pruning_plan(module, tp.prune_conv, idxs=pruning_idxs)
		print( (pruning_plan))
		pruning_plan.exec()
	
	print_size_of_model(model)