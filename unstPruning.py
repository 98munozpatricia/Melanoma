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
		
	validation(model)
	print("Number of parameters after pruning: ", count_parameters(model))