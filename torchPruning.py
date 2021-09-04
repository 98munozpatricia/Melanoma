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