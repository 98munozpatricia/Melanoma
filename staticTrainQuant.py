def QATraining (model_path):
	model_fp32 = torch.load(model_path)
	model_fp32.train()
	model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
	model_fp32_prepared = torch.quantization.prepare_qat(model_fp32)
	model = model_fp32_prepared
	
	train_loop(model)
	
	model_fp32_prepared.to(device)
	model_int8 = torch.quantization.convert(model_fp32_prepared)
	print_size_of_model(model_int8)
	start = timer()
	with torch.no_grad():
		for i, x_test in enumerate(test_loader):
			x_test = torch.tensor(x_test.cpu().detach().numpy(), dtype=torch.float32)
			x_test = x_test.to(device)
			res = model_int8(x_test)
			z_test = torch.sigmoid(res)
			preds[i * x_test.shape[0]:i * x_test.shape[0] + x_test.shape[0]] += z_test.to("cpu")
	end = timer()
	print(end - start)