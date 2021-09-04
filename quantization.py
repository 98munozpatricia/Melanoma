#!/usr/bin/env python

"""
   The following module quantize a pretrained model
   using the PyTorch API for Quantization
   in three diferent methods.
"""

def print_size_of_model(model):
	torch.save(model.state_dict(), "temp.p")
	print('Size (MB):', os.path.getsize("temp.p")/1e6)
	os.remove('temp.p')

def DynamicQuantization (model_path):	
	os.environ["CUDA_VISIBLE_DEVICES"]=""
	device = torch.device("cpu")
	model = torch.load(model_path)
	model_fp32 = model.to(device)
	
	device = torch.device("cpu")
	print_size_of_model(model_fp32)
	model_int8 = torch.quantization.quantize_dynamic(model_fp32, {torch.nn.Linear}, dtype=torch.qint8)
	model_int8.to(device)
	print_size_of_model(model_int8)
	model_float16 = torch.quantization.quantize_dynamic(model_fp32, {torch.nn.Linear}, dtype=torch.float16)
	model_float16.to(device)
	print_size_of_model(model_float16)
	
	model_int8.to(device)
	preds = preds.to("cpu")
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
	
	model_float16.to(device)
	preds = preds.to("cpu")
	start = timer()
	with torch.no_grad():
		for i, x_test in enumerate(test_loader):
			x_test = torch.tensor(x_test.cpu().detach().numpy(), dtype=torch.float32)
			x_test = x_test.to(device)
			res = model_float16(x_test)
			z_test = torch.sigmoid(res)
			preds[i * x_test.shape[0]:i * x_test.shape[0] + x_test.shape[0]] += z_test.to("cpu")
	end = timer()
	print(end - start)

def StaticQuantization (model_path):
	os.environ["CUDA_VISIBLE_DEVICES"]=""
	device = torch.device("cpu")
	model = torch.load(model_path)
	model_fp32 = model
	model_fp32 = model.to(device)
	model_fp32.eval()
	
	model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
	model_fp32_prepared = torch.quantization.prepare(model_fp32)
	
	model_fp32_prepared.to(device)
	model_int8 = torch.quantization.convert(model_fp32_prepared)
	print_size_of_model(model_int8)
	device = torch.device("cpu")
	model_int8 = model_int8.to(device)
	preds=preds.to("cpu")
	
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