from read_trainset import *
from output_nn import *
import torch
import os


def init_optimizer(tin, model):
	if tin.method == "adam":
		model.optimizer = torch.optim.Adam(model.parameters(), lr = tin.lr, weight_decay=tin.regularization)

	if tin.method == "adadelta":
		model.optimizer = torch.optim.Adadelta(model.parameters(), lr = tin.lr, weight_decay=tin.regularization)

	if tin.method == "adagrad":
		model.optimizer = torch.optim.Adagrad(model.parameters(), lr = tin.lr, weight_decay=tin.regularization)

	if tin.method == "adamw":
		model.optimizer = torch.optim.AdamW(model.parameters(), lr = tin.lr, weight_decay=tin.regularization)

	if tin.method == "adamax":
		model.optimizer = torch.optim.Adamax(model.parameters(), lr = tin.lr, weight_decay=tin.regularization)



def init_train(tin, model):
	# If RESTART, read previous model and optimizer state_dict
	if os.path.exists("./model.restart"):
		#print("Restarting...")
		model_restart = torch.load("./model.restart")
		model.load_state_dict( model_restart["model"] )
		model.optimizer.load_state_dict( model_restart["optimizer"] )

		model.eval()

		model.alpha = torch.tensor(tin.alpha, device=model.device)

	#else:
		#print("Starting from scratch...")


def init_transfer_stp(tin):

	transfer_setup_params = FPSetupParameter(tin.N_species)

	for iesp in range( tin.N_species ):
		with open(tin.transfer_param["names"][iesp], "r") as f:
			read_network_iesp( None, iesp, f)
			neval_i, sfval_min_i, sfval_max_i, sfval_avg_i, sfval_cov_i = read_setup_iesp(f)
			transfer_setup_params.neval    [iesp] = neval_i
			transfer_setup_params.sfval_min[iesp] = sfval_min_i
			transfer_setup_params.sfval_max[iesp] = sfval_max_i
			transfer_setup_params.sfval_avg[iesp] = sfval_avg_i
			transfer_setup_params.sfval_cov[iesp] = sfval_cov_i

			#transfer_setup_params.add_specie( iesp, None, None, None, None, None, None, 
			#		None, None, None, None, None, None, neval_i,
			#		sfval_min_i, sfval_max_i, sfval_avg_i, sfval_cov_i)

	tin.transfer_setup_params = transfer_setup_params


def init_transfer(tin, trainset_params, model):
	E_min, E_max, E_avg          = trainset_params.E_min, trainset_params.E_max, trainset_params.E_avg
	E_scaling, E_shift, E_atomic = trainset_params.E_scaling, trainset_params.E_shift, trainset_params.E_atomic

	# If TRANSFER, read previous model parameters, and SFT/energy normalization parameters
	if os.path.exists("./model.transfered.restart"):

		#print("Restarting...")
		model_restart = torch.load("./model.transfered.restart")
		model.load_state_dict( model_restart["model"] )
		model.optimizer.load_state_dict( model_restart["optimizer"] )

		model.eval()

		model.alpha = torch.tensor(tin.alpha, device=model.device)



	else:
		print("Reading for transfer...")
		model_transfer = torch.load(tin.transfer_param["transfer_model"])
		model.load_state_dict( model_transfer["model"] )
		model.optimizer.load_state_dict( model_transfer["optimizer"] )

		model.eval()


	# Freeze layer
	for iesp in range(tin.N_species):
		for ifun in  range(len(model.hidden_size[iesp])): 
			if tin.transfer_param["freeze_layers"][ifun]:
				weight = model.functions[iesp][2*ifun].weight.requires_grad = False
				bias   = model.functions[iesp][2*ifun].bias.requires_grad = False

	for g in model.optimizer.param_groups:
		g['lr'] = tin.lr
		g["weight_decay"] = tin.regularization



	#for i in model.functions:
	#	print(i)

	#for n, p in model.named_parameters():
	#	print("{:40} {:} {:}".format(n, p.requires_grad, tuple(p.shape)))
