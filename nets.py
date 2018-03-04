# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import sys, time, os
sys.path.append('./utils')
from nets_pixel_shuffle import *
from data import *
from image_pool import *
from scipy.misc import imsave
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# from hyperboard import Agent
import argparse



# def sample_z(batch_size, z_dim):
# 	return np.random.uniform(-1., 1., size=[batch_size, z_dim])


class CycleGAN():
	def __init__(self, G_AB, G_BA, D_A, D_B, data, use_lsgan, exp, cuda, port, pair):
		self.G_AB = G_AB
		self.G_BA = G_BA
		self.D_A = D_A
		self.D_B = D_B
		self.data = data
		self.cuda = cuda
		self.port = port
		self.use_lsgan = use_lsgan
		self.exp = exp
		self.pair = pair
		# self.registe_curves()

		if self.cuda:
			self.G_AB.cuda()
			self.G_BA.cuda()
			self.D_A.cuda()
			self.D_B.cuda()
	
	# def registe_curves(self):
	# 	self.agent = Agent(username = '', password = '', address = '127.0.0.1', port = self.port)
	# 	d_a_loss = {self.exp: 'adv loss of D_A'}
	# 	d_b_loss = {self.exp: 'adv loss of D_B'}
	# 	g_ab_loss = {self.exp: 'adv loss of G_AB'}
	# 	g_ba_loss = {self.exp: 'adv loss of G_BA'}
	# 	a_recon_loss = {self.exp: 'reconstruction loss of A (A -> B -> A)'}
	# 	b_recon_loss = {self.exp: 'reconstruction loss of B (B -> A -> B)'}
	# 	self.da_loss = self.agent.register(d_a_loss, 'loss', overwrite=True)
	# 	self.db_loss = self.agent.register(d_b_loss, 'loss', overwrite=True)
	# 	self.g_loss_ab = self.agent.register(g_ab_loss, 'loss', overwrite=True)
	# 	self.g_loss_ba = self.agent.register(g_ba_loss, 'loss', overwrite=True)
	# 	self.g_recon_loss_a = self.agent.register(a_recon_loss, 'loss', overwrite=True)
	# 	self.g_recon_loss_b = self.agent.register(b_recon_loss, 'loss', overwrite=True)

	
	def train(self, sample_dir, ckpt_dir, batch_size, training_epochs=50000, from_epoch=0):
		fig_count = 0
		g_lr = 2e-3
		d_lr = 5e-4
		lam = 10
		n_g = 1

		if self.pair:
			self.data.reset(sort=True)

		# if self.cuda:
		# 	label_true = Variable(torch.ones(batch_size).cuda())
		# 	label_false = Variable(torch.zeros(batch_size).cuda())
		# else:
		# 	label_true = Variable(torch.ones(batch_size))
		# 	label_false = Variable(torch.zeros(batch_size))
	
		label_true = 1.0
		label_false = 0.0
		if self.use_lsgan:
			criterion = lambda u,v: torch.mean((u - v)**2)
		else:
			criterion = lambda u,v: -torch.mean(v * torch.log(u+1e-8) + (1-v) * torch.log(1-u+1e-8))

		optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=d_lr, betas=(0.5, 0.999), weight_decay=1e-5)
		optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=d_lr, betas=(0.5, 0.999), weight_decay=1e-5)
		optimizer_G_AB = optim.Adam(self.G_AB.parameters(), lr=g_lr, betas=(0.5, 0.999), weight_decay=1e-5)
		optimizer_G_BA = optim.Adam(self.G_BA.parameters(), lr=g_lr, betas=(0.5, 0.999), weight_decay=1e-5)

		scheduler_G_BA = lr_scheduler.StepLR(optimizer_G_BA, step_size=5000, gamma=0.92)
		scheduler_G_AB = lr_scheduler.StepLR(optimizer_G_AB, step_size=5000, gamma=0.92)
		scheduler_D_A = lr_scheduler.StepLR(optimizer_D_A, step_size=5000, gamma=0.92)
		scheduler_D_B = lr_scheduler.StepLR(optimizer_D_B, step_size=5000, gamma=0.92)

		self.pool_A = ImagePool(50)
		self.pool_B = ImagePool(50)


		for epoch in range(from_epoch, from_epoch+training_epochs):
			scheduler_G_BA.step()
			scheduler_G_AB.step()
			scheduler_D_A.step()
			scheduler_D_B.step()

			begin_time = time.time()

			# update D_A and D_B
			self.D_A.zero_grad()  # clear previous gradients
			self.D_B.zero_grad()
			
			images_A, images_B = self.data(batch_size)
			real_images_A = Variable(torch.from_numpy(images_A))
			real_images_B = Variable(torch.from_numpy(images_B))

			if self.cuda:
				real_images_A = real_images_A.cuda()
				real_images_B = real_images_B.cuda()

			d_real_A = self.D_A(real_images_A)
			d_loss_real_A = criterion(d_real_A, label_true)
			d_loss_real_A.backward()
			d_loss_A = d_loss_real_A

			fake_images_AB = self.G_AB(real_images_A)
			d_fake_AB = self.D_B(self.pool_B.query(fake_images_AB.detach()))
			d_loss_fake_AB = criterion(d_fake_AB, label_false)
			d_loss_fake_AB.backward()
			d_loss_B = d_loss_fake_AB

			fake_images_ABA = self.G_BA(fake_images_AB.detach())
			# d_fake_ABA = self.D_A(fake_images_ABA.detach())  # detach() makes it a leaf node, so that backward will not apply to G
			# d_loss_fake_ABA = criterion(d_fake_ABA, label_false)
			# d_loss_fake_ABA.backward()  # gradients accmulate at D's nodes
			# d_loss_A += d_loss_fake_ABA

			d_real_B = self.D_B(real_images_B)
			d_loss_real_B = criterion(d_real_B, label_true)
			d_loss_real_B.backward()
			d_loss_B += d_loss_real_B

			fake_images_BA = self.G_BA(real_images_B)
			d_fake_BA = self.D_A(self.pool_A.query(fake_images_BA.detach()))
			d_loss_fake_BA = criterion(d_fake_BA, label_false)
			d_loss_fake_BA.backward()
			d_loss_A += d_loss_fake_BA

			fake_images_BAB = self.G_AB(fake_images_BA.detach())
			# d_fake_BAB = self.D_B(fake_images_BAB.detach())
			# d_loss_fake_BAB = criterion(d_fake_BAB, label_false)
			# d_loss_fake_BAB.backward()
			# d_loss_B += d_loss_fake_BAB

			optimizer_D_A.step()
			optimizer_D_B.step()

			# self.agent.append(self.da_loss, epoch, float(d_loss_A.data[0]))
			# self.agent.append(self.db_loss, epoch, float(d_loss_B.data[0]))

			# for G_AB and G_BA
			self.G_BA.zero_grad()
			self.G_AB.zero_grad()

			d_fake_AB = self.D_B(fake_images_AB)
			g_AB_loss_fake = criterion(d_fake_AB, label_true)
			g_AB_loss_fake.backward(retain_graph=True)
			g_AB_loss = g_AB_loss_fake
			# d_fake_ABA = self.D_A(fake_images_ABA)
			# g_ABA_loss_fake = criterion(d_fake_ABA, label_true)
			# g_ABA_loss_fake.backward(retain_graph=True)
			g_ABA_loss_recon = lam * torch.mean(torch.abs(fake_images_ABA - real_images_A)) / batch_size  # lam * torch.mean((fake_images_ABA - real_images_A)**2)
			g_ABA_loss_recon.backward(retain_graph=True)
			g_BA_loss = g_ABA_loss_recon # + g_ABA_loss_fake

			d_fake_BA = self.D_A(fake_images_BA)
			g_BA_loss_fake = criterion(d_fake_BA, label_true)
			g_BA_loss_fake.backward(retain_graph=True)
			g_BA_loss += g_BA_loss_fake
			# d_fake_BAB = self.D_B(fake_images_BAB)
			# g_BAB_loss_fake = criterion(d_fake_BAB, label_true)
			# g_BAB_loss_fake.backward(retain_graph=True)
			g_BAB_loss_recon = lam * torch.mean(torch.abs(fake_images_BAB - real_images_B)) / batch_size  # lam * torch.mean((fake_images_BAB - real_images_B)**2) 
			g_BAB_loss_recon.backward(retain_graph=True)
			g_AB_loss += g_BAB_loss_recon # + g_BAB_loss_fake
			
			if self.pair:
				g_AB_recon_loss = lam * torch.mean(torch.abs(real_images_B - fake_images_AB))
				g_AB_recon_loss.backward()
				g_AB_loss += g_AB_recon_loss
				g_BA_recon_loss = lam * torch.mean(torch.abs(real_images_A - fake_images_BA))
				g_BA_recon_loss.backward()
				g_BA_loss += g_BA_recon_loss
			else:
				g_AB_recon_loss = g_BA_recon_loss = 0.0

			optimizer_G_BA.step()
			optimizer_G_AB.step()

			# self.agent.append(self.g_recon_loss_a, epoch, float((g_ABA_loss_recon+g_BA_recon_loss).data[0]))
			# self.agent.append(self.g_recon_loss_b, epoch, float((g_BAB_loss_recon+g_AB_recon_loss).data[0]))
			# self.agent.append(self.g_loss_ab, epoch, float(g_AB_loss.data[0]))
			# self.agent.append(self.g_loss_ba, epoch, float(g_BA_loss.data[0]))			

			elapse_time = time.time() - begin_time
			print('Iter[%s], d_a_loss: %.4f, d_b_loss: %.4f, g_ab_loss: %s, g_ba_loss: %s, time elapsed: %.4fsec' % \
					(epoch+1, d_loss_A.data[0], d_loss_B.data[0], g_AB_loss.data[0], g_BA_loss.data[0], elapse_time))

			if epoch % 500 == 0 or epoch == from_epoch+training_epochs-1:
				images_A, images_B = self.data(batch_size)
				real_images_A = Variable(torch.from_numpy(images_A))
				if self.cuda:
					real_images_A = real_images_A.cuda()
				fake_images_AB = self.G_AB(real_images_A)
				fake_images_ABA = self.G_BA(fake_images_AB)
				A = torch.cat([real_images_A[0], fake_images_AB[0], fake_images_ABA[0]], 2)
				imsave(os.path.join(sample_dir, 'A-%s.png'%(str(epoch+1).zfill(7))), np.transpose(A.cpu().data.numpy(), [1,2,0]))
				
				real_images_B = Variable(torch.from_numpy(images_B))
				if self.cuda:
					real_images_B = real_images_B.cuda()
				fake_images_BA = self.G_BA(real_images_B)
				fake_images_BAB = self.G_AB(fake_images_BA)
				B = torch.cat([real_images_B[0], fake_images_BA[0], fake_images_BAB[0]], 2)
				imsave(os.path.join(sample_dir, 'B-%s.png'%(str(epoch+1).zfill(7))), np.transpose(B.cpu().data.numpy(), [1,2,0]))

			if epoch % 5000 == 4999:
				torch.save(self.G_AB.state_dict(), os.path.join(ckpt_dir, 'G_AB_epoch-%s.pth' % str(epoch).zfill(7)))
				torch.save(self.G_BA.state_dict(), os.path.join(ckpt_dir, 'G_BA_epoch-%s.pth' % str(epoch).zfill(7)))
				torch.save(self.D_A.state_dict(), os.path.join(ckpt_dir, 'D_A_epoch-%s.pth' % str(epoch).zfill(7)))
				torch.save(self.D_B.state_dict(), os.path.join(ckpt_dir, 'D_B_epoch-%s.pth' % str(epoch).zfill(7)))

	def pretrain(self, save_dir, batch_size, training_epochs=10000):
		'''
		Pretraining using pair data.
		'''
		g_lr = 1e-5
		
		if self.pair:
			self.data.reset(sort=True)

		optimizer_G_AB = optim.SGD(self.G_AB.parameters(), lr=g_lr)
		optimizer_G_BA = optim.SGD(self.G_BA.parameters(), lr=g_lr)

		
		for epoch in range(training_epochs):
			begin_time = time.time()
			
			images_A, images_B = self.data(batch_size)
			real_images_A = Variable(torch.from_numpy(images_A))
			real_images_B = Variable(torch.from_numpy(images_B))

			if self.cuda:
				real_images_A = real_images_A.cuda()
				real_images_B = real_images_B.cuda()

			fake_images_AB = self.G_AB(real_images_A)
			fake_images_ABA = self.G_BA(fake_images_AB)
			fake_images_BA = self.G_BA(real_images_B)
			fake_images_BAB = self.G_AB(fake_images_BA)
		
			if self.pair:
				recon_loss_AB = torch.mean(torch.abs(fake_images_AB - real_images_B))
				recon_loss_AB.backward(retain_graph=True)
				recon_loss_BA = torch.mean(torch.abs(fake_images_BA - real_images_A))
				recon_loss_BA.backward()
			else:
				recon_loss_AB = Variable(torch.zeros(1))
				recon_loss_BA = Variable(torch.zeros(1))

			recon_loss_A = torch.mean(torch.abs(fake_images_ABA - real_images_A))
			recon_loss_A.backward(retain_graph=True)
			recon_loss_B = torch.mean(torch.abs(fake_images_BAB - real_images_B))
			recon_loss_B.backward()

			optimizer_G_AB.step()
			optimizer_G_BA.step()


			elapse_time = time.time() - begin_time
			print('Iter[%s], recon loss A: %.4f, recon loss B: %.4f, recon loss AB: %.4f, recon loss BA: %.4f, time elapsed: %.4fsec' % \
					(epoch+1, recon_loss_A.data[0], recon_loss_B.data[0], recon_loss_AB.data[0], recon_loss_BA.data[0], elapse_time))

			if epoch % 500 == 0:
				images_, images_B = self.data(batch_size)
				real_images_A = Variable(torch.from_numpy(images_A))
				if self.cuda:
					real_images_A = real_images_A.cuda()
				fake_images_AB = self.G_AB(real_images_A)
				fake_images_ABA = self.G_BA(fake_images_AB)
				A = torch.cat([real_images_A[0], fake_images_AB[0], fake_images_ABA[0]], 2)
				imsave(os.path.join(save_dir, 'A-%s.png'%(str(epoch+1).zfill(7))), np.transpose(A.cpu().data.numpy(), [1,2,0]))
				
				real_images_B = Variable(torch.from_numpy(images_B))
				if self.cuda:
					real_images_B = real_images_B.cuda()
				fake_images_BA = self.G_BA(real_images_B)
				fake_images_BAB = self.G_AB(fake_images_BA)
				B = torch.cat([real_images_B[0], fake_images_BA[0], fake_images_BAB[0]], 2)
				imsave(os.path.join(save_dir, 'B-%s.png'%(str(epoch+1).zfill(7))), np.transpose(B.cpu().data.numpy(), [1,2,0]))


if __name__ == '__main__':

	data_path_dict = {'yellowman': (combined_images, 'datasets/yellowman/yelloman_self_train'),
			  'shoe': (combined_images, 'datasets/edges2shoes/train'),
			  'face': (combined_images, 'datasets/Combine_out1'),
			  'apple2orange': (image_folder, 'datasets/apple2orange', 'trainA', 'trainB'),
			  'season': (image_folder, '/data/rui.wu/gapeng/data_collection/seasons/', 'selected_autumn', 'selected_summer')}

	import argparse, psutil
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu', default='', type=str, help='gpu(s) to use.')
	parser.add_argument('-ls', '--use_lsgan', default=0, type=int, help='to use lsgan or not.')
	parser.add_argument('-e', '--exp', default='yellowman', type=str, help='experiment to run.')
	parser.add_argument('-pe', '--pretrain', default=0, type=int, help='number of pretrain epochs.')
	parser.add_argument('-p', '--port', default=5000, type=int, help='port for hyperboard to use.')
	parser.add_argument('-n', '--norm', default='batch', type=str, help='normalization to use(batch/instance which represents batchnorm or instancenorm respectively).')
	parser.add_argument('-gn', '--g_net', default='ae', type=str, help='network artchitecture of G, should be one of ["ae", "skip_ae", "u_net"], default: "ae"')
	parser.add_argument('-pair', '--pair', default=1, type=int, help='whether to use pair data or not.')
	parser.add_argument('-mhw', '--max_hw', default=512, type=int, help='max height and width.')
	parser.add_argument('-dn', '--d_net', default='gan', type=str, help='network artchitecture of G, should be one of ["gan", "patchgan"], default: "gan"')
	parser.add_argument('-ne', '--num_epochs', default=500000, type=int, help='number of epochs to run.')
	parser.add_argument('-s', '--size', default=-1, type=int, help='resize images.')
	parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size.')
	# parser.add_argument('-pid', '--pid', default=-1, type=int, help='waiting for process # to finish and run after that.')
	args = parser.parse_args()
	gpu = args.gpu
	use_lsgan = args.use_lsgan
	exp = args.exp
	cuda = True if len(gpu) else False
	pretrain = args.pretrain
	port = args.port
	norm = args.norm
	max_hw = args.max_hw
	g_net = args.g_net
	assert g_net in ['ae', 'skip_ae', 'u_net']
	d_net = args.d_net
	assert d_net in ['gan', 'patchgan']
	pair = args.pair
	num_epochs = args.num_epochs
	size = args.size
	batch_size = args.batch_size
	if size < 0:
		size = None
		if batch_size != 1:
			batch_size = 1
			print('Inputs might have different size. Reset batch_size=1.')
	if pair:
		assert data_path_dict[exp][0] == combined_images and len(data_path_dict[exp]) == 2
	else:
		assert len(data_path_dict[exp]) == 4

	os.environ['CUDA_VISIBLE_DEVICES'] = gpu
	# ready = False
	# print('Waiting', end='')
	#	while not ready:
	#		if not psutil.pid_exists(pid):
	#			ready = True
	#		else:
	#			print('.', end='')
	#			time.sleep(60)
	#	print()

	# save generated images
	exp_name = 'CycleGAN_pixel_shuffle+imagepool.l1.%s.%s.%s.%s.%s.%s.%s.%s'%(norm, 'lsgan' if use_lsgan else 'gan', exp, g_net, d_net, 'pair' if pair else 'unpair', max_hw, size)
	sample_dir = 'Samples/%s'%exp_name
	ckpt_dir = 'Models/%s'%exp_name
	pretrain_dir = 'Pretrain/%s'%exp_name
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	if not os.path.exists(pretrain_dir):
		os.makedirs(pretrain_dir)
	
	if g_net == 'ae':
		G_AB = G_autoencoder_flexible(channel=3, norm_type=norm, skip=False)
		G_BA = G_autoencoder_flexible(channel=3, norm_type=norm, skip=False)
	elif g_net == 'skip_ae':
		G_AB = G_autoencoder_flexible(channel=3, norm_type=norm, skip=True)
		G_BA = G_autoencoder_flexible(channel=3, norm_type=norm, skip=True)
	elif g_net == 'u_net':
		G_AB = UnetGenerator(input_nc=3, output_nc=3, num_downs=7, \
			norm_layer=nn.BatchNorm2d if norm=='batch' else nn.InstanceNorm2d)
		G_BA = UnetGenerator(input_nc=3, output_nc=3, num_downs=7, \
			norm_layer=nn.BatchNorm2d if norm=='batch' else nn.InstanceNorm2d)

	if d_net == 'gan':
		D_A = D_conv_flexible(channel=3, activation='sigmoid')
		D_B = D_conv_flexible(channel=3, activation='sigmoid')
	else:
		D_A = NLayerDiscriminator(input_nc=3, use_sigmoid=True, n_layers=3)
		D_B = NLayerDiscriminator(input_nc=3, use_sigmoid=True, n_layers=3)

	print('G_AB:\n', G_AB)
	print('G_BA:\n', G_BA)
	print('D_A:\n', D_A)
	print('D_B:\n', D_B)

	model_files = os.listdir(ckpt_dir)
	from_epoch = 0
	if len(model_files) > 0:
		g_ab_model = sorted([f for f in model_files if 'G_AB' in f])
		g_ba_model = sorted([f for f in model_files if 'G_BA' in f])
		d_a_model = sorted([f for f in model_files if 'D_A' in f])
		d_b_model = sorted([f for f in model_files if 'D_B' in f])
		assert int(g_ab_model[-1].strip('.pth').split('-')[-1]) == int(g_ba_model[-1].strip('.pth').split('-')[-1]) == \
			int(d_a_model[-1].strip('.pth').split('-')[-1]) == int(d_b_model[-1].strip('.pth').split('-')[-1])
		from_epoch = int(g_ab_model[-1].strip('.pth').split('-')[-1])
		G_AB.load_state_dict(torch.load(os.path.join(ckpt_dir, g_ab_model[-1])))
		G_BA.load_state_dict(torch.load(os.path.join(ckpt_dir, g_ba_model[-1])))
		D_A.load_state_dict(torch.load(os.path.join(ckpt_dir, d_a_model[-1])))
		D_B.load_state_dict(torch.load(os.path.join(ckpt_dir, d_b_model[-1])))
		print('Successfully restored from history models(epoch=%s).' % from_epoch)

		pretrain = 0  # reset pretrain, no more pretrain is needed.

	
	if pair:
		data = data_path_dict[exp][0](data_path_dict[exp][1], max_hw=max_hw, img_size=size)
	else:
		data_loader, base_path, path_A, path_B = data_path_dict[exp]
		data_A = data_loader(datapath=os.path.join(base_path, path_A), max_hw=max_hw, img_size=size)
		data_B = data_loader(datapath=os.path.join(base_path, path_B), max_hw=max_hw, img_size=size)
		data = lambda batch_size: (data_A(batch_size), data_B(batch_size))

	cyclegan = CycleGAN(G_AB, G_BA, D_A, D_B, data, use_lsgan, exp_name, cuda, port, pair)
	
	if pretrain:
		cyclegan.pretrain(pretrain_dir, batch_size, training_epochs=pretrain)
	
	cyclegan.train(sample_dir, ckpt_dir, batch_size, training_epochs=500000, from_epoch=from_epoch)






# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import sys, time, os
sys.path.append('./utils')
from nets import *
from data import *
from scipy.misc import imsave
import scipy.misc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from hyperboard import Agent
import argparse



def sample_y(batch_size, y_dim):
	return np.random.normal(0., 1., size=[batch_size, y_dim])


class CycleGAN():
	def __init__(self, y_dim, G_AB, G_BA, D_A, D_B, data, use_lsgan, exp, cuda, port, pair):
		self.y_dim = y_dim
		self.G_AB = G_AB
		self.G_BA = G_BA
		self.D_A = D_A
		self.D_B = D_B
		self.data = data
		self.cuda = cuda
		self.port = port
		self.use_lsgan = use_lsgan
		self.exp = exp
		self.pair = pair
		self.registe_curves()

		if self.cuda:
			self.G_AB.cuda()
			self.G_BA.cuda()
			self.D_A.cuda()
			self.D_B.cuda()
	
	def registe_curves(self):
		self.agent = Agent(username = '', password = '', address = '127.0.0.1', port = self.port)
		d_a_loss = {self.exp: 'adv loss of D_A'}
		d_b_loss = {self.exp: 'adv loss of D_B'}
		g_ab_loss = {self.exp: 'adv loss of G_AB'}
		g_ba_loss = {self.exp: 'adv loss of G_BA'}
		a_recon_loss = {self.exp: 'reconstruction loss of A (A -> B -> A)'}
		b_recon_loss = {self.exp: 'reconstruction loss of B (B -> A -> B)'}
		self.da_loss = self.agent.register(d_a_loss, 'loss', overwrite=True)
		self.db_loss = self.agent.register(d_b_loss, 'loss', overwrite=True)
		self.g_loss_ab = self.agent.register(g_ab_loss, 'loss', overwrite=True)
		self.g_loss_ba = self.agent.register(g_ba_loss, 'loss', overwrite=True)
		self.g_recon_loss_a = self.agent.register(a_recon_loss, 'loss', overwrite=True)
		self.g_recon_loss_b = self.agent.register(b_recon_loss, 'loss', overwrite=True)

	
	def train(self, sample_dir, ckpt_dir, training_epochs=50000):
		fig_count = 0
		g_lr = 2e-3
		d_lr = 1e-3
		batch_size = 1
		lam = 10
		n_g = 1

		if self.pair:
			self.data.reset(sort=True)

		# if self.cuda:
		# 	label_true = Variable(torch.ones(batch_size).cuda())
		# 	label_false = Variable(torch.zeros(batch_size).cuda())
		# else:
		# 	label_true = Variable(torch.ones(batch_size))
		# 	label_false = Variable(torch.zeros(batch_size))
		
		y1 = Variable(torch.FloatTensor(batch_size, self.y_dim))
		y2 = Variable(torch.FloatTensor(batch_size, self.y_dim))
		if self.cuda:
			y1 = y1.cuda()
			y2 = y2.cuda()

		label_true = 1.0
		label_false = 0.0
		if self.use_lsgan:
			criterion = lambda u,v: torch.mean((u - v)**2)
		else:
			criterion = lambda u,v: lambda u,v: -torch.mean(v * torch.log(u+1e-8) + (1-v) * torch.log(1-u+1e-8))

		optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=d_lr, betas=(0.5, 0.999), weight_decay=1e-5)
		optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=d_lr, betas=(0.5, 0.999), weight_decay=1e-5)
		optimizer_G_AB = optim.Adam(self.G_AB.parameters(), lr=g_lr, betas=(0.5, 0.999), weight_decay=1e-5)
		optimizer_G_BA = optim.Adam(self.G_BA.parameters(), lr=g_lr, betas=(0.5, 0.999), weight_decay=1e-5)

		scheduler_G_BA = lr_scheduler.StepLR(optimizer_G_BA, step_size=5000, gamma=0.92)
		scheduler_G_AB = lr_scheduler.StepLR(optimizer_G_AB, step_size=5000, gamma=0.92)
		scheduler_D_A = lr_scheduler.StepLR(optimizer_D_A, step_size=5000, gamma=0.92)
		scheduler_D_B = lr_scheduler.StepLR(optimizer_D_B, step_size=5000, gamma=0.92)


		for epoch in range(training_epochs):
			scheduler_G_BA.step()
			scheduler_G_AB.step()
			scheduler_D_A.step()
			scheduler_D_B.step()

			begin_time = time.time()

			# update D_A and D_B
			self.D_A.zero_grad()  # clear previous gradients
			self.D_B.zero_grad()
			
			images_A, images_B = self.data(batch_size)
			y1.data.copy_(torch.from_numpy(sample_y(batch_size, self.y_dim)))
			y2.data.copy_(torch.from_numpy(sample_y(batch_size, self.y_dim)))
			real_images_A = Variable(torch.from_numpy(images_A))
			real_images_B = Variable(torch.from_numpy(images_B))

			if self.cuda:
				real_images_A = real_images_A.cuda()
				real_images_B = real_images_B.cuda()

			d_real_A, _ = self.D_A(real_images_A)
			d_loss_real_A = criterion(d_real_A, label_true)
			d_loss_real_A.backward()
			d_loss_A = d_loss_real_A

			fake_images_AB = self.G_AB(real_images_A, y1)
			d_fake_AB, d_AB_y1 = self.D_B(fake_images_AB.detach())
			d_loss_fake_AB = criterion(d_fake_AB, label_false) + torch.mean((d_AB_y1 - y1)**2)
			d_loss_fake_AB.backward()
			d_loss_B = d_loss_fake_AB

			fake_images_ABA = self.G_BA(fake_images_AB.detach(), y1)
			d_fake_ABA, d_ABA_y1 = self.D_A(fake_images_ABA.detach())  # detach() makes it a leaf node, so that backward will not apply to G
			d_loss_fake_ABA = criterion(d_fake_ABA, label_false) + torch.mean((d_ABA_y1 - y1)**2)
			d_loss_fake_ABA.backward()  # gradients accmulate at D's nodes
			d_loss_A += d_loss_fake_ABA

			d_real_B, _ = self.D_B(real_images_B)
			d_loss_real_B = criterion(d_real_B, label_true)
			d_loss_real_B.backward()
			d_loss_B += d_loss_real_B

			fake_images_BA = self.G_BA(real_images_B, y2)
			d_fake_BA, d_BA_y2 = self.D_A(fake_images_BA.detach())
			d_loss_fake_BA = criterion(d_fake_BA, label_false) + torch.mean((d_BA_y2 - y2)**2)
			d_loss_fake_BA.backward()
			d_loss_A += d_loss_fake_BA

			fake_images_BAB = self.G_AB(fake_images_BA.detach(), y2)
			d_fake_BAB, d_BAB_y2 = self.D_B(fake_images_BAB.detach())
			d_loss_fake_BAB = criterion(d_fake_BAB, label_false) + torch.mean((d_BAB_y2 - y2)**2)
			d_loss_fake_BAB.backward()
			d_loss_B += d_loss_fake_BAB

			optimizer_D_A.step()
			optimizer_D_B.step()

			self.agent.append(self.da_loss, epoch, float(d_loss_A.data[0]))
			self.agent.append(self.db_loss, epoch, float(d_loss_B.data[0]))

			# for G_AB and G_BA
			self.G_BA.zero_grad()
			self.G_AB.zero_grad()

			d_fake_AB, d_AB_y1 = self.D_B(fake_images_AB)
			g_AB_loss_fake = criterion(d_fake_AB, label_true) + torch.mean((d_AB_y1 - y1)**2)
			g_AB_loss_fake.backward(retain_graph=True)
			g_AB_loss = g_AB_loss_fake
			d_fake_ABA, d_ABA_y1 = self.D_A(fake_images_ABA)
			g_ABA_loss_fake = criterion(d_fake_ABA, label_true) + torch.mean((d_ABA_y1 - y1)**2)
			g_ABA_loss_fake.backward(retain_graph=True)
			g_ABA_loss_recon = lam * torch.mean(torch.abs(fake_images_ABA - real_images_A)) / batch_size  # lam * torch.mean((fake_images_ABA - real_images_A)**2)
			g_ABA_loss_recon.backward(retain_graph=True)
			g_BA_loss = g_ABA_loss_recon + g_ABA_loss_fake

			d_fake_BA, d_BA_y2 = self.D_A(fake_images_BA)
			g_BA_loss_fake = criterion(d_fake_BA, label_true) + torch.mean((d_BA_y2 - y2)**2)
			g_BA_loss_fake.backward(retain_graph=True)
			g_BA_loss += g_BA_loss_fake
			d_fake_BAB, d_BAB_y2 = self.D_B(fake_images_BAB)
			g_BAB_loss_fake = criterion(d_fake_BAB, label_true) + torch.mean((d_BAB_y2 - y2)**2)
			g_BAB_loss_fake.backward(retain_graph=True)
			g_BAB_loss_recon = lam * torch.mean(torch.abs(fake_images_BAB - real_images_B)) / batch_size  # lam * torch.mean((fake_images_BAB - real_images_B)**2) 
			g_BAB_loss_recon.backward(retain_graph=True)
			g_AB_loss += (g_BAB_loss_recon + g_BAB_loss_fake)
			
			if self.pair:
				g_AB_recon_loss = lam * torch.mean(torch.abs(real_images_B - fake_images_AB))
				g_AB_recon_loss.backward()
				g_AB_loss += g_AB_recon_loss
				g_BA_recon_loss = lam * torch.mean(torch.abs(real_images_A - fake_images_BA))
				g_BA_recon_loss.backward()
				g_BA_loss += g_BA_recon_loss
			else:
				g_AB_recon_loss = g_BA_recon_loss = 0.0

			optimizer_G_BA.step()
			optimizer_G_AB.step()

			self.agent.append(self.g_recon_loss_a, epoch, float((g_ABA_loss_recon+g_BA_recon_loss).data[0]))
			self.agent.append(self.g_recon_loss_b, epoch, float((g_BAB_loss_recon+g_AB_recon_loss).data[0]))
			self.agent.append(self.g_loss_ab, epoch, float(g_AB_loss.data[0]))
			self.agent.append(self.g_loss_ba, epoch, float(g_BA_loss.data[0]))			

			elapse_time = time.time() - begin_time
			print('Iter[%s], d_a_loss: %.4f, d_b_loss: %.4f, g_ab_loss: %s, g_ba_loss: %s, time elapsed: %.4fsec' % \
					(epoch+1, d_loss_A.data[0], d_loss_B.data[0], g_AB_loss.data[0], g_BA_loss.data[0], elapse_time))

			if epoch % 500 == 0:
				images_A, images_B = self.data(batch_size)
				real_images_A = Variable(torch.from_numpy(images_A))
				y1.data.copy_(torch.from_numpy(sample_y(batch_size, self.y_dim)))
				if self.cuda:
					real_images_A = real_images_A.cuda()
				fake_images_AB = self.G_AB(real_images_A, y1)
				fake_images_ABA = self.G_BA(fake_images_AB, y1)
				A = torch.cat([real_images_A[0], fake_images_AB[0], fake_images_ABA[0]], 2)
				imsave(os.path.join(sample_dir, 'A-%s.png'%(str(epoch+1).zfill(7))), np.transpose(A.cpu().data.numpy(), [1,2,0]))
				
				y2.data.copy_(torch.from_numpy(sample_y(batch_size, self.y_dim)))
				real_images_B = Variable(torch.from_numpy(images_B))
				if self.cuda:
					real_images_B = real_images_B.cuda()
				fake_images_BA = self.G_BA(real_images_B, y2)
				fake_images_BAB = self.G_AB(fake_images_BA, y2)
				B = torch.cat([real_images_B[0], fake_images_BA[0], fake_images_BAB[0]], 2)
				imsave(os.path.join(sample_dir, 'B-%s.png'%(str(epoch+1).zfill(7))), np.transpose(B.cpu().data.numpy(), [1,2,0]))

			if epoch % 10000 == 0:
				torch.save(self.G_AB.state_dict(), os.path.join(ckpt_dir, 'G_AB_epoch-%s.pth' % str(epoch).zfill(7)))
				torch.save(self.G_BA.state_dict(), os.path.join(ckpt_dir, 'G_BA_epoch-%s.pth' % str(epoch).zfill(7)))
				torch.save(self.D_A.state_dict(), os.path.join(ckpt_dir, 'D_A_epoch-%s.pth' % str(epoch).zfill(7)))
				torch.save(self.D_B.state_dict(), os.path.join(ckpt_dir, 'D_B_epoch-%s.pth' % str(epoch).zfill(7)))

	"""
	def pretrain(self, save_dir, training_epochs=10000):
		'''
		Pretraining using pair data.
		'''
		batch_size = 1
		g_lr = 1e-5
		
		if self.pair:
			self.data.reset(sort=True)

		optimizer_G_AB = optim.SGD(self.G_AB.parameters(), lr=g_lr)
		optimizer_G_BA = optim.SGD(self.G_BA.parameters(), lr=g_lr)

		
		for epoch in range(training_epochs):
			begin_time = time.time()
			
			images_A, images_B = self.data(batch_size)
			real_images_A = Variable(torch.from_numpy(images_A))
			real_images_B = Variable(torch.from_numpy(images_B))

			if self.cuda:
				real_images_A = real_images_A.cuda()
				real_images_B = real_images_B.cuda()

			fake_images_AB = self.G_AB(real_images_A)
			fake_images_ABA = self.G_BA(fake_images_AB)
			fake_images_BA = self.G_BA(real_images_B)
			fake_images_BAB = self.G_AB(fake_images_BA)
		
			if self.pair:
				recon_loss_AB = torch.mean(torch.abs(fake_images_AB - real_images_B))
				recon_loss_AB.backward(retain_graph=True)
				recon_loss_BA = torch.mean(torch.abs(fake_images_BA - real_images_A))
				recon_loss_BA.backward()
			else:
				recon_loss_AB = Variable(torch.zeros(1))
				recon_loss_BA = Variable(torch.zeros(1))

			recon_loss_A = torch.mean(torch.abs(fake_images_ABA - real_images_A))
			recon_loss_A.backward(retain_graph=True)
			recon_loss_B = torch.mean(torch.abs(fake_images_BAB - real_images_B))
			recon_loss_B.backward()

			optimizer_G_AB.step()
			optimizer_G_BA.step()


			elapse_time = time.time() - begin_time
			print('Iter[%s], recon loss A: %.4f, recon loss B: %.4f, recon loss AB: %.4f, recon loss BA: %.4f, time elapsed: %.4fsec' % \
					(epoch+1, recon_loss_A.data[0], recon_loss_B.data[0], recon_loss_AB.data[0], recon_loss_BA.data[0], elapse_time))

			if epoch % 500 == 0:
				images_, images_B = self.data(batch_size)
				real_images_A = Variable(torch.from_numpy(images_A))
				if self.cuda:
					real_images_A = real_images_A.cuda()
				fake_images_AB = self.G_AB(real_images_A)
				fake_images_ABA = self.G_BA(fake_images_AB)
				A = torch.cat([real_images_A[0], fake_images_AB[0], fake_images_ABA[0]], 2)
				imsave(os.path.join(save_dir, 'A-%s.png'%(str(epoch+1).zfill(7))), np.transpose(A.cpu().data.numpy(), [1,2,0]))
				
				real_images_B = Variable(torch.from_numpy(images_B))
				if self.cuda:
					real_images_B = real_images_B.cuda()
				fake_images_BA = self.G_BA(real_images_B)
				fake_images_BAB = self.G_AB(fake_images_BA)
				B = torch.cat([real_images_B[0], fake_images_BA[0], fake_images_BAB[0]], 2)
				imsave(os.path.join(save_dir, 'B-%s.png'%(str(epoch+1).zfill(7))), np.transpose(B.cpu().data.numpy(), [1,2,0]))
	"""

def translate(directory, write_to_directory, G_AB, G_BA, channel, y_dim, from_domain, cuda):
	files = os.listdir(directory)
	batch_size = 1
	G_dict = {'translate':None, 'reconstruct':None}
	domain = {'A': 'B', 'B': 'A'}
	y1 = Variable(torch.FloatTensor(batch_size, y_dim))
	X = Variable(torch.FloatTensor(batch_size, channel, 256, 256))
	if cuda:
		y1 = y1.cuda()
		X = X.cuda()
		G_AB.cuda()
		G_BA.cuda()
	if from_domain.upper() == 'A':  # restrict images size to 256*256 to support U-net 
		G_dict['translate'] = G_AB
		G_dict['reconstruct'] = G_BA
	else:
		G_dict['translate'] = G_BA
		G_dict['reconstruct'] = G_AB
	for f in files:
		img = scipy.misc.imresize(scipy.misc.imread(os.path.join(directory, f), mode='RGB'), [256, 256]).reshape([1, 256, 256, channel]) / 255.
		img = np.transpose(img, [0, 3, 1, 2])
		X.data.copy_(torch.from_numpy(img.astype(np.float32)))
		_y1 = sample_y(batch_size, y_dim)
		_y2 = sample_y(batch_size, y_dim)
		for lam in np.arange(0, 1, 0.1):
			y1.data.copy_(torch.from_numpy((_y1 + lam * (_y2 - _y1)).astype(np.float32)))
			X_tran = G_dict['translate'](X, y1)
			X_recon = G_dict['reconstruct'](X_tran, y1)
			scipy.misc.imsave(os.path.join(write_to_directory, 'Tranto_'+domain[from_domain]+'_'+str(lam)+'_'+f), X_tran.cpu().data.numpy()[0].transpose([1, 2, 0]))
			scipy.misc.imsave(os.path.join(write_to_directory, 'Recon_'+domain[from_domain]+'_'+str(lam)+'_'+f), X_recon.cpu().data.numpy()[0].transpose([1, 2, 0]))
	


if __name__ == '__main__':

	data_path_dict = {'yellowman': (combined_images, 'datasets/yellowman/yelloman_self_train'),
			  'shoe': (combined_images, 'datasets/edges2shoes/train'),
			  'face': (combined_images, 'datasets/Combine_out1'),
			  'apple2orange': (image_folder, 'datasets/apple2orange', 'trainA', 'trainB')}

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-g', '--gpu', default='', type=str, help='gpu(s) to use.')
	parser.add_argument('-ls', '--use_lsgan', default=0, type=int, help='to use lsgan or not.')
	parser.add_argument('-e', '--exp', default='yellowman', type=str, help='experiment to run.')
	parser.add_argument('-p', '--port', default=5000, type=int, help='port for hyperboard to use.')
	parser.add_argument('-n', '--norm', default='batch', type=str, help='normalization to use(batch/instance which represents batchnorm or instancenorm respectively).')
	parser.add_argument('-gn', '--g_net', default='ae', type=str, help='network artchitecture of G, should be one of ["ae", "skip_ae"], default: "ae"')
	parser.add_argument('-pair', '--pair', default=1, type=int, help='whether to use pair data or not.')
	parser.add_argument('-y', '--y_dim', default=100, type=int, help='y_dim, the condition(gaussian, unsupervised using infogan).')
	parser.add_argument('-m', '--mode', default='train', type=str, help='Mode: translation or train, should be one of ["translate", "train"]')
	parser.add_argument('-fd', '--from_directory', default='test', type=str, help='Reading evaluation images from # directory.')
	parser.add_argument('-td', '--write_to_directory', default='result', type=str, help='Writing translation results to # directory.')
	parser.add_argument('-md', '--model_directory', default='', type=str, help='Loading model from # directory for translation mode.')
	parser.add_argument('-me', '--model_epoch', default=0, type=int, help="model's epoch for translation model")
	parser.add_argument('-d', '--from_domain', default='A', type=str, help="Translating from domain #")
	args = parser.parse_args()
	gpu = args.gpu
	use_lsgan = args.use_lsgan
	exp = args.exp
	cuda = True if len(gpu) else False
	port = args.port
	norm = args.norm
	g_net = args.g_net
	assert g_net in ['ae', 'skip_ae']
	pair = args.pair
	mode = args.mode
	assert mode in ['translate', 'train']
	from_directory = args.from_directory
	write_to_directory = args.write_to_directory
	model_directory = args.model_directory
	model_epoch = args.model_epoch
	from_domain = args.from_domain
	y_dim = args.y_dim
	if pair:
		assert data_path_dict[exp][0] == combined_images and len(data_path_dict[exp]) == 2
	else:
		assert len(data_path_dict[exp]) == 4

	os.environ['CUDA_VISIBLE_DEVICES'] = gpu

	# save generated images
	if mode == 'train':
		exp_name = 'CycleGAN.l1.%s.%s.%s.%s.%s.%s'%(norm, 'lsgan' if use_lsgan else 'gan', exp, g_net, 'pair' if pair else 'unpair', y_dim)
		sample_dir = 'Samples/%s'%exp_name
		ckpt_dir = 'Models/%s'%exp_name
		if not os.path.exists(sample_dir):
			os.makedirs(sample_dir)
		if not os.path.exists(ckpt_dir):
			os.makedirs(ckpt_dir)
		
		D_A = D_conv_with_recon_flexible(y_dim, channel=3, activation='sigmoid')
		D_B = D_conv_with_recon_flexible(y_dim, channel=3, activation='sigmoid')
		print('D_A:\n', D_A)
		print('D_B:\n', D_B)

	if g_net == 'ae':
		G_AB = G_cond_autoencoder_flexible(y_dim, channel=3, norm_type=norm, skip=False)
		G_BA = G_cond_autoencoder_flexible(y_dim, channel=3, norm_type=norm, skip=False)
	elif g_net == 'skip_ae':
		G_AB = G_cond_autoencoder_flexible(y_dim, channel=3, norm_type=norm, skip=True)
		G_BA = G_cond_autoencoder_flexible(y_dim, channel=3, norm_type=norm, skip=True)

	print('G_AB:\n', G_AB)
	print('G_BA:\n', G_BA)
	
	if mode == 'translate':
		_succ = 0
		model_files = os.listdir(model_directory)
		for model_file in model_files:
			if str(model_epoch) in model_file and 'G_AB' in model_file:
				G_AB.load_state_dict(torch.load(os.path.join(model_directory, model_file)))
				_succ += 1
			elif str(model_epoch) in model_file and 'G_BA' in model_file:
				G_BA.load_state_dict(torch.load(os.path.join(model_directory, model_file)))
				_succ += 1
		if _succ == 2:
			print('Model restored.')
		else:
			raise ValueError('Can not find model at epoch %s in directory: %s' % (model_epoch, model_directory))
	
	if mode == 'train':
		if pair:
			data = data_path_dict[exp][0](data_path_dict[exp][1], max_hw=512)
		else:
			data_loader, base_path, path_A, path_B = data_path_dict[exp]
			data_A = data_loader(datapath=os.path.join(base_path, path_A), max_hw=512)
			data_B = data_loader(datapath=os.path.join(base_path, path_B), max_hw=512)
			data = lambda batch_size: (data_A(batch_size), data_B(batch_size))
	

		cyclegan = CycleGAN(y_dim, G_AB, G_BA, D_A, D_B, data, use_lsgan, exp_name, cuda, port, pair)
	
		cyclegan.train(sample_dir, ckpt_dir, training_epochs=500000)
	else:
		translate(from_directory, write_to_directory, G_AB, G_BA, 3, y_dim, from_domain, cuda)
