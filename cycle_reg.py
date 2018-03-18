import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class CycleGANMaskModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        assert (opt.output_nc == 1)
        if self.isTrain:
            self.label_noise = opt.label_noise
            if self.label_noise:
                self.noise_scale = opt.label_noise_scale
            else:
                self.noise_scale = 0
        self.thres = opt.thres
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_MA = self.Tensor(nb, 1, size, size)
        self.input_MB = self.Tensor(nb, 1, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_MA_pool = ImagePool(opt.pool_size)
            self.fake_MB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.regLoss == 'MSE':
                self.criterionCycle = torch.nn.MSELoss()
                self.criterionIdt = torch.nn.MSELoss()
            elif opt.regLoss == 'L1':
                self.criterionCycle = torch.nn.L1Loss()
                self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        # real_A = AMA, real_B = BMB, fake_AMB = A+fake_MB, fake_BMA = B+fake_MA
        # rec_MA = G_B(A+fake_MB) ~ MA, rec_MB = G_A(B+fake_MA) ~ MB
        # idt_MA = G_B(A+MA) ~ MA, idt_MB = G_A(B+MB) ~ MB
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_MA = input['MA']
        input_B = input['B' if AtoB else 'A']
        input_MB = input['MB']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_MA.resize_(input_MA.size()).copy_(input_MA)
        self.input_MB.resize_(input_MB.size()).copy_(input_MB)
        if self.isTrain:
            MA_noise = self.input_MA + self.noise_scale * (np.random.normal(0, 1))
            MB_noise = self.input_MB + self.noise_scale * (np.random.normal(0, 1))
            self.input_MA = torch.clamp(MA_noise, 0, 1)
            self.input_MB = torch.clamp(MB_noise, 0, 1)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def ToDiscrete(self, continuous):
        tensor = continuous.cpu().numpy()
        mask = np.ceil(tensor * (tensor>0.5))
        return torch.from_numpy(mask).cuda().type(torch.cuda.FloatTensor)

    def forward(self):
        self.real_A = Variable(torch.cat([self.input_A, self.input_MA], 1))
        self.real_B = Variable(torch.cat([self.input_B, self.input_MB], 1))

    def test(self):
        self.real_A = Variable(torch.cat([self.input_A, self.input_MA], 1))
        self.real_B = Variable(torch.cat([self.input_B, self.input_MB], 1))
        fake_MB= self.netG_A(self.real_A)
        self.rec_MA = self.netG_B(Variable(torch.cat([self.input_A, fake_MB.data], 1))).data
        self.fake_MB = fake_MB.data
        fake_MA = self.netG_B(self.real_B)
        self.rec_MB = self.netG_A(Variable(torch.cat([self.input_B, fake_MA.data], 1))).data
        self.fake_MA = fake_MA.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    # def add_noise(self):

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Cycle loss
        fake_MB = self.netG_A(self.real_A)

        # fake_AMB = Variable(torch.cat([self.input_A, fake_MB.data], 1))
        fake_AMB_dis = Variable(torch.cat([self.input_A, self.ToDiscrete(fake_MB.data)], 1))
        rec_MA = self.netG_B(fake_AMB_dis)
        loss_cycle_A = self.criterionCycle(rec_MA, Variable(self.input_MA)) * lambda_A

        fake_MA = self.netG_B(self.real_B)
        # fake_BMA = Variable(torch.cat([self.input_B, fake_MA.data], 1))
        fake_BMA_dis = Variable(torch.cat([self.input_B, self.ToDiscrete(fake_MA.data)], 1))
        rec_MB = self.netG_A(fake_BMA_dis)
        loss_cycle_B = self.criterionCycle(rec_MB, Variable(self.input_MB)) * lambda_B

        # GAN loss
        pred_fake = self.netD_A(fake_AMB_dis)
        loss_G_A = self.criterionGAN(pred_fake, True)
        pred_fake = self.netD_B(fake_BMA_dis)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Identity loss
        if lambda_idt > 0:
            # loss_idt_A:
            idt_MA = self.netG_B(self.real_A)
            loss_idt_A = self.criterionIdt(idt_MA, Variable(self.input_MA)) * lambda_A * lambda_idt
            # loss_idt_B:
            idt_MB = self.netG_A(self.real_B)
            loss_idt_B = self.criterionIdt(idt_MB, Variable(self.input_MB)) * lambda_B * lambda_idt

            self.idt_MA = idt_MA.data
            self.idt_MB = idt_MB.data
            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        loss_G = loss_cycle_A + loss_cycle_B + loss_G_A + loss_G_B + loss_idt_A + loss_idt_B
        loss_G.backward()

        self.fake_MA = fake_MA.data
        self.fake_MB = fake_MB.data
        self.rec_MA = rec_MA.data
        self.rec_MB = rec_MB.data

        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]
        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]


    def backward_D_A(self):
        real = self.real_A
        pred_real = self.netD_A(real)
        loss_D_A_real = self.criterionGAN(real, True)

        fake_MA = self.fake_MA_pool.query(self.fake_MA)
        fake = Variable(torch.cat([self.input_B, self.ToDiscrete(fake_MA.data)], 1), requires_grad=True)
        # fake = Variable(torch.cat([self.input_B, fake_MA.data], 1), requires_grad=True)
        pred_fake = self.netD_A(fake.detach())
        loss_D_A_fake = self.criterionGAN(fake, False)

        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_A.backward()
        self.loss_D_A = loss_D_A.data[0]

    def backward_D_B(self):
        real = self.real_B
        pred_real = self.netD_B(real)
        loss_D_B_real = self.criterionGAN(real, True)

        fake_MB = self.fake_MB_pool.query(self.fake_MB)
        fake = Variable(torch.cat([self.input_A, self.ToDiscrete(fake_MB.data)], 1), requires_grad=True)
        # fake = Variable(torch.cat([self.input_A, fake_MB.data], 1), requires_grad=True)
        pred_fake = self.netD_B(fake.detach())
        loss_D_B_fake = self.criterionGAN(fake, False)

        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D_B.backward()
        self.loss_D_B = loss_D_B.data[0]


    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                 ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B',  self.loss_cycle_B)])
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        real_B = util.tensor2im(self.input_B)
        if self.isTrain:
            real_MA = util.tensor2im(self.input_MA)
            fake_MB = util.tensor2im(self.fake_MB)
            rec_MA = util.tensor2im(self.rec_MA)
            real_MB = util.tensor2im(self.input_MB)
            fake_MA = util.tensor2im(self.fake_MA)
            rec_MB = util.tensor2im(self.rec_MB)
        else:
            real_MA = util.tensor2biim(self.input_MA, self.thres)
            fake_MB = util.tensor2biim(self.fake_MB, self.thres)
            rec_MA = util.tensor2biim(self.rec_MA, self.thres)
            real_MB = util.tensor2biim(self.input_MB, self.thres)
            fake_MA = util.tensor2biim(self.fake_MA, self.thres)
            rec_MB = util.tensor2biim(self.rec_MB, self.thres)


        ret_visuals = OrderedDict([('real_A', real_A), ('real_MA', real_MA), ('fake_MB', fake_MB), ('rec_MA', rec_MA),
                                   ('real_B', real_B), ('real_MB', real_MB), ('fake_MA', fake_MA), ('rec_MB', rec_MB)])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_MA'] = util.tensor2biim(self.idt_MA, self.thres)
            ret_visuals['idt_MB'] = util.tensor2biim(self.idt_MB, self.thres)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
