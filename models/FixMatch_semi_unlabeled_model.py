import numpy as np
import torch
from torch.nn import functional as F
import pdb
from random import uniform
from tqdm import tqdm
from .base_model import BaseModel
from . import networks3D
from .densenet import *
from .hypergraph_utils import *
from .hypergraph import *
from utils import ema, contrastive_loss, cutout

class FixMatchSemiUnlabeledModel(BaseModel):
    def name(self):
        return 'FixMatchSemiUnlabeledModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        class_num = 2
        self.K_neigs = opt.K_neigs
        self.beta = opt.beta
        
        self.conf_thre = 0.95
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.using_focalloss = opt.focal
        self.loss_names = ['cls']
        self.train_encoders = opt.train_encoders

        self.loss_names.append('ce_u')
        
        self.model_names = ['Encoder_MRI', 'Encoder_PET', 'Encoder_NonImage']

        self.netEncoder_MRI = networks3D.init_net_update(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        self.netEncoder_PET = networks3D.init_net_update(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        self.netEncoder_NonImage = networks3D.init_net_update(networks3D.Encoder_NonImage(in_channels=7, out_channels=1024, dropout_prob=0.5), self.gpu_ids)
        
        self.num_graph_update = opt.num_graph_update
        self.weight_u = opt.weight_u # loss weight for unlabeled data
        self.netClassifier = torch.nn.Linear(1024*3, class_num)
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netClassifier.to(self.gpu_ids[0])
            self.netClassifier = torch.nn.DataParallel(self.netClassifier, self.gpu_ids)
        # self.netDecoder_HGIB = networks3D.init_net_update(HGIB_v1(1024*3, 1024, 3, use_bn=False, heads=1), self.gpu_ids)
        self.ema_model_mri = ema.ModelEMA(self.device, self.netEncoder_MRI, 0.999)
        self.ema_model_pet = ema.ModelEMA(self.device, self.netEncoder_PET, 0.999)
        self.ema_model_non_img = ema.ModelEMA(self.device, self.netEncoder_NonImage, 0.999)
        self.ema_model_cls = ema.ModelEMA(self.device, self.netClassifier, 0.999)

        self.criterionCE = torch.nn.CrossEntropyLoss()
        # self.contrastive_loss = contrastive_loss.ContrastiveLoss(opt.batch_size)
        # initialize optimizers
        if self.isTrain:
            self.optimizer = torch.optim.AdamW([#{'params': self.netDecoder_HGIB.parameters()}, 
                                                {'params': self.netEncoder_MRI.parameters()}, 
                                                {'params': self.netEncoder_PET.parameters()}, 
                                                {'params': self.netEncoder_NonImage.parameters()},
                                                {'params': self.netClassifier.parameters()}
                                                ],
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers = []
            self.optimizers.append(self.optimizer)

    def info(self, lens):
        self.len_train = lens[0]
        self.len_test = lens[1]

    def set_input(self, input):
        self.MRI = input[0].to(self.device)
        self.PET = input[1].to(self.device)
        self.target = input[2].to(self.device)
        self.nonimage = input[3].to(self.device)

    def set_HGinput(self, input=None):
        self.embedding = self.embedding.to(self.device)
        if input is not None:
            self.target = input.to(self.device)

    def ExtractFeatures(self, phase='test'):
        if (next(self.netEncoder_MRI.parameters()).device != self.MRI.device) or (next(self.netEncoder_NonImage.parameters()).device != self.nonimage.device):
            self.MRI = self.MRI.to(self.device)
            self.PET = self.PET.to(self.device)
            self.nonimage = self.nonimage.to(self.device)
            self.netEncoder_MRI.to(self.device)
            self.netEncoder_PET.to(self.device)
            self.netEncoder_NonImage.to(self.device)
        if phase == 'test':
            with torch.no_grad():
                self.embedding_MRI = self.ema_model_mri.ema(self.MRI)
                self.embedding_PET = self.ema_model_pet.ema(self.PET)
                self.embedding_NonImage = self.ema_model_non_img.ema(self.nonimage)
                # self.embedding_MRI = self.netEncoder_MRI(self.MRI)
                # self.embedding_PET = self.netEncoder_PET(self.PET)
                # self.embedding_NonImage = self.netEncoder_NonImage(self.nonimage)
        else:
            self.embedding_MRI = self.netEncoder_MRI(self.MRI)
            self.embedding_PET = self.netEncoder_PET(self.PET)
            self.embedding_NonImage = self.netEncoder_NonImage(self.nonimage)
        return self.embedding_MRI, self.embedding_PET, self.embedding_NonImage

    def HGconstruct(self, embedding_MRI, embedding_PET, embedding_NonImage):
        G = Hypergraph.from_feature_kNN(embedding_MRI, self.K_neigs, self.device)
        G.add_hyperedges_from_feature_kNN(embedding_PET, self.K_neigs)
        G.add_hyperedges_from_feature_kNN(embedding_NonImage, self.K_neigs)
        self.G = G  # construct graph for the forward pass
        self.embedding = torch.Tensor(np.hstack((embedding_MRI, embedding_PET, embedding_NonImage))).to(self.device)

    def forward(self, phase='train', train_loader=None, test_loader=None, train_loader_u=None, epoch=None):
        if phase == 'train':
            assert train_loader is not None, 'train_loader is None, please provide train_loader for training'
            len_train_loader = len(train_loader)
            train_loader_x_iter = iter(train_loader)
            if train_loader_u is not None:
                len_train_loader = max(len(train_loader), len(train_loader_u))
                train_loader_u_iter = iter(train_loader_u)
            for i in range(len_train_loader): 
                try:
                    data = next(train_loader_x_iter)
                except StopIteration:
                    train_loader_x_iter = iter(train_loader)
                    data = next(train_loader_x_iter)
                try:
                    data_u = next(train_loader_u_iter)
                except StopIteration:
                    train_loader_u_iter = iter(train_loader_u)
                    data_u = next(train_loader_u_iter)
                
                self.set_input(data)
                self.ExtractFeatures(phase='train')
                embedding = torch.cat((self.embedding_MRI, self.embedding_PET, self.embedding_NonImage), dim=1)
                prediction = self.netClassifier(embedding)
                self.loss_cls = self.criterionCE(prediction, self.target)
                
                self.loss = self.loss_cls
                # Pseudo-labeling for unlabeled data
                MRI_np, PET_np, _, non_image, MRI_str_aug, PET_str_aug = data_u
                with torch.no_grad():
                    for i in range(MRI_str_aug.shape[0]):
                        MRI_str_aug[i, :, :, :, :] = cutout.CutoutAbs(MRI_str_aug[i, :, :, :, :], 8)
                        PET_str_aug[i, :, :, :, :] = cutout.CutoutAbs(PET_str_aug[i, :, :, :, :], 8)
                embedding_MRI = self.netEncoder_MRI(MRI_np)
                embedding_PET = self.netEncoder_PET(PET_np)
                #embedding_u = torch.cat((embedding_MRI, embedding_PET, embedding_NonImage), dim=1)
                #prediction_u_weak = self.netClassifier(embedding_u)
                #MRI_str_aug = fda.mix_amplitude(MRI_str_aug, PET_str_aug)
                #PET_str_aug = fda.mix_amplitude(PET_str_aug, MRI_str_aug)
                embedding_MRI_str_aug = self.netEncoder_MRI(MRI_str_aug)
                embedding_PET_str_aug = self.netEncoder_PET(PET_str_aug)
                with torch.no_grad():
                    embedding_NonImage = self.netEncoder_NonImage(non_image)
                    embedding_u = torch.cat((embedding_MRI, embedding_PET, embedding_NonImage), dim=1)
                    prediction_u = self.netClassifier(embedding_u)
                    output_u = F.softmax(prediction_u, 1).clone().detach()
                    max_prob, label_u = output_u.max(1)
                    mask_u = (max_prob >= self.conf_thre).float()

                # lmda = uniform(0., 1.0)
                # lmda = max(lmda, 1-lmda)
                # embedding_MRI_str_aug = lmda * embedding_MRI_str_aug + (1-lmda) * embedding_PET_str_aug
                # lmda = uniform(0., 1.0)
                # lmda = max(lmda, 1-lmda)
                # embedding_PET_str_aug = (1-lmda) * embedding_MRI_str_aug + lmda * embedding_PET_str_aug
                embedding_u_aug = torch.cat((embedding_MRI_str_aug, embedding_PET_str_aug, embedding_NonImage), dim=1)
                prediction_u_aug = self.netClassifier(embedding_u_aug)
                loss_ce_u = F.cross_entropy(prediction_u_aug, label_u, reduction='none')

                self.loss_ce_u = (loss_ce_u * mask_u).mean()
                self.loss = self.loss + self.loss_ce_u
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                self.ema_model_mri.update(self.netEncoder_MRI)
                self.ema_model_pet.update(self.netEncoder_PET)
                self.ema_model_non_img.update(self.netEncoder_NonImage)
                self.ema_model_cls.update(self.netClassifier)
                self.prediction_cur = prediction
                self.target_cur = self.target
                self.accuracy = (torch.softmax(self.prediction_cur, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
                if i % 100 == 0:
                    print('Epoch {} iteration {}, loss for encoders {}, loss_u_ce {}, mask_sum {}'.format(
                        epoch, i, self.loss.item(), loss_ce_u.item(), mask_u.mean().item()))
        elif phase == 'test':
            self.loss_cls = torch.tensor(0.)
            self.loss_ce_u = torch.tensor(0.)
            MRI, PET, Non_Img, Label, length = self.get_features([train_loader, test_loader])
            # create hypergraph
            self.HGconstruct(MRI, PET, Non_Img)
            self.info(length)
            self.set_HGinput(Label)
            idx = torch.tensor(range(self.len_test)).to(self.device) + self.len_train
            prediction_encoder = self.ema_model_cls.ema(self.embedding)
            # prediction_encoder = self.netClassifier(self.embedding)
            self.prediction_cur = prediction_encoder[idx]
            self.target_cur = self.target[idx]
            self.pred_encoder = prediction_encoder[idx]
            self.accuracy = (torch.softmax(self.prediction_cur, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
            self.acc_encoder = (torch.softmax(self.pred_encoder, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target.size(0))
        else:
            print('Wrong in loss calculation')
            exit(-1)


    def optimize_parameters(self, train_loader, test_loader, train_loader_u=None, epoch=None):
        #self.optimizer.zero_grad()
        # forward pass is here
        self.netClassifier.train()
        self.train()
        self.forward('train', train_loader, test_loader, train_loader_u, epoch)
        #self.loss.backward()
        #self.optimizer.step()

    def validation(self):
        self.netClassifier.eval()
        self.eval()
        with torch.no_grad():
            self.forward('test', self.train_loader, self.test_loader)

    def get_pred_encoder(self):
        return self.pred_encoder
    
    def get_acc_encoder(self):
        return self.acc_encoder
    
    def get_features(self, loaders):
        # extract featrues from pre-trained model
        # stack them 
        MRI = None
        PET = None
        Non_Img = None
        Label = None
        length = [0, 0]
        for idx, loader in enumerate(loaders):
            for i, data in enumerate(tqdm(loader)):
                self.set_input(data)
                i_MRI, i_PET, i_Non_Img = self.ExtractFeatures()
                if MRI is None:
                    MRI = i_MRI
                    PET = i_PET
                    Non_Img = i_Non_Img
                    Label = data[2]
                else:
                    MRI = torch.cat([MRI, i_MRI], 0)
                    PET = torch.cat([PET, i_PET], 0)
                    Non_Img = torch.cat([Non_Img, i_Non_Img], 0)
                    Label = torch.cat([Label, data[2]], 0)
            length[idx] = MRI.size(0)
        length[1] = length[1] - length[0]
        return MRI.cpu().detach().numpy(), PET.cpu().detach().numpy(), Non_Img.cpu().detach().numpy(), Label, length
