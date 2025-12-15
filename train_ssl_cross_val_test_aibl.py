import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import torch
import matplotlib.pyplot as plt
import numpy as np
import pdb
import copy
from utils.NiftiDataset_cls_adni_cross_val_aibl import NifitDataSetADNICrossValAIBL
from utils.NiftiDataset_cls_semisup_non_imaging import NifitSemiSupDataSet
from utils.NiftiDataset_cls_AIBL_cross_val import NifitAIBLCrossVal
from options.train_options import TrainOptions
import pandas as pd
from models import create_model
from utils.visualizer import Visualizer
from tqdm import tqdm
from monai.data import DataLoader
from torchmetrics.classification import (
    BinaryConfusionMatrix, 
    BinaryF1Score, 
    BinaryAccuracy, 
    BinarySpecificity, 
    BinaryRecall, 
    BinaryAUROC, 
    BinaryNegativePredictiveValue
    )
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    ScaleIntensity,
    CenterSpatialCrop,
    RandAdjustContrast,
    RandFlip,
    RandRotate
)
from torch.utils.tensorboard import SummaryWriter
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.utils import GetFeatures, cal_metrics

import random


def validation(model, epoch, save_best_result=None, test=False, str_metric=None):
    visualizer = Visualizer(opt)
    total_steps = 0
    print("Validating model...")

    losses = []
    acc = []
    best_for_test = False
    if test:
        phase = 'Test'
    else:
        phase = 'Validation'

    visualizer.reset()
    total_steps += opt.batch_size

    model.validation()
    loss= model.get_current_losses(train=False)
    losses.append(loss)
    acc.append(model.get_current_acc())

    prediction = model.get_prediction_cur()
    target = model.get_target_cur()

    prediction = prediction.squeeze().cpu().detach()

    target = target.cpu().detach()

    metric_encoder = BinaryAccuracy()
    acc = metric_encoder(prediction, target)
    metric_encoder = BinaryF1Score()
    F1 = metric_encoder(prediction, target)
    metric_encoder = BinaryAUROC()
    AUROC = metric_encoder(prediction, target)

    metric_encoder = BinarySpecificity()
    Specificity = metric_encoder(prediction, target)
    metric_encoder = BinaryRecall()
    recall = metric_encoder(prediction, target)

    confmat_encoder = BinaryConfusionMatrix()
    CM = confmat_encoder(prediction, target)
    PPV, NPV_calc = cal_metrics(CM)
    PPV = PPV[1]
    npv_metric = BinaryNegativePredictiveValue()
    NPV = npv_metric(prediction, target)
    print('PPV, NPV from confusion matrix {}, {}, NPV from torchmetric {}'.format(PPV, NPV_calc, NPV))
    print('['+phase+'] loss: ', np.sum(losses, 0) / len(losses), '\t\t Accuracy:', acc.item(),
          'AUC', AUROC.item(), 'PPV:', PPV.item(), 'NPV:', NPV.item(), 
          'F1:', F1.item(), 'Spec.:', Specificity.item(), 'Recall:', recall.item())
    if save_best_result is not None:
        save_results = 'Epoch {}, Accuracy {}, AUC {}, PPV {}, NPV {}, F1 {}, Spec. {}, Recall {}'.format(
            epoch, acc.item(), AUROC.item(), PPV.item(), NPV.item(), F1.item(), Specificity.item(), recall.item())
        if save_best_result['graph_ROC'] <= AUROC.item():
            save_best_result['graph_ROC'] = AUROC.item()
            save_best_result['graph_ROC_all'] = save_results
            # best_for_test = True
        if save_best_result['graph_PPV'] <= PPV.item():
            save_best_result['graph_PPV'] = PPV.item()
            save_best_result['graph_PPV_all'] = save_results
        if save_best_result['graph_NPV'] <= NPV.item():
            save_best_result['graph_NPV'] = NPV.item()
            save_best_result['graph_NPV_all'] = save_results
        if save_best_result['graph_ACC'] <= acc.item():
            save_best_result['graph_ACC'] = acc.item()
            save_best_result['graph_ACC_all'] = save_results
    
    if True:
        pred_encoder = model.get_pred_encoder()
        pred_encoder = pred_encoder.squeeze().cpu().detach()

        metric_encoder = BinaryAccuracy()
        acc_pred = metric_encoder(pred_encoder, target)
        metric_encoder = BinaryF1Score()
        F1 = metric_encoder(pred_encoder, target)
        metric_encoder = BinaryAUROC()
        AUROC_encoder = metric_encoder(pred_encoder, target)

        metric_encoder = BinarySpecificity()
        Specificity = metric_encoder(pred_encoder, target)
        metric_encoder = BinaryRecall()
        recall = metric_encoder(pred_encoder, target)

        confmat_encoder = BinaryConfusionMatrix()
        CM = confmat_encoder(pred_encoder, target)
        PPV_encoder, NPV_calc = cal_metrics(CM)
        PPV_encoder = PPV_encoder[1]
        npv_metric = BinaryNegativePredictiveValue()
        NPV_encoder = npv_metric(pred_encoder, target)
        print('PPV, NPV from confusion matrix {}, {}, NPV from torchmetric {}'.format(PPV_encoder, NPV_calc, NPV_encoder))
        print('['+phase+'] for encoders: ', '\t\t Accuracy:', acc_pred.item(),
            'AUC', AUROC_encoder.item(), 'PPV:', PPV_encoder.item(), 'NPV:', NPV_encoder.item(),
            'F1:', F1.item(), 'Spec.:', Specificity.item(), 'Recall:', recall.item()
            )
        fig = plt.figure()
        plt.imshow(CM)
        plt.colorbar()
        plt.savefig('fig/confusion_matrix.png')
        if save_best_result is not None:
            save_results = 'Epoch {}, Accuracy {}, AUC {}, PPV {}, NPV {}, F1 {}, Spec. {}, Recall {}'.format(
                epoch, acc_pred.item(), AUROC_encoder.item(), 
                PPV_encoder.item(), NPV_encoder.item(), F1.item(), Specificity.item(), recall.item())
            if test is False:
                str_metric = []
                if save_best_result['Enc_ROC'] <= AUROC_encoder.item():
                    save_best_result['Enc_ROC'] = AUROC_encoder.item()
                    save_best_result['Enc_ROC_all'] = save_results
                    best_for_test = True
                    str_metric.append('ROC')
                if save_best_result['Enc_PPV'] <= PPV_encoder.item():
                    save_best_result['Enc_PPV'] = PPV_encoder.item()
                    save_best_result['Enc_PPV_all'] = save_results
                    best_for_test = True
                    str_metric.append('PPV')
                if save_best_result['Enc_NPV'] <= NPV_encoder.item():
                    save_best_result['Enc_NPV'] = NPV_encoder.item()
                    save_best_result['Enc_NPV_all'] = save_results
                if save_best_result['Enc_ACC'] <= acc_pred.item():
                    save_best_result['Enc_ACC'] = acc_pred.item()
                    save_best_result['Enc_ACC_all'] = save_results
            else:
                assert str_metric is not None, 'str_metric parameter should not be None during testing phase'
                for metr in str_metric:
                    save_best_result['Enc_' + metr + '_all'] = save_results
        
        pred_avg = (pred_encoder + prediction) / 2.
        CM = confmat_encoder(pred_avg, target)
        NPV_avg = npv_metric(pred_avg, target)
        PPV_avg, NPV_calc = cal_metrics(CM)
        PPV_avg = PPV_avg[1]
        metric_encoder = BinaryAUROC()
        AUROC_avg = metric_encoder(pred_avg, target)
        metric_encoder = BinaryF1Score()
        F1 = metric_encoder(pred_avg, target)
        metric_encoder = BinarySpecificity()
        Specificity = metric_encoder(pred_avg, target)
        metric_encoder = BinaryRecall()
        recall = metric_encoder(pred_avg, target)
        print('['+phase+'] Averaged Prediction: ', '\t\t AUC', AUROC_avg.item(), 'PPV:', PPV_avg.item(), 'NPV:', 
              NPV_avg.item(), 'F1:', F1.item(), 'Spec.:', Specificity.item(), 'Recall:', recall.item())
        if save_best_result is not None:
            save_results = 'Epoch {}, AUC {}, PPV {}, NPV {}, F1 {}, Spec. {}, Recall {}'.format(
                epoch, AUROC_avg.item(), PPV_avg.item(), NPV_avg.item(), F1.item(), Specificity.item(), recall.item())
            if save_best_result['Avg_ROC'] <= AUROC_avg.item():
                save_best_result['Avg_ROC'] = AUROC_avg.item()
                save_best_result['Avg_ROC_all'] = save_results
            if save_best_result['Avg_PPV'] <= PPV_avg.item():
                save_best_result['Avg_PPV'] = PPV_avg.item()
                save_best_result['Avg_PPV_all'] = save_results
            if save_best_result['Avg_NPV'] <= NPV_avg.item():
                save_best_result['Avg_NPV'] = NPV_avg.item()
                save_best_result['Avg_NPV_all'] = save_results
        if best_for_test:
            model.save_networks('best')
            print(f'save the best model at epoch {epoch}')
    return save_best_result, best_for_test, str_metric

if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = TrainOptions().parse()
    seed=opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    use_writer = False
    if use_writer:
        writer = SummaryWriter(log_dir=os.path.join(opt.checkpoints_dir, opt.name, 'tensorboard/logs'))
    else:
        writer = None
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    # load dataset
    trainTransforms = Compose([RandFlip(prob=0.5), 
                               ScaleIntensity(), EnsureChannelFirst(), CenterSpatialCrop(opt.patch_size)])
    train_set = NifitDataSetADNICrossValAIBL(opt.data_path, transforms=trainTransforms, train=True, 
                                     phase='train', label_time=opt.label_time, control = opt.control, split=opt.split, label_ratio=opt.label_ratio, fold=opt.fold)
    print('length labeled train list:', len(train_set))
    if len(train_set) < opt.batch_size:
        drop_last = False
    else:
        drop_last = True
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers*len(opt.gpu_ids), 
                              pin_memory=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g)  # Here are then fed to the network with a defined batch size
    # unlabeled data
    trainTransformsStrong = Compose([RandAdjustContrast(prob=0.3, gamma=(0.6, 1.5)), 
                                     RandFlip(prob=0.5), 
                                     RandRotate(range_x=15.0, range_y=15.0, range_z=15.0, prob=0.5),
                                     ScaleIntensity(), EnsureChannelFirst(), CenterSpatialCrop(opt.patch_size)])
    train_set_u = NifitDataSetADNICrossValAIBL(opt.data_path, transforms=trainTransforms, unlabeled_only=True,
                                      train=True, phase='train', label_time=opt.label_time, 
                                      control = opt.control, split=opt.split, use_strong_aug=opt.use_strong_aug,
                                      transforms_strong=trainTransformsStrong, binary_class=True)
    train_loader_u = DataLoader(train_set_u, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                              num_workers=opt.workers*len(opt.gpu_ids), pin_memory=True, worker_init_fn=seed_worker, generator=g)
    testTransforms = Compose([ScaleIntensity(), EnsureChannelFirst(), CenterSpatialCrop(opt.patch_size)])
    val_set = NifitDataSetADNICrossValAIBL(opt.data_path, transforms=testTransforms,
                           test=True, phase='val', label_time=opt.label_time, control=opt.control, split=opt.split, 
                           fold=opt.fold, binary_class=True)
    print('length val list:', len(val_set))
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                            pin_memory=True)
    test_set = NifitAIBLCrossVal(opt.data_path, transforms=testTransforms, control=opt.control, phase='test_only',
                                 img_time='bl', label_time=opt.label_time, binary_class=True)
    print('length test list on AIBL:', len(test_set))
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                            pin_memory=True)
    # create model
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print("Creating model...")
    model = create_model(opt)  # creation of the model
    model.setup(opt)
    if opt.epoch_count > 1:
        model.load_networks(opt.epoch_count)
    if opt.load_weight:
        if not opt.continue_train:
            model.load_pretrained_networks(opt.epoch_count)
        else:
            print('opt.continue_train is True, expecting resuming training from epoch {}, \
                  but the weights at that epoch are over-written by {}'.format(opt.epoch_count, opt.load_weight))

    visualizer = Visualizer(opt)
    total_steps = 0
    num_classes = 4 if opt.label_time == 'bl' else 3
    model.train_loader, model.test_loader = train_loader, val_loader
    MRI, Non_Img, Label, length = model.get_features([train_loader, val_loader])
    # create hypergraph
    model.HGconstruct(MRI, Non_Img)
    model.info(length)
    
    print("Training model...")
    save_best_result = {'graph_ACC': 0, 'graph_PPV': 0, 'graph_NPV': 0, 'graph_ROC': 0, 
                        'Enc_ACC': 0, 'Enc_PPV': 0, 'Enc_NPV': 0, 'Enc_ROC': 0,
                        'Avg_PPV': 0, 'Avg_NPV': 0, 'Avg_ROC': 0}
    save_test_result = copy.deepcopy(save_best_result)
    best_model = None
    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1), desc="Current epoch during training."):
        if epoch > 100:
            break
        epoch_iter = 0
        losses = []
        acc = []
        prediction = None
        target = None

        visualizer.reset()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size

        model.set_HGinput(Label)
        if epoch == 0:
            validation(model, epoch)
        try:
            model.optimize_parameters(train_loader, val_loader, train_loader_u, epoch)
        except:
            if 'DHGNN' in opt.model:
                model.optimize_parameters(epoch)
            else:
                model.optimize_parameters()
        loss = model.get_current_losses()
        losses.append(loss)
        acc.append(model.get_current_acc())
        if use_writer:
            writer.add_scalar("train_loss/CE", loss[0],  epoch )
            if opt.focal:
                writer.add_scalar("train_loss/focal", loss[1], epoch)
            writer.add_scalar("train_loss/acc", model.get_current_acc(), epoch)
        prediction = model.get_prediction_cur()
        target = model.get_target_cur()
        prediction = prediction.cpu().detach()

        # evaluation results
        target = target.cpu().detach()
        metric = BinaryAccuracy()
        acc = metric(prediction, target)
        metric = BinaryF1Score() 
        F1 = metric(prediction, target)
        metric = BinaryAUROC()
        AUROC = metric(prediction, target)
        print('Train loss: ', np.sum(losses, 0) / len(losses), '\t\t Accuracy:', acc, 'AUC', AUROC, 'F1', F1)

        if opt.focal:
            visualizer.print_val_losses(epoch, np.sum(losses, 0) / len(losses), acc, 'Train')
        else:
            visualizer.print_val_losses(epoch, losses, acc, 'Train')

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, total_steps))
            model.save_networks('latest')

        if epoch % opt.save_epoch_freq == 0 and epoch != 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d ' % (epoch, opt.niter + opt.niter_decay))
        model.update_learning_rate(model.get_current_acc())

        if epoch % 10 == 0 and epoch != 0:
            MRI, Non_Img, Label, length = model.get_features([train_loader, val_loader])
            # create hypergraph
            model.HGconstruct(MRI, Non_Img)
            model.info(length)
            model.set_HGinput(Label)
            save_best_result, best_for_test, str_metric = validation(model, epoch, save_best_result)
            df = pd.DataFrame(save_best_result, index=[0])
            df.T.to_csv('checkpoints/{}/Best_results_fold{}.csv'.format(opt.name, opt.fold), index=True, header=False)
            if best_for_test:
                model.train_loader, model.test_loader = None, None
                best_model = copy.deepcopy(model)
                model.train_loader, model.test_loader = train_loader, val_loader
                best_model.train_loader, best_model.test_loader = train_loader, test_loader
                MRI, Non_Img, Label, length = best_model.get_features([train_loader, test_loader])
                # create hypergraph
                best_model.HGconstruct(MRI, Non_Img)
                best_model.info(length)
                best_model.set_HGinput(Label)
                save_test_result, _, _ = validation(best_model, epoch=epoch, save_best_result=save_test_result, test=True, str_metric=str_metric)
                
                df_to_append = pd.DataFrame(save_test_result, index=[0])
                df_new = pd.concat([df, df_to_append], ignore_index=True)
                df_new.T.to_csv('checkpoints/{}/Best_results_fold{}.csv'.format(opt.name, opt.fold), mode='a', index=True, header=False)
        if use_writer:
            writer.close()