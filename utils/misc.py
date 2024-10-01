import torch
import os
from datetime import datetime, timedelta
import logging
from torch.utils.tensorboard import SummaryWriter

def save_model(args, epoch, loss, model):
    # remove old models
    if epoch > 0:
        best_snapshot = 'best_epoch_{}_loss_{:.8f}.pth'.format(
            args.best_record['epoch'], args.best_record['loss'])
        best_snapshot = os.path.join(args.exp_path, best_snapshot)
        assert os.path.exists(best_snapshot), 'cant find old snapshot {}'.format(best_snapshot)
        os.remove(best_snapshot)

    # save new best
    args.best_record['epoch'] = epoch
    args.best_record['loss'] = loss

    best_snapshot = 'best_epoch_{}_loss_{:.8f}.pth'.format(
        args.best_record['epoch'], args.best_record['loss'])
    best_snapshot = os.path.join(args.exp_path, best_snapshot)

    torch.save({
        'state_dict': model.state_dict(),
        'epoch': epoch,
    }, best_snapshot)
    logging.info('save best models in ' + best_snapshot)



def prep_experiment_dir(args):
    """
    Make output directories, setup logging, Tensorboard, snapshot code.
    """
    ckpt_path = args.ckpt
    os.makedirs(ckpt_path + f'/plot', exist_ok=True)
    ckpt_path = os.path.join(ckpt_path, args.model)
    args.exp_path = os.path.join(ckpt_path, args.exp)
    os.makedirs(args.exp_path, exist_ok=True)
    args.fig_path = args.exp_path
    os.makedirs(args.fig_path, exist_ok=True)
    os.makedirs(args.fig_path + f'/best', exist_ok=True)
    os.makedirs(args.fig_path + f'/obs_log', exist_ok=True)
    os.makedirs(args.fig_path + f'/model', exist_ok=True)
    os.makedirs(args.fig_path + f'/loss_log', exist_ok=True)

