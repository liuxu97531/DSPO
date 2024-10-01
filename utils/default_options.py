import configargparse
import torch

def parses():
    # Argument Parser
    parser = configargparse.ArgumentParser(description='physics_field_reconstruction')
    # problem settings
    parser.add_argument('--num_obs', type=int, default=10)
    parser.add_argument('--idx_obs', type=int, default=None)
    parser.add_argument('--idx_optima', type=int, default=None)
    parser.add_argument('--P', type=int, default=400)
    parser.add_argument('--N_fun', type=int, default=60)
    parser.add_argument('--delta_u_obs', type=int, default=None)
    parser.add_argument('--sample_method', type=str, default=None)
    parser.add_argument('--pod_basis_num', type=int, default=None)
    parser.add_argument('--model', type=str, default=None)
    # training settings
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    # logging and models save settings
    parser.add_argument('--plot_freq', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--exp', type=str, default='recon',
                        help='experiment directory name')
    parser.add_argument('--ckpt', type=str, default='logs',
                        help='Save Checkpoint Point')
    parser.add_argument('--tb_path', type=str, default='logs/tb',
                        help='Save Tensorboard Path')
    parser.add_argument('--snapshot', type=str, default=None)

    # training environment settings
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    return args
