import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='SR', help='SR, RE')

# LF_SR
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--scale_factor", type=int, default=2, help="4, 2")
parser.add_argument('--model_name', type=str, default='LF_DGT', help="model name")
parser.add_argument("--use_pre_ckpt", type=bool, default=True, help="use pre model ckpt")
parser.add_argument("--path_pre_pth", type=str, default='./pre/', help="path for pre model ckpt")
parser.add_argument('--data_name', type=str, default='ALL',
                    help='EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, ALL(of Five Datasets)')
parser.add_argument('--path_for_train', type=str, default='./data_for_training/')
parser.add_argument('--path_for_test', type=str, default='./data_for_test/')
parser.add_argument('--path_log', type=str, default='./log/')
parser.add_argument('--use_mixed_precision', type=bool, default=True, help='Enable mixed precision or not')
parser.add_argument('--use_grad_accumulation', type=bool, default=True, help='Enable gradient accumulation or not')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate')
parser.add_argument('--decay_rate', type=float, default=0, help='weight decay [default: 1e-4]')
parser.add_argument('--n_steps', type=int, default=10, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument('--epoch', default=60, type=int, help='Epoch to run [default: 50]')
parser.add_argument('--seed', default=10, type=int, help='seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_workers', type=int, default=1, help='num workers of the Data Loader')
parser.add_argument('--local_rank', dest='local_rank', type=int, default=0)

args = parser.parse_args()

if args.task == 'SR':
    args.angRes_in = args.angRes
    args.angRes_out = args.angRes
    args.patch_size_for_test = 32
    args.stride_for_test = 16
    args.minibatch_for_test = 1

if args.scale_factor == 4:
    args.batch_size           = 4
    args.epoch                = 90
    args.n_steps              = 15
    args.use_mixed_precision  = False
    args.use_grad_accumulation = True

elif args.scale_factor == 2:
    args.batch_size           = 8
    args.epoch                = 60
    args.n_steps              = 10
    args.use_mixed_precision  = True
    args.use_grad_accumulation = True