from collections import OrderedDict
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import imageio
import importlib
import random
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader



def main(args):
    setup_seed(args.seed)

    ''' Create Dir for Save'''
    log_dir, checkpoints_dir, val_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA Training LOADING '''
    logger.log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)
    logger.log_string("The number of training data is: %d" % len(train_Dataset))
    effective_batch_size = 1 if args.use_grad_accumulation else args.batch_size
    train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
                                               batch_size=effective_batch_size, shuffle=True,)

    ''' DATA Validation LOADING '''
    logger.log_string('\nLoad Validation Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of validation data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args, mode='train')
    net = net.to(device)
    ''' Optimizer '''
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)
    scaler = GradScaler() if args.use_mixed_precision else None
    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
        start_epoch = 0
        logger.log_string('Do not use pre-trained model!')
    else:
        try:
            ckpt_path = args.path_pre_pth
            checkpoint = torch.load(ckpt_path, map_location=device)
            if checkpoint['training_complete']:
                raise RuntimeError(
                    "Error: Loaded a model with an equivalent layer (testing phase) during training. Please load a "
                    "full training model.")
            start_epoch = checkpoint['epoch']
            try:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = 'module.' + k  # add `module.`
                    new_state_dict[name] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
            except:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                logger.log_string('Loaded scheduler state.')
            if args.use_mixed_precision and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
                logger.log_string('Scaler state loaded!')
        except:
            net = MODEL.get_model(args, mode='train')
            net.apply(MODEL.weights_init)
            start_epoch = 0
            logger.log_string('No existing model, starting training from scratch...')
            pass
        pass


    cudnn.benchmark = True

    ''' Print Parameters '''
    logger.log_string('PARAMETER ...')
    logger.log_string(args)


    ''' LOSS LOADING '''
    criterion = MODEL.get_loss(args).to(device)

    ''' TRAINING & TEST '''
    logger.log_string('\nStart training...')
    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))

        ''' Training '''
        net.train()
        loss_epoch_train, psnr_epoch_train, ssim_epoch_train = train(train_loader, device, net, criterion, optimizer, MODEL, scaler, args)
        logger.log_string('The %dth Train, loss is: %.5f, psnr is %.5f, ssim is %.5f' %
                          (idx_epoch + 1, loss_epoch_train, psnr_epoch_train, ssim_epoch_train))

        ''' scheduler '''
        scheduler.step()

        ''' Save PTH  '''
        if args.local_rank == 0:
            save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%dx_epoch_%02d_model.pth' % (
            args.model_name, args.angRes_in, args.angRes_in, args.scale_factor, idx_epoch + 1)
            state = {
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict() if args.use_mixed_precision else None,
                'optimizer': optimizer.state_dict(),
                'training_complete': False,
            }
            torch.save(state, save_ckpt_path)
            logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))


        ''' Validation '''
        step = 5
        if (idx_epoch )%step==0 or idx_epoch > 45:
            with torch.no_grad():
                ''' Create Excel for PSNR/SSIM '''
                excel_file = ExcelFile()

                psnr_testset = []
                ssim_testset = []
                for index, test_name in enumerate(test_Names):
                    test_loader = test_Loaders[index]

                    epoch_dir = val_dir.joinpath('VAL_epoch_%02d' % (idx_epoch + 1))
                    epoch_dir.mkdir(exist_ok=True)
                    save_dir = epoch_dir.joinpath(test_name)
                    save_dir.mkdir(exist_ok=True)

                    psnr_iter_test, ssim_iter_test, LF_name = test(test_loader, device, net, MODEL, save_dir)
                    excel_file.write_sheet(test_name, LF_name, psnr_iter_test, ssim_iter_test)


                    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
                    ssim_epoch_test = float(np.array(ssim_iter_test).mean())


                    psnr_testset.append(psnr_epoch_test)
                    ssim_testset.append(ssim_epoch_test)
                    logger.log_string('The %dth Test on %s, psnr/ssim is %.2f/%.3f' % (
                    idx_epoch + 1, test_name, psnr_epoch_test, ssim_epoch_test))
                    pass
                psnr_mean_test = float(np.array(psnr_testset).mean())
                ssim_mean_test = float(np.array(ssim_testset).mean())
                logger.log_string('The mean psnr on testsets is %.5f, mean ssim is %.5f'
                                  % (psnr_mean_test, ssim_mean_test))
                excel_file.xlsx_file.save(str(epoch_dir) + '/evaluation.xls')
                pass
            pass


        pass


    ''' Save Test PTH  '''
    # After training, create the equivalent layer
    MODEL.create_equivalent_layers_recursively(net)
    print("Creating equivalent layer")
    if args.local_rank == 0:
        save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%dx_model_test.pth' % (
            args.model_name, args.angRes_in, args.angRes_in, args.scale_factor)
        state = {
            'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
            'training_complete': True,
        }
        torch.save(state, save_ckpt_path)
        logger.log_string('Saving the model_test')
    pass




def train(train_loader, device, net, criterion, optimizer, MODEL, scaler, args):
    ''' training one epoch '''
    psnr_iter_train = []
    loss_iter_train = []
    ssim_iter_train = []
    disparity_info = MODEL.gen_disp_match_idx(args.angRes, args.angRes, args.patch_size_for_test, args.patch_size_for_test, args.scale_factor)
    for idx_iter, (data, label, data_info) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
        # [Lr_angRes_in, Lr_angRes_out] = data_info
        # data_info[0] = Lr_angRes_in[0].item()
        # data_info[1] = Lr_angRes_out[0].item()

        data = data.to(device)      # low resolution
        label = label.to(device)    # high resolution
        # flops = FlopCountAnalysis(net, inputs=(data, data_info))
        # flops = FlopCountAnalysis(net, inputs=(data, disparity_info))
        # print("FLOPs: ", flops.total())
        # print(parameter_count_table(net))

        if args.use_mixed_precision:
            with autocast():
                # out = net(data, data_info)
                out = net(data, disparity_info)
                loss = criterion(out, label)
        else:
            out = net(data, disparity_info)
            loss = criterion(out, label)

        (scaler.scale(loss) if args.use_mixed_precision else loss).backward()

        if (idx_iter + 1) % args.batch_size == 0 or not args.use_grad_accumulation or (idx_iter + 1) == len(train_loader):
            if args.use_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        torch.cuda.empty_cache()

        loss_iter_train.append(loss.data.cpu())
        psnr, ssim = cal_metrics(args, label, out)
        psnr_iter_train.append(psnr)
        ssim_iter_train.append(ssim)
        pass

    loss_epoch_train = float(np.array(loss_iter_train).mean())
    psnr_epoch_train = float(np.array(psnr_iter_train).mean())
    ssim_epoch_train = float(np.array(ssim_iter_train).mean())

    return loss_epoch_train, psnr_epoch_train, ssim_epoch_train


def test(test_loader, device, net, MODEL, save_dir=None):
    LF_iter_test = []
    psnr_iter_test = []
    ssim_iter_test = []
    disparity_info = MODEL.gen_disp_match_idx(args.angRes, args.angRes, args.patch_size_for_test, args.patch_size_for_test, args.scale_factor)
    for idx_iter, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        Lr_SAI_y = Lr_SAI_y.squeeze().to(device)  # numU, numV, h*angRes, w*angRes
        Hr_SAI_y = Hr_SAI_y
        Sr_SAI_cbcr = Sr_SAI_cbcr

        ''' Crop LFs into Patches '''
        subLFin = LFdivide(Lr_SAI_y, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
        numU, numV, H, W = subLFin.size()
        subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
        subLFout = torch.zeros(numU * numV, 1, args.angRes_in * args.patch_size_for_test * args.scale_factor,
                               args.angRes_in * args.patch_size_for_test * args.scale_factor)

        ''' SR the Patches '''
        for i in range(0, numU * numV, args.minibatch_for_test):
            tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
            with torch.no_grad():
                net.eval()
                torch.cuda.empty_cache()
                out = net(tmp.to(device), disparity_info)

                subLFout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = out
        subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

        ''' Restore the Patches to LFs '''
        Sr_4D_y = LFintegrate(subLFout, args.angRes_out, args.patch_size_for_test * args.scale_factor,
                              args.stride_for_test * args.scale_factor, Hr_SAI_y.size(-2)//args.angRes_out, Hr_SAI_y.size(-1)//args.angRes_out)
        Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')

        ''' Calculate the PSNR & SSIM '''
        psnr, ssim = cal_metrics(args, Hr_SAI_y, Sr_SAI_y)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        LF_iter_test.append(LF_name[0])


        ''' Save RGB '''
        if save_dir is not None:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            views_dir = save_dir_.joinpath('views')
            views_dir.mkdir(exist_ok=True)
            Sr_SAI_ycbcr = torch.cat((Sr_SAI_y, Sr_SAI_cbcr), dim=1)
            Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0,1)*255).astype('uint8')
            Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)

            # save the SAI
            # path = str(save_dir_) + '/' + LF_name[0] + '_SAI.bmp'
            # imageio.imwrite(path, Sr_SAI_rgb)
            # save the center view
            img = Sr_4D_rgb[args.angRes_out // 2, args.angRes_out // 2, :, :, :]
            path = str(save_dir_) + '/' + LF_name[0] + '_' + 'CenterView.bmp'
            imageio.imwrite(path, img)
            # save all views
            for i in range(args.angRes_out):
                for j in range(args.angRes_out):
                    img = Sr_4D_rgb[i, j, :, :, :]
                    path = str(views_dir) + '/' + LF_name[0] + '_' + str(i) + '_' + str(j) + '.bmp'
                    imageio.imwrite(path, img)
                    pass
                pass
            pass
        pass

    return psnr_iter_test, ssim_iter_test, LF_iter_test

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    from option import args

    main(args)

