import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib
import matplotlib.pyplot as plt
import time

matplotlib.use('Agg')

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(args, optimizer, iter_num, max_iter):
    decay = (1 + args.lr_gamma * iter_num / max_iter) ** (-args.lr_power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
        
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        # transforms.RandomCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
    
def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)
    
    return dset_loaders

def gmm(all_fea, pi, mu, all_output):    
    Cov = []
    dist = []
    log_probs = []
    
    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:,i].unsqueeze(dim=-1)
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / (predi.sum()) + args.epsilon * torch.eye(temp.shape[1]).cuda()
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += args.epsilon * torch.eye(temp.shape[1]).cuda() * 100
            chol = torch.linalg.cholesky(Covi)
        chol_inv = torch.inverse(chol)
        Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
        logdet = torch.logdet(Covi)
        mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
        log_prob = -0.5*(Covi.shape[0] * np.log(2*math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        Cov.append(Covi)
        log_probs.append(log_prob)
        dist.append(mah_dist)
    Cov = torch.stack(Cov, dim=0)
    dist = torch.stack(dist, dim=0).t()
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    gamma = torch.exp(zz)
    
    return zz, gamma

def evaluation(loader, netF, netB, netC, args, cnt):
    start_test = True
    iter_test = iter(loader)
    for _ in tqdm(range(len(loader))):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1].cuda()
        inputs = inputs.cuda()
        feas = netB(netF(inputs))
        outputs = netC(feas)
        if start_test:
            all_fea = feas.float()
            all_output = outputs.float()
            all_label = labels.float()
            start_test = False
        else:
            all_fea = torch.cat((all_fea, feas.float()), 0)
            all_output = torch.cat((all_output, outputs.float()), 0)
            all_label = torch.cat((all_label, labels.float()), 0)
            
    _, predict = torch.max(all_output, 1)
    accuracy_return = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).data.item()

    if args.dset=='VISDA-C':
        matrix = confusion_matrix(all_label.cpu().numpy(), torch.squeeze(predict).float().cpu().numpy())
        acc_return = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc_return.mean()
        aa = [str(np.round(i, 2)) for i in acc_return]
        acc_return = ' '.join(aa)

    all_output_logit = all_output
    all_output = nn.Softmax(dim=1)(all_output)
    all_fea_orig = all_fea
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon2), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float()
    K = all_output.shape[1]
    aff = all_output.float()
    initc = torch.matmul(aff.t(), (all_fea))
    initc = initc / (1e-8 + aff.sum(dim=0)[:,None])

    if args.pickle and (cnt==0):
        data = {
            'all_fea': all_fea,
            'all_output': all_output,
            'all_label': all_label,
            'all_fea_orig': all_fea_orig,
        }
        filename = osp.join(args.output_dir, 'data_{}'.format(args.names[args.t]) + args.prefix + '.pickle')
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('data_{}.pickle finished\n'.format(args.names[args.t]))
        
        
    ############################## Gaussian Mixture Modeling #############################

    uniform = torch.ones(len(all_fea),args.class_num)/args.class_num
    uniform = uniform.cuda()

    pi = all_output.sum(dim=0)
    mu = torch.matmul(all_output.t(), (all_fea))
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

    zz, gamma = gmm((all_fea), pi, mu, uniform)
    pred_label = gamma.argmax(dim=1)
    
    for round in range(1):
        pi = gamma.sum(dim=0)
        mu = torch.matmul(gamma.t(), (all_fea))
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

        zz, gamma = gmm((all_fea), pi, mu, gamma)
        pred_label = gamma.argmax(axis=1)
            
    aff = gamma
    
    acc = (pred_label==all_label).float().mean()
    log_str = 'Model Prediction : Accuracy = {:.2f}%'.format(accuracy * 100) + '\n'

    if args.dset=='VISDA-C':
        log_str += 'VISDA-C classwise accuracy : {:.2f}%\n{}'.format(aacc, acc_return) + '\n'

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)
    
    ############################## Computing JMDS score #############################

    sort_zz = zz.sort(dim=1, descending=True)[0]
    zz_sub = sort_zz[:,0] - sort_zz[:,1]
    
    LPG = zz_sub / zz_sub.max()

    if args.coeff=='JMDS':
        PPL = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
        JMDS = (LPG * PPL)
    elif args.coeff=='PPL':
        JMDS = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
    elif args.coeff=='NO':
        JMDS=torch.ones_like(LPG)
    else:
        JMDS = LPG

    sample_weight = JMDS

    if args.dset=='VISDA-C':
        return aff, sample_weight, aacc/100
    return aff, sample_weight, accuracy
    
def KLLoss(input_, target_, coeff, args):
    softmax = nn.Softmax(dim=1)(input_)
    kl_loss = (- target_ * torch.log(softmax + args.epsilon2)).sum(dim=1)
    kl_loss *= coeff
    return kl_loss.mean(dim=0)

def mixup(x, c_batch, t_batch, netF, netB, netC, args):
    # weight mixup
    if args.alpha==0:
        outputs = netC(netB(netF(x)))
        return KLLoss(outputs, t_batch, c_batch, args)
    lam = (torch.from_numpy(np.random.beta(args.alpha, args.alpha, [len(x)]))).float().cuda()
    t_batch = torch.eye(args.class_num)[t_batch.argmax(dim=1)].cuda()
    shuffle_idx = torch.randperm(len(x))
    mixed_x = (lam * x.permute(1,2,3,0) + (1 - lam) * x[shuffle_idx].permute(1,2,3,0)).permute(3,0,1,2)
    mixed_c = lam * c_batch + (1 - lam) * c_batch[shuffle_idx]
    mixed_t = (lam * t_batch.permute(1,0) + (1 - lam) * t_batch[shuffle_idx].permute(1,0)).permute(1,0)
    mixed_x, mixed_c, mixed_t = map(torch.autograd.Variable, (mixed_x, mixed_c, mixed_t))
    mixed_outputs = netC(netB(netF(mixed_x)))
    return KLLoss(mixed_outputs, mixed_t, mixed_c, args)

def train_target(args):
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    ####################################################################
    modelpath = args.output_dir_src + '/source_F_{}.pt'.format(args.seed)
    print('modelpath: {}'.format(modelpath))
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B_{}.pt'.format(args.seed)
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C_{}.pt'.format(args.seed)
    netC.load_state_dict(torch.load(modelpath))
        
    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    
    for k, v in netC.named_parameters():
        if args.lr_decay3 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay3}]
        else:
            v.requires_grad = False
    
    resize_size = 256
    crop_size = 224
    augment1 = transforms.Compose([
        # transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
    ])
            
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    cnt = 0

    dset_loaders = data_load(args)
    
    epochs = []
    accuracies = []
    
    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        # Compute JMDS score at offline & evaluation.
        soft_pseudo_label, coeff, accuracy = evaluation(
            dset_loaders["test"], netF, netB, netC, args, cnt
        )
        epochs.append(cnt)
        accuracies.append(np.round(accuracy*100, 2))
    netF.train()
    netB.train()
    netC.train()
    
    uniform_ent = np.log(args.class_num)
    
    
    
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // (args.interval)
    iter_num = 0
    
    print('\nTraining start\n')
    while iter_num < max_iter:
        try:
            inputs_test, label, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, label, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue
        
        iter_num += 1
        lr_scheduler(args, optimizer, iter_num=iter_num, max_iter=max_iter)
        pred = soft_pseudo_label[tar_idx]
        pred_label = pred.argmax(dim=1)
        
        coeff, pred = map(torch.autograd.Variable, (coeff, pred))
        images1 = torch.autograd.Variable(augment1(inputs_test))
        images1 = images1.cuda()
        coeff = coeff.cuda()
        pred = pred.cuda()
        pred_label = pred_label.cuda()
        
        CoWA_loss = mixup(images1, coeff[tar_idx], pred, netF, netB, netC, args)
        
        # For warm up the start.
        if iter_num < args.warm * interval_iter + 1:
            CoWA_loss *= 1e-6
            
        optimizer.zero_grad()
        CoWA_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            print('Evaluation iter:{}/{} start.'.format(iter_num, max_iter))
            log_str = 'Task: {}, Iter:{}/{};'.format(args.name, iter_num, max_iter)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
            
            netF.eval()
            netB.eval()
            netC.eval()
            
            cnt += 1
            with torch.no_grad():
                # Compute JMDS score at offline & evaluation.
                soft_pseudo_label, coeff, accuracy = evaluation(dset_loaders["test"], netF, netB, netC, args, cnt)
                epochs.append(cnt)
                accuracies.append(np.round(accuracy*100, 2))

            print('Evaluation iter:{}/{} finished.\n'.format(iter_num, max_iter))
            netF.train()
            netB.train()
            netC.train()

    ####################################################################
    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, 'ckpt_F_' + args.prefix + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, 'ckpt_B_' + args.prefix + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, 'ckpt_C_' + args.prefix + ".pt"))
        
        
    log_str = '\nAccuracies history : {}\n'.format(accuracies)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(epochs, accuracies, 'o-')
    plt.savefig(osp.join(args.output_dir,'png_{}.png'.format(args.prefix)))
    plt.close()
    
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'DomainNet'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")
 
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--warm', type=float, default=0.0)
    parser.add_argument('--coeff', type=str, default='LPG', choices=['LPG', 'JMDS', 'PPL','NO'])
    parser.add_argument('--pickle', default=False, action='store_true')
    parser.add_argument('--lr_gamma', type=float, default=10.0)
    parser.add_argument('--lr_power', type=float, default=0.75)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--lr_decay3', type=float, default=0.1)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--epsilon2', type=float, default=1e-6)
    parser.add_argument('--delta', type=float, default=2.0)
    parser.add_argument('--n_power', type=int, default=1)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        args.names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        args.names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        args.names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        args.names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'DomainNet':
        args.names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.class_num = 345
        
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    
    ############# If you want to obtain the stochastic result, comment following lines. #############
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)
    
    for i in range(len(args.names)):
        start = time.time()
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' +args.names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + args.names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + args.names[args.t] + '_list.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, args.names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, args.names[args.s][0].upper()+args.names[args.t][0].upper())
        args.name = args.names[args.s][0].upper()+args.names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)
        
        args.prefix = '{}_alpha{}_lr{}_epoch{}_interval{}_seed{}_warm{}'.format(
            args.coeff, args.alpha, args.lr, args.max_epoch, args.interval, args.seed, args.warm
        )
        
        ####################################################################
        if not osp.exists(osp.join(args.output_dir, 'ckpt_F_' + args.prefix + ".pt")):
            args.out_file = open(osp.join(args.output_dir, 'log' + args.prefix + '.txt'), 'w')
            args.out_file.write(print_args(args)+'\n')
            args.out_file.flush()
            train_target(args)

            total_time = time.time() - start
            log_str = 'Consumed time : {} h {} m {}s'.format(total_time // 3600, (total_time // 60) % 60, np.round(total_time % 60, 2))
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
        else:
            print('{} Already exists'.format(osp.join(args.output_dir, 'log' + args.prefix + '.txt')))
