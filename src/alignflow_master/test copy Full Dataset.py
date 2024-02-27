'''UNUSED'''


import json
import models
import numpy as np
import os
import shutil
import torch.nn as nn

from args import TestArgParser
from dataset.test_dataset import TestDataset
from datetime import datetime
from evaluation import evaluate
from evaluation.criteria import mse
from PIL import Image
from saver import ModelSaver
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from util import make_grid, un_normalize


import pathlib, torch
import matplotlib.pyplot as plt

class Visualizer():   

    def visualize(self, model, data, y_inst, epoch):
        clrs = ['b','g','r','c','m','k','y', 'lime','deeppink','aqua','yellow','gray','darkorange','saddlebrown','salmon']
        from itertools import cycle
        cycol = cycle(clrs)
        
        '''fig, ax = plt.subplots(2,3)
        ax[0,0].scatter(a_data[:,0], a_data[:,1])
        ax[0,0].set_title("original A")
        ax[0,0].set_xlabel("Mean: "+str(np.mean(a_data[:,0])))
        ax[0,0].set_ylabel("Mean: "+str(np.mean(a_data[:,1])))
        ax[1,0].scatter(b_data[:,0], b_data[:,1])
        ax[1,0].set_title("original B")
        ax[1,0].set_xlabel("Mean: "+str(np.mean(b_data[:,0])))
        ax[1,0].set_ylabel("Mean: "+str(np.mean(b_data[:,1])))'''
        if model != None:
            print("Testing")
            newdata = [torch.from_numpy(d) for d in data.values()]
            model.set_inputs(newdata)
            lats, transfs = model.test()
            #newb = transfs[0][1].numpy()
            D = {}
            i = 0
            for lat in lats:
                D[i] = lat.numpy()
                i += 1
        else: 
            D = data
        #newa =  transfs[1][0].numpy()
        #latb = lats[1].numpy()
        '''reb = transfs[1][1].numpy()
        lata = lats[0].numpy()
        rea =  transfs[0][0].numpy()
        latb = lats[1].numpy()'''

        '''for batch in data_loader:
            model.set_inputs(batch['src'], batch['tgt'])
            newb, newa = model.test()'''
        '''ax[0,1].scatter(lata[:,0], lata[:,1])
        ax[0,1].set_title("A->latent")
        ax[0,1].set_xlabel("Mean: "+str(np.mean(lata[:,0])))
        ax[0,1].set_ylabel("Mean: "+str(np.mean(lata[:,1])))
        ax[1,1].scatter(latb[:,0], latb[:,1])
        ax[1,1].set_title("B->latent")
        ax[1,1].set_xlabel("Mean: "+str(np.mean(latb[:,0])))
        ax[1,1].set_ylabel("Mean: "+str(np.mean(latb[:,1])))
        ax[0,2].scatter(newb[:,0], newb[:,1])
        ax[0,2].set_title("A->latent->B")
        ax[0,2].set_xlabel("Mean: "+str(np.mean(newb[:,0])))
        ax[0,2].set_ylabel("Mean: "+str(np.mean(newb[:,1])))
        ax[1,2].scatter(newa[:,0], newa[:,1])
        ax[1,2].set_title("B->latent->A")
        ax[1,2].set_xlabel("Mean: "+str(np.mean(newa[:,0])))
        ax[1,2].set_ylabel("Mean: "+str(np.mean(newa[:,1])))'''
        '''ax[0,3].scatter(rea[:,0], rea[:,1])
        ax[0,3].set_title("A->latent->A")
        ax[0,3].set_xlabel("Mean: "+str(np.mean(rea[:,0])))
        ax[1,3].scatter(reb[:,0], reb[:,1])
        ax[1,3].set_title("B->latent->B")
        ax[1,3].set_xlabel("Mean: "+str(np.mean(reb[:,0])))'''
        '''plt.tight_layout()
        plt.show()'''
        

        fig, ax = plt.subplots( nrows=1, ncols=1, figsize = (15,9) )  # create figure & 1 axis

        for bag in y_inst.keys():
            domain = D[bag]
            anomalies = []
            normals = []
            D[bag] = np.asarray(domain.tolist())

            for idx in range(len(domain)):
                if y_inst[bag][idx] == 1:
                    anomalies.append(domain[idx])
                else:
                    normals.append(domain[idx])

            anomalies = np.asarray(anomalies)
            normals = np.asarray(normals)

            c=next(cycol)
            if (len(normals)>0):
                ax.scatter(normals[:,0], normals[:,1], marker='.', c=c, s=200, label = "Bag "+str(bag))#, c= 'b')
            if (len(anomalies)>0):
                ax.scatter(anomalies[:,0], anomalies[:,1],  marker='+', c=c, s=200, label = "Anomalies")#,c= 'b')   

        '''for bag in range(len(bags)):
            domain = bags[bag]
            if (len(domain)>0):
                ax.scatter(domain[:,0], domain[:,1], c=next(cycol))'''
        print(epoch)
        plt.title('Aligned 2D Toy Data Set - Epoch '+str(epoch))
        plt.legend()
        fig.savefig(os.path.join(parser.current, "img", "epoch"+str(epoch)+'.png'),bbox_inches='tight')
        plt.close(fig)


def test2(args):
    # Get dataset
    '''dataset = TestDataset(args.data_dir,
                            phase=args.phase,
                            direction=args.direction)
    data_loader = DataLoader(dataset,
                             args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)'''

    # Get model
    model : nn.Module= models.__dict__[args.model](args)
    
    model = ModelSaver.load_model(model, args.ckpt_path, args.gpu_ids)
    model.train()

    ''' # Set up image saving
    if args.save_images is None:
        save_hook = None
    else:
        saver = ImageSaver(args.save_images, args.results_dir, args.name, args.phase)
        save_hook = saver.save'''

    '''# Test model
    criteria = {'MSE_src2tgt': mse, 'MSE_tgt2src': mse}
    stats = evaluate(model, data_loader, criteria, batch_hook=save_hook)'''

    '''# Add model info to stats
    stats.update({
        'name': args.name,
        'ckpt_path': args.ckpt_path
    })'''

    '''# Write stats to disk
    stats_path = os.path.join(args.results_dir, 'stats.json')
    print('Saving stats to {}...'.format(stats_path))
    with open(stats_path, 'w') as json_fh:
        json.dump(stats, json_fh, sort_keys=True, indent=4)

    # Copy training args for reference
    args_src = os.path.join(args.save_dir, 'args.json')
    args_dst = os.path.join(args.results_dir, 'args.json')
    print('Copying args to {}...'.format(args_dst))
    shutil.copy(args_src, args_dst)'''

    a_file = os.path.join(current, "data", "testdata", "testA.csv")
    a_data = np.genfromtxt(a_file, delimiter=',', dtype=np.float32)
    b_file = os.path.join(current, "data", "testdata", "testB.csv")
    b_data = np.genfromtxt(b_file, delimiter=',', dtype=np.float32)

    ix, ax, iy, ay = min(a_data[:,0]), max(a_data[:,0]), min(a_data[:,1]),max(a_data[:,1])
    zax = np.arange(np.floor(ix), np.ceil(ax), (np.ceil(ax)-np.floor(ix))/10)
    zay = np.arange(np.floor(iy), np.ceil(ay), (np.ceil(ay)-np.floor(iy))/10)
    ix, ax, iy, ay = min(b_data[:,0]), max(b_data[:,0]), min(b_data[:,1]),max(b_data[:,1])
    zbx = np.arange(np.floor(ix), np.ceil(ax), (np.ceil(ax)-np.floor(ix))/10)
    zby =  np.arange(np.floor(iy), np.ceil(ay), (np.ceil(ay)-np.floor(iy))/10)

    fig, ax = plt.subplots(2,3)
    ax[0,0].scatter(a_data[:,0], a_data[:,1])
    ax[0,0].set_title("original A")
    ax[0,0].set_xlabel("Mean: "+str(np.mean(a_data[:,0])))
    ax[0,0].set_ylabel("Mean: "+str(np.mean(a_data[:,1])))
    ax[1,0].scatter(b_data[:,0], b_data[:,1])
    ax[1,0].set_title("original B")
    ax[1,0].set_xlabel("Mean: "+str(np.mean(b_data[:,0])))
    ax[1,0].set_ylabel("Mean: "+str(np.mean(b_data[:,1])))
    newa = torch.from_numpy(a_data)
    newb = torch.from_numpy(b_data)
    model.set_inputs([newa, newb])
    lats, transfs = model.test()
    newb = transfs[0][1].numpy()
    lata = lats[0].numpy()
    newa =  transfs[1][0].numpy()
    latb = lats[1].numpy()
    '''reb = transfs[1][1].numpy()
    lata = lats[0].numpy()
    rea =  transfs[0][0].numpy()
    latb = lats[1].numpy()'''

    '''for batch in data_loader:
        model.set_inputs(batch['src'], batch['tgt'])
        newb, newa = model.test()'''
    ax[0,1].scatter(lata[:,0], lata[:,1])
    ax[0,1].set_title("A->latent")
    ax[0,1].set_xlabel("Mean: "+str(np.mean(lata[:,0])))
    ax[0,1].set_ylabel("Mean: "+str(np.mean(lata[:,1])))
    ax[1,1].scatter(latb[:,0], latb[:,1])
    ax[1,1].set_title("B->latent")
    ax[1,1].set_xlabel("Mean: "+str(np.mean(latb[:,0])))
    ax[1,1].set_ylabel("Mean: "+str(np.mean(latb[:,1])))
    ax[0,2].scatter(newb[:,0], newb[:,1])
    ax[0,2].set_title("A->latent->B")
    ax[0,2].set_xlabel("Mean: "+str(np.mean(newb[:,0])))
    ax[0,2].set_ylabel("Mean: "+str(np.mean(newb[:,1])))
    ax[1,2].scatter(newa[:,0], newa[:,1])
    ax[1,2].set_title("B->latent->A")
    ax[1,2].set_xlabel("Mean: "+str(np.mean(newa[:,0])))
    ax[1,2].set_ylabel("Mean: "+str(np.mean(newa[:,1])))
    '''ax[0,3].scatter(rea[:,0], rea[:,1])
    ax[0,3].set_title("A->latent->A")
    ax[0,3].set_xlabel("Mean: "+str(np.mean(rea[:,0])))
    ax[1,3].scatter(reb[:,0], reb[:,1])
    ax[1,3].set_title("B->latent->B")
    ax[1,3].set_xlabel("Mean: "+str(np.mean(reb[:,0])))'''
    plt.tight_layout()
    plt.show()

    Xa = np.array(zax).repeat(10)
    Ya = np.array([zay.copy() for _ in range(10)]).flatten()
    newa = torch.from_numpy(np.array(np.vstack((Xa, Ya)).T, dtype = np.float32))
    Xb= np.array(zbx).repeat(10)
    Yb = np.array([zby.copy() for _ in range(10)]).flatten()
    newb = torch.from_numpy(np.array(np.vstack((Xb, Yb)).T,dtype = np.float32))
    model.set_inputs([newa, newb])
    lats, transfs = model.test()
    newb = transfs[0][1].numpy()
    lata_ = lats[0].numpy()
    newa =  transfs[1][0].numpy()
    latb_ = lats[1].numpy()

    fig, ax = plt.subplots(2,2)
    for i in range(10):
        ax[0,0].plot(Xa[10*i:10*(i+1)], Ya[10*i:10*(i+1)], c= "b")
        ax[0,0].plot(Xa[[10*k+i for k in range(10)]], Ya[[10*k+i for k in range(10)]], c= "b")
        ax[1,0].plot(Xb[10*i:10*(i+1)], Yb[10*i:10*(i+1)], c= "b")
        ax[1,0].plot(Xb[[10*k+i for k in range(10)]], Yb[[10*k+i for k in range(10)]], c= "b")
        ax[0,1].plot(lata_[10*i:10*(i+1),0], lata_[10*i:10*(i+1),1], c= "b")
        ax[0,1].plot(lata_[[10*k+i for k in range(10)],0], lata_[[10*k+i for k in range(10)],1], c= "b")
        ax[1,1].plot(latb_[10*i:10*(i+1),0], latb_[10*i:10*(i+1),1], c= "b")
        ax[1,1].plot(latb_[[10*k+i for k in range(10)],0], latb_[[10*k+i for k in range(10)],1], c= "b")
    ax[0,0].scatter(a_data[:,0], a_data[:,1], c= "orange")
    ax[1,0].scatter(b_data[:,0], b_data[:,1], c= "orange")
    ax[0,1].scatter(lata[:,0], lata[:,1], c= "orange")
  
    ax[1,1].scatter(latb[:,0], latb[:,1], c= "orange")

    plt.tight_layout()
    plt.show()

def test(args):
    # Get model
    model : nn.Module= models.__dict__[args.model](args)
    
    model = ModelSaver.load_model(model, args.ckpt_path, args.gpu_ids)
    model.train()

    from dataInput import datA, y_inst

    Visualizer().visualize(model, datA, y_inst, "result")

    



'''class ImageSaver(object):
    """Saver for logging images during testing.

    Set `saver = ImageSaver(...)`, pass `saver.save` as the `batch_hook`
    argument to `evaluate`. Then every batch will get saved.

    Args:
        save_format (str): One of 'tensorboard' or 'disk'.
        results_dir (str): Directory for saving output images (disk only).
        name (str): Experiment name for saving to disk or TensorBoard.
        phase (str): One of 'train', 'val', or 'test'.
    """
    def __init__(self, save_format, results_dir, name, phase):
        self.phase = phase
        if save_format == 'tensorboard':
            log_dir = 'logs/{}_{}_{}'\
                .format(name, phase, datetime.now().strftime('%b%d_%H%M'))
            self.summary_writer = SummaryWriter(log_dir=log_dir)
        elif save_format == 'disk':
            self.summary_writer = None
            self.save_dir = os.path.join(results_dir, 'images')
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            raise ValueError('Invalid save format: {}'.format(save_format))
        self.global_step = 1

    def save(self, src, src2tgt, tgt, tgt2src):
        # Un-normalize
        src, src2tgt = un_normalize(src), un_normalize(src2tgt)
        tgt, tgt2src = un_normalize(tgt), un_normalize(tgt2src)

        # Make grid of images
        i = 0
        for s, s2t, t, t2s in zip(src, src2tgt, tgt, tgt2src):
            # Save image
            if self.summary_writer is None:
                images_concat = make_grid([s, s2t], nrow=2, padding=0)
                file_name = 'src2tgt_{:04d}.png'.format(self.global_step)
                self._write(images_concat, file_name)

                images_concat = make_grid([t, t2s], nrow=2, padding=0)
                file_name = 'tgt2src_{:04d}.png'.format(self.global_step)
                self._write(images_concat, file_name)
            else:
                images_concat = make_grid([s, s2t], nrow=2, padding=4, pad_value=(243, 124, 42))
                self.summary_writer.add_image('src/src_src2tgt_{}'.format(i), images_concat, self.global_step)

                images_concat = make_grid([t, t2s], nrow=2, padding=4, pad_value=(243, 124, 42))
                self.summary_writer.add_image('tgt/tgt_tgt2src_{}'.format(i), images_concat, self.global_step)

            i += 1
            self.global_step += 1

    def _write(self, img, img_name):
        img_path = os.path.join(self.save_dir, img_name)
        img_np = np.transpose(img.numpy(), [1, 2, 0])
        img = Image.fromarray(img_np)
        img.save(img_path)
'''

if __name__ == '__main__':
    current = pathlib.Path(__file__).parent.resolve()
    parser = TestArgParser()
    parser.model = "Flow2Flow"
    parser.crop_shape = (128, 128)
    parser.resize_shape = (144, 144)
    parser.data_dir = os.path.join(current, "data", "testdata")
    parser.gpu_ids = []#[0]#[0,1,2,3]
    parser.iters_per_visual =320 
    parser.norm_type= "instance" 
    parser.num_blocks= 0#4 
    parser.num_channels_g = 1#32 
    parser.num_scales= 2 
    parser.use_dropout= False 
    parser.use_mixer =True 
    parser.num_channels = 1
    parser.num_channels_d = 1
    parser.initializer = "normal"
    parser.kernel_size_d = 1
    parser.iters_per_visual = 256
    parser.iters_per_eval = 100
    parser.max_ckpts = 5
    parser.metric_name = "MSE_src2tgt"
    parser.maximize_metric = False
    parser.lambda_src = 10.
    parser.lambda_tgt = 10.
    parser.lambda_id = 0.5
    parser.lambda_l1 = 100.
    parser.beta_1 = .5
    parser.beta_2 = .999
    parser.rnvp_beta_1 = .5
    parser.rnvp_beta_2 = .999
    parser.weight_norm_l2 = 5e-5
    parser.lr_policy = "linear"
    parser.lr_step_epochs = 100
    parser.lr_warmup_epochs = 100
    parser.lr_decay_epochs = 100
    parser.num_visuals = 4
    parser.clip_gradient = 0.
    parser.clamp_jacobian = False
    parser.jc_lambda_min = 1.
    parser.jc_lambda_max = 20.
    parser.checkpoints_dir = os.path.join(current, "ckpts")
    parser.ckpt_path = ""
    parser.kernel_size_d = 4
    parser.norm_type = "instance"
    parser.num_workers = 1#16
    parser.phase = "train"
    parser.use_dropout=False
    parser.current = current
    parser.direction = "ab"
    parser.num_sources = 2 #dataReplacer.getNumSources()
    ##
    parser.name = "normalaligner"
    parser.num_epochs = 300
    parser.features = 2
    parser.model = "Flow2Flow"
    parser.batch_size = 150#16
    parser.iters_per_print= parser.batch_size
    parser.lr =.005#.005# 2e-4
    parser.rnvp_lr =.005#.005# 2e-4
    parser.lambda_mle = 1. # 1.
    parser.is_training = False


    # Set up available GPUs
    if len(parser.gpu_ids) > 0:
        # Set default GPU for `tensor.to('cuda')`
        torch.cuda.set_device(parser.gpu_ids[0])
        parser.device = 'cuda:{}'.format(parser.gpu_ids[0])
    else:
        parser.device = 'cpu'

    # Set up save dir and output dir (test mode only)
    parser.save_dir = os.path.join(current, parser.checkpoints_dir, parser.name)
    os.makedirs(parser.save_dir, exist_ok=True)
    '''if parser.is_training:
        with open(os.path.join(parser.save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(parser), fh, indent=4, sort_keys=True)
            fh.write('\n')
    else:
        parser.results_dir = os.path.join(parser.results_dir, parser.name)
        os.makedirs(parser.results_dir, exist_ok=True)'''
    parser.ckpt_path =  os.path.join(parser.save_dir, "best.pth.tar")




    test(parser)
