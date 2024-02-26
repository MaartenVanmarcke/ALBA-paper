'''
A script to train the normalizing flows with the data FOR 2 BAGS in alignflow_master\\data\\testdata\\trainA.csv (bag 1) and trainB.csv (bag 2).
Specify on line 71 the labels of the data in that csv file for nice plotting in the alignflow_master\\img\\ folder. 
(for plotting, the data in testA.csv and testB.csv is used) '''

import models

from args import TrainArgParser
from dataset import PairedDataset, UnpairedDataset
from evaluation import evaluate
from evaluation.criteria import mse
from logger import TrainLogger
from saver import ModelSaver
from torch.utils.data import DataLoader

import torch
import os
import pathlib
import json
import matplotlib.pyplot as plt 
import numpy as np

class Args(object):
    def __init__(self) -> None:
        super().__init__()

class Visualizer():
    def __init__(self) -> None:
        self.loss_dict = {
            # Generator loss
            'loss_gan': [],
            'loss_jc': [],
            'loss_mle': [],
            'loss_g': [],

            # Discriminator loss
            #'loss_d_src': [],
            #'loss_d_tgt': [],
            'loss_d': []
        }
        self.totalloss = []

    def addLoss(self, loss_dict):
        total = 0
        for key, val in loss_dict.items():
            if key in self.loss_dict.keys():
                self.loss_dict[key].append(val)
                if key == "loss_g" or key == "loss_d":
                    total += val
        self.totalloss.append(total)

    def visualizeLoss(self):
        fig, ax = plt.subplots( nrows=1, ncols=1)#, figsize = (15,9) )  # create figure & 1 axis

        for key, val in self.loss_dict.items():
            if len(val)>0:
                ax.plot(range(len(val)), val, label = key)#, c= 'b')

        ax.plot(range(len(self.totalloss)), self.totalloss,  linewidth=2.5, label = "Total Loss")
        plt.title("Loss over the epochs")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        fig.savefig('img/loss.png',bbox_inches='tight')
        plt.close(fig)

    def visualize(self, model, a_data, b_data, epoch, parser):
        y_inst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        y_inst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y_inst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        y_inst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

        
        '''fig, ax = plt.subplots(2,3)
        ax[0,0].scatter(a_data[:,0], a_data[:,1])
        ax[0,0].set_title("original A")
        ax[0,0].set_xlabel("Mean: "+str(np.mean(a_data[:,0])))
        ax[0,0].set_ylabel("Mean: "+str(np.mean(a_data[:,1])))
        ax[1,0].scatter(b_data[:,0], b_data[:,1])
        ax[1,0].set_title("original B")
        ax[1,0].set_xlabel("Mean: "+str(np.mean(b_data[:,0])))
        ax[1,0].set_ylabel("Mean: "+str(np.mean(b_data[:,1])))'''
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
        
        D = {0:lata, 1: latb}
        y_inst = {0: y_inst[:round(len(y_inst)/2)], 1: y_inst[round(len(y_inst)/2):]}
        color = ["b","g"]

        fig, ax = plt.subplots( nrows=1, ncols=1, figsize = (15,9) )  # create figure & 1 axis

        for bag in range(2):
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
            
            if (len(normals)>0):
                ax.scatter(normals[:,0], normals[:,1], marker='.', c=color[bag], s=200, label = "Bag "+str(bag))#, c= 'b')
            if (len(anomalies)>0):
                ax.scatter(anomalies[:,0], anomalies[:,1],  marker='+', c=color[bag], s=200, label = "Anomalies")#,c= 'b')   

        '''for bag in range(len(bags)):
            domain = bags[bag]
            if (len(domain)>0):
                ax.scatter(domain[:,0], domain[:,1], c=next(cycol))'''
        print(epoch)
        if epoch == 0:
            self.ylim = ax.get_ylim()
            self.ylim= (min(-3,self.ylim[0]),max(3,self.ylim[1]))
            self.xlim = ax.get_xlim()
            self.xlim = (min(-3,self.xlim[0]),max(3,self.xlim[1]))
        ax.set_ylim(self.ylim )
        ax.set_xlim(self.xlim)
        plt.title('Aligned 2D Toy Data Set - Epoch '+str(epoch))
        plt.legend()
        fig.savefig(os.path.join(parser.current,'img','epoch'+str(epoch)+'.png'),bbox_inches='tight')
        plt.close(fig)

    def makevideo(self):
        import cv2
        import os

        image_folder = 'alignflow-master/img'
        video_name = 'video.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith(".png") and "loss" not in img]
        images.sort(key=lambda item: (len(item), item))
        print(images)
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(os.path.join("alignflow-master", video_name), 0, 1, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        video.release()
    


def train(args):
    # Get model
    model = models.__dict__[args.model](args)
    '''if args.ckpt_path:
        model = ModelSaver.load_model(model, args.ckpt_path, args.gpu_ids, is_training=True)'''
    model = model.to(args.device)
    model.train()

    a_file = os.path.join(current, "data", "testdata", "testA.csv")
    a_data = np.genfromtxt(a_file, delimiter=',', dtype=np.float32)
    b_file = os.path.join(current, "data", "testdata", "testB.csv")
    b_data = np.genfromtxt(b_file, delimiter=',', dtype=np.float32)

    # Get loader, logger, and saver
    train_loader, val_loader = get_data_loaders(args)
    logger = TrainLogger(args, model, dataset_len=len(train_loader.dataset))
    saver = ModelSaver(args.save_dir, args.max_ckpts, metric_name=args.metric_name,
                       maximize_metric=args.maximize_metric, keep_topk=True)
    
    vis = Visualizer()
    vis.visualize(model, a_data, b_data, 0, parser)

    last = np.inf

    # Train
    while not logger.is_finished_training():
        logger.start_epoch()
        for batch in train_loader:
            logger.start_iter()
            # Train over one batch
            model.set_inputs([batch[0], batch[1]])
            model.train_iter()

            logger.end_iter()

            # Evaluate
            #if logger.global_step % args.iters_per_eval < args.batch_size:
                #criteria = {'MSE_src2tgt': mse, 'MSE_tgt2src': mse}
                #stats = evaluate(moSdel, val_loader, criteria)
                #logger.log_scalars({'val_' + k: v for k, v in stats.items()})
                #saver.save(logger.global_step, model,
                           #0, args.device)
            if  (logger.loss_dict["loss_g"] <= last):
                last = logger.loss_dict["loss_g"]
                saver.save(logger.global_step, model,
                            0, args.device)
                
        vis.addLoss(logger.loss_dict)
        vis.visualize(model, a_data, b_data, logger.epoch, parser)

        logger.end_epoch()
    vis.makevideo()
    vis.visualizeLoss()


def get_data_loaders(args):
    train_dataset = UnpairedDataset(args.data_dir,
                                    phase='train',
                                    shuffle_pairs=True,
                                    direction=args.direction)
    train_loader = DataLoader(train_dataset,
                              args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

    val_dataset = PairedDataset(args.data_dir,
                                phase='val',
                                resize_shape=args.resize_shape,
                                crop_shape=args.crop_shape,
                                direction=args.direction)
    val_loader = DataLoader(val_dataset,
                            args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            drop_last=True)

    return train_loader, val_loader


if __name__ == '__main__':
    current = pathlib.Path(__file__).parent.resolve()
    parser = TrainArgParser()
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
    ##
    parser.name = "mmoons"
    parser.num_epochs = 300
    parser.num_sources = 2
    parser.features = 2
    parser.model = "Flow2Flow"
    parser.batch_size = 300#16
    parser.iters_per_print= parser.batch_size
    parser.lr =.005#.005# 2e-4
    parser.rnvp_lr =.005#.005# 2e-4
    parser.lambda_mle = 1. # 1.

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
    if parser.is_training:
        '''with open(os.path.join(parser.save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(parser), fh, indent=4, sort_keys=True)
            fh.write('\n')'''
    else:
        parser.results_dir = os.path.join(parser.results_dir, parser.name)
        os.makedirs(parser.results_dir, exist_ok=True)



    train(parser)
