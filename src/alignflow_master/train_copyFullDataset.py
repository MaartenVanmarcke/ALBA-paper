'''
A script to train the normalizing flows (FOR MULTIPLE BAGS) with the data in the given dataPlacer.
If this script is run as main file, then the dataPlacer comes from the datA and y_inst fields in dataInput.py.'''


import sys
import os
import pathlib
current = pathlib.Path().parent.absolute()
sys.path.insert(1, os.path.join(current, "alignflow_master"))

import models

from args import TrainArgParser
from dataset import PairedDataset, UnpairedDataset
from evaluation import evaluate
from evaluation.criteria import mse
from logger import TrainLogger
from saver import ModelSaver
from torch.utils.data import DataLoader
from dataReplacer import DataReplacer
from dataInput import datA, y_inst

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

    def visualizeLoss(self, args = None):
        fig, ax = plt.subplots( nrows=1, ncols=1)#, figsize = (15,9) )  # create figure & 1 axis

        for key, val in self.loss_dict.items():
            if len(val)>0:
                ax.plot(range(len(val)), val, label = key)#, c= 'b')

        ax.plot(range(len(self.totalloss)), self.totalloss,  linewidth=2.5, label = "Total Loss")
        plt.title("Loss over the epochs")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        fig.savefig(os.path.join(args.current,'img','loss.png'),bbox_inches='tight')
        plt.close(fig)

    

    def visualize(self, model, data,y_inst, epoch, isNewEpoch: bool = False, args = None):
        dataReplacer = data
        data = dataReplacer.getData()
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
            newdata = [torch.from_numpy(d[:,:-1]) for d in data.values()]
            weights = [torch.from_numpy(d[:,-1]) for d in data.values()]
            model.set_inputs(newdata, weights)
            lats, transfs = model.test()
            #newb = transfs[0][1].numpy()
            D = {}
            w = {}
            i = 0
            for lat in lats:
                D[i] = lat.numpy()
                w[i] = data[i][:,-1]
                i += 1
            if isNewEpoch:
                dataReplacer.setLatent(D)
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

        plt.rcParams.update({'font.size': 18})
        for bag in y_inst.keys():
            domain = D[bag]
            anomalies = []
            normals = []
            D[bag] = np.asarray(domain.tolist())
            s1 = []
            s2 = []
            for idx in range(len(domain)):
                if y_inst[bag][idx] == 1:
                    anomalies.append(domain[idx])
                    s1.append(w[bag][idx])

                else:
                    normals.append(domain[idx])
                    s2.append(w[bag][idx])

            s1 = np.asarray(s1)
            s2 = np.asarray(s2)
            anomalies = np.asarray(anomalies)
            normals = np.asarray(normals)

            c=next(cycol)
            if (len(normals)>0):
                ax.scatter(normals[:,0], normals[:,1], marker='.', c=c, s=250-200*s2, label = "Bag "+str(bag))#, c= 'b')
            if (len(anomalies)>0):
                ax.scatter(anomalies[:,0], anomalies[:,1],  marker='+', c=c, s=250-200*s1, label = "Anomalies bag "+str(bag))#,c= 'b')   

        '''for bag in range(len(bags)):
            domain = bags[bag]
            if (len(domain)>0):
                ax.scatter(domain[:,0], domain[:,1], c=next(cycol))'''
        """if epoch == 0:
            self.ylim = ax.get_ylim()
            self.ylim= (min(-3,self.ylim[0]),max(3,self.ylim[1]))
            self.xlim = ax.get_xlim()
            self.xlim = (min(-3,self.xlim[0]),max(3,self.xlim[1]))
        ax.set_ylim(self.ylim )
        ax.set_xlim(self.xlim)"""
        if epoch == "result":
            plt.title('Aligned 2D Toy Data Set')
        else:
            plt.title('Aligned 2D Toy Data Set - Epoch '+str(epoch))
        #plt.legend()
        fig.savefig(os.path.join(args.current, 'img','epoch'+str(epoch)+'.png'),bbox_inches='tight')
        plt.close(fig)

    def makevideo(self):
        return
        import cv2
        import os

        image_folder = 'alignflow_master/img'
        video_name = 'video.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith(".png") and "loss" not in img]
        images.sort(key=lambda item: (len(item), item))
        print(images)
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(os.path.join("alignflow_master", video_name), 0, 1, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        video.release()
    


def train(args, dataReplacer: DataReplacer, y_inst):
    # Get model
    model = models.__dict__[args.model](args)
    if args.modelload and os.path.exists(os.path.join(args.save_dir, "best.pth.tar")):
        print("Model loaded.")
        model = ModelSaver.load_model(model, os.path.join(args.save_dir, "best.pth.tar"), args.gpu_ids, is_training=True)
    else:
        print("Starting new model")
    model = model.to(args.device)
    model.train()

    '''a_file = os.path.join(current, "data", "testdata", "testA.csv")
    a_data = np.genfromtxt(a_file, delimiter=',', dtype=np.float32)
    b_file = os.path.join(current, "data", "testdata", "testB.csv")
    b_data = np.genfromtxt(b_file, delimiter=',', dtype=np.float32)
    '''


    # Get loader, logger, and saver
    train_loader, val_loader = get_data_loaders(args, dataReplacer.getData())
    logger = TrainLogger(args, model, dataset_len=len(train_loader.dataset))
    saver = ModelSaver(args.save_dir, args.max_ckpts, metric_name=args.metric_name,
                       maximize_metric=args.maximize_metric, keep_topk=True)
    
    vis = Visualizer()
    #vis.visualize(None, datA, y_inst, 0)
    vis.visualize(model, dataReplacer, y_inst, 0, True, args = args)

    last = np.inf
    print(torch.cuda.is_available())
    import time
    xxx  =time.time()
    print("START", xxx)

    summm = 0
    thedata = None
    # Train
    while not logger.is_finished_training():
        logger.start_epoch()
        if thedata == None:
            thedata = []
            for batch in train_loader:
                thedata.append(batch)
        for batch in thedata:
            logger.start_iter()

            # Train over one batch
            model.set_inputs([batch[i][:,:-1] for i in range(args.num_sources)], [batch[i][:,-1] for i in range(args.num_sources)])
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
        vis.visualize(model, dataReplacer, y_inst, logger.epoch, True, args = args)

        logger.end_epoch()  
    print("END", time.time()-xxx)
    print("summm", summm)
    vis.makevideo()
    vis.visualizeLoss(args = args)


def get_data_loaders(args,data):
    train_dataset = UnpairedDataset(data,#args.data_dir,
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


def main(dataReplacer:DataReplacer, y_inst, load: bool = False):
    current = pathlib.Path(__file__).parent.resolve()
    parser = TrainArgParser()
    parser.model = "Flow2Flow"
    parser.crop_shape = (128, 128)
    parser.resize_shape = (144, 144)
    parser.data_dir = os.path.join(current, "data", "testdata")
    parser.gpu_ids = []#[0]#[0,1,2,3]
    parser.iters_per_visual =320 
    parser.norm_type= "instance" 
    parser.num_blocks= np.NaN#0#4 
    parser.num_channels_g = np.NaN#1#32 
    parser.num_scales= 2 
    parser.use_dropout= False 
    parser.use_mixer =True 
    parser.num_channels = np.NaN#1
    parser.num_channels_d = np.NaN#1
    parser.initializer = "normal"
    parser.kernel_size_d = np.NaN#1
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
    parser.num_sources = dataReplacer.getNumSources()
    ##
    parser.modelload = load
    parser.name = "normalaligner"
    parser.num_epochs = 200#300
    parser.features = 2
    parser.model = "Flow2Flow"
    parser.batch_size = 30# 30#16
    parser.iters_per_print= parser.batch_size
    parser.lr =.005#.005# 2e-4
    parser.rnvp_lr =.005#.005# 2e-4
    parser.lambda_mle = 1. # 1e-4
    parser.epochs_per_print = 5


    # Set up available GPUs
    if len(parser.gpu_ids) > 0:
        # Set default GPU for `tensor.to('cuda')`
        torch.cuda.set_device(parser.gpu_ids[0])
        parser.device = 'cuda:{}'.format(parser.gpu_ids[0])
    else:
        parser.device = 'cpu'

    # Set up save dir and output dir (test mode only)
    parser.save_dir = os.path.join(parser.checkpoints_dir, parser.name)
    os.makedirs(parser.save_dir, exist_ok=True)
    if parser.is_training:
        '''with open(os.path.join(parser.save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(parser), fh, indent=4, sort_keys=True)
            fh.write('\n')'''
    else:
        parser.results_dir = os.path.join(parser.results_dir, parser.name)
        os.makedirs(parser.results_dir, exist_ok=True)

    if not (isinstance(y_inst, dict)):
        res = {}
        idx = 0
        for k, v in dataReplacer.getData().items():
            res[k] = y_inst[idx:idx+len(v[:,0])]
            idx += len(v[:,0])
    else:
        res = y_inst


    train(parser, dataReplacer, res)
    
    model = models.__dict__[parser.model](parser)
    model = ModelSaver.load_model(model, os.path.join(parser.save_dir, "best.pth.tar"), parser.gpu_ids, is_training=True)
    #model.train()
    Visualizer().visualize(model, dataReplacer, res, "result", True, args = parser)


if __name__ == '__main__':
    np.random.seed(1302)

    ff = np.zeros((90,2))
    for i in range(len(ff)):
            ff[i,:] = np.random.uniform(-1,1,size = ((1,2)))
            while ff[i,1]>-.3 and abs(ff[i,0]) < .5:
                ff[i,:] = np.random.uniform(-1,1,size = ((1,2)))

    dd = np.hstack((np.random.uniform(-2,-1.25,size=((100,1))), np.random.uniform(-1.5,0.5, size = ((100,1)))))

    ## TODO: delete this
    datA = {0: np.concatenate((ff, np.hstack((np.random.uniform(-.25,.25,size=((10,1))), np.random.uniform(0,1,size = ((10,1))))))), 1: dd}
    y_inst = {0: np.concatenate((np.zeros((90)), np.ones((10)))), 1: np.zeros((100))}
    weights = {0: np.concatenate((np.zeros((90)), 0.9*np.ones((10)))), 1: np.zeros((100))}


    dataReplacer = DataReplacer(num_sources=len(datA))
    dataReplacer.setInitData(datA)


    ## TODO delete this
    dataReplacer.setWeights(weights)

    main(dataReplacer, y_inst, load = False)


    D = datA
    fig, ax = plt.subplots( nrows=1, ncols=1, figsize = (15,9) )  # create figure & 1 axis
    clrs = ['b','g','r','c','m','k','y', 'lime','deeppink','aqua','yellow','gray','darkorange','saddlebrown','salmon']
    from itertools import cycle
    cycol = cycle(clrs)
    w = weights
    
    tick_font_size = 18
    plt.rcParams.update({'font.size': 18})

    for bag in y_inst.keys():
        domain = D[bag]
        anomalies = []
        normals = []
        D[bag] = np.asarray(domain.tolist())
        s1 = []
        s2 = []
        for idx in range(len(domain)):
            if y_inst[bag][idx] == 1:
                anomalies.append(domain[idx])
                s1.append(w[bag][idx])

            else:
                normals.append(domain[idx])
                s2.append(w[bag][idx])

        s1 = np.asarray(s1)
        s2 = np.asarray(s2)
        anomalies = np.asarray(anomalies)
        normals = np.asarray(normals)

        c=next(cycol)
        if (len(normals)>0):
            ax.scatter(normals[:,0], normals[:,1], marker='.', c=c, s=250-200*s2, label = "Bag "+str(bag))#, c= 'b')
        if (len(anomalies)>0):
            ax.scatter(anomalies[:,0], anomalies[:,1],  marker='+', c=c, s=250-200*s1, label = "Anomalies bag"+str(bag))#,c= 'b')   
        
        plt.title('Generated 2D Toy Data Set')
        fig.savefig(os.path.join(pathlib.Path(__file__).parent.resolve(), 'img','originalSimple.png'),bbox_inches='tight')
        plt.close(fig)

        