from time import time
from logger.base_logger import BaseLogger


class TrainLogger(BaseLogger):
    """Logs training info."""
    def __init__(self, args, model, dataset_len):
        super(TrainLogger, self).__init__(args, model, dataset_len)

        assert args.is_training
        assert args.iters_per_print % args.batch_size == 0, \
            'iters_per_print must be divisible by batch_size'

        self.batch_size = args.batch_size
        self.iters_per_print = args.iters_per_print
        self.epochs_per_print = args.epochs_per_print
        self.num_visuals = args.num_visuals
        self.num_epochs = args.num_epochs

    def start_iter(self, src_filenames=None):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

        # Periodically write to the log and TensorBoard
        if self.global_step % self.iters_per_print == 0:
            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / self.batch_size
            message = '[epoch: {}, iter: {} / {}, time: {:.2f}] '\
                      .format(self.epoch, self.iter, self.dataset_len, avg_time)

            # Write the current error report
            self.loss_dict = self.model.get_loss_dict()
            loss_keys = ['loss_g', 'loss_gan', 'loss_mle', 'loss_d']  # Can add other losses here (e.g., 'loss_g_l1').
            for k in loss_keys:
                if k not in self.loss_dict.keys():
                    loss_keys.remove(k)
            loss_strings = ['{}: {:.3g}'.format(k, self.loss_dict[k]) for k in loss_keys]
            message += ', '.join(loss_strings)

            # Write all errors as scalars to the graph
            for k, v in self.loss_dict.items():
                # Group generator and discriminator losses
                if k.startswith('loss_d'):
                    k = 'd/' + k
                else:
                    k = 'g/' + k
                #self.summary_writer.add_scalar(k, v, self.global_step)

            if (self.epoch) % self.epochs_per_print == 0 or self.epoch == 1:
                self.write(message)

        

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        if (self.epoch) % self.epochs_per_print == 0 or self.epoch == 1:
            self.write('[start of epoch {}]'.format(self.epoch))

    def end_epoch(self):
        """Log info for end of an epoch. Save model parameters and update learning rate."""
        
        if (self.epoch) % self.epochs_per_print == 0 or self.epoch == 1:
            self.write('[end of epoch {}, epoch time: {:.2f}]'.format(self.epoch, time() - self.epoch_start_time))

        # Update the learning rate according to the LR schedulers
        self.model.on_epoch_end()
        learning_rate = self.model.get_learning_rate()
        #self.summary_writer.add_scalar('hpm/lr', learning_rate, self.global_step)

        self.epoch += 1

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch
