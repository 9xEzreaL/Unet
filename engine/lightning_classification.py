import pytorch_lightning as pl
import time, torch
import numpy as np
import torch.nn as nn
import os
from utils.imagesc import imagesc


class LitClassification(pl.LightningModule):
    def __init__(self, args, train_loader, eval_loader, net, loss_function, metrics):
        super().__init__()

        self.hparams.update(args)

        self.args = args
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.net = net
        self.loss_function = loss_function
        self.get_metrics = metrics

        self.optimizer = self.configure_optimizers()

        # parameters to optimize
        for param in self.net.par_freeze:
            param.requires_grad = False
        model_parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        print('Number of parameters: ' + str(sum([np.prod(p.size()) for p in model_parameters])))

        # Begin of training
        self.tini = time.time()
        self.all_label = []
        self.all_out = []
        self.epoch = 0

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=self.args['weight_decay'])
        optimizer = torch.optim.SGD(list(set(self.net.parameters()) - set(self.net.par_freeze)),
                                    lr=self.args['lr'],
                                    momentum=0.9,
                                    weight_decay=self.args['weight_decay'])
        return optimizer

    def training_step(self, batch, batch_idx=0):
        # training_step defined the train loop. It is independent of forward
        imgs, labels, _ = batch
        if (self.args['legacy']) and (not self.args['cpu']):
            imgs = imgs.cuda()
            labels = labels.cuda()
        output = self.net(imgs)
        loss, _ = self.loss_function(output, labels)
        if not self.args['legacy']:
            self.log('train_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        if batch_idx == 5:
            img = torch.cat([(imgs[i,0,::]/imgs[i,0,::].max())*255 for i in range(imgs.shape[0])], 1).detach().cpu()
            label = torch.cat([labels[i,0,::]*255/4 for i in range(labels.shape[0])], 1).detach().cpu()
            masks_probs = output.permute(0, 2, 3, 1)
            _, masks_pred = torch.max(masks_probs, 3)
            pred = torch.cat([masks_pred[i,::]*255/4 for i in range(masks_pred.shape[0])], 1).detach().cpu()
            all = torch.cat([img, pred, label], 0)
            imagesc(all, show=False, save='sample_visualization.png')
        return loss

    def validation_step(self, batch, batch_idx=0):
        imgs, labels, _ = batch
        if (self.args['legacy']) and (not self.args['cpu']):
            imgs = imgs.cuda()
            labels = labels.cuda()
        output = self.net(imgs)
        loss, _ = self.loss_function(output, labels)
        if not self.args['legacy']:
            self.log('val_loss', loss, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)

        # show image and result
        if batch_idx == 5:
            img = torch.cat([imgs[i,0,::]*255 for i in range(imgs.shape[0])], 1).detach().cpu()
            label = torch.cat([labels[i,0,::]*255/4 for i in range(labels.shape[0])], 1).detach().cpu()
            masks_probs = output.permute(0, 2, 3, 1)
            _, masks_pred = torch.max(masks_probs, 3)
            pred = torch.cat([masks_pred[i,::]*255/4 for i in range(masks_pred.shape[0])], 1).detach().cpu()
            all = torch.cat([img, pred, label], 0)
            imagesc(all, show=False, save='images/{}.png'.format(self.epoch))
            self.epoch += 1

        # metrics
        self.all_label.append(labels.cpu())
        # self.all_out.append(output[0].cpu().detach())
        self.all_out.append(output.cpu().detach())
        # metrics = self.get_metrics(labels.cpu(), output[0].cpu().detach())
        return loss

    # def training_epoch_end(self, x):
    def validation_epoch_end(self, x):
        all_out = torch.cat(self.all_out, 0)
        all_label = torch.cat(self.all_label, 0)
        metrics = self.get_metrics(all_label, all_out)
        auc = torch.from_numpy(np.array(metrics)).cuda()
        if not self.args['legacy']:
            for i in range(len(auc)):
                self.log('auc' + str(i), auc[i], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.all_label = []
        self.all_out = []

        self.tini = time.time()
        return metrics

    """ Original Pytorch Code """
    def training_loop(self, train_loader):
        self.net.train(mode=True)
        epoch_loss = 0
        # Loop over the train loader
        for i, batch in enumerate(train_loader):
            loss = self.training_step(batch=batch)
            loss.backward()
            epoch_loss += loss
            if i % (self.args['batch_update'] // self.args['batch_size']) == 0 or i == len(train_loader):
                self.optimizer.step()
                self.optimizer.zero_grad()
        return epoch_loss / i

    def eval_loop(self, eval_loader):
        self.net.train(mode=False)
        self.net.eval()
        epoch_loss = 0
        with torch.no_grad():
            # Loop over the eavluation loader
            for i, batch in enumerate(eval_loader):
                loss = self.validation_step(batch=batch)
                epoch_loss += loss
            metrics = self.validation_epoch_end(x=None)
            return epoch_loss / i, metrics

    def overall_loop(self):

        # We loop over the training epochs
        for epoch in range(self.args['epochs']):
            tini = time.time()
            train_loss = self.training_loop(self.train_loader)
            with torch.no_grad():
                eval_loss, eval_metrics = self.eval_loop(self.eval_loader)

            print_out = {
                'Epoch: {}': [epoch],
                'Time: {:.2f} ': [time.time() - tini],
                'Train Loss: ' + '{:.4f} ': [train_loss],
                'Val Loss: ' + '{:.4f} ': [eval_loss],
                'Metrics: ' + '{:.4f} ' * len(eval_metrics): eval_metrics,
            }

            print(' '.join(print_out.keys()).format(*[j for i in print_out.values() for j in i]))

            if (epoch % 5) == 0:
                os.makedirs(self.args['dir_checkpoint'], exist_ok=True)
                torch.save(self.net, self.args['dir_checkpoint'] + str(epoch) + '.pth')


