import os
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter, AverageMeterSet, Ranker, create_optimizer_and_scheduler
from models.modules import kl_divergence
from flylora_dual import set_dual_flylora_task


class Trainer:
    def __init__(self, args, accelerator, model, train_loader, dev_loader, test_loader):
        self.args = args
        self.logger = args.logger
        self.accelerator = accelerator

        num_steps = (len(train_loader) * args.num_train_epochs) // args.gradient_accumulation_steps
        optimizer, scheduler = create_optimizer_and_scheduler(model, num_steps, args)
        self.model, self.train_loader, self.dev_loader, self.test_loader, self.optimizer, self.scheduler = accelerator.prepare(
            model,
            train_loader,
            dev_loader,
            test_loader,
            optimizer,
            scheduler,
        )

        if self.accelerator.is_local_main_process:
            self.writer = SummaryWriter(f'{args.output_path}')

    def save_checkpoints(self, save_path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(unwrapped_model.state_dict(), save_path)

    def load_checkpoints(self, load_path):
        self.accelerator.wait_for_everyone()
        self.logger.info(f'Load best model: {load_path}')
        if dist.is_initialized():
            self.model.module.load_state_dict(torch.load(load_path, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(load_path, map_location='cpu'))

    def _forward_text(self, input_ids, attention_mask, target_ids=None):
        set_dual_flylora_task(self.model, 'fused')
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids, is_text=True)

    def _forward_id(self, batch, target_ids=None):
        set_dual_flylora_task(self.model, 'fused')
        if self.args.use_gate:
            item_input_ids, item_seq_mask, item_target_ids = batch['item_data']
            labels = None if target_ids is None else item_target_ids
            return self.model(
                input_ids=item_input_ids,
                attention_mask=item_seq_mask,
                labels=labels,
                is_text=False,
            )

        interactions = batch['interactions']
        return self.model(labels=target_ids, interactions=interactions, is_text=False)

    def predict(self, epoch, data_loader=None):
        if data_loader is None:
            data_loader = self.dev_loader

        ranker = Ranker(self.args.metric_ks)
        average_meter_set = AverageMeterSet()

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, ncols=100, desc='Evaluate', disable=(not self.accelerator.is_local_main_process)):
                input_ids, attention_mask, _, labels = batch['user_seq_data']
                scores, _ = self._forward_text(input_ids, attention_mask)

                scores, labels = self.accelerator.gather_for_metrics((scores, labels))
                user_ids = self.accelerator.gather_for_metrics((batch['user_ids']))

                res, _ = ranker(scores, labels, user_ids)
                metrics = {}
                for i, k in enumerate(self.args.metric_ks):
                    metrics[f'NDCG@{k}'] = res[2 * i]
                    metrics[f'Recall@{k}'] = res[2 * i + 1]
                metrics['MRR'] = res[-2]

                for k, v in metrics.items():
                    average_meter_set.update(k, v)

        return average_meter_set.averages()

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        loss_meter_id = AverageMeter()
        loss_meter_kl = AverageMeter()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            ncols=100,
            disable=(not self.accelerator.is_local_main_process),
        )

        for _, batch in pbar:
            with self.accelerator.accumulate(self.model):
                loss_fct = nn.CrossEntropyLoss()
                input_ids, attention_mask, target_ids, _ = batch['user_seq_data']

                text_logits, text_labels = self._forward_text(input_ids, attention_mask, target_ids)
                text_loss = loss_fct(text_logits.view(-1, text_logits.size(-1)), text_labels.view(-1))

                id_loss = torch.zeros((), device=text_logits.device, dtype=text_logits.dtype)
                bi_kl = torch.zeros((), device=text_logits.device, dtype=text_logits.dtype)

                if self.args.use_gate or self.args.late_fusion:
                    id_logits, id_labels = self._forward_id(batch, target_ids=target_ids)
                    id_loss = loss_fct(id_logits.view(-1, id_logits.size(-1)), id_labels.view(-1))

                    text_detached = text_logits.detach()
                    id_detached = id_logits.detach()

                    if self.args.late_fusion:
                        kl_text = kl_divergence(id_detached[:, 1:], text_logits, self.args.kl_temperature)
                        kl_id = kl_divergence(text_detached, id_logits[:, 1:], self.args.kl_temperature)
                    else:
                        kl_text = kl_divergence(id_detached, text_logits, self.args.kl_temperature)
                        kl_id = kl_divergence(text_detached, id_logits, self.args.kl_temperature)

                    bi_kl = 0.5 * (kl_text + kl_id)

                loss = text_loss + self.args.cf_loss_weight * id_loss
                if self.args.kl_loss_weight > 0:
                    loss = loss + self.args.kl_loss_weight * bi_kl

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                loss_meter.update(text_loss.item())
                loss_meter_id.update(id_loss.item())
                loss_meter_kl.update(bi_kl.item())

                pbar.set_description(
                    f'Epoch: {epoch + 1}, text_l: {loss_meter.avg:.4f}, id_l: {loss_meter_id.avg:.4f}, '
                    f'kl_l: {loss_meter_kl.avg:.4f}, lr: {self.scheduler.get_last_lr()[0] * 1e5:.2f}*1e-5'
                )

        self.logger.info(
            f'Epoch: {epoch + 1}, training loss: {loss_meter.avg:.5f}, '
            f'id_loss: {loss_meter_id.avg:.5f}, kl_loss: {loss_meter_kl.avg:.5f}, '
            f'lr: {self.scheduler.get_last_lr()[0]:.7f}'
        )

        if self.accelerator.is_local_main_process:
            self.writer.add_scalar('train/loss_text', loss_meter.avg, epoch)
            self.writer.add_scalar('train/loss_id', loss_meter_id.avg, epoch)
            self.writer.add_scalar('train/loss_kl', loss_meter_kl.avg, epoch)
            self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], epoch)

    def evaluate(self):
        if not self.args.only_test:
            self.load_checkpoints(os.path.join(self.args.output_path, 'pytorch_model.bin'))
        test_metrics = self.predict(self.args.num_train_epochs, data_loader=self.test_loader)
        self.logger.info(f'==Test set==: {test_metrics}\n')
        if self.accelerator.is_local_main_process:
            self.writer.add_scalar('test/NDCG@10', test_metrics['NDCG@10'])
            self.writer.add_scalar('test/Recall@10', test_metrics['Recall@10'])
            self.writer.add_scalar('test/MRR', test_metrics['MRR'])
        return test_metrics[self.args.valid_metric]

    def train(self):
        best_target = float('-inf')
        patient = self.args.patient

        if self.args.only_test:
            return self.evaluate()

        for epoch in range(self.args.num_train_epochs):
            self.train_one_epoch(epoch)

            if (epoch + 1) % self.args.save_interval == 0:
                self.logger.info(f'Save model epoch {epoch + 1}')
                save_path = os.path.join(self.args.output_path, f'epoch_{epoch + 1}.bin')
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_local_main_process:
                    self.save_checkpoints(save_path)

            if (epoch + 1) % self.args.interval == 0 and (epoch + 1) > self.args.skip_valid:
                dev_metrics = self.predict(epoch)
                self.logger.info(f'Epoch: {epoch + 1}. Dev set: {dev_metrics}')

                if self.accelerator.is_local_main_process:
                    self.writer.add_scalar('dev/NDCG@10', dev_metrics['NDCG@10'], epoch)
                    self.writer.add_scalar('dev/Recall@10', dev_metrics['Recall@10'], epoch)
                    self.writer.add_scalar('dev/MRR', dev_metrics['MRR'], epoch)

                if dev_metrics[self.args.valid_metric] > best_target:
                    best_target = dev_metrics[self.args.valid_metric]
                    patient = self.args.patient
                    self.logger.info('Save the best model.')
                    save_path = os.path.join(self.args.output_path, 'pytorch_model.bin')
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_local_main_process:
                        self.save_checkpoints(save_path)
                else:
                    patient -= 1
                    if patient == 0:
                        break

        return self.evaluate()
