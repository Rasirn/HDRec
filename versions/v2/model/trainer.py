import os
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter, AverageMeterSet, Ranker, create_optimizer_and_scheduler
from models.modules import confidence_fusion, kl_divergence, symmetric_kl
from flylora import set_flylora_task


class Trainer:
    def __init__(self, args, accelerator, model, train_loader, dev_loader, test_loader):
        self.args = args
        self.logger = args.logger
        self.accelerator = accelerator
        num_train_optimization_steps = (len(train_loader) * args.num_train_epochs) // args.gradient_accumulation_steps
        optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)
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
        state_dict = unwrapped_model.state_dict()
        self.accelerator.save(state_dict, save_path)

    def load_checkpoints(self, load_path):
        self.accelerator.wait_for_everyone()
        self.logger.info(f'Load best model: {load_path}')
        if dist.is_initialized():
            self.model.module.load_state_dict(torch.load(load_path, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(load_path, map_location='cpu'))

    def _forward_text(self, input_ids, attention_mask, target_ids=None):
        set_flylora_task(self.model, 'text')
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids, is_text=True)

    def _forward_cf(self, batch, target_ids=None):
        if self.args.use_gate:
            set_flylora_task(self.model, 'cf')
            # When use_small_model is False, negative item ids in item_data cannot be embedded.
            # Fallback to user_seq_data while keeping cf head/task active.
            if self.args.use_small_model:
                item_input_ids, item_seq_mask, item_target_ids = batch['item_data']
                # Eval path should not pass labels, otherwise logits are flattened by valid positions.
                labels = None if target_ids is None else item_target_ids
                return self.model(
                    input_ids=item_input_ids,
                    attention_mask=item_seq_mask,
                    labels=labels,
                    is_text=False,
                )

            input_ids, attention_mask, _, _ = batch['user_seq_data']
            labels = target_ids
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                is_text=False,
            )

        interactions = batch['interactions']
        set_flylora_task(self.model, 'cf')
        labels = target_ids
        return self.model(labels=labels, interactions=interactions, is_text=False)

    def predict(self, epoch, data_loader=None):
        if data_loader is None:
            data_loader = self.dev_loader

        ranker = Ranker(self.args.metric_ks)
        average_meter_set = AverageMeterSet()

        self.model.eval()
        with torch.no_grad():
            res_users = []
            for batch in tqdm(data_loader, ncols=100, desc='Evaluate', disable=(not self.accelerator.is_local_main_process)):
                input_ids, attention_mask, _, labels = batch['user_seq_data']
                scores, _ = self._forward_text(input_ids, attention_mask)

                if self.args.use_gate or self.args.late_fusion:
                    scores_cf, _ = self._forward_cf(batch)
                    scores = confidence_fusion(
                        scores,
                        scores_cf,
                        self.args.fusion_temperature,
                        self.args.fusion_alpha,
                        self.args.fusion_type,
                    )

                scores, labels = self.accelerator.gather_for_metrics((scores, labels))
                user_ids = batch['user_ids']
                user_ids = self.accelerator.gather_for_metrics((user_ids))

                res, batch_res_users = ranker(scores, labels, user_ids)
                res_users.extend(batch_res_users)

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
        loss_meter_cf = AverageMeter()
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

                text_logits, pool_target_ids = self._forward_text(input_ids, attention_mask, target_ids)
                text_loss = loss_fct(text_logits.view(-1, text_logits.size(-1)), pool_target_ids.view(-1))
                loss_meter.update(text_loss.item())

                # Stage-1 backward (text) to avoid keeping text/cf graphs in memory simultaneously.
                self.accelerator.backward(text_loss)
                text_logits_detached = text_logits.detach()

                if self.args.use_gate or self.args.late_fusion:
                    cf_logits, pool_item_target_ids = self._forward_cf(batch, target_ids=target_ids)

                    if self.args.fusion_before_loss:
                        # This branch keeps both branches coupled at loss level and may consume more memory.
                        fusion_logits = confidence_fusion(
                            text_logits_detached,
                            cf_logits,
                            self.args.fusion_temperature,
                            self.args.fusion_alpha,
                            self.args.fusion_type,
                        )
                        text_loss = loss_fct(fusion_logits.view(-1, fusion_logits.size(-1)), pool_target_ids.view(-1))

                        if self.args.late_fusion:
                            cf_target = pool_item_target_ids.view(-1)
                            cf_loss = loss_fct(fusion_logits.view(-1, fusion_logits.size(-1)), cf_target)
                        else:
                            cf_loss = loss_fct(fusion_logits.view(-1, fusion_logits.size(-1)), pool_item_target_ids.view(-1))
                    else:
                        cf_loss = loss_fct(cf_logits.view(-1, cf_logits.size(-1)), pool_item_target_ids.view(-1))

                    loss_meter_cf.update(cf_loss.item())

                    total_cf_loss = self.args.cf_loss_weight * cf_loss

                    if self.args.kl_loss_weight > 0:
                        if self.args.late_fusion:
                            kl_loss = kl_divergence(text_logits_detached, cf_logits[:, 1:], self.args.kl_temperature)
                        else:
                            kl_loss = kl_divergence(text_logits_detached, cf_logits, self.args.kl_temperature)
                        total_cf_loss = total_cf_loss + self.args.kl_loss_weight * kl_loss
                        loss_meter_kl.update(kl_loss.item())

                    self.accelerator.backward(total_cf_loss)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                pbar.set_description(
                    f"Epoch: {epoch + 1}, text_l: {loss_meter.avg:.4f}, cf_l: {loss_meter_cf.avg:.4f}, "
                    f"kl_l: {loss_meter_kl.avg:.4f}, lr: {self.scheduler.get_last_lr()[0] * 1e5:.2f}*1e-5"
                )

        if self.args.use_gate or self.args.late_fusion:
            self.logger.info(
                f'Epoch: {epoch + 1}, training loss: {loss_meter.avg:.5f}, '
                f'cf_loss: {loss_meter_cf.avg:.5f}, kl_loss: {loss_meter_kl.avg:.5f}, '
                f'lr: {self.scheduler.get_last_lr()[0]:.7f}'
            )
        else:
            self.logger.info(f'Epoch: {epoch + 1}, training loss: {loss_meter.avg:.5f}, lr: {self.scheduler.get_last_lr()[0]:.7f}')

        if self.accelerator.is_local_main_process:
            self.writer.add_scalar('train/loss', loss_meter.avg, epoch)
            self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], epoch)
            if self.args.use_gate or self.args.late_fusion:
                self.writer.add_scalar('train/cf_loss', loss_meter_cf.avg, epoch)
                self.writer.add_scalar('train/kl_loss', loss_meter_kl.avg, epoch)

    def evaluate(self):
        if not self.args.only_test:
            save_path = os.path.join(self.args.output_path, 'pytorch_model.bin')
            self.load_checkpoints(load_path=save_path)
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
