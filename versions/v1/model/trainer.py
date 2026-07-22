import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter, AverageMeterSet, Ranker, create_optimizer_and_scheduler
from models.modules import confidence_fusion, kl_divergence, symmetric_kl

class Trainer:
    def __init__(self, args, accelerator, model, train_loader, dev_loader, test_loader):
        self.args = args
        self.logger = args.logger
        self.accelerator = accelerator
        num_train_optimization_steps = (len(train_loader) * args.num_train_epochs) // args.gradient_accumulation_steps
        optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)
        self.model, self.train_loader, self.dev_loader, self.test_loader, self.optimizer, self.scheduler = accelerator.prepare(model, train_loader, dev_loader, test_loader, optimizer, scheduler)
        
        if self.accelerator.is_local_main_process:
            self.writer = SummaryWriter(f'{args.output_path}')
        self.best_epoch = None
        self.best_validation_metric = None
        self.last_test_metrics = None

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
    
    def predict(self, epoch, data_loader=None):
        
        if data_loader is None:
            data_loader = self.dev_loader

        ranker = Ranker(self.args.metric_ks)
        average_meter_set = AverageMeterSet()
        
        self.model.eval()
        with torch.no_grad():
            res_users = []
            # u1_attention = []
            # u1_query_token_indices = []
            # u1_id_indices = []
            # u2_attention = []
            # u2_query_token_indices = []
            # u2_id_indices = []
            for batch in tqdm(data_loader, ncols=100, desc='Evaluate', disable=(not self.accelerator.is_local_main_process)):
                input_ids, attention_mask, _, labels = batch["user_seq_data"]
                # context_mask = batch["context_mask"]
                item_input_ids, item_seq_mask, item_target_ids = batch["item_data"]
                self.accelerator.unwrap_model(self.model).set_adapter("lora_text")
                scores, _ = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        adapter_name="lora_text"
                        )
                if self.args.use_gate:
                    if self.args.alternating_learning > 0:
                        self.accelerator.unwrap_model(self.model).set_adapter("lora_cf")
                    scores1, _ = self.model(
                            input_ids=item_input_ids,
                            attention_mask=item_seq_mask,
                            adapter_name="lora_cf",
                            is_text=False
                            )
                    scores = confidence_fusion(scores, scores1, self.args.fusion_temperature, self.args.fusion_alpha, self.args.fusion_type)
                elif self.args.late_fusion:
                    interactions = batch["interactions"]
                    scores1, _ = self.model(
                        interactions=interactions,
                        is_text=False
                    )
                    scores = confidence_fusion(scores, scores1, self.args.fusion_temperature, self.args.fusion_alpha, self.args.fusion_type)
                scores, labels = self.accelerator.gather_for_metrics((scores, labels))
                
                user_ids = batch["user_ids"]
                user_ids = self.accelerator.gather_for_metrics((user_ids))
                
                # query_token_indices, id_indices = batch["query_token_indices"], batch["id_indices"]
                # user_ids1 = 
                # user_ids2 = 
                # for i, user_id in enumerate(user_ids):
                #     if user_id in user_ids1:
                #         u1_attention.append(attentions[i])
                #         u1_query_token_indices.append(query_token_indices[i])
                #         u1_id_indices.append(id_indices[i])
                #     if user_id in user_ids2:
                #         u2_attention.append(attentions[i])
                #         u2_query_token_indices.append(query_token_indices[i])
                #         u2_id_indices.append(id_indices[i])
                res, batch_res_users = ranker(scores, labels, user_ids)
                res_users.extend(batch_res_users)

                metrics = {}
                for i, k in enumerate(self.args.metric_ks):
                    metrics["NDCG@%d" % k] = res[2*i]
                    metrics["Recall@%d" % k] = res[2*i+1]
                metrics["MRR"] = res[-2]
                # metrics["AUC"] = res[-2]

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
            # if data_loader != self.dev_loader and self.accelerator.is_local_main_process:
            #     torch.save(u1_attention, os.path.join(self.args.output_path, 'u1_attention.pt'))
            #     torch.save(u2_attention, os.path.join(self.args.output_path, 'u2_attention.pt'))
            #     torch.save(u1_query_token_indices, os.path.join(self.args.output_path, 'u1_query_token_indices.pt'))
            #     torch.save(u2_query_token_indices, os.path.join(self.args.output_path, 'u2_query_token_indices.pt'))
            #     torch.save(u1_id_indices, os.path.join(self.args.output_path, 'u1_id_indices.pt'))
            #     torch.save(u2_id_indices, os.path.join(self.args.output_path, 'u2_id_indices.pt'))
            
        average_metrics = average_meter_set.averages()
        # if data_loader != self.dev_loader and self.accelerator.is_local_main_process:
        #     self.logger.info(f'good user nums: {len(res_users)}, user_ids: {res_users}')
        return average_metrics
    
    def train_one_epoch(self, epoch):

        self.model.train()
        
        loss_meter = AverageMeter()
        if self.args.use_gate or self.args.late_fusion:
            loss_meter_cf = AverageMeter()
            loss_meter_kl = AverageMeter()
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=100, disable=(not self.accelerator.is_local_main_process))
        
        for step, batch in pbar:
            with self.accelerator.accumulate(self.model):
                if not self.args.use_gate and not self.args.late_fusion: # only id/text or early fusion
                    input_ids, attention_mask, target_ids, labels = batch["user_seq_data"]
                    self.model.module.set_adapter("lora_text")
                    pooled_logits, target_ids = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=target_ids,
                            )
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(pooled_logits.view(-1, pooled_logits.size(-1)), target_ids.view(-1))
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    loss_meter.update(loss.item())
                    pbar.set_description(f"Epoch: {epoch + 1}, train loss: {loss_meter.avg:.5f}, lr: {self.scheduler.get_last_lr()[0]:.7f}")
                elif self.args.late_fusion:
                    for cl_step in range(self.args.alternating_learning):
                        loss_fct = nn.CrossEntropyLoss()
                        # text
                        input_ids, attention_mask, target_ids, labels = batch["user_seq_data"]
                        text_logits, pool_target_ids = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=target_ids,
                                )
                        if cl_step > 0:
                            if self.args.fusion_before_loss:
                                fusion_logits = confidence_fusion(text_logits, cf_logits, self.args.fusion_temperature, self.args.fusion_alpha, self.args.fusion_type)
                                text_loss = loss_fct(fusion_logits.view(-1, fusion_logits.size(-1)), pool_target_ids.view(-1))
                            else:
                                text_loss = loss_fct(text_logits.view(-1, text_logits.size(-1)), pool_target_ids.view(-1))
                            loss_meter.update(text_loss.item())
                            # kl loss
                            if self.args.kl_loss_weight > 0:
                                kl_loss = kl_divergence(cf_logits[:, 1:], text_logits, self.args.kl_temperature)
                                text_loss += self.args.kl_loss_weight * kl_loss
                                loss_meter_kl.update(kl_loss.item())

                            self.accelerator.backward(text_loss)
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                        text_logits = text_logits.detach()
                        
                        if cl_step != self.args.alternating_learning - 1:
                            # 
                            interactions = batch["interactions"]
                            cf_logits, pool_item_target_ids = self.model(
                                labels=target_ids, # Placeholder for training
                                interactions=interactions,
                                is_text=False
                            )
                            if self.args.fusion_before_loss:
                                fusion_logits = confidence_fusion(text_logits, cf_logits, self.args.fusion_temperature, self.args.fusion_alpha, self.args.fusion_type)
                                cf_loss = loss_fct(fusion_logits.view(-1, fusion_logits.size(-1)), pool_item_target_ids.view(-1))
                            else:
                                cf_loss = loss_fct(cf_logits.view(-1, cf_logits.size(-1)), pool_item_target_ids.view(-1))
                            loss_meter_cf.update(cf_loss.item())
                            # kl loss
                            if self.args.kl_loss_weight > 0:
                                kl_loss = kl_divergence(text_logits, cf_logits[:, 1:], self.args.kl_temperature)
                                cf_loss += self.args.kl_loss_weight * kl_loss
                                loss_meter_kl.update(kl_loss.item())
                            self.accelerator.backward(cf_loss)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            cf_logits = cf_logits.detach()
                    self.scheduler.step()
                    pbar.set_description(f"Epoch: {epoch + 1}, text_l: {loss_meter.avg:.4f}, cf_l: {loss_meter_cf.avg:.4f}, kl_l: {loss_meter_kl.avg:.4f}, lr: {self.scheduler.get_last_lr()[0]*1e5:.2f}*1e-5")
                elif self.args.use_gate and self.args.alternating_learning == 0: # no use alternating learning(only one LoRA)
                    loss_fct = nn.CrossEntropyLoss()
                    input_ids, attention_mask, target_ids, labels = batch["user_seq_data"]
                    text_logits, target_ids = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=target_ids,
                            )
                    loss = loss_fct(text_logits.view(-1, text_logits.size(-1)), target_ids.view(-1))
                    loss_meter.update(loss.item())
                    item_input_ids, item_seq_mask, item_target_ids = batch["item_data"]
                    cf_logits, item_target_ids = self.model(
                            input_ids=item_input_ids,
                            attention_mask=item_seq_mask,
                            labels=item_target_ids,
                            is_text=False
                            )
                    loss_cf = loss_fct(cf_logits.view(-1, cf_logits.size(-1)), item_target_ids.view(-1))
                    loss_meter_cf.update(loss_cf.item())
                    # kl loss
                    if self.args.kl_loss_weight > 0:
                        kl_loss = symmetric_kl(cf_logits, text_logits, self.args.kl_temperature)
                        loss += self.args.kl_loss_weight * kl_loss
                        loss_meter_kl.update(kl_loss.item())
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    pbar.set_description(f"Epoch: {epoch + 1}, text_l: {loss_meter.avg:.4f}, cf_l: {loss_meter_cf.avg:.4f}, kl_l: {loss_meter_kl.avg:.4f}, lr: {self.scheduler.get_last_lr()[0]*1e5:.2f}*1e-5")
                else: # use alternating learning
                    for cl_step in range(self.args.alternating_learning):
                        loss_fct = nn.CrossEntropyLoss()
                        # text
                        input_ids, attention_mask, target_ids, labels = batch["user_seq_data"]
                        self.accelerator.unwrap_model(self.model).set_adapter("lora_text")
                        text_logits, pool_target_ids = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=target_ids,
                                adapter_name="lora_text"
                                )
                        if cl_step > 0:
                            text_loss = loss_fct(text_logits.view(-1, text_logits.size(-1)), pool_target_ids.view(-1))
                            loss_meter.update(text_loss.item())
                            # kl loss
                            if self.args.kl_loss_weight > 0:
                                kl_loss = kl_divergence(cf_logits, text_logits, self.args.kl_temperature)
                                text_loss += self.args.kl_loss_weight * kl_loss
                                loss_meter_kl.update(kl_loss.item())

                            self.accelerator.backward(text_loss)
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                        text_logits = text_logits.detach()
                        
                        if cl_step != self.args.alternating_learning - 1:
                            # cf
                            item_input_ids, item_seq_mask, item_target_ids = batch["item_data"]
                            self.accelerator.unwrap_model(self.model).set_adapter("lora_cf")
                            cf_logits, pool_item_target_ids = self.model(
                                input_ids=item_input_ids,
                                attention_mask=item_seq_mask,
                                labels=item_target_ids,
                                adapter_name="lora_cf",
                                is_text=False
                            )
                            cf_loss = loss_fct(cf_logits.view(-1, cf_logits.size(-1)), pool_item_target_ids.view(-1))
                            loss_meter_cf.update(cf_loss.item())
                            # kl loss
                            if self.args.kl_loss_weight > 0:
                                kl_loss = kl_divergence(text_logits, cf_logits, self.args.kl_temperature)
                                cf_loss += self.args.kl_loss_weight * kl_loss
                                loss_meter_kl.update(kl_loss.item())
                            self.accelerator.backward(cf_loss)
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                            cf_logits = cf_logits.detach()
                    self.scheduler.step()
                    pbar.set_description(f"Epoch: {epoch + 1}, text_l: {loss_meter.avg:.4f}, cf_l: {loss_meter_cf.avg:.4f}, kl_l: {loss_meter_kl.avg:.4f}, lr: {self.scheduler.get_last_lr()[0]*1e5:.2f}*1e-5")
        
        if self.args.use_gate or self.args.late_fusion:
            self.logger.info(f'Epoch: {epoch + 1}, training loss: {loss_meter.avg:.5f}, cf_loss: {loss_meter_cf.avg:.5f}, kl_loss: {loss_meter_kl.avg:.5f}, lr: {self.scheduler.get_last_lr()[0]:.7f}')
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
        self.last_test_metrics = test_metrics
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
                    self.best_epoch = epoch + 1
                    self.best_validation_metric = best_target
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
