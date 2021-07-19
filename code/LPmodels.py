import os
import logging
import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# +
# from data import BatchType, TestDataset
# -

class KGEModel(nn.Module):
    def __init__(self, num_relation, hidden_dim, mode = "HAKE", gamma=12.0, modulus_weight=1.0, phase_weight=0.5):
        super(KGEModel, self).__init__()
#         self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))

        self.pi = 3.14159262358979323846
        self.LPmode = mode

    def func(self, head, rel, tail):
        if self.LPmode == "HAKE":        
            phase_head, mod_head = torch.chunk(head, 2, dim=2)
            phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
            phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

            phase_head = phase_head / (self.embedding_range.item() / self.pi)
            phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
            phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

            phase_score = (phase_head + phase_relation) - phase_tail

            mod_relation = torch.abs(mod_relation)
            bias_relation = torch.clamp(bias_relation, max=1)
            indicator = (bias_relation < -mod_relation)
            bias_relation[indicator] = -mod_relation[indicator]

            r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

            phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
            r_score = torch.norm(r_score, dim=2) * self.modulus_weight

            return self.gamma.item() - (phase_score + r_score)
            
        
        return self.gamma.item() - torch.norm(head * rel - tail, p=1, dim=2)
        
        
        

    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)
    

        
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

        # negative scores
        negative_score = model((positive_sample, negative_sample), batch_type=batch_type)

        positive_sample_loss = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score = model(positive_sample)

        negative_sample_loss = F.logsigmoid(positive_score).squeeze(dim=1)

#         positive_sample_loss = positive_score- (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
#         negative_sample_loss = negative_score- (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        loss.backward()

        optimizer.step()

        log = {
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, data_reader, mode, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        test_dataloader_head = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.HEAD_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.TAIL_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, batch_type in test_dataset:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score = model((positive_sample, negative_sample), batch_type)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if batch_type == BatchType.HEAD_BATCH:
                        positive_arg = positive_sample[:, 0]
                    elif batch_type == BatchType.TAIL_BATCH:
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... ({}/{})'.format(step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics



class HAKE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, modulus_weight=1.0, phase_weight=0.5):
        super(HAKE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))

        self.pi = 3.14159262358979323846


    def func(self, head, rel, tail, batch_type):
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        return self.gamma.item() - (phase_score + r_score)

