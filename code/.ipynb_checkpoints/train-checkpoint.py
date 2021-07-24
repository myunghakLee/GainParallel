from LPmodels import KGEModel, HAKE
import torch.nn.functional as F




# +
def p_score(vector, subsampling_weight, mode="HAKE"):
    positive_score = F.logsigmoid(vector).squeeze(dim=1)
    positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
    return positive_sample_loss

def n_score(vector, subsampling_weight, mode="HAKE", adversarial_temperature=1.0):
    negative_score = (F.softmax(vector * adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-vector)).sum(dim=1)
    negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
    return negative_sample_loss


# -

import time

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


from config import *
from data import DGLREDataset, DGLREDataloader, BERTDGLREDataset
from models.GAIN import GAIN_GloVe, GAIN_BERT
from test import test
from utils import Accuracy, get_cuda, logging, print_params

matplotlib.use('Agg')


# for ablation
# from models.GAIN_nomention import GAIN_GloVe, GAIN_BERT

# +
def train(opt):
    ##################################################data loader##################################################
    if opt.use_model == 'bert':
        # datasets
        train_set = BERTDGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                     opt=opt)
        dev_set = BERTDGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='dev',
                                   instance_in_train=train_set.instance_in_train, opt=opt)

        # dataloaders
        train_loader = DGLREDataloader(train_set, batch_size=opt.batch_size, shuffle=True,
                                       negativa_alpha=opt.negativa_alpha)
        dev_loader = DGLREDataloader(dev_set, batch_size=opt.test_batch_size, dataset_type='dev')

        model = GAIN_BERT(opt)

    elif opt.use_model == 'bilstm':
        # datasets
        train_set = DGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',
                                 opt=opt)
        dev_set = DGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='dev',
                               instance_in_train=train_set.instance_in_train, opt=opt)

        # dataloaders
        train_loader = DGLREDataloader(train_set, batch_size=opt.batch_size, shuffle=True,
                                       negativa_alpha=opt.negativa_alpha)
        dev_loader = DGLREDataloader(dev_set, batch_size=opt.test_batch_size, dataset_type='dev')

        model = GAIN_GloVe(opt)
    else:
        assert 1 == 2, 'please choose a model from [bert, bilstm].'
    ##################################################################################################################
    
    relation_emb_size = opt.bert_hid_size + opt.entity_id_size + opt.entity_type_size
    assert relation_emb_size % 2 == 0, "relation embedding dimension is wrong"

    if opt.LPmode == "HAKE":
        relation_emb_size = int(relation_emb_size * 1.5)
        assert relation_emb_size % 3 == 0, "relation embedding dimension is wrong"
    relation_embedding = nn.Parameter(torch.randn(opt.relation_nums, relation_emb_size, requires_grad=True , device= "cuda"))
    print("relation_embedding : ", relation_embedding.shape)
    # exit(True)
#     assert relation_emb_size % 3 == 0, "relation embedding dimension is wrong"
    print(opt.relation_nums, relation_embedding//3)
    LPmodel = KGEModel(num_relation = opt.relation_nums, hidden_dim = relation_emb_size//3, mode = opt.LPmode)

    print_params(model)

    start_epoch = 1
    pretrain_model = opt.pretrain_model
    lr = opt.lr
    model_name = opt.model_name

    if pretrain_model != '':
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load model from {}'.format(pretrain_model))
        start_epoch = chkpt['epoch'] + 1
        lr = chkpt['lr']
        logging('resume from epoch {} with lr {}'.format(start_epoch, lr))
    else:
        logging('training from scratch with lr {}'.format(lr))

    model = get_cuda(model)
    LPmodel = get_cuda(LPmodel)
    if opt.use_model == 'bert':
        bert_param_ids = list(map(id, model.bert.parameters()))
        base_params = filter(lambda p: p.requires_grad and id(p) not in bert_param_ids, model.parameters())

        optimizer = optim.AdamW([
            {'params': model.bert.parameters(), 'lr': lr * 0.01},
            {'params': relation_embedding, 'lr': lr * opt.RE_lr_scale},
            {'params': base_params, 'weight_decay': opt.weight_decay}
        ], lr=lr)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                weight_decay=opt.weight_decay)

    BCE = nn.BCEWithLogitsLoss(reduction='none')
    
    
    
    if opt.coslr:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(opt.epoch // 4) + 1)

    checkpoint_dir = opt.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    fig_result_dir = opt.fig_result_dir
    if not os.path.exists(fig_result_dir):
        os.mkdir(fig_result_dir)

    best_ign_auc = 0.0
    best_ign_f1 = 0.0
    best_epoch = 0

    model.train()

    global_step = 0
    total_loss = 0
    pn_total_loss = 0
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.title('Precision-Recall')
    plt.grid(True)

    acc_NA, acc_not_NA, acc_total = Accuracy(), Accuracy(), Accuracy()
    logging('begin..')

    for epoch in range(start_epoch, opt.epoch + 1):
        start_time = time.time()
        for acc in [acc_NA, acc_not_NA, acc_total]:
            acc.clear()

        for ii, d in enumerate(train_loader):
            relation_multi_label = d['relation_multi_label']
            relation_mask = d['relation_mask']
            relation_label = d['relation_label']
            
            '''
                predictions: last output (1,95,97)
                encoder_outputs: BERT output
                output_feature: GCN output
                entity_graph_feature: GCN output2
                h_entity: head entity  (1,35,2424)
                t_entity: tail entity
            '''

            predictions, encoder_outputs, output_feature, \
            entity_graph_feature, h_entity, t_entity = model(words=d['context_idxs'],
                                                    src_lengths=d['context_word_length'],
                                                    mask=d['context_word_mask'],
                                                    entity_type=d['context_ner'],
                                                    entity_id=d['context_pos'],
                                                    mention_id=d['context_mention'],
                                                    distance=None,
                                                    entity2mention_table=d['entity2mention_table'],
                                                    graphs=d['graphs'],
                                                    h_t_pairs=d['h_t_pairs'],
                                                    relation_mask=relation_mask,
                                                    path_table=d['path_table'],
                                                    entity_graphs=d['entity_graphs'],
                                                    ht_pair_distance=d['ht_pair_distance'])
            
            
            loss = torch.sum(BCE(predictions, relation_multi_label) * relation_mask.unsqueeze(2)) / (
                    opt.relation_nums * torch.sum(relation_mask))
            
            start_idx = 0
            positive_sample_loss = 0.0
            negative_sample_loss = 0.0
#             print(d['entity_graphs'])
            for i, batch in enumerate(relation_multi_label):
#                 print("relation_multi_label dddddddddddddddddddd: ",len(relation_multi_label))
                h_t_pairs = d['h_t_pairs'][i]
                
                ent_size = len(d['entity_graphs'][i].nodes())
            
                rel = torch.nonzero(batch[:ent_size] == 1)
#                 print("batch : ", batch.shape)
#                 print("h_t_pairs : ", d['h_t_pairs'][i])
                
#                 print(rel)
#                 print(rel.shape)
                p_idx = rel[:,0][torch.where(rel[:,1]>0)]
                if len(p_idx) == 0:
                    continue
#                 n_idx = rel[:,0][torch.where(rel[:,1]==0)]
                head_p = entity_graph_feature[h_t_pairs[p_idx][:,0] + start_idx-1]
                tail_p = entity_graph_feature[h_t_pairs[p_idx][:,1] + start_idx-1]
                head_p = head_p.reshape(len(head_p), 1, -1)
                tail_p = tail_p.reshape(len(tail_p), 1, -1)
                
                
                ent_n = torch.cat((entity_graph_feature[:start_idx], entity_graph_feature[start_idx+ent_size:]))
                ent_n = ent_n.repeat((len(head_p),1,1))

                
                if len(head_p > 0):
                    rel_p = rel[:,1][p_idx]
                    rel_p = relation_embedding[rel_p]
                    rel_p = rel_p.reshape(len(rel_p), 1, -1)
#                     print("HeadP : ", head_p.shape)
#                     print("tail_p : ", tail_p.shape)
#                     print("rel_p : ", rel_p.shape)
#                     print("ent_n : ", ent_n.shape)
#                     print("="*100)
                    output = LPmodel.func(head_p,rel_p,tail_p)
                    subsampling_weight = torch.tensor([1/len(head_p) for i in range(len(head_p))]).cuda()
                    positive_sample_loss = p_score(output, subsampling_weight)

#                 if len(head_n > 0):
#                     rel_n = (torch.randint(opt.relation_nums-1, (len(head_n),))+1).cuda()
#                     rel_n = relation_embedding[rel_n]
                    
                    if torch.randint(2,(1,)) == 1:
                        output = LPmodel.func(head_p.reshape(len(head_p), 1, -1),
                                              rel_p.reshape(len(head_p), 1, -1),
                                              ent_n)
                    else:
                        output = LPmodel.func(ent_n,
                                              rel_p.reshape(len(head_p), 1, -1),
                                              tail_p.reshape(len(head_p), 1, -1))
                        
#                     subsampling_weight = torch.tensor([1/len(head_n) for i in range(len(head_n))]).cuda()
                    negative_sample_loss = n_score(output, subsampling_weight)
#                     negative_sample_loss += -(subsampling_weight * negative_score).sum() / subsampling_weight.sum()
                start_idx += ent_size                   
                                   
                                   
                                
                
            positive_sample_loss = positive_sample_loss / len(relation_multi_label)
            negative_sample_loss = negative_sample_loss / len(relation_multi_label)
#             print("positive_sample_loss : ",positive_sample_loss)
#             print("negative_sample_loss : ",negative_sample_loss)
#             print("="*120)
            if epoch < 10:
                pn_loss = (positive_sample_loss + negative_sample_loss)*0.5
            elif epoch < 80:
                pn_loss = (positive_sample_loss + negative_sample_loss)*0.05
            else:
                pn_loss = (positive_sample_loss + negative_sample_loss)*0.005
#             print("pn_loss: ", pn_loss)
#             print("loss: ", loss)
            
            loss = loss + pn_loss
            optimizer.zero_grad()
            loss.backward()

            
            if opt.clip != -1:
                nn.utils.clip_grad_value_(model.parameters(), opt.clip)
            optimizer.step()
            if opt.coslr:
                scheduler.step(epoch)
            

            output = torch.argmax(predictions, dim=-1)
            output = output.data.cpu().numpy()
            relation_label = relation_label.data.cpu().numpy()

            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    label = relation_label[i][j]
                    if label < 0:
                        break

                    is_correct = (output[i][j] == label)
                    if label == 0:
                        acc_NA.add(is_correct)
                    else:
                        acc_not_NA.add(is_correct)

                    acc_total.add(is_correct)

            global_step += 1
            total_loss += loss.item()
            pn_total_loss += pn_loss
            log_step = opt.log_step
            if global_step % log_step == 0:
                cur_loss = total_loss / log_step
                pn_cur_loss = pn_total_loss/log_step
                elapsed = time.time() - start_time
                logging(
                    '| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | pn loss {:5.3f} |train loss {:5.3f} | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(
                        epoch, global_step, elapsed * 1000 / log_step, pn_cur_loss * 1000,cur_loss * 1000, acc_NA.get(), acc_not_NA.get(),
                        acc_total.get()))
                total_loss = 0
                pn_total_loss = 0
                start_time = time.time()

        if epoch % opt.test_epoch == 0:
            logging('-' * 89)
            eval_start_time = time.time()
            model.eval()
            ign_f1, ign_auc, pr_x, pr_y = test(model, dev_loader, model_name, id2rel=id2rel)
            model.train()
            logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
            logging('-' * 89)

            if ign_f1 > best_ign_f1:
                best_ign_f1 = ign_f1
                best_ign_auc = ign_auc
                best_epoch = epoch
                path = os.path.join(checkpoint_dir, model_name + '_best.pt')
                torch.save({
                    'epoch': epoch,
                    'checkpoint': model.state_dict(),
                    'lr': lr,
                    'best_ign_f1': ign_f1,
                    'best_ign_auc': ign_auc,
                    'best_epoch': epoch
                }, path)

                plt.plot(pr_x, pr_y, lw=2, label=str(epoch))
                plt.legend(loc="upper right")
                plt.savefig(os.path.join(fig_result_dir, model_name) + ".jpg")

        if epoch % opt.save_model_freq == 0:
            path = os.path.join(checkpoint_dir, model_name + '_{}.pt'.format(epoch))
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'checkpoint': model.state_dict()
            }, path)

    print("Finish training")
    print("Best epoch = %d | Best Ign F1 = %f" % (best_epoch, best_ign_f1))
    print("Storing best result...")
    print("Finish storing")
# -

if __name__ == '__main__':
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    opt = get_opt()
    print(json.dumps(opt.__dict__, indent=4))
    opt.data_word_vec = word2vec
    train(opt)
