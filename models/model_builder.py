#!/usr/bin/env python
"""
    model_builder.py, to build the model.

"""



import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from pytorch_transformers import BertModel, BertConfig
from models.encoder import Classifier, ExtTransformerEncoder
from models.decoder import TransformerDecoder
from models.optimizers import Optimizer

class Bert(nn.Module):
    
    """
    bertmodel 初始分为三个部分，分别为bertembedding，bertencoder和bertpooler。
    bert模型一般对应的embedding包括三个部分，即word_embedding, position_embedding,和 segment_embedding.对应input_ids[batch_size, sequence_length],position_embedding(就是每个单词对应的在原文的位置) ,和token_type_ids.最终输出应该是[batch_size. seq_len, hidden_size]
    第二层为encoder层,包括众多bertlayer层,其中需要注意，mask要从[batch_size, seq_len]变成[batch_size,1,1,seq_len]，为了方便[batch_size, num_heads, from_seq_len(dim_Q), to_seq_len(dim_K)]的broadcast
    # 问题1：新版本的transformers库的情况？
    这一层里，输出的是最后一层的hiddenstate，前面层的hidden_state输出由config中的output_hidden_states控制
    注意，某个tensor a的维度为n维时，a[:,:,...,0]，假设a前面有m个:，则0
是第m+1个维度上选择的向量
    最终第二层输出的大小是（[batch_size,seq_len,hidden_size],(num_layers_dim * []), (attention) ）
    第三层是pooled层，是把第二层的tuple[0]拿来，取[CLS]对应的embddding，做一层dnn和tanh即可，得到输出。
    当然整个模型最终只要最后一层的各个序列token对应的隐向量，即[batch_size, seq_len, hidden_size]

    model.train()启用 BatchNormalization 和 Dropout
    model.eval()不启用 BatchNormalization 和 Dropout
    https://blog.csdn.net/cpluss/article/details/88418176 Bert-pretrained库（old）版本讲解
    某些config里没有但是代码里有的attri在modeling_utils里有定义，通常长这个样子：(放在jupyter里)
    () + (3,)+(4,) = (3,4)


   """ 
    
    def __init__(self, bert_path,finetune = False):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained(bert_path)
        self.finetune = finetune
    def forward(self, x, segs, mask):# 这里的segs就是token_type_ids
        if self.finetune:
            top_encoded_vec, _ = self.model(input_ids = x, token_type_ids = segs, attention_mask = mask)
        else:
            self.eval()
            with torch.no_grad():
                top_encoded_vec, _ = self.model(x, segs, attention_mask = mask)
        return top_encoded_vec# [batch_size , seq_len, hidden_size]


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, ckpt):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.bert_path, args.finetune_bert)
        self.ext_layer = ExtTransformerEncoder()
        """
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
            num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        """  
        if ckpt is not None:
            self.load_state_dict(ckpt['model'], strict = True)# 注意这里strict用于检测model和ckpt里的keys是否严格一一对应，false则可以放缓
        else:
            if args.param_init != 0 :
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            elif args.param_init_glorot:# 即选用xavier均匀分布
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device) # 关键，一定记住
    """
    some points:
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        to id: 1,2,3
        此外，抽取式摘要需要对原文的每个句子是否被选中做一个0,1的标签标记，称为src_sent_labels
        为什么clss的pad用-1？因为clss是cls的token的位置，不是token的值
    """
    def forward(self, x, segs, clss, mask_src, mask_cls):
        """
        inputs:
            x: input_ids [batch_size, seq_len] , len(max(seq_len)) < args.max_pos
            segs: token_type_ids [batch_size, seq_len]
            clss : cls_token_pos_ids [batch_size, clss_len], clss[:,] < args.max_pos
            mask_src : 0,1 [batch_size, seq_len]
            mask_cls: 0,1 [batch_size, clss_len]]
        """
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss] # batch_size, clss_token, hidden_size
        sents_vec =  sents_vec * mask_cls[:,:,None].float()# bsz, clstoken, hidden_size
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
        """
            如何得到[batch_size, clss_token, hidden_size]？当二维的tensor用作索引时，需要注意，第一维的tensor[a,b,。。。]中的每一个元素都要用来提取[:, seq_len, hiddensize]，在此处，需要让第二维的tensor的num_dim = 第一维的tensor的num_dim.如果想在seq_len维度上抽取多个，则必须要batch_size的基础上unsqueeze(1)。
            mask[:,:,None]将原来的mask的最后一维度扩充，等同于unsqueeze(2),两个向量前面的维度相同时，等同于最后一个维度（只可能长度相同，或者其中一个长度为1）每个数字两两相乘
            squeeze 即判断最后一个维度是否为1，是的话去除
        """

        pass
       
