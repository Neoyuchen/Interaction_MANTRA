import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import scipy.sparse as sp
import pandas as pd
import networkx as nx
import warnings


class model_encdec_GCN(nn.Module):
    def __init__(self, settings, model_pretrained):
        super(model_encdec_GCN, self).__init__()

        self.name_model = 'autoencoder_GCN'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]  # 48
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]

        # encoder_decoder layers
        self.conv_past = model_pretrained.conv_past
        self.conv_fut = model_pretrained.conv_fut

        self.encoder_past = model_pretrained.encoder_past
        self.encoder_fut = model_pretrained.encoder_fut
        self.decoder = nn.GRU(self.dim_embedding_key * 3, self.dim_embedding_key * 3, 1, batch_first=False)
        self.FC_output = torch.nn.Linear(self.dim_embedding_key * 3, 2)

        # GCN layers
        self.conv1 = GCNConv(self.dim_embedding_key, self.dim_embedding_key,
                             normalize=True)
        self.conv2 = GCNConv(self.dim_embedding_key, self.dim_embedding_key,
                             normalize=True)

    def forward(self, past, future, x, edge_index, edge_weight):
        """
        Incorporate interaction representation with past encoding
        Then forward pass that encodes past and future and decodes the future.
        :param past: past trajectory [batch_size, len_past, 2]
        :param future: future trajectory [batch_size, len_future, 2]
        :param x: surrounding vehicle's past trajectory [total_node_number, len_past, 2]
        :param edge_index: pairs of interactive agents [batch_size,2,edge_number] diagonal
        :param edge_weight: degree of interaction [batch_size, edge_number]
        :return: decoded future
        """

        dim_batch = past.size()[0]
        zero_padding = torch.zeros(1, dim_batch, self.dim_embedding_key * 3)
        prediction = torch.Tensor()
        present = past[:, -1, :2].unsqueeze(1)
        if self.use_cuda:
            zero_padding = zero_padding.cuda()
            prediction = prediction.cuda()

        # temporal encoding for graph nodes
        inter_past = torch.transpose(x.view(-1, self.past_len, 2), 1, 2)
        story_embed = F.relu(self.conv_past(inter_past))
        story_embed = torch.transpose(story_embed, 1, 2)
        output_past, state_past = self.encoder_past(story_embed)

        # GCN
        if edge_weight.size()[1] != 0:
            x1 = state_past.view(state_past.size()[1], state_past.size()[2])
            edge_weight = edge_weight.view(-1)
            edge_index = edge_index.view(2, -1)
            x1 = F.relu(self.conv1(x1, edge_index, edge_weight))
            x1 = F.dropout(x1, training=self.training)
            x1 = self.conv2(x1, edge_index, edge_weight)
            # get the target encoding
            interact_past = x1[0].view(dim_batch, 1, -1)
        else:
            interact_past = torch.zeros(dim_batch, 1, 48)

        # temporal encoding for target past
        past = torch.transpose(past, 1, 2)
        past_embed = F.relu(self.conv_past(past))
        past_embed = torch.transpose(past_embed, 1, 2)

        # temporal encoding for target future
        future = torch.transpose(future, 1, 2)
        future_embed = F.relu(self.conv_fut(future))
        future_embed = torch.transpose(future_embed, 1, 2)

        # sequence encoding
        output_past, target_past = self.encoder_past(past_embed)
        output_fut, state_fut = self.encoder_fut(future_embed)

        # incorporate target past encoding with interaction encoding
        # co_past = interact_past + target_past
        co_past = torch.cat((interact_past, target_past), 2)

        # state concatenation and decoding
        state_conc = torch.cat((co_past, state_fut), 2)
        input_fut = state_conc
        state_fut = zero_padding
        for i in range(self.future_len):
            output_decoder, state_fut = self.decoder(input_fut, state_fut)
            displacement_next = self.FC_output(output_decoder)
            coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            prediction = torch.cat((prediction, coords_next), 1)
            present = coords_next
            input_fut = zero_padding
        return prediction

