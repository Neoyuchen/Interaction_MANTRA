import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

class model_memory_IRM(nn.Module):
    """
    Memory Network model with Iterative Refinement Module.
    """

    def __init__(self, settings, model_pretrained):
        super(model_memory_IRM, self).__init__()
        self.name_model = 'MANTRA'

        # parameters
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.num_prediction = settings["num_prediction"]
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]

        # similarity criterion
        self.weight_read = []
        self.index_max = []
        self.similarity = nn.CosineSimilarity(dim=1)

        # Memory
        self.memory_past = model_pretrained.memory_past
        self.memory_fut = model_pretrained.memory_fut
        self.memory_count = []

        # layers
        self.conv_past = model_pretrained.conv_past
        self.conv_fut = model_pretrained.conv_fut

        self.encoder_past = model_pretrained.encoder_past
        self.encoder_fut = model_pretrained.encoder_fut
        self.decoder = model_pretrained.decoder
        self.FC_output = model_pretrained.FC_output
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.maxpool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # writing controller
        self.linear_controller = model_pretrained.linear_controller

        # scene: input shape (batch, classes, 360, 360)
        self.convScene_1 = nn.Sequential(nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2), nn.ReLU(),
                                         nn.BatchNorm2d(8))
        self.convScene_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), nn.BatchNorm2d(16))

        self.RNN_scene = nn.GRU(16, self.dim_embedding_key, 1, batch_first=True)

        # refinement fc layer
        self.fc_refine = nn.Linear(self.dim_embedding_key, self.future_len * 2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.RNN_scene.weight_ih_l0)
        nn.init.kaiming_normal_(self.RNN_scene.weight_hh_l0)
        nn.init.kaiming_normal_(self.RNN_scene.weight_ih_l0)
        nn.init.kaiming_normal_(self.RNN_scene.weight_hh_l0)
        nn.init.kaiming_normal_(self.convScene_1[0].weight)
        nn.init.kaiming_normal_(self.convScene_2[0].weight)
        nn.init.kaiming_normal_(self.fc_refine.weight)

        nn.init.zeros_(self.RNN_scene.bias_ih_l0)
        nn.init.zeros_(self.RNN_scene.bias_hh_l0)
        nn.init.zeros_(self.RNN_scene.bias_ih_l0)
        nn.init.zeros_(self.RNN_scene.bias_hh_l0)
        nn.init.zeros_(self.convScene_1[0].bias)
        nn.init.zeros_(self.convScene_2[0].bias)
        nn.init.zeros_(self.fc_refine.bias)

    def init_memory(self, data_train):
        """
        Initialization: write element in memory.
        :param data_train: dataset
        :return: None
        """

        self.memory_past = torch.Tensor()  # .cuda()
        self.memory_fut = torch.Tensor()  # .cuda()

        j = random.randint(0, len(data_train)-1)
        past = data_train[j][1].unsqueeze(0)
        future = data_train[j][2].unsqueeze(0)

        # past = past.cuda()
        # future = future.cuda()

        # past encoding
        past = torch.transpose(past, 1, 2)
        story_embed = self.relu(self.conv_past(past))
        story_embed = torch.transpose(story_embed, 1, 2)
        output_past, state_past = self.encoder_past(story_embed)

        # future encoding
        future = torch.transpose(future, 1, 2)
        future_embed = self.relu(self.conv_fut(future))
        future_embed = torch.transpose(future_embed, 1, 2)
        output_fut, state_fut = self.encoder_fut(future_embed)

        state_past = state_past.squeeze(0)
        state_fut = state_fut.squeeze(0)

        self.memory_past = torch.cat((self.memory_past, state_past), 0)
        self.memory_fut = torch.cat((self.memory_fut, state_fut), 0)

        # #ablation study
        # future = torch.transpose(future, 1, 2)
        # self.memory_count = torch.cat((self.memory_count, future), 0)

    def forward(self, past, scene=None):
        """
        Forward pass. Refine predictions generated by MemNet with IRM.
        :param past: past trajectory
        :param scene: surrounding map
        :return: predicted future
        """
        dim_batch = past.size()[0]
        zero_padding = torch.zeros(1, dim_batch * self.num_prediction, self.dim_embedding_key * 2)  # .cuda()
        prediction = torch.Tensor()  # .cuda()
        present_temp = past[:, -1].unsqueeze(1)

        # past temporal encoding
        past = torch.transpose(past, 1, 2)
        story_embed = self.relu(self.conv_past(past))
        story_embed = torch.transpose(story_embed, 1, 2)
        output_past, state_past = self.encoder_past(story_embed)

        # Cosine similarity and memory read
        past_normalized = F.normalize(self.memory_past, p=2, dim=1)
        state_normalized = F.normalize(state_past.squeeze(), p=2, dim=1)
        self.weight_read = torch.matmul(past_normalized, state_normalized.transpose(0, 1)).transpose(0, 1)
        self.index_max = torch.sort(self.weight_read, descending=True)[1].cpu()[:, :self.num_prediction]

        present = present_temp.repeat_interleave(self.num_prediction, dim=0)
        state_past = state_past.repeat_interleave(self.num_prediction, dim=1)
        ind = self.index_max.flatten()

        info_future = self.memory_fut[ind]
        info_total = torch.cat((state_past, info_future.unsqueeze(0)), 2)
        input_dec = info_total
        state_dec = zero_padding
        for i in range(self.future_len):
            output_decoder, state_dec = self.decoder(input_dec, state_dec)
            displacement_next = self.FC_output(output_decoder)
            coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            prediction = torch.cat((prediction, coords_next), 1)
            present = coords_next
            input_dec = zero_padding

        if scene is not None:
            # scene encoding
            # scene = scene.permute(0, 3, 1, 2)
            scene = scene.view(dim_batch, -1, 360, 360)
            scene_1 = self.convScene_1(scene)
            scene_2 = self.convScene_2(scene_1)
            scene_2 = scene_2.repeat_interleave(self.num_prediction, dim=0)

            # Iteratively refine predictions using context
            for i_refine in range(4):
                pred_map = prediction + 90
                pred_map = pred_map.unsqueeze(2)
                indices = pred_map.permute(0, 2, 1, 3)
                # rescale between -1 and 1
                indices = 2 * (indices / 180) - 1
                output = F.grid_sample(scene_2, indices, mode='nearest')
                output = output.squeeze(2).permute(0, 2, 1)

                state_rnn = state_past
                output_rnn, state_rnn = self.RNN_scene(output, state_rnn)
                prediction_refine = self.fc_refine(state_rnn).view(-1, self.future_len, 2)
                prediction = prediction + prediction_refine

        prediction = prediction.view(dim_batch, self.num_prediction, self.future_len, 2)
        return prediction

    def write_in_memory(self, past, future, scene=None):
        """
        Writing controller decides if the pair past-future will be inserted in memory.
        :param past: past trajectory
        :param future: future trajectory
        """

        if (self.memory_past.shape[0] < self.num_prediction):
            num_prediction = self.memory_past.shape[0]
        else:
            num_prediction = self.num_prediction

        dim_batch = past.size()[0]
        zero_padding = torch.zeros(1, dim_batch * num_prediction, self.dim_embedding_key * 2).cuda()
        prediction = torch.Tensor().cuda()
        present_temp = past[:, -1].unsqueeze(1)

        # past temporal encoding
        past = torch.transpose(past, 1, 2)
        story_embed = self.relu(self.conv_past(past))
        story_embed = torch.transpose(story_embed, 1, 2)
        output_past, state_past = self.encoder_past(story_embed)

        # Cosine similarity and memory read
        past_normalized = F.normalize(self.memory_past, p=2, dim=1)
        state_normalized = F.normalize(state_past.squeeze(0), p=2, dim=1)
        weight_read = torch.matmul(past_normalized, state_normalized.transpose(0, 1)).transpose(0, 1)
        index_max = torch.sort(weight_read, descending=True)[1].cpu()[:, :num_prediction]

        present = present_temp.repeat_interleave(num_prediction, dim=0)
        state_past_repeat = state_past.repeat_interleave(num_prediction, dim=1)
        ind = index_max.flatten()
        info_future = self.memory_fut[ind]
        info_total = torch.cat((state_past_repeat, info_future.unsqueeze(0)), 2)
        input_dec = info_total
        state_dec = zero_padding
        for i in range(self.future_len):
            output_decoder, state_dec = self.decoder(input_dec, state_dec)
            displacement_next = self.FC_output(output_decoder)
            coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            prediction = torch.cat((prediction, coords_next), 1)
            present = coords_next
            input_dec = zero_padding

        # Iteratively refine predictions using context
        if scene is not None:
            # scene encoding
            scene = scene.permute(0, 3, 1, 2)
            scene_1 = self.convScene_1(scene)
            scene_2 = self.convScene_2(scene_1)
            scene_2 = scene_2.repeat_interleave(num_prediction, dim=0)
            for i_refine in range(4):
                pred_map = prediction + 90
                pred_map = pred_map.unsqueeze(2)
                indices = pred_map.permute(0, 2, 1, 3)
                # rescale between -1 and 1
                indices = 2 * (indices / 180) - 1
                output = F.grid_sample(scene_2, indices, mode='nearest')
                output = output.squeeze(2).permute(0, 2, 1)

                state_rnn = state_past_repeat
                output_rnn, state_rnn = self.RNN_scene(output, state_rnn)
                prediction_refine = self.fc_refine(state_rnn).view(-1, self.future_len, 2)
                prediction = prediction + prediction_refine
        prediction = prediction.view(dim_batch, num_prediction, self.future_len, 2)

        future_rep = future.unsqueeze(1).repeat(1, num_prediction, 1, 1)
        distances = torch.norm(prediction - future_rep, dim=3)
        tolerance_1s = torch.sum(distances[:, :, :10] < 0.5, dim=2)
        tolerance_2s = torch.sum(distances[:, :, 10:20] < 1, dim=2)
        tolerance_3s = torch.sum(distances[:, :, 20:30] < 1.5, dim=2)
        tolerance_4s = torch.sum(distances[:, :, 30:40] < 2, dim=2)
        tolerance = tolerance_1s + tolerance_2s + tolerance_3s + tolerance_4s
        tolerance_rate = torch.max(tolerance, dim=1)[0].type(torch.FloatTensor) / 40
        tolerance_rate = tolerance_rate.unsqueeze(1).cuda()

        # controller
        writing_prob = torch.sigmoid(self.linear_controller(tolerance_rate))

        # future encoding
        future = torch.transpose(future, 1, 2)
        future_embed = self.relu(self.conv_fut(future))
        future_embed = torch.transpose(future_embed, 1, 2)
        output_fut, state_fut = self.encoder_fut(future_embed)

        # ablation study: all tracks in memory
        # index_writing = np.where(writing_prob.cpu() > 0)[0]
        index_writing = np.where(writing_prob.cpu() > 0.5)[0]
        past_to_write = state_past.squeeze()[index_writing]
        future_to_write = state_fut.squeeze()[index_writing]

        self.memory_past = torch.cat((self.memory_past, past_to_write), 0)
        self.memory_fut = torch.cat((self.memory_fut, future_to_write), 0)

        # #ablation study: future track in memory
        # future = torch.transpose(future, 1, 2)
        # future_track_to_write = future[index_writing]
        # self.memory_count = torch.cat((self.memory_count, future_track_to_write), 0)
