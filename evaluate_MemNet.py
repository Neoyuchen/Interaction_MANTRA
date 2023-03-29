import os
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
import datetime
import cv2
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset_invariance
import index_qualitative
from torch.autograd import Variable
import csv
import time
import tqdm
import pdb
from tensorboardX import SummaryWriter
from reasonability_compute import RationCompute


class Validator():
    def __init__(self, config):
        """
        class to evaluate Memnet
        :param config: configuration parameters (see test.py)
        """
        self.online = config.online_learning
        self.ration_c = RationCompute(config.osm_path)
        self.index_qualitative = index_qualitative.dict_test
        self.name_test = str(datetime.datetime.now())[:13]
        self.folder_test = 'test/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.pic_path = r'D:\BaiduNetdiskWorkspace\轨迹预测\picture\interaction'+config.info
        if not os.path.exists(self.pic_path):
            os.makedirs(self.pic_path)

        print('creating dataset...')
        self.dim_clip = 180
        tracks = json.load(open(config.dataset_file))
        if not config.saved_memory:
            self.data_train = dataset_invariance.TrackDataset(tracks,
                                                              len_past=config.past_len,
                                                              len_future=config.future_len,
                                                              train=True,
                                                              dim_clip=self.dim_clip,
                                                              graph_train=True)

            self.train_loader = DataLoader(self.data_train,
                                           batch_size=config.batch_size,
                                           num_workers=1,
                                           shuffle=False)

        self.data_test = dataset_invariance.TrackDataset(tracks,
                                                         len_past=config.past_len,
                                                         len_future=config.future_len,
                                                         train=False,
                                                         dim_clip=self.dim_clip,
                                                         graph_train=True)

        self.test_loader = DataLoader(self.data_test,
                                      batch_size=config.batch_size,
                                      num_workers=1,
                                      shuffle=False)

        # self.data_train = dataset_invariance.InteractionDataset(json_path=config.train_path)
        # self.train_loader = DataLoader(self.data_train,
        #                                batch_size=config.batch_size,
        #                                num_workers=1)
        # self.data_test = dataset_invariance.InteractionDataset(json_path=config.val_path)
        # self.test_loader = DataLoader(self.data_test,
        #                               batch_size=config.batch_size,
        #                               num_workers=1)

        print('dataset created')
        if config.visualize_dataset:
            print('save examples in folder test')
            self.data_train.save_dataset(self.folder_test + 'dataset_train/')
            self.data_test.save_dataset(self.folder_test + 'dataset_test/')
            print('Saving complete!')

        # load model to evaluate
        self.mem_n2n = torch.load(config.model, map_location=torch.device('cpu'))
        self.mem_n2n.num_prediction = config.preds
        self.mem_n2n.future_len = config.future_len
        self.mem_n2n.past_len = config.past_len

        self.EuclDistance = nn.PairwiseDistance(p=2)
        if config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

        # Tensorboard summary: configuration
        self.writer = SummaryWriter(self.folder_test + '_' + config.info)

    def test_model(self):
        """
        Memory selection and evaluation!
        :return: None
        """
        # populate the memory
        start = time.time()
        self._memory_writing(self.config.saved_memory)
        end = time.time()
        print('writing time: ' + str(end-start))

        # run test!
        dict_metrics_test = self.evaluate(self.test_loader)
        self.save_results(dict_metrics_test)

    def save_results(self, dict_metrics_test):
        """
        Serialize results
        :param dict_metrics_test: dictionary with test metrics
        :param dict_metrics_train: dictionary with train metrics
        :param epoch: epoch index (default: 0)
        :return: None
        """
        self.file = open(self.folder_test + "results.txt", "w")
        self.file.write("TEST:" + '\n')

        self.file.write("model:" + self.config.model + '\n')
        # self.file.write("split test: " + str(self.data_test.ids_split_test) + '\n')
        self.file.write("num_predictions:" + str(self.config.preds) + '\n')
        self.file.write("memory size: " + str(len(self.mem_n2n.memory_past)) + '\n')
        # self.file.write("TRAIN size: " + str(len(self.data_train)) + '\n')
        self.file.write("TEST size: " + str(len(self.data_test)) + '\n')

        self.file.write("error 0.5s: " + str(dict_metrics_test['horizon_05']) + 'm \n')
        self.file.write("error 1.0s: " + str(dict_metrics_test['horizon_10']) + 'm \n')
        self.file.write("error 1.53s: " + str(dict_metrics_test['horizon_15']) + 'm \n')
        self.file.write("error 2.0s: " + str(dict_metrics_test['horizon_20']) + 'm \n')
        self.file.write("error 2.5s: " + str(dict_metrics_test['horizon_25']) + 'm \n')
        self.file.write("error 3.0s: " + str(dict_metrics_test['horizon_30']) + 'm \n')

        if self.config.future_len == 40:
            self.file.write("error 3.5s: " + str(dict_metrics_test['horizon_35']) + 'm \n')
            self.file.write("error 4s: " + str(dict_metrics_test['horizon_40']) + 'm \n')

        self.file.write("ADE 0.5s: " + str(dict_metrics_test['ADE_05']) + 'm \n')
        self.file.write("ADE 1.0s: " + str(dict_metrics_test['ADE_10']) + 'm \n')
        self.file.write("ADE 1.5s: " + str(dict_metrics_test['ADE_15']) + 'm \n')
        self.file.write("ADE 2.0s: " + str(dict_metrics_test['ADE_20']) + 'm \n')
        self.file.write("ADE 2.5s: " + str(dict_metrics_test['ADE_25']) + 'm \n')
        self.file.write("ADE 3.0s: " + str(dict_metrics_test['ADE_30']) + 'm \n')

        if self.config.future_len == 40:
            self.file.write("ADE 3.5s: " + str(dict_metrics_test['ADE_35']) + 'm \n')
            self.file.write("ADE 4s: " + str(dict_metrics_test['eucl_mean']) + 'm \n')
        else:
            self.file.write("Intersect Area: " + str(dict_metrics_test['intersect_area']) + 'm \n')

            self.file.write("Original_ration: " + str(dict_metrics_test['Original_ration']) + 'm \n')
            self.file.write("New_ration: " + str(dict_metrics_test['New_ration']) + 'm \n')
            self.file.write("Original_dist: " + str(dict_metrics_test['Original_dist']) + 'm \n')
            self.file.write("New_dist: " + str(dict_metrics_test['New_dist']) + 'm \n')
            self.file.write("Original_idx: " + str(dict_metrics_test['Original_idx']) + 'm \n')
            self.file.write("New_idx: " + str(dict_metrics_test['New_idx']) + 'm \n')


        self.file.close()

    def draw_track(self, past, future, scene_track, surr_tracks, pred=None, angle=0, video_id='', vec_id='', index_tracklet=0,
                    path='', horizon_dist=None):
        """
        Plot past and future trajectory and save it to test folder.
        :param past: the observed trajectory
        :param future: ground truth future trajectory
        :param pred: predicted future trajectory
        :param angle: rotation angle to plot the trajectory in the original direction
        :param video_id: video index of the trajectory
        :param vec_id: vehicle type of the trajectory
        :param pred: predicted future trajectory
        :param: the observed scene where is the trajectory
        :param index_tracklet: index of the trajectory in the dataset (default 0)
        :param num_epoch: current epoch (default 0)
        :return: None
        """

        colors = [(0, 0, 0), (0.87, 0.87, 0.87), (0.54, 0.54, 0.54), (0.49, 0.33, 0.16), (0.29, 0.57, 0.25)]
        cmap_name = 'scene_cmap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=5)
        fig = plt.figure()
        plt.imshow(scene_track, cmap=cm)
        colors = pl.cm.Reds(np.linspace(1, 0.3, pred.shape[0]))

        matRot_track = cv2.getRotationMatrix2D((0, 0), -angle, 1)
        past = cv2.transform(past.cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
        future = cv2.transform(future.cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()

        for i in range(1, len(surr_tracks)):
            sur_past = cv2.transform(surr_tracks[i].cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
            sur_past_scene = sur_past * 2 + self.dim_clip
            plt.plot(sur_past_scene[:, 0], sur_past_scene[:, 1], c='purple', marker='o', markersize=1)

        story_scene = past * 2 + self.dim_clip
        future_scene = future * 2 + self.dim_clip
        plt.plot(story_scene[:, 0], story_scene[:, 1], c='blue', linewidth=1, marker='o', markersize=1)
        if pred is not None:
            for i_p in reversed(range(pred.shape[0])):
                pred_i = cv2.transform(pred[i_p].cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
                pred_scene = pred_i * 2 + self.dim_clip
                plt.plot(pred_scene[:, 0], pred_scene[:, 1], color=colors[i_p], linewidth=1, marker='o', markersize=0.5)
        plt.plot(future_scene[:, 0], future_scene[:, 1], c='green', linewidth=1.5, marker='o', markersize=1)
        # plt.title('FDE 1s: ' + str(horizon_dist[0]) + ' FDE 2s: ' + str(horizon_dist[1]) + ' FDE 3s: ' +
        #           str(horizon_dist[2]) + ' FDE 4s: ' + str(horizon_dist[3]))
        plt.axis('off')
        plt.savefig(path + video_id + '_' + vec_id + '_' + str(index_tracklet).zfill(3) + '.png')
        plt.close(fig)


    def evaluate(self, loader):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :return: dictionary of performance metrics
        """
        online_learning = self.online
        self.mem_n2n.eval()
        with torch.no_grad():
            dict_metrics = {}
            inter_area = eucl_mean = ADE_05 = ADE_10 = ADE_15 = ADE_20 = ADE_25 = ADE_30 = ADE_35 =\
                horizon_05 = horizon_10 = horizon_15 = horizon_20 = horizon_25 = horizon_30 = horizon_35 = horizon_40 = 0

            o_ration = n_ration = o_dist = n_dist = o_idx = n_idx = 0

            # KITTI:(index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene, graph)
            # INTERACTION:(index, past, future, presents, angle_presents, videos, vehicles, number_vec, graph)
            pred_list = []
            for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, graph) \
                    in enumerate(tqdm.tqdm(loader)):
                ia = []
                online_metrics = {}
                past = Variable(past)
                future = Variable(future)
                if self.config.cuda:
                    past = past.cuda()
                    future = future.cuda()
                # if self.config.withIRM:
                #     scene_one_hot = Variable(scene_one_hot)
                #     scene_one_hot = scene_one_hot.cuda()
                #     pred = self.mem_n2n(past, scene_one_hot)
                # else:
                #     pred = self.mem_n2n(past)

                start = time.time()
                pred = self.mem_n2n(past, graph['x'], graph['edge_index'], graph['edge_weight'])  # , graph['x'], graph['edge_index'], graph['edge_weight']
                end = time.time()
                run_time = end - start

                future_rep = future.unsqueeze(1).repeat(1, self.config.preds, 1, 1)
                distances = torch.norm(pred - future_rep, dim=3)
                mean_distances = torch.mean(distances, dim=2)

                # Use buffer area to compute the size of intersection area
                future_buf = LineString(future[0]).buffer(0.5)
                for k in range(self.mem_n2n.num_prediction):
                    pred_buf = LineString(pred[0][k]).buffer(0.5)
                    int_area = future_buf.intersection(pred_buf)
                    ia.append(int_area.area/future_buf.area)

                index_min = torch.argmin(mean_distances, dim=1)
                distance_pred = distances[torch.arange(0, len(index_min)), index_min]

                # index_min = ia.index(max(ia))
                # distance_pred = distances[0][index_min].unsqueeze(0)

                # transfer to the real world coordinate
                matRot_track = cv2.getRotationMatrix2D((0, 0), -float(angle_presents[0]), 1)
                pred_l = []
                for i_p in range(pred.shape[1]):
                    pred_i = cv2.transform(pred[0][i_p].cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
                    pred_i = pred_i + np.array(presents[0])
                    pred_l.append(pred_i.tolist())
                past_l = cv2.transform(past.cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
                past_l = past_l + np.array(presents[0])
                future_l = cv2.transform(future.cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
                future_l = future_l + np.array(presents[0])

                surr = []
                for i in range(graph['x'].shape[1]):
                    sur = graph['x'][0][i]
                    sur_l = cv2.transform(sur.cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
                    sur_l = sur_l + np.array(presents[0])
                    surr.append(sur_l)

                # save predict results
                pred_dict = {'pred': pred_l, 'distance': mean_distances.tolist(), 'ia': ia}
                pred_list.append(pred_dict)

                # evaluation
                inter_area += ia[index_min]

                horizon_05 += sum(distance_pred[:, 5])
                horizon_10 += sum(distance_pred[:, 9])
                horizon_15 += sum(distance_pred[:, 15])
                horizon_20 += sum(distance_pred[:, 19])
                horizon_25 += sum(distance_pred[:, 25])
                horizon_30 += sum(distance_pred[:, 29])

                ADE_05 += sum(torch.mean(distance_pred[:, :5], dim=1))
                ADE_10 += sum(torch.mean(distance_pred[:, :10], dim=1))
                ADE_15 += sum(torch.mean(distance_pred[:, :15], dim=1))
                ADE_20 += sum(torch.mean(distance_pred[:, :20], dim=1))
                ADE_25 += sum(torch.mean(distance_pred[:, :25], dim=1))
                ADE_30 += sum(torch.mean(distance_pred[:, :30], dim=1))

                online_metrics['ADE_05'] = round((ADE_05 / (step + 1)).item(), 3)
                online_metrics['ADE_10'] = round((ADE_10 / (step + 1)).item(), 3)
                online_metrics['ADE_15'] = round((ADE_15 / (step + 1)).item(), 3)
                online_metrics['ADE_20'] = round((ADE_20 / (step + 1)).item(), 3)
                online_metrics['ADE_25'] = round((ADE_25 / (step + 1)).item(), 3)
                online_metrics['ADE_30'] = round((ADE_30 / (step + 1)).item(), 3)
                online_metrics['horizon_05'] = round((horizon_05 / (step + 1)).item(), 3)
                online_metrics['horizon_10'] = round((horizon_10 / (step + 1)).item(), 3)
                online_metrics['horizon_15'] = round((horizon_15 / (step + 1)).item(), 3)
                online_metrics['horizon_20'] = round((horizon_20 / (step + 1)).item(), 3)
                online_metrics['horizon_25'] = round((horizon_25 / (step + 1)).item(), 3)
                online_metrics['horizon_30'] = round((horizon_30 / (step + 1)).item(), 3)
                online_metrics['intersect_area'] = round((inter_area / (step + 1)), 3)
                online_metrics['memory size'] = len(self.mem_n2n.memory_past)
                online_metrics['run_time'] = run_time

                # online continues learning from testing sample
                if online_learning:
                    self.mem_n2n.write_in_memory(past, future,  graph['x'], graph['edge_index'], graph['edge_weight'])  # , graph['x'], graph['edge_index'], graph['edge_weight']

                if step%10 == 0:
                    self.writer.add_scalar('memory_size/memory_size_test', online_metrics['memory size'], step)
                    self.writer.add_scalar('predict_error/ADE_1s', online_metrics['ADE_10'], step)
                    self.writer.add_scalar('predict_error/ADE_2s', online_metrics['ADE_20'], step)
                    self.writer.add_scalar('predict_error/ADE_3s', online_metrics['ADE_30'], step)
                    self.writer.add_scalar('predict_error/horizon_1s', online_metrics['horizon_10'], step)
                    self.writer.add_scalar('predict_error/horizon_2s', online_metrics['horizon_20'], step)
                    self.writer.add_scalar('predict_error/horizon_3s', online_metrics['horizon_30'], step)
                    self.writer.add_scalar('predict_error/run_time', online_metrics['run_time'], step)

                if self.config.future_len == 40:
                    horizon_35 += sum(distance_pred[:, 35])
                    horizon_40 += sum(distance_pred[:, 39])
                    ADE_35 += sum(torch.mean(distance_pred[:, :35], dim=1))
                    eucl_mean += sum(torch.mean(distance_pred[:, :40], dim=1))
                    online_metrics['ADE_35'] = round((ADE_30 / (step + 1)).item(), 3)
                    online_metrics['eucl_mean'] = round((eucl_mean / (step + 1)).item(), 3)
                    online_metrics['horizon_35'] = round((horizon_35 / (step + 1)).item(), 3)
                    online_metrics['horizon_40'] = round((horizon_40 / (step + 1)).item(), 3)
                    self.writer.add_scalar('predict_error/ADE_4s', online_metrics['eucl_mean'], step)
                    self.writer.add_scalar('predict_error/horizon_4s', online_metrics['horizon_40'], step)
                else:
                    o_r, r_r, o_d, r_d, min_dist_idx, new_min_idx, n_pred = self.ration_c.evaluate(pred_l, mean_distances)
                    # self.ration_c.track_plot(pred_l, future_l, past_l, np.array(surr), self.pic_path, step)
                    # self.ration_c.track_plot(n_pred, future_l, past_l, np.array(surr), self.pic_path, step)
                    # ration
                    o_ration += o_r
                    n_ration += r_r
                    o_dist += o_d
                    n_dist += r_d
                    o_idx += min_dist_idx
                    n_idx += new_min_idx

                # if (self.config.saveImages is not None) & (step%10 == 0):
                #     for i in range(len(past)):
                #         horizon_dist = [round(distance_pred[i, 9].item(), 3), round(distance_pred[i, 19].item(), 3),
                #                         round(distance_pred[i, 29].item(), 3), round(distance_pred[i, 39].item(), 3)]
                #         vid = videos[i]
                #         vec = vehicles[i]
                #         num_vec = number_vec[i]
                #         index_track = index[i].numpy()
                #         angle = angle_presents[i].cpu()
                #
                #         if self.config.saveImages == 'All':
                #             if not os.path.exists(self.folder_test + vid):
                #                 os.makedirs(self.folder_test + vid)
                #             video_path = self.folder_test + vid + '/'
                #             if not os.path.exists(video_path + vec + num_vec):
                #                 os.makedirs(video_path + vec + num_vec)
                #             vehicle_path = video_path + vec + num_vec + '/'
                #             self.draw_track(past[i], future[i], scene[i], graph['x'][i], pred[i], angle, vid, vec + num_vec,
                #                             index_tracklet=index_track, path=vehicle_path, horizon_dist=horizon_dist)
                #         if self.config.saveImages == 'Subset':
                #             if index_track.item() in self.index_qualitative[vid][vec + num_vec]:
                #                 # Save interesting results
                #                 if not os.path.exists(self.folder_test + 'highlights'):
                #                     os.makedirs(self.folder_test + 'highlights')
                #                 highlights_path = self.folder_test + 'highlights' + '/'
                #                 self.draw_track(past[i], future[i], scene[i], graph['x'][i], pred[i], angle, vid, vec + num_vec,
                #                                 index_tracklet=index_track, path=highlights_path, horizon_dist=horizon_dist)


            dict_metrics['ADE_05'] = round((ADE_05 / len(loader.dataset)).item(), 3)
            dict_metrics['ADE_10'] = round((ADE_10 / len(loader.dataset)).item(), 3)
            dict_metrics['ADE_15'] = round((ADE_15 / len(loader.dataset)).item(), 3)
            dict_metrics['ADE_20'] = round((ADE_20 / len(loader.dataset)).item(), 3)
            dict_metrics['ADE_25'] = round((ADE_25 / len(loader.dataset)).item(), 3)
            dict_metrics['ADE_30'] = round((ADE_30 / len(loader.dataset)).item(), 3)
            dict_metrics['horizon_05'] = round((horizon_05 / len(loader.dataset)).item(), 3)
            dict_metrics['horizon_10'] = round((horizon_10 / len(loader.dataset)).item(), 3)
            dict_metrics['horizon_15'] = round((horizon_15 / len(loader.dataset)).item(), 3)
            dict_metrics['horizon_20'] = round((horizon_20 / len(loader.dataset)).item(), 3)
            dict_metrics['horizon_25'] = round((horizon_25 / len(loader.dataset)).item(), 3)
            dict_metrics['horizon_30'] = round((horizon_30 / len(loader.dataset)).item(), 3)
            dict_metrics['intersect_area'] = round((inter_area / len(loader.dataset)), 3)


            if self.config.future_len == 40:
                dict_metrics['ADE_35'] = round((ADE_35 / len(loader.dataset)).item(), 3)
                dict_metrics['eucl_mean'] = round((eucl_mean / len(loader.dataset)).item(), 3)
                dict_metrics['horizon_35'] = round((horizon_35 / len(loader.dataset)).item(), 3)
                dict_metrics['horizon_40'] = round((horizon_40 / len(loader.dataset)).item(), 3)
            else:
                dict_metrics['Original_ration'] = round((o_ration / len(loader.dataset)), 3)
                dict_metrics['New_ration'] = round((n_ration / len(loader.dataset)), 3)
                dict_metrics['Original_dist'] = round((o_dist / len(loader.dataset)), 3)
                dict_metrics['New_dist'] = round((n_dist / len(loader.dataset)), 3)
                dict_metrics['Original_idx'] = round((o_idx / len(loader.dataset)), 3)
                dict_metrics['New_idx'] = round((n_idx / len(loader.dataset)), 3)

            # save memory
            if online_learning:
                torch.save(self.mem_n2n.memory_past, self.folder_test + 'memory_past.pt')
                torch.save(self.mem_n2n.memory_fut, self.folder_test + 'memory_fut.pt')

        # save predict result
        pre_save_path = self.folder_test + 'pred_results.json'
        with open(pre_save_path, 'w', encoding='utf-8') as file:
            json.dump(pred_list, file)
            print("successfully writing prediction in the json file")

        return dict_metrics

    def _memory_writing(self, saved_memory):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """

        if saved_memory:
            self.mem_n2n.memory_past = torch.load(self.config.memories_path + 'memory_past.pt')
            self.mem_n2n.memory_fut = torch.load(self.config.memories_path + 'memory_fut.pt')

            # self.mem_n2n.memory_past = self.mem_n2n.memory_past[5915:]
            # self.mem_n2n.memory_fut = self.mem_n2n.memory_fut[5915:]

            # config = self.config
            # with torch.no_grad():
            #     for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, graph)\
            #                 in enumerate(tqdm.tqdm(self.train_loader)):
            #         past = Variable(past)
            #         future = Variable(future)
            #         if config.cuda:
            #             past = past.cuda()
            #             future = future.cuda()
            #         self.mem_n2n.write_in_memory(past, future, graph['x'], graph['edge_index'], graph['edge_weight'])
            #
            #     # save memory
            #     torch.save(self.mem_n2n.memory_past, self.folder_test + 'memory_past.pt')
            #     torch.save(self.mem_n2n.memory_fut, self.folder_test + 'memory_fut.pt')

        else:
            self.mem_n2n.init_memory(self.data_train)
            config = self.config
            with torch.no_grad():
                for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, graph)\
                            in enumerate(tqdm.tqdm(self.train_loader)):
                    past = Variable(past)
                    future = Variable(future)
                    if config.cuda:
                        past = past.cuda()
                        future = future.cuda()
                    # if self.config.withIRM:
                    #     scene_one_hot = Variable(scene_one_hot)
                    #     scene_one_hot = scene_one_hot.cuda()
                    #     self.mem_n2n.write_in_memory(past, future, scene_one_hot)
                    # else:
                    #     self.mem_n2n.write_in_memory(past, future)

                    self.mem_n2n.write_in_memory(past, future, graph['x'], graph['edge_index'], graph['edge_weight']) #

                # save memory
                torch.save(self.mem_n2n.memory_past, self.folder_test + 'memory_past.pt')
                torch.save(self.mem_n2n.memory_fut, self.folder_test + 'memory_fut.pt')
