import json
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from torch_geometric.data import Data
from itertools import cycle, islice
import cv2
import math
from scipy import spatial
import scipy.sparse as sp
import ijson
import pdb


# colormap
colors = [(0, 0, 0), (0.87, 0.87, 0.87), (0.54, 0.54, 0.54), (0.29, 0.57, 0.25)]
cmap_name = 'scene_list'
cm = LinearSegmentedColormap.from_list(
    cmap_name, colors, N=4)


class TrackDataset(data.Dataset):
    """
    Dataset class for KITTI.
    The building class is merged into the background class
    0:background 1:street 2:sidewalk, 3:building 4: vegetation ---> 0:background 1:street 2:sidewalk, 3: vegetation
    """
    def __init__(self, tracks, len_past=20, len_future=40, train=False, dim_clip=180, neighbor_distance=10, graph_train=True):

        self.tracks = tracks      # dataset dict
        self.dim_clip = dim_clip  # dim_clip*2 is the dimension of scene (pixel)
        self.is_train = train

        self.video_track = []     # '0001'
        self.vehicles = []        # 'Car'
        self.number_vec = []      # '4'
        self.index = []           # '50'
        self.pasts = []           # [len_past, 2]
        self.presents = []        # position in complete scene
        self.angle_presents = []  # trajector
        self.neighbor_distance = neighbor_distance

        # y angle in complete scene
        self.futures = []         # [len_future, 2]
        self.scene = []           # [dim_clip, dim_clip, 1], scene fot qualitative examples
        self.scene_crop = []      # [dim_clip, dim_clip, 4], input to IRM

        # graph
        self.graph = []
        self.surr_traj = []       # [visible_number, len_past, 2]
        # self.edge_index = []      # [2, edge_number]
        # self.edge_weight = []       # [edge_number, ]

        num_total = len_past + len_future
        self.video_split, self.ids_split_test = self.get_desire_track_files(train)

        # save as json 1. straight 2. turning 3. interaction 4. accelerate 5. decelerate
        self.dict_list = []
        self.straight = []
        self.turning = []
        self.interaction = []
        self.acc = []
        self.dec = []

        if train:
            json_path = 'graph_dataset/graph_dataset_train.json'
        else:
            json_path = 'graph_dataset/graph_dataset_test.json'

        check_flag = 0

        # Preload data
        for video in self.video_split:
            vehicles = self.tracks[video].keys()
            video_id = video[-9:-5]
            # print('video: ' + video_id)
            video_sample_n = 0
            path_scene = 'maps/2011_09_26__2011_09_26_drive_' + video_id + '_sync_map.png'
            scene_track = cv2.imread(path_scene, 0)
            scene_track_onehot = scene_track.copy()

            # Remove building class
            scene_track_onehot[np.where(scene_track_onehot == 3)] = 0
            scene_track_onehot[np.where(scene_track_onehot == 4)] -= 1

            # Removing still object
            temp_list = list(vehicles).copy()
            for vec1 in vehicles:
                if np.var(tracks[video][vec1]['trajectory'][0]) < 0.1 and np.var(tracks[video][vec1]['trajectory'][1]) < 0.1:
                    temp_list.remove(vec1)

            for vec in temp_list:
                class_vec = tracks[video][vec]['cls']
                num_vec = vec.split('_')[1]
                start_frame = tracks[video][vec]['start']
                end_frame = tracks[video][vec]['end']
                points = np.array(tracks[video][vec]['trajectory']).T
                len_track = len(points)
                for count in range(0, len_track, 1):
                    if len_track - count > num_total:
                        tag = []

                        last_frame = count + start_frame + len_past - 1
                        temp_past = points[count:count + len_past].copy()
                        temp_future = points[count + len_past:count + num_total].copy()
                        # non-centralize by minus position of last history frame
                        origin = temp_past[-1]

                        # filter out noise for non-moving vehicles
                        if np.var(temp_past[:, 0]) < 0.1 and np.var(temp_past[:, 1]) < 0.1:
                            temp_past = np.zeros((20, 2))
                        else:
                            temp_past = temp_past - origin

                        if np.var(temp_past[:, 0]) < 0.1 and np.var(temp_past[:, 1]) < 0.1:
                            temp_future = np.zeros((40, 2))
                        else:
                            temp_future = temp_future - origin

                        scene_track_clip = scene_track[
                                           int(origin[1]) * 2 - self.dim_clip:int(origin[1]) * 2 + self.dim_clip,
                                           int(origin[0]) * 2 - self.dim_clip:int(origin[0]) * 2 + self.dim_clip]

                        scene_track_onehot_clip = scene_track_onehot[
                                                  int(origin[1]) * 2 - self.dim_clip:int(origin[1]) * 2 + self.dim_clip,
                                                  int(origin[0]) * 2 - self.dim_clip:int(origin[0]) * 2 + self.dim_clip]

                        # rotation invariance
                        unit_y_axis = torch.Tensor([0, -1])
                        vector = temp_past[-5]
                        if vector[0] > 0.0:
                            angle = np.rad2deg(self.angle_vectors(vector, unit_y_axis))
                        else:
                            angle = -np.rad2deg(self.angle_vectors(vector, unit_y_axis))
                        matRot_track = cv2.getRotationMatrix2D((0, 0), angle, 1)
                        matRot_scene = cv2.getRotationMatrix2D((self.dim_clip, self.dim_clip), angle, 1)

                        past_rot = cv2.transform(temp_past.reshape(-1, 1, 2), matRot_track).squeeze()
                        future_rot = cv2.transform(temp_future.reshape(-1, 1, 2), matRot_track).squeeze()
                        scene_track_onehot_clip = cv2.warpAffine(scene_track_onehot_clip, matRot_scene,
                                           (scene_track_onehot_clip.shape[0], scene_track_onehot_clip.shape[1]),
                                           borderValue=0,
                                           flags=cv2.INTER_NEAREST)  # (1, 0, 0, 0)

                        video_sample_n = video_sample_n + 1

                        # self.scene_crop.append(scene_track_onehot_clip)
                        self.index.append(count + 19 + start_frame)
                        self.pasts.append(past_rot)
                        self.futures.append(future_rot)
                        self.presents.append(origin)
                        self.angle_presents.append(angle)
                        self.video_track.append(video_id)
                        self.vehicles.append(class_vec)
                        self.number_vec.append(num_vec)
                        # self.scene.append(scene_track_clip)

                        if graph_train:
                            x_traj, edge_index, edge_weight = self.graph_generation(tracks[video], temp_list,
                                                                                    last_frame, vec)
                            # rotation invariance
                            for i in range(len(x_traj)):
                                x_traj[i] = x_traj[i] - origin
                                x_traj[i] = cv2.transform(x_traj[i].reshape(-1, 1, 2), matRot_track).squeeze()
                                x_traj[i] = x_traj[i].tolist()

                            g_data = dict(x=torch.FloatTensor(x_traj), edge_index=torch.LongTensor(edge_index),
                                          edge_weight=torch.FloatTensor(edge_weight))

                            self.graph.append(g_data)
                            # self.surr_traj.append(torch.FloatTensor(x_traj))

                            # save to local file
                            # columns: [index, past, future, presents, angle_presents, videos,
                            #           vehicles, number_vec, scene, scene_one_hot, graph]
                            #
                            # "scene_one_hot": scene_track_onehot_clip.tolist(),
                            graph_dict = {'index': str(count + 19 + start_frame), "past": past_rot.tolist(),
                                          "future": future_rot.tolist(),
                                          "presents": origin.tolist(), "angle_presents": str(angle), "videos": video_id,
                                          "vehicles": class_vec,
                                          "number_vec": num_vec,  "scene": scene_track_clip.tolist(), "x": x_traj,
                                          "edge_index": edge_index.tolist(), "edge_weight": edge_weight.tolist()}
                            # self.dict_list.append(graph_dict)

                            # divide different trajectory condition
                            # 1. straight 2. turning 3. interaction 4. accelerate 5. decelerate
                            track_all = np.concatenate((past_rot, future_rot), axis=0)
                            s_diff = (track_all[::5][1:][:, 0] - track_all[::5][:-1][:, 0]) ** 2 + \
                                     (track_all[::5][1:][:, 1] - track_all[::5][:-1][:, 1]) ** 2

                            if np.var(track_all[:, 0]) > 0.3:
                                # self.turning.append(graph_dict)
                                tag.append("turning")

                            elif np.var(track_all[:, 0]) < 0.1:
                                # self.straight.append(graph_dict)
                                tag.append('straight')

                            if np.sum(s_diff[1:] - s_diff[:-1] <= -1) > 6:
                                # self.dec.append(graph_dict)
                                tag.append('dec')

                            elif np.sum(s_diff[1:] - s_diff[:-1] >= 1) > 6:
                                # self.acc.append(graph_dict)
                                tag.append('acc')

                            if edge_index.shape[1]/2 > 1:
                                # self.interaction.append(graph_dict)
                                tag.append('interaction')


            print('video: ' + video_id + 'sample number:' + str(video_sample_n))

        self.index = np.array(self.index)
        self.pasts = torch.FloatTensor(self.pasts)
        self.futures = torch.FloatTensor(self.futures)
        self.presents = torch.FloatTensor(self.presents)
        self.video_track = np.array(self.video_track)
        self.vehicles = np.array(self.vehicles)
        self.number_vec = np.array(self.number_vec)
        # self.scene = np.array(self.scene)

        # self.surr_traj = torch.FloatTensor(self.surr_traj)
        # self.edge_index = torch.LongTensor(self.edge_index)
        # self.edge_weight = torch.FloatTensor(self.edge_weight)

        # print("straight:{}".format(len(self.straight)))
        # print("Turning:{}".format(len(self.turning)))
        # print("acc:{}".format(len(self.acc)))
        # print("dec:{}".format(len(self.dec)))
        # print("interaction:{}".format(len(self.interaction)))

        # with open(json_path, 'w', encoding='utf-8') as file:
        #     json.dump(self.dict_list, file)
        #     print("successfully writing in the json file")

    def graph_generation(self, track, vehicles, last_frame, vec, len_past=20):
        # object vehicle id
        vec_id = vehicles.index(vec)
        # record the exits frame ang type of all agent in this scene
        frame_start = []
        frame_end = []
        class_all = []
        for i in vehicles:
            frame_start.append(track[i]['start'])
            frame_end.append(track[i]['end'])
            class_all.append(track[i]['cls'])
        frame_start = np.array(frame_start)
        frame_end = np.array(frame_end)
        class_all = np.array(class_all)

        # agents appear in the last history frame
        visible_object_id_list = np.intersect1d(np.where(frame_start <= last_frame), np.where(frame_end >= last_frame))
        visible_class = class_all[list(visible_object_id_list)]

        # change object index in the graph to 0
        vec_index = np.where(visible_object_id_list == vec_id)[0][0]
        visible_object_id_list = np.roll(visible_object_id_list, -vec_index)

        # coordinate in last frame and history trajectory of all agent
        xy = []
        traj = []
        for i in visible_object_id_list:
            last_frame_v = last_frame - frame_start[i]
            tra_index = vehicles[i]
            xy.append([track[tra_index]['trajectory'][0][last_frame_v], track[tra_index]['trajectory'][1][last_frame_v]])
            if last_frame_v - len_past + 1 >= 0:
                t_x = track[tra_index]['trajectory'][0][last_frame_v - len_past + 1:last_frame_v + 1]
                t_y = track[tra_index]['trajectory'][1][last_frame_v - len_past + 1:last_frame_v + 1]
            else:
                t_x = track[tra_index]['trajectory'][0][0:last_frame_v + 1]
                t_y = track[tra_index]['trajectory'][1][0:last_frame_v + 1]
            # padding surrounding agent trajectory to the length of past history
            traj.append((np.array([list(np.pad(t_x, (len_past-len(t_x), 0), mode='edge')),
                                       list(np.pad(t_y, (len_past-len(t_y), 0), mode='edge'))]).T).tolist())

        # compute the distance between each agent
        dist_xy = spatial.distance.cdist(xy, xy)
        # only regard agent which distance less than neighbor_distance as neighbors
        dist_xy[dist_xy > self.neighbor_distance] = 0
        # normalization and adverse the attribute(long distance has less impact)
        # norm_dist = 1 - dist_xy / np.linalg.norm(dist_xy)
        norm_dist = self.neighbor_distance - dist_xy
        norm_dist[norm_dist == self.neighbor_distance] = 0
        # transfer adjacency matrix to list of vertex index and attribute
        edge_index_temp = sp.coo_matrix(norm_dist)
        edge_weight = edge_index_temp.data
        edge_index = np.vstack((edge_index_temp.row, edge_index_temp.col))

        return traj, edge_index, edge_weight

    def save_dataset(self, folder_save):
        for i in range(self.pasts.shape[0]):
            video = self.video_track[i]
            vehicle = self.vehicles[i]
            number = self.number_vec[i]
            past = self.pasts[i]
            future = self.futures[i]
            scene_track = self.scene_crop[i]

            saving_list = ['only_tracks', 'only_scenes', 'tracks_on_scene']
            for sav in saving_list:
                folder_save_type = folder_save + sav + '/'
                if not os.path.exists(folder_save_type + video):
                    os.makedirs(folder_save_type + video)
                video_path = folder_save_type + video + '/'
                if not os.path.exists(video_path + vehicle + number):
                    os.makedirs(video_path + vehicle + number)
                vehicle_path = video_path + '/' + vehicle + number + '/'
                if sav == 'only_tracks':
                    self.draw_track(past, future, index_tracklet=self.index[i], path=vehicle_path)
                if sav == 'only_scenes':
                    self.draw_scene(scene_track, index_tracklet=self.index[i], path=vehicle_path)
                if sav == 'tracks_on_scene':
                    self.draw_track_in_scene(past, scene_track, index_tracklet=self.index[i], future=future, path=vehicle_path)

    def draw_track(self, past, future, index_tracklet, path):
        plt.plot(past[:, 0], -past[:, 1], c='blue', marker='o', markersize=1)
        if future is not None:
            future = future.cpu().numpy()
            plt.plot(future[:, 0], -future[:, 1], c='green', marker='o', markersize=1)
        plt.axis('equal')
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()

    def draw_multi_track(self, past, future, surr_tracks, index_tracklet, path):
        for i in range(len(surr_tracks)):
            plt.plot(surr_tracks[i][:, 0], -surr_tracks[i][:, 1], marker='o', markersize=1)
        if future is not None:
            # future = future.cpu().numpy()
            plt.plot(future[:, 0], -future[:, 1], c='green', marker='o', markersize=1)
        plt.plot(past[:, 0], -past[:, 1], c='blue', marker='o', markersize=1)
        plt.axis('equal')
        plt.show()
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()


    def draw_scene(self, scene_track, index_tracklet, path):
        # print semantic map
        cv2.imwrite(path + str(index_tracklet) + '.png', scene_track)

    def draw_track_in_scene(self, story, scene_track, index_tracklet, future=None, path=''):
        plt.imshow(scene_track, cmap=cm)
        plt.plot(story[:, 0] * 2 + self.dim_clip, story[:, 1] * 2 + self.dim_clip, c='blue', marker='o', markersize=1)
        plt.plot(future[:, 0] * 2 + self.dim_clip, future[:, 1] * 2 + self.dim_clip, c='green', marker='o', markersize=1)
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()

    @staticmethod
    def get_desire_track_files(train):
        """ Get videos only from the splits defined in DESIRE: https://arxiv.org/abs/1704.04394
        Splits obtained from the authors:
        all: [1, 2, 5, 9, 11, 13, 14, 15, 17, 18, 27, 28, 29, 32, 48, 51, 52, 56, 57, 59, 60, 70, 84, 91]
        train: [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
        test: [1, 2, 15, 18, 29, 32, 52, 70]
        new_train: [11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91, 52, 70]
        new_test: [1, 5, 9, 2, 15, 18, 29, 32]
        """
        if train:
            desire_ids = [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
        else:
            desire_ids = [1, 2, 15, 18, 29, 32, 52, 70]

        tracklet_files = ['video_2011_09_26__2011_09_26_drive_' + str(x).zfill(4) + '_sync'
                          for x in desire_ids]
        return tracklet_files, desire_ids

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_vectors(self, v1, v2):
        """ Returns angle between two vectors.  """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        if math.isnan(angle):
            return 0.0
        else:
            return angle

    def __len__(self):
        return self.pasts.shape[0]

    def __getitem__(self, idx):
        # np.eye(4, dtype=np.float32)[self.scene_crop[idx]], , self.graph[idx], self.scene[idx], self.scene_crop[idx], self.presents[idx],
        # (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene,
        #                        graph)
        return self.index[idx], self.pasts[idx], self.futures[idx], self.presents[idx], self.angle_presents[idx], self.video_track[idx], \
               self.vehicles[idx], self.number_vec[idx], self.graph[idx]  # , self.surr_traj[idx] self.scene[idx],


class trackIterableDataset(data.IterableDataset):
    def __init__(self, file_path, train=True):
        self.file_path = file_path
        self.train = train

    def __len__(self):
        if self.train:
            return 7063  # 7063
        else:
            return 2312  # 2312
        # return self.pasts.shape[0]

    def __iter__(self):
        with open(self.file_path, "r", encoding='utf-8') as file_obj:
            objects = ijson.items(file_obj, "item")
            # objects_cycle = cycle(objects)
            while True:
                try:
                    dict_data = objects.__next__()
                    # dict_data = next(objects_cycle)
                    self.index = np.array(int(dict_data['index']))
                    self.pasts = torch.FloatTensor(dict_data['past'])
                    self.futures = torch.FloatTensor(dict_data['future'])
                    self.presents = torch.FloatTensor(dict_data['presents'])
                    self.angle_presents = np.array(float(dict_data['angle_presents']))
                    self.video_track = dict_data['videos']
                    # self.vehicles =
                    self.number_vec = dict_data['number_vec']
                    self.scene = np.array(dict_data['scene'])
                    # self.scene_crop = np.eye(4, dtype=np.float32)[np.array(dict_data['scene_one_hot'])]
                    self.graph = dict(x=torch.FloatTensor(dict_data['x']),
                                      edge_index=torch.LongTensor(dict_data['edge_index']),
                                      edge_weight=torch.FloatTensor(dict_data['edge_weight']))

                except:  # StopIteration as e
                    if self.train:
                        objects = ijson.items(file_obj, "item")
                    else:
                        break

                yield self.index, self.pasts, self.futures, self.presents, self.angle_presents, self.video_track, \
                      self.number_vec, self.scene, self.graph

class InteractDatasetITR(data.IterableDataset):
    def __init__(self, json_path, train=True):
        self.train = train
        self.tracks = pd.DataFrame(json.load(open(json_path)))

    def __len__(self):
        return len(self.tracks)

    def __iter__(self):

        for i in range(len(self.tracks)):
            surr_traj = torch.FloatTensor(self.tracks['x'][i])
            edge_index = torch.LongTensor(self.tracks['edge_index'][i])
            edge_weight = torch.FloatTensor(self.tracks['edge_weight'][i])
            self.index = self.tracks['index'][i]
            self.pasts = torch.FloatTensor(self.tracks['past'][i])
            self.futures = torch.FloatTensor(self.tracks['future'][i])
            self.presents = torch.FloatTensor(self.tracks['presents'][i])
            self.video_track = self.tracks['videos'][i]
            self.vehicles = self.tracks['vehicles'][i]
            self.number_vec = self.tracks['number_vec'][i]
            self.angle_presents = self.tracks['angle_presents'][i]

            self.graph = dict(x=torch.FloatTensor(surr_traj), edge_index=torch.LongTensor(edge_index),
                          edge_weight=torch.FloatTensor(edge_weight))


            yield self.index, self.pasts, self.futures, self.presents, self.angle_presents, self.video_track, \
                  self.vehicles, self.number_vec, self.graph


class InteractionDataset(data.Dataset):
    """
    Dataset class for Interaction.

    """
    def __init__(self, json_path=None, data_name='interaction', tracks=None, len_past=10, len_future=30, train=False, dim_clip=180, neighbor_distance=5, graph_train=True,
                 data_generate=False):

        self.len_past = len_past
        self.len_future = len_future
        self.train = train
        self.data_generate = data_generate
        self.graph_train = graph_train
        self.data_name = data_name[:-4]

        self.tracks = tracks

        # dataset dict (csv:'case_id', 'track_id', 'frame_id', 'timestamp_ms', 'agent_type', 'x','y', 'vx', 'vy',
        # 'psi_rad', 'length', 'width')
        self.dim_clip = dim_clip  # dim_clip*2 is the dimension of scene (pixel)
        self.is_train = train

        self.video_track = []     # '1'
        self.vehicles = []        # 'car'
        self.number_vec = []      # '4'
        self.index = []           # '1'
        self.pasts = []           # [len_past, 2]
        self.presents = []        # position in complete scene
        self.angle_presents = []  # trajector
        self.neighbor_distance = neighbor_distance

        # y angle in complete scene
        self.futures = []         # [len_future, 2]
        self.scene = []           # [dim_clip, dim_clip, 1], scene fot qualitative examples
        self.scene_crop = []      # [dim_clip, dim_clip, 4], input to IRM

        # graph
        self.graph = []
        self.surr_traj = []       # [visible_number, len_past, 2]
        self.edge_index = []      # [2, edge_number]
        self.edge_weight = []       # [edge_number, ]

        # self.video_split, self.ids_split_test = self.get_desire_track_files(train)

        if self.data_generate:
            self.data_generation()
        else:

            self.tracks = pd.DataFrame(json.load(open(json_path)))

            self.index = np.array(self.tracks['index'])
            self.pasts = torch.FloatTensor(self.tracks['past'])
            self.futures = torch.FloatTensor(self.tracks['future'])
            self.presents = torch.FloatTensor(self.tracks['presents'])
            self.video_track = np.array(self.tracks['videos'])
            self.vehicles = np.array(self.tracks['vehicles'])
            self.number_vec = np.array(self.tracks['number_vec'])
            self.angle_presents = np.array(self.tracks['angle_presents'])

            if self.graph_train:
                for i in range(len(self.tracks)):
                    surr_traj = torch.FloatTensor(self.tracks['x'][i])
                    edge_index = torch.LongTensor(self.tracks['edge_index'][i])
                    edge_weight = torch.FloatTensor(self.tracks['edge_weight'][i])
                    self.graph.append(dict(x=torch.FloatTensor(surr_traj), edge_index=torch.LongTensor(edge_index),
                                      edge_weight=torch.FloatTensor(edge_weight)))

    def data_generation(self):
        # save as json
        self.dict_list = []

        json_path = 'graph_dataset/{}.json'.format(self.data_name)

        check_flag = 0
        print("total case number: {}".format(len(self.tracks['case_id'].unique())))

        # Preload data
        for video_id in self.tracks['case_id'].unique():
            print("generating case {}".format(video_id))
            # if video_id == 500 & self.train:
            #     break
            # if not self.train & video_id == 100:
            #     break
            case = self.tracks[self.tracks['case_id'] == video_id]
            vehicles = case['track_id'].unique()

            # Removing still object
            temp_list = list(vehicles).copy()
            # for vec1 in vehicles:
            #     if np.var(case[case['track_id'] == vec1]['x']) < 0.1 and np.var(case[case['track_id'] == vec1]['y']) < 0.1:
            #         temp_list.remove(vec1)

            for vec in temp_list:
                case_vec = case[case['track_id'] == vec]
                case_vec.index = range(len(case_vec))
                if len(case_vec) <= 39:
                    continue

                class_vec = case_vec['agent_type'][0]
                temp_past = np.array(case[(case['track_id'] == vec) & (case['frame_id'] <= self.len_past)][['x', 'y']])
                temp_future = np.array(case[(case['track_id'] == vec) & (case['frame_id'] > self.len_past)][['x', 'y']])
                origin = temp_past[-1]

                temp_past = temp_past - origin
                temp_future = temp_future - origin

                # rotation invariance
                unit_y_axis = torch.Tensor([0, -1])
                vector = temp_past[-5]
                if vector[0] > 0.0:
                    angle = np.rad2deg(self.angle_vectors(vector, unit_y_axis))
                else:
                    angle = -np.rad2deg(self.angle_vectors(vector, unit_y_axis))
                matRot_track = cv2.getRotationMatrix2D((0, 0), angle, 1)
                # matRot_scene = cv2.getRotationMatrix2D((self.dim_clip, self.dim_clip), angle, 1)

                past_rot = cv2.transform(temp_past.reshape(-1, 1, 2), matRot_track).squeeze()
                future_rot = cv2.transform(temp_future.reshape(-1, 1, 2), matRot_track).squeeze()
                # scene_track_onehot_clip = cv2.warpAffine(scene_track_onehot_clip, matRot_scene,
                #                                          (scene_track_onehot_clip.shape[0],
                #                                           scene_track_onehot_clip.shape[1]),
                #                                          borderValue=0,
                #                                          flags=cv2.INTER_NEAREST)  # (1, 0, 0, 0)

                self.index.append(9)
                self.pasts.append(past_rot)
                self.futures.append(future_rot)
                self.presents.append(origin)
                self.angle_presents.append(angle)
                self.video_track.append(video_id)
                self.vehicles.append(class_vec)
                self.number_vec.append(vec)
                # self.scene.append(scene_track_clip)

                if self.graph_train:
                    x_traj, edge_index, edge_weight = self.graph_generation(case, temp_list, vec)
                    # rotation invariance
                    for i in range(len(x_traj)):
                        x_traj[i] = x_traj[i] - origin
                        x_traj[i] = cv2.transform(x_traj[i].reshape(-1, 1, 2), matRot_track).squeeze()
                        x_traj[i] = x_traj[i].tolist()

                    g_data = dict(x=torch.FloatTensor(x_traj), edge_index=torch.LongTensor(edge_index),
                                  edge_weight=torch.FloatTensor(edge_weight))

                    self.graph.append(g_data)

                    # save to local file
                    # columns: [index, past, future, presents, angle_presents, videos,
                    #           vehicles, number_vec, graph]
                    #
                    # "scene_one_hot": scene_track_onehot_clip.tolist(),
                    graph_dict = {'index': 19, "past": past_rot.tolist(),
                                  "future": future_rot.tolist(),
                                  "presents": origin.tolist(), "angle_presents": str(angle),
                                  "videos": video_id,
                                  "vehicles": class_vec,
                                  "number_vec": int(vec),
                                  "x": x_traj, "edge_index": edge_index.tolist(), "edge_weight": edge_weight.tolist()
                                  }
                    self.dict_list.append(graph_dict)


        self.index = np.array(self.index)
        self.pasts = torch.FloatTensor(self.pasts)
        self.futures = torch.FloatTensor(self.futures)
        self.presents = torch.FloatTensor(self.presents)
        self.video_track = np.array(self.video_track)
        self.vehicles = np.array(self.vehicles)
        self.number_vec = np.array(self.number_vec)
        # self.scene = np.array(self.scene)
        # self.surr_traj = torch.FloatTensor(self.surr_traj)
        # self.edge_index = torch.LongTensor(self.edge_index)
        # self.edge_weight = torch.FloatTensor(self.edge_weight)

        # with open(json_path, 'w', encoding='utf-8') as file:
        #     json.dump(self.dict_list, file)
        #     print("successfully writing in the json file")

    def graph_generation(self, case, vehicles, vec, len_past=10):
        # object vehicle id
        vec_id = vehicles.index(vec)
        # record the exits frame ang type of all agent in this scene
        frame_start = []
        frame_end = []
        class_all = []
        for i in vehicles:
            frame_start.append(case[case['track_id'] == i]['frame_id'].min())
            frame_end.append(case[case['track_id'] == i]['frame_id'].max())
            class_all.append(case[case['track_id'] == i]['agent_type'].iloc[0])
        frame_start = np.array(frame_start)
        frame_end = np.array(frame_end)
        class_all = np.array(class_all)

        # agents appear in the last history frame
        visible_object_id_list = np.intersect1d(np.where(frame_start < len_past), np.where(frame_end >= len_past))
        visible_class = class_all[list(visible_object_id_list)]

        # change object index in the graph to 0
        vec_index = np.where(visible_object_id_list == vec_id)[0][0]
        visible_object_id_list = np.roll(visible_object_id_list, -vec_index)

        # coordinate in last frame and history trajectory of all agent
        xy = []
        traj = []
        for i in visible_object_id_list:
            last_frame_v = len_past - frame_start[i]
            tra_index = vehicles[i]
            xy.append([case[case['track_id'] == tra_index]['x'].iloc[last_frame_v],
                       case[case['track_id'] == tra_index]['x'].iloc[last_frame_v]])
            if last_frame_v - len_past + 1 >= 0:
                t_x = list(case[case['track_id'] == tra_index]['x'])[last_frame_v - len_past + 1:last_frame_v + 1]
                t_y = list(case[case['track_id'] == tra_index]['y'])[last_frame_v - len_past + 1:last_frame_v + 1]
            else:
                t_x = list(case[case['track_id'] == tra_index]['x'])[0:last_frame_v + 1]
                t_y = list(case[case['track_id'] == tra_index]['y'])[0:last_frame_v + 1]
            # padding surrounding agent trajectory to the length of past history
            traj.append((np.array([list(np.pad(t_x, (len_past-len(t_x), 0), mode='edge')),
                                       list(np.pad(t_y, (len_past-len(t_y), 0), mode='edge'))]).T).tolist())

        # compute the distance between each agent
        dist_xy = spatial.distance.cdist(xy, xy)
        # only regard agent which distance less than neighbor_distance as neighbors
        dist_xy[dist_xy > self.neighbor_distance] = 0
        # no car have interaction with each other
        if np.any(dist_xy):
            # normalization and adverse the attribute(long distance has less impact)
            norm_dist = 1 - dist_xy / np.linalg.norm(dist_xy)
            # norm_dist = self.neighbor_distance - dist_xy
            norm_dist[norm_dist == 1] = 0
        else:
            norm_dist = dist_xy.copy()

        # transfer adjacency matrix to list of vertex index and attribute
        edge_index_temp = sp.coo_matrix(norm_dist)
        edge_weight = edge_index_temp.data
        edge_index = np.vstack((edge_index_temp.row, edge_index_temp.col))

        return traj, edge_index, edge_weight

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_vectors(self, v1, v2):
        """ Returns angle between two vectors.  """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        if math.isnan(angle):
            return 0.0
        else:
            return angle

    def __len__(self):
        return self.pasts.shape[0]

    def __getitem__(self, idx):
        # np.eye(4, dtype=np.float32)[self.scene_crop[idx]], , self.graph[idx], self.scene[idx], self.scene_crop[idx]
        if self.graph_train:
            return self.index[idx], self.pasts[idx], self.futures[idx], self.presents[idx], self.angle_presents[idx], self.video_track[idx], \
                   self.vehicles[idx], self.number_vec[idx], self.graph[idx]
            # , self.surr_traj[idx], self.edge_index[idx], self.edge_weight[idx]
        else:
            return self.index[idx], self.pasts[idx], self.futures[idx], self.presents[idx], self.angle_presents[idx], self.video_track[idx], \
                   self.vehicles[idx], self.number_vec[idx]

