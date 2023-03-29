import osmnx
import pandas as pd
import json
import interaction.utils.lanelet2_parser as parser
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import cv2
import numpy as np
import geopandas as gpd
import xml.etree.ElementTree as xml
import osmium as osm
import math
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from interaction.utils.map_vis_without_lanelet import LL2XYProjector, get_x_y_lists, XY2LLProjector


class Rationality():
    def __init__(self, map_data, G):
        self.map_data = map_data
        self.osm_g = G
        self.projector = XY2LLProjector(0.0, 0.0)

    def compute(self, track):
        self.s1 = 0  # destination outside drivable area
        self.s2 = 0  # trajectory direction is abnormal with the lanelet
        self.s3 = 0  # avg speed above 15mph
        self.s4 = 0  # irrational lane change

        track_line = LineString(track)
        f_x, f_y = track[-1]
        o_x, o_y = track[0]

        # s1:not contains in drivable area
        if not self.map_data.drivable_polygon.contains(track_line):
            self.s1 = 1
            return self.s1, self.s2, self.s3, self.s4

        o_lane_key = self.find_lanelet(o_x, o_y)
        f_lane_key = self.find_lanelet(f_x, f_y)
        # f_lon, f_lat = self.projector.xy2latlon(f_x, f_y)
        # nearest_n, dist = osmnx.distance.nearest_nodes(self.osm_g, f_lon, f_lat, return_dist=True)
        o_lanelet = self.map_data.lanelets[o_lane_key]
        f_lanelet = self.map_data.lanelets[f_lane_key]


        # s2: irrational direction
        # difference between lanelet's left bound and track direction

        for p in range(1, len(track), 5):
            p_x, p_y = track[p]
            l_x, l_y = track[p-1]

            lane_key = self.find_lanelet(p_x, p_y)
            lanelet_p = self.map_data.lanelets[lane_key]
            right_bound = lanelet_p.right_bound.linestring.coords

            lane_v = [right_bound[-1][0] - right_bound[0][0], right_bound[-1][1] - right_bound[0][1]]
            track_v = [p_x-l_x, p_y-l_y]
            angle = self.angle_diff(lane_v, track_v)
            if abs(angle) > 60:
                self.s2 = 1

        # plt.figure()
        # plt.plot(lanelet_p.polygon.exterior.xy[0], lanelet_p.polygon.exterior.xy[1])
        # plt.plot(right_bound.xy[0], right_bound.xy[1])
        # plt.plot([l_x, p_x], [l_y, p_y])
        # plt.scatter(right_bound.xy[0][-1], right_bound.xy[1][-1])

        # s3: irratianal speed
        # speed > 15mph = 0.44704m/s
        speed_limit = 6.7056
        if track_line.length/3 >= speed_limit:
            self.s3 = abs(track_line.length/3 - speed_limit) / speed_limit

        # s4: irrational lane_change
        if (o_lanelet.left_bound == f_lanelet.right_bound) or (o_lanelet.right_bound == f_lanelet.left_bound):
            track_v = [f_x-o_x, f_y-o_y]
            angle = self.angle_diff(lane_v,track_v)
            if (angle < 0 and o_lanelet.left_bound.lanechange_tag != 'yes') or \
                (angle > 0 and o_lanelet.right_bound.lanechange_tag != 'yes'):
                self.s4 = 1

        return self.s1, self.s2, self.s3, self.s4

    def find_lanelet(self, x, y):
        for key in list(self.map_data.lanelets.keys()):
            if self.map_data.lanelets[key].polygon.contains(Point(x, y)):
                return key

        return None

    def lane_regulatory(self, lanelet):
        rules = []
        for ele in range(len(lanelet.regulatory_elements)):
            rules.append({
                'id': lanelet.regulatory_elements[ele].id_,
                'type': lanelet.regulatory_elements[ele].subtype,
                'sign_tag': lanelet.regulatory_elements[ele].sign_tag
            })
        return rules

    def angle_diff(self, lane_v, track_v):
        # angle:(-pi,pi], angle between x axis and o->xy
        # lean leftward: angle<0, lean rightward: angle>0
        angle1 = math.atan2(lane_v[1], lane_v[0])*180/math.pi
        angle1_v = math.atan2(lane_v[1], lane_v[0])*180/math.pi + 180
        angle2 = math.atan2(track_v[1], track_v[0])*180/math.pi
        if angle1 < 0:
            angle1 = angle1 + 360
        if angle2 < 0:
            angle2 = angle2 + 360
        angle = min(abs(angle1-angle2), abs(angle1_v-angle2))
        return angle


def traj2LL(traj):
    projector = XY2LLProjector(0.0, 0.0)
    traj_ll = []
    for i in range(len(traj)):
        lon, lat = projector.xy2latlon(traj[i][0], traj[i][1])
        traj_ll.append([lon, lat])
    return np.array(traj_ll)


def track_plot(pred, osm_gpd):
    colors = pl.cm.OrRd(np.linspace(1, 0.3, len(pred)))
    osm_gpd.plot()
    for t in range(len(pred)-1, -1, -1):
        traj_ll =traj2LL(pred[t])
        plt.plot(traj_ll[:, 0], traj_ll[:, 1], color=colors[t])
    plt.show()


class RationCompute():
    def __init__(self, osm_file):
        # read osm file to lanelets2 format
        self.filename = osm_file
        self.data = parser.MapData(buffer_=0)
        self.data.parse(self.filename, align_range=0.5)

        # to graph and gpd
        self.osm_gpd = osmnx.geometries.geometries_from_xml(self.filename)
        self.osm_g = osmnx.graph.graph_from_xml(self.filename)

        self.ra = Rationality(self.data, self.osm_g)

    def evaluate(self, pred, dist):
        pred = np.array(pred)
        s = []
        for j in range(len(pred)):
            traj = pred[j]
            s1, s2, s3, s4 = self.ra.compute(traj)
            s.append((s1 * 0.5 + s2 * 0.3 + s3 * 0.1 + s4 * 0.1))

        s = np.array(s)
        min_dist_idx = sorted(range(len(dist[0])), key=lambda k: dist[0][k], reverse=False)[0]
        rat_idx = np.array(list(range(0, 5))) / 10 + s
        rat_idx = sorted(range(len(rat_idx)), key=lambda k: rat_idx[k], reverse=False)
        sorted_s = s[rat_idx]

        # top 3 rationality
        o_r = 1 - sum(s[:3]) / 3
        r_r = 1 - sum(sorted_s[:3]) / 3
        # top 3 dist
        n_dist = np.array(dist[0][rat_idx])
        o_d = sum(np.array(dist[0][:3])) / 3
        r_d = sum(n_dist[:3]) / 3
        new_min_idx = rat_idx.index(min_dist_idx)
        # print('original rationality: {}, rationalized:{}, original min idx:{}, rationalized:{}'
        #       .format(o_r, r_r, min_dist_idx, new_min_idx))

        n_pred = pred[rat_idx]
        return o_r, r_r, o_d, r_d, min_dist_idx, new_min_idx, n_pred

    def traj2LL(self, traj):
        projector = XY2LLProjector(0.0, 0.0)
        traj_ll = []
        for i in range(len(traj)):
            lon, lat = projector.xy2latlon(traj[i][0], traj[i][1])
            traj_ll.append([lon, lat])
        return np.array(traj_ll)

    def track_plot(self, pred, future, past, surr, path, step):
        colors = pl.cm.OrRd(np.linspace(1, 0.3, len(pred)))
        self.osm_gpd.plot(linestyle='--')
        for t in range(len(pred)-1, -1, -1):
            traj_ll = traj2LL(pred[t])
            plt.plot(traj_ll[:, 0], traj_ll[:, 1], color=colors[t])
        for t in range(len(surr)):
            traj_ll = traj2LL(surr[t])
            plt.plot(traj_ll[:10, 0], traj_ll[:10, 1], color='purple')
            if (traj_ll[-10][0] != traj_ll[-12][0]) & (traj_ll[-10][1] != traj_ll[-12][1]):
                plt.arrow(x=traj_ll[-12][0], y=traj_ll[-12][1], dx=traj_ll[-10][0] - traj_ll[-12][0],
                          dy=traj_ll[-10][1] - traj_ll[-12][1], width=0.00001, head_width=0, fc='aquamarine', edgecolor='aquamarine')
        future_ll = traj2LL(future)
        past_ll = traj2LL(past)
        plt.plot(future_ll[:, 0], future_ll[:, 1], color='g')
        plt.plot(past_ll[:, 0], past_ll[:, 1], color='b')
        # plt.savefig(path + '/{}.png'.format(step), dpi=200)



if __name__ == '__main__':
    # read osm file to lanelets2 format
    filename = r"D:\data\INTERACTION-Dataset-DR-multi-v1_2\maps\DR_DEU_Roundabout_OF.osm"
    data = parser.MapData(buffer_=0)
    data.parse(filename, align_range=0.5)

    # to graph and gpd
    osm_gpd = osmnx.geometries.geometries_from_xml(filename)
    osm_g = osmnx.graph.graph_from_xml(filename)

    # predict trajectory
    json_path = r'test/2023-01-31 16_DR_DEU_Roundabout_OF_offline/pred_results.json'
    tracks = json.load(open(json_path))
    tracks = pd.DataFrame(tracks)

    ra = Rationality(data, osm_g)

    for i in range(len(tracks)):
        pred = np.array(tracks['pred'][i])
        dist = tracks['distance'][i]
        track_plot(pred, osm_gpd)
        s = []
        for j in range(len(pred)):
            traj = pred[j]
            s1, s2, s3, s4 = ra.compute(traj)
            s.append((s1*0.5 + s2*0.3 + s3*0.1 + s4*0.1))
            print("s1:{}, s2:{}, s3:{} s4:{}, s:{}".format(s1, s2, s3, s4, s))

        s = np.array(s)
        min_dist_idx = sorted(range(len(dist[0])), key=lambda k:dist[0][k], reverse=False)[0]
        rat_idx = np.array(list(range(0,5)))/10 + s
        rat_idx = sorted(range(len(rat_idx)), key=lambda k:rat_idx[k], reverse=False)
        sorted_s = s[rat_idx]

        # top 3 rationality
        o_r = 1 - sum(s[:3])/3
        r_r = 1 - sum(sorted_s[:3])/3
        new_min_idx = rat_idx.index(min_dist_idx)
        print('original rationality: {}, rationalized:{}, original min idx:{}, rationalized:{}'
              .format(o_r, r_r, min_dist_idx, new_min_idx))

        n_pred = pred[rat_idx]
        track_plot(n_pred, osm_gpd)




# x, y = track_line.xy;plt.plot(x, y)
# p_x, p_y = data.drivable_polygon.geoms[0].exterior.xy
# plt.plot(p_x, p_y)





# projector = LL2XYProjector(0.0, 0.0)
# projector1 = XY2LLProjector(0.0, 0.0)
#
# e = xml.parse(filename).getroot()
# point_dict = dict()
# for node in e.findall("node"):
#     point = Point()
#     point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
#     point_dict[int(node.get('id'))] = point
#
# for way in e.findall('way'):
#     x_list, y_list = get_x_y_lists(way, point_dict)
#
# [x, y] = [1022.0149, 952.52631]
# X, Y = projector1.xy2latlon(x, y)
# # X:lon, Y:lat
# (ne, dist) = osmnx.distance.nearest_edges(osm_g, X, Y, interpolate=None, return_dist=True)


        # matRot_track = cv2.getRotationMatrix2D((0, 0), -angle, 1)
        # past = cv2.transform(past.cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
        # future = cv2.transform(future.cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()



