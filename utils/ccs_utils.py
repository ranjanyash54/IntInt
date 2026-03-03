from scipy.optimize import minimize
import math
import numpy as np
from dtw import dtw

"""Cluster id to direction_id, maneuver_id, lane_id"""
cluster2attr = {
    0: (0, 0, 0),
    1: (0, 2, 2),
    2: (0, 1, 2),
    3: (0, 1, 1),

    4: (1, 0, 0),
    5: (1, 0, 0),
    6: (1, 2, 1),
    7: (1, 1, 1),

    8: (2, 0, 0),
    9: (2, 2, 2),
    10: (2, 1, 2),
    11: (2, 1, 1)
}

"""Direction_id, maneuver_id to list of cluster_ids"""
attr2cluster = {(0, 0): [0],
 (0, 1): [2, 3],
 (0, 2): [1],
 (1, 0): [4, 5],
 (1, 1): [7],
 (1, 2): [6],
 (2, 0): [8],
 (2, 1): [10, 11],
 (2, 2): [9]}

new_old_clus_map = {
    0 : 9,
    1: 10,
    2: 11,
    3: 8,
    4: 6,
    5: 7,
    6: 4,
    7: 1,
    8: 2,
    9: 3,
    10: 0
}

def convert_to_img_coord(pts_ccs, centroid_spl):
    """
    Convert the points in curvilinear coordinates to image coordinates.
    """
    x_spl, y_spl = centroid_spl
    xds, yds = x_spl.derivative(), y_spl.derivative()
    res_pts = []
    for pt in pts_ccs:
        curr_s, dis = pt
        perpen_vec = np.array([-yds(curr_s), xds(curr_s)])
        perpen_vec = perpen_vec / np.linalg.norm(perpen_vec)
        perpen_vec = perpen_vec * dis
        x = x_spl(curr_s) + perpen_vec[0]
        y = y_spl(curr_s) + perpen_vec[1]
        res_pts.append((x,y))
    return res_pts


def convert_to_CCS_coord(pts, centroid_spl):
    """
    Convert the points in image coordinates to curvilinear coordinates.
    """
    x_spl, y_spl = centroid_spl

    def Q(s, x):
        """Distance function to be minimized
        """
        return (x_spl(s[0]) - x[0]) ** 2 + (y_spl(s[0]) - x[1]) ** 2

    def isleft(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0

    last_s = 0
    s_list = []
    dist_list = []
    mask = []
    for x, y in pts:
        bounds = [(-1, len(x_spl.breaks))]
        res = minimize(Q, last_s, args=([x, y]), bounds=bounds, method='L-BFGS-B')
        curr_s = res.x[0]
        if (curr_s < 0) or (curr_s > len(x_spl.breaks) - 1):
            curr_s = 0 if curr_s < 0 else len(x_spl.breaks) - 1
            s_list.append(curr_s)
            if len(dist_list) > 0:
                dist_list.append(dist_list[-1])
            else:
                dist_list.append(0)
            mask.append(0)
            continue
        mask.append(1)
        sign = isleft((x_spl(curr_s - 0.1), y_spl(curr_s - 0.1)), (x_spl(curr_s), y_spl(curr_s)), (x, y))
        s_list.append(curr_s)

        if (sign):
            dist_list.append(-math.sqrt(res.fun))
        else:
            dist_list.append(math.sqrt(res.fun))

        last_s = curr_s

    return np.array([(x, y) for (x, y) in zip(s_list, dist_list)]), np.array(mask)


def find_matching_centroid(track_seq, centroid_df_in, cluster_list = None):
    """
    Find the matching cluster of a query trajectory. If cluster_list is not None, then only the cluster_id in cluster_list
    will be considered.
    :param cluster_list: List of target cluster IDs.
    """
    try:
        if cluster_list:
            centroid_df = centroid_df_in[centroid_df_in.id.isin(cluster_list)]
        else:
            centroid_df = centroid_df_in
        cluster_names = centroid_df['name'].unique()
        min_dis = 100
        matched_centroid = ''
        track_match = []
        centroid_match = []
        if len(track_seq) == 1: track_seq.append(track_seq[0])

        for name in cluster_names:
            cluster = centroid_df[centroid_df['name'] == name]
            cluster_seq = list(zip(cluster.center_x, cluster.center_y))
            alignment = dtw(track_seq, cluster_seq, open_begin=True, open_end=True, step_pattern='asymmetricP0')
            dis = alignment.distance / len(np.unique(alignment.index1s))

            if dis < min_dis:
                min_dis = dis
                matched_centroid = name
                track_match = alignment.index1s
                centroid_match = alignment.index2s
    except:
        import pdb; pdb.set_trace()

    return matched_centroid, min_dis, track_match, centroid_match


def process_ccs_for_vehicles(data_df, centroid_dict):
    """
    This function will change the data_df in place. It calculates the curvilinear coordinates of input trajectories and
    adds 'ccs_x' and 'ccs_y' columns to data_df.
    """
    try:
        centroid_spl_dict = centroid_dict['centroid_spl_dict']
        vehicle_df = data_df[data_df['class'] == 0.]
        centroid_df = centroid_dict['centroid_df']
        centroid_map = centroid_dict['centroid_map']
        cluster_id2name = {i: c for c, i in centroid_map.items()}
        for track_id in vehicle_df.track_id.unique():
            track = vehicle_df[vehicle_df.track_id == track_id]
            cluster_id = track.cluster.iloc[0]
            track_seq = list(zip(track.pos_x, track.pos_y))
            cluster_list = None
            if 'direction_id' in track.columns:
                dir_id, man_id = track.direction_id.iloc[0], track.maneuver_id.iloc[0]
                cluster_list = attr2cluster[(dir_id, man_id)]
            # cluster_id, min_dis, _, _ = find_matching_centroid(track_seq, centroid_df, cluster_list)
            # data_df.loc[data_df.track_id == track_id, 'cluster'] = centroid_map[cluster_id]
            track_ccs, mask = convert_to_CCS_coord(list(zip(track.pos_x, track.pos_y)), centroid_spl_dict[cluster_id])
            # vehicle_df.loc[index, 'mask'] = mask
            # import pdb; pdb.set_trace()
            data_df.loc[data_df.track_id == track_id, 'ccs_x'] = np.array(track_ccs)[:, 0]
            data_df.loc[data_df.track_id == track_id, 'ccs_y'] = np.array(track_ccs)[:, 1]
    except:
        import pdb; pdb.set_trace()

def process_ccs(data_df, centroid_dict):
    """
    This function will change the data_df in place. It matches the vehicle trajectories to cluster centroids and
    calculates the curvilinear coordinates for vehicle trajectories.
    """
    process_ccs_for_vehicles(data_df, centroid_dict)
    data_df.loc[data_df['class'] == 1., 'cluster'] = -1.
    data_df.loc[data_df['class'] == 1., 'ccs_x'] = data_df[data_df['class'] == 1.].pos_x
    data_df.loc[data_df['class'] == 1., 'ccs_y'] = data_df[data_df['class'] == 1.].pos_y
    # print (data_df)
