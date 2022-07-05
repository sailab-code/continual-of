import numpy as np
from scipy.spatial.transform import Rotation


class StreamBuilder():
    def __init__(self, random_seed=0, fps=25, def_time_step=0.001):
        self.transforms = None
        self.path = None
        self.bounds = None
        self.avatars = None
        self.fps = fps
        np.random.seed(random_seed)
        self.def_time_step = def_time_step

    def get_time_step(self, i):
        return self.def_time_step

    def process_init_commands(self):
        pass

    def get_object_id_by_name(self, commands, name):
        l = []
        for c in commands:
            if c['$type'] == 'add_object' and c['name'] == name:
                l.append(c['id'])
        if len(l) == 1: return l[0]
        else: return l

    def create_rotate_command(self, id, euler_angles=None, quaternion=None):
        if quaternion is None:
            rot = Rotation.from_euler('xyz', [euler_angles['x'], euler_angles['y'], euler_angles['z']], degrees=True).as_quat()
            quaternion = {'x': rot[0], 'y': rot[1], 'z': rot[2], 'w': rot[3]}
        return {'$type': 'rotate_object_to', 'rotation': quaternion, 'id': id}

    def create_move_command(self, id, position):
        return {'$type': 'teleport_object', 'position': position, 'id': id}

    def update_info(self, frame):
        if 'transforms' in frame: self.transforms = frame['transforms']
        if 'path' in frame:  self.path = frame['path']
        if 'bounds' in frame:  self.bounds = frame['bounds']
        if 'avatars' in frame:  self.avatars = frame['avatars']


    def get_min_dist_from_scene_bounds(self, id):
        current_transform = self.transforms[id]
        current_pos = current_transform['position']
        current_pos = {'x': current_pos[0], 'y': current_pos[1], 'z': current_pos[2]}
        dist = []
        for axis in self.view_bounds.keys():
            for t in ['min', 'max']:
                dist.append(abs(current_pos[axis] - self.view_bounds[axis][t]))
        return min(dist)


    def get_view_center(self):
        pos = {}
        for axis in self.view_bounds.keys():
            pos[axis] = np.mean(list(self.view_bounds[axis].values()))
        return pos

    def get_delta_to_center(self, pos):
        v = {'x': 0.0, 'y':0.0, 'z':0.0}
        pos = {'x': pos[0], 'y': pos[1], 'z': pos[2]}
        for axis in self.get_view_center().keys():
            v[axis] = self.get_view_center()[axis] - pos[axis]
        return v


