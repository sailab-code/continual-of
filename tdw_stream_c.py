import traceback
from pathlib import Path
import random

import cv2
import numpy as np
from tqdm import tqdm

from lve.unity_client.tdw_agent import TDWAgent
from lve.unity_client.tdw_controllers19 import CachedFloorPlanController
from tdw_streams.stream_builders import StreamBuilder

from tdw_streams.utils import showInMovedWindow, CommandsBuffer, Force, Object, StreamSaver
from tdw.controller import Controller
from tdw.add_ons.floorplan import Floorplan

class CStreamBuilder(StreamBuilder):
    ## floorplan 2a, bedroom
    def __init__(self, agent, commands_buf, random_seed=0):
        super().__init__(random_seed=random_seed)
        self.time_factor = 10
        self.avatar_initial_pos = {"x": -10.21021, "y": 1.8103, "z": 4.90}
        self.avatar_initial_euler = {"x": 19.114, "y": 155.853, "z": -0.006}
        self.avatar_pos = Force(**self.avatar_initial_pos)
        self.avatar_force = Force()
        self.look_at_pos = {"x": -9.187, "y":0.874, "z": 0.382}
        self.next_shift = Force()
        self.objects = {}
        self.agent = agent
        self.commands_buf = commands_buf
        self.low_speed_counter = 0
        self.last_dir_applied = None

    def process_init_commands(self, commands):
        print(commands)
        new_commands = []
        new_commands.append({"$type": "set_time_step", "time_step": 0.01/self.time_factor})
        new_commands.append({"$type": "set_camera_clipping_planes", "near": 0.001, "far": 100, "avatar_id": self.agent.id})
        new_commands.append({
            "$type": "teleport_avatar_to",
            "position": self.avatar_initial_pos,
            "avatar_id": self.agent.id
        })

        new_commands.append({"$type": "rotate_avatar_to_euler_angles",
                             "euler_angles": self.avatar_initial_euler,
                             "avatar_id": self.agent.id})
        new_commands.append({"$type": "set_field_of_view", "field_of_view": 96, "avatar_id": self.agent.id})

        self.objects['black_lamp'] = Object(id=self.get_object_id_by_name(commands, name='black_lamp'))
        self.objects['bed01_blue'] = Object(id=self.get_object_id_by_name(commands, name='bed01_blue'))
        self.objects['trunck'] = Object(id=self.get_object_id_by_name(commands, name='trunck'))
        self.objects['side_table_wood'] = Object(id=self.get_object_id_by_name(commands, name='side_table_wood')[1])

        new_commands.append({"$type": "set_mass", "id": self.objects['bed01_blue'].id,
                             "mass": 1.4})

        new_commands.append({"$type": "set_kinematic_state", "id": self.objects['trunck'].id, "is_kinematic": False, "use_gravity": True})
        new_commands.append({"$type": "set_mass", "id": self.objects['trunck'].id,
                             "mass": 5.4})

        new_commands.append({"$type": "set_kinematic_state", "id": self.objects['side_table_wood'].id, "is_kinematic": False,
                             "use_gravity": True})
        new_commands.append({"$type": "set_mass", "id": self.objects['side_table_wood'].id,
                             "mass": 1.4})

        self.close_box_id = 9999
        new_commands.append({"$type": "load_primitive_from_resources", "primitive_type": "Cube", "id": self.close_box_id,
                       "position": {"x": -5.7, "y": 1.49, "z": 0.479}, "orientation": {"x": 0, "y": 0, "z": 0}})
        new_commands.append({"$type": "scale_object", "id": self.close_box_id,
                             "scale_factor": {"x": 1, "y": 2.96, "z": 1.92}})
        new_commands.append({"$type": "set_mass", "id": self.close_box_id,
                             "mass": 10000.0})
        new_commands.append({"$type": "set_avatar_mass", "mass": 0.125, "avatar_id": self.agent.id})
        new_commands.append({"$type": "set_avatar_drag", "drag": 0.0005, "angular_drag": 0.0005, "avatar_id": self.agent.id})
        new_commands.append({"$type": "scale_avatar", "scale_factor": {"x": 0.01, "y": 0.01, "z": 0.01}, "avatar_id": self.agent.id})
        new_commands.append({"$type": "send_collisions", "enter": True, "stay": False, "exit": False, "collision_types": ["obj", "env"]})
        new_commands.append({"$type": "set_avatar_physic_material", "dynamic_friction": 0.00001, "static_friction":0.00001, "bounciness": 0.4,
         "avatar_id": self.agent.id})
        new_commands.append({"$type": "set_anti_aliasing", "mode": "temporal", "sensor_name": "SensorContainer", "avatar_id": self.agent.id})
        new_commands.append({"$type": "set_avatar_kinematic_state", "is_kinematic": False, "use_gravity": False, "avatar_id":  self.agent.id})
        #new_commands.append({"$type": "set_avatar_rigidbody_constraints", "rotate": True, "translate": True, "avatar_id": self.agent.id}) only for sticky mitten avatar
        return new_commands

    def get_commands_for_transforms(self):
        return [{"$type": "send_avatars", "ids": [self.agent.id], "frequency": "once"}]

    def pick_new_dir(self):
        c = np.random.randint(3)
        dir = np.zeros(3)
        dir[c] = np.random.randint(2) * 2 - 1
        dir = {'x': dir[0], 'y': dir[1], 'z': dir[2]}
        if dir == self.last_dir_applied:
            dir = self.pick_new_dir()
        return dir

    def get_commands_for_dynamics(self, i):
        avatar_current_velocity = None
        if self.avatars is not None:
            avatar_current_velocity = np.linalg.norm(self.avatars[self.agent.id]['velocity'])
            avatar_current_velocity_ = np.max(np.abs(avatar_current_velocity))
        cmds = []
        torque_scale = 0.1
        k = int(100 * self.time_factor * 0.2)

        m = np.random.uniform(0.8, 1.1)
        dir = self.pick_new_dir()

        high_speed_flag = avatar_current_velocity is not None and avatar_current_velocity > 8.5
        high_component_speed_flag = avatar_current_velocity is not None and avatar_current_velocity_ > 6.0
        low_speed_flag = avatar_current_velocity is not None and avatar_current_velocity < 3.0
        if low_speed_flag:
            self.low_speed_counter += 1
        else:
            self.low_speed_counter = 0

        action_on_low_speed = self.low_speed_counter >= 20
        action_on_period = i % k == 0 and not high_speed_flag and not high_component_speed_flag
        if high_speed_flag:
            v = self.avatars[self.agent.id]['velocity']
            dir = np.asarray(v)
            dir /= np.linalg.norm(dir)
            dir *= -1
            self.last_dir_applied = {'x': dir[0], 'y': dir[1], 'z': dir[2]}
            self.avatar_force.add(x=dir[0]*m, y=dir[1]*m, z=dir[2]*m)
            cmds.append({"$type": "apply_force_to_avatar",
                         "magnitude": m,
                         "direction": {'x': dir[0]*m, 'y': dir[1]*m, 'z': dir[2]*m},
                         "avatar_id": self.agent.id})
        elif action_on_period or action_on_low_speed:
            print('action_on_period:', action_on_period, 'action_on_low_speed', action_on_low_speed)
            print('avatar current velocity', avatar_current_velocity)
            print('applying a force ', dir)
            self.last_dir_applied = dir
            self.avatar_force.add(x=dir['x']*m, y=dir['y']*m, z=dir['z']*m)
            print('total force on avatar', self.avatar_force.get())
            cmds.append({"$type": "apply_force_to_avatar",
                         "magnitude": m,
                         "direction": dir,
                         "avatar_id": self.agent.id})

        cmds.append({"$type": "rotate_sensor_container_towards_position",
                     "position": {"x": self.look_at_pos['x'], "y": self.look_at_pos['y'], "z": self.look_at_pos['z']}, "speed": 0.5,
                     "sensor_name": "SensorContainer", "avatar_id": self.agent.id})

        if i % self.time_factor == 0:
            for name, obj in self.objects.items():
                init_rand = 5 * np.random.randn(3*2) * obj.weight_factor
                f = {"x": init_rand[0], "y": init_rand[1], "z": init_rand[2]}
                t = {"x": torque_scale*init_rand[3], "y": torque_scale*init_rand[4], "z": torque_scale*init_rand[5]}
                cmds.append({"$type": "apply_force_to_object", "id": obj.id, "force": f})
                cmds.append({"$type": "apply_torque_to_object", "id": obj.id, "torque": t})
                obj.force.add(**f)
                obj.torque.add(**t)
        return cmds

transition_frames = 70
def main():
    try:
        c = CachedFloorPlanController(launch_build=False, port=1071)
        commands_buf = CommandsBuffer()
        tdw_agent = TDWAgent(
            True, False, True, True, False, controller=c, width=1024, height=1024
        )
        init_commands = c.get_scene_init_commands(scene="2a", layout=0, audio=False)
        print(init_commands)

        builder = CStreamBuilder(tdw_agent, commands_buf, random_seed=1)
        saver = StreamSaver('data', stream_name='stream_c', agent=tdw_agent, resize=(256,256))

        c.communicate(init_commands)

        print("register agent")
        tdw_agent.register()
        # tdw_agent.create_avatar(type='A_Img_Caps_Kinematic')
        tdw_agent.create_avatar(type='A_Img_Caps')
        print("registered agent")

        new_commands = builder.process_init_commands(init_commands)
        print('----------')
        print(new_commands)
        frame = tdw_agent.get_frame(commands_list=new_commands)


        avatar_id = tdw_agent.id
        i = 0
        while i < int(transition_frames):
            print("getting warmup frame", i)
            frame = tdw_agent.get_frame()
            i += 1

        for i in tqdm(range(25*60*60)):
            commands_buf.push(builder.get_commands_for_dynamics(i))
            commands_buf.push(builder.get_commands_for_transforms())
            frame = tdw_agent.get_frame(commands_list=commands_buf.get())
            #print('avatar velocity:', np.linalg.norm(frame['avatars'][tdw_agent.id]['velocity']), frame['avatars'][tdw_agent.id]['velocity'])
            if bool(frame['collisions']):
                pass
                #print('detected collision!', frame['collisions'])
            builder.update_info(frame)
            saver.save(i, frame=frame['main'], motion=frame['flow'])

            #flow_orig = cv2.cvtColor(frame['orig_flow'], cv2.COLOR_RGB2BGR)

            # normalize flow_orig to 0-255 with cv2
            #flow_orig = cv2.normalize(flow_orig, None, 0, 255, cv2.NORM_MINMAX)

            #showInMovedWindow("Optical Flow (ORIG)", flow_orig, 100, 50)
            #cv2.waitKey(10)
            i += 1
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        cv2.waitKey()
        # communicate terminate command
        c.communicate([{"$type": "terminate"}])
        saver.finalize()
        print('saved stream', saver.path)


if __name__ == "__main__":
    main()
