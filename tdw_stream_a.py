import argparse
import traceback
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from lve.unity_client.tdw_agent import TDWAgent
from tdw_streams.stream_builders import StreamBuilder

from tdw_streams.utils import showInMovedWindow, CommandsBuffer, Force, StreamSaver
# v1.8
#from lve.unity_client.tdw_controllers import CachedFloorPlanController

# v1.9
from tdw.controller import Controller
from tdw.add_ons.floorplan import Floorplan
from lve.unity_client.tdw_controllers19 import CachedFloorPlanController

parser = argparse.ArgumentParser(description='Stream A generation')
parser.add_argument('--minutes', type=float, default=60)

args_cmd = parser.parse_args()

class AStreamBuilder(StreamBuilder):
    ## floorplan 2a, little room with laptop and sofa
    def __init__(self, agent, commands_buf, fps=25, def_time_step=0.001):
        super().__init__(fps=fps, def_time_step=def_time_step)
        self.avatar_initial_pos = {"x": 0.862, "y": 0.995, "z": 2.046}
        self.avatar_initial_euler = {"x": 0, "y": -90, "z": 0.0}
        self.view_bounds = {'y':{'min': 0.615, 'max':1.39}, 'z': {'min': 1.613, 'max':2.467}}
        self.laptop_force = Force()
        self.laptop_torque = Force()
        self.agent = agent
        self.commands_buf = commands_buf

    def process_init_commands(self, commands):
        print(commands)
        new_commands = []
        #new_commands.append({"$type": "add_material", "name": "glass_clear", "url": "file:///"+str(Path('./cache/files/glass_clear').absolute())})
        new_commands.append({
            "$type": "teleport_avatar_to",
            "position": self.avatar_initial_pos,
            "avatar_id": self.agent.id
        })

        new_commands.append({"$type": "rotate_avatar_to_euler_angles",
                             "euler_angles": self.avatar_initial_euler,
                             "avatar_id": self.agent.id})
        new_commands.append({"$type": "set_time_step", "time_step": self.def_time_step})

        # identify laptop object
        self.laptop_id = self.get_object_id_by_name(commands, name='macbook_air')
        self.framed_painting_id = self.get_object_id_by_name(commands, name='framed_painting')[0]

        # move laptop
        new_commands.append(self.create_rotate_command(self.laptop_id,
                                                       euler_angles={'x':0.0, 'y': 90.0, 'z': 0.0}))
        new_commands.append(self.create_move_command(self.laptop_id,
                                                       position={'x': -0.053, 'y': 0.829, 'z': 2.113}))
        new_commands.append({'$type': 'set_kinematic_state', 'id': self.laptop_id, 'is_kinematic': False, 'use_gravity': False})

        # move framed_painting
        new_commands.append(self.create_rotate_command(self.framed_painting_id,
                                                       euler_angles={'x':0.0, 'y': 90.0, 'z': 0.0}))
        new_commands.append(self.create_move_command(self.framed_painting_id,
                                                       position={'x': -2.5227, 'y': 1.251, 'z': 3.028}))
        new_commands.append({"$type": "adjust_point_lights_intensity_by", "intensity": 1.8})
        new_commands.append({"$type": "set_anti_aliasing", "mode": "temporal", "sensor_name": "SensorContainer", "avatar_id": self.agent.id})

        return new_commands

    def get_other_commands(self, i):
        return [{"$type": "send_transforms", "ids": [self.laptop_id], "frequency": "once"},
                {"$type": "set_time_step", "time_step": self.get_time_step(i)}
                #{"$type": "send_bounds", "ids": [self.laptop_id], "frequency": "once"}
                ]

    def get_commands_for_dynamics(self, i, plan='square'):
        cmds = []
        if plan == 'square':
            if i == 0:
                f = {"x": 0, "y": 0, "z": 0.1}
                cmds.append({"$type": "apply_force_to_object", "id": self.laptop_id, "force": f})
                self.laptop_force.add(**f)
            elif i == 10:
                f = {"x": 0, "y": 0.1, "z": -0.1}
                cmds.append({"$type": "apply_force_to_object", "id": self.laptop_id, "force": f})
                self.laptop_force.add(**f)
            elif i == 20:
                f = {"x": 0, "y": -0.1, "z": -0.1}
                cmds.append({"$type": "apply_force_to_object", "id": self.laptop_id, "force": f})
                self.laptop_force.add(**f)
            elif i == 30:
                f = {"x": 0, "y": -0.1, "z": +0.1}
                cmds.append({"$type": "apply_force_to_object", "id": self.laptop_id, "force": f})
                self.laptop_force.add(**f)
            elif i==40:
                f = {"x": 0, "y": +0.1, "z": 0.0}
                cmds.append({"$type": "apply_force_to_object", "id": self.laptop_id, "force": f})
                self.laptop_force.add(**f)
        elif plan == 'random':
            if i==0:
                init_rand = np.random.randn(3)
                f = {"x": 0.0, "y": init_rand[1], "z": init_rand[2]}
                t = {"x": 0.03, "y":0.0, "z":0.0}
                print('initial force', f)
                print('initial torque', t)
                cmds.append({"$type": "apply_force_to_object", "id": self.laptop_id, "force": f})
                cmds.append({"$type": "apply_torque_to_object", "id": self.laptop_id, "torque": t})
                self.laptop_force.add(**f)
                self.laptop_torque.add(**t)
            else:
                current_laptop_transform = self.transforms[self.laptop_id]
                current_laptop_pos = current_laptop_transform['position']
                dist_from_bounds = self.get_min_dist_from_scene_bounds(self.laptop_id)
                if dist_from_bounds < 0.2:
                    l_force = self.laptop_force.get()
                    correction_force = Force(-l_force['x'], -l_force['y'], -l_force['z'])

                    delta_to_center = self.get_delta_to_center(current_laptop_pos)
                    move_force = Force(**delta_to_center)
                    move_force.scale(5.0)

                    move_force.add(**(correction_force.get()))

                    cmds.append({"$type": "apply_force_to_object", "id": self.laptop_id, "force": move_force.get()})
                    self.laptop_force.add(**(move_force.get()))
                else:
                    if np.random.rand() > 0.99:
                        init_rand = np.random.randn(2)
                        init_rand = 0.9 * init_rand / np.linalg.norm(init_rand)
                        print('applying force', init_rand)
                        l_force = self.laptop_force.get()
                        random_force = Force(-l_force['x'], -l_force['y']+init_rand[0], -l_force['z']+init_rand[1])
                        cmds.append({"$type": "apply_force_to_object", "id": self.laptop_id, "force": random_force.get()})
                        self.laptop_force.add(**(random_force.get()))
        cmds.append({"$type": "focus_on_object", "object_id": self.laptop_id, "avatar_id": self.agent.id})
        return cmds

def main():
    try:
        transition_frames = 15
        c = CachedFloorPlanController(launch_build=False, port=1071)
        commands_buf = CommandsBuffer()
        tdw_agent = TDWAgent(
            True, False, True, True, False, controller=c, width=1024, height=1024
        )
        init_commands = c.get_scene_init_commands(scene="2a", layout=0, audio=False)
        print(init_commands)

        saver = StreamSaver('data', stream_name='stream_a', resize=(256, 256), agent=tdw_agent, options=vars(args_cmd))
        builder = AStreamBuilder(tdw_agent, commands_buf)

        c.communicate(init_commands)

        print("register agent")
        tdw_agent.register()
        tdw_agent.create_avatar()
        print("registered agent")

        new_commands = builder.process_init_commands(init_commands)
        print('----------')
        print(new_commands)
        frame = tdw_agent.get_frame(commands_list=new_commands)
        avatar_id = tdw_agent.id

        i = 0
        while i < transition_frames:
            print("getting warmup frame", i)
            frame = tdw_agent.get_frame()
            i += 1

        i = 0
        for i in tqdm(range(int(25*60*args_cmd.minutes))):
            if builder.get_time_step(i) > 0.0:
                commands_buf.push(builder.get_commands_for_dynamics(i, plan='random'))
                commands_buf.push(builder.get_other_commands(i))
                frame = tdw_agent.get_frame(commands_list=commands_buf.get())
                builder.update_info(frame)
            else:
                frame['flow'] *= 0
            saver.save(i, frame=frame['main'], motion=frame['flow'])

            #flow_orig = cv2.cvtColor(frame['orig_flow'], cv2.COLOR_RGB2BGR)

            # normalize flow_orig to 0-255 with cv2
            #flow_orig = cv2.normalize(flow_orig, None, 0, 255, cv2.NORM_MINMAX)

            #showInMovedWindow("Optical Flow (ORIG)", flow_orig, 100, 50)
            #cv2.waitKey(10)
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

