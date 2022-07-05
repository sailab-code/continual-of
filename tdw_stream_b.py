from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from lve.unity_client.tdw_agent import TDWAgent

from tdw_streams.stream_builders import StreamBuilder

from tdw_streams.utils import showInMovedWindow, CommandsBuffer, Force, Object, StreamSaver

# v1.8
# from lve.unity_client.tdw_controllers import CachedFloorPlanController

# v1.9
from tdw.controller import Controller
from tdw.add_ons.floorplan import Floorplan
from lve.unity_client.tdw_controllers19 import CachedFloorPlanController


class BStreamBuilder(StreamBuilder):
    ## floorplan 2a, bedroom
    def __init__(self, agent, commands_buf):
        super().__init__()
        self.avatar_initial_pos = {"x": -10.21021, "y": 1.8103, "z": 4.90}
        self.avatar_initial_euler = {"x": 19.114, "y": 155.853, "z": -0.006}
        self.objects = {}
        self.agent = agent
        self.commands_buf = commands_buf

    def process_init_commands(self, commands):
        print(commands)
        new_commands = []
        new_commands.append({"$type": "set_time_step", "time_step": 0.001})
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

        for obj_name in ['black_lamp', 'trunck', 'side_table_wood']:
            new_commands.append({"$type": "set_kinematic_state", "id": self.objects[obj_name].id, "is_kinematic": False,
                                 "use_gravity": True})
            new_commands.append({"$type": "set_mass", "id": self.objects[obj_name].id, "mass": 1.0})

        self.close_box_id = 9999
        new_commands.append({"$type": "load_primitive_from_resources", "primitive_type": "Cube", "id": self.close_box_id,
                       "position": {"x": -5.7, "y": 1.54, "z": 0.493}, "orientation": {"x": 0, "y": 0, "z": 0}})
        new_commands.append({"$type": "scale_object", "id": self.close_box_id,
                             "scale_factor": {"x": 1, "y": 2.83, "z": 1.67}})
        new_commands.append({"$type": "set_mass", "id": self.close_box_id,
                             "mass": 10000.0})
        return new_commands

    def get_commands_for_transforms(self):
        return []

    def get_commands_for_dynamics(self, i):
        cmds = []
        torque_scale = 0.1
        for name, obj in self.objects.items():
            init_rand = 5 * np.random.randn(3*2)
            f = {"x": init_rand[0], "y": init_rand[1], "z": init_rand[2]}
            t = {"x": torque_scale*init_rand[3], "y": torque_scale*init_rand[4], "z": torque_scale*init_rand[5]}
            cmds.append({"$type": "apply_force_to_object", "id": obj.id, "force": f})
            cmds.append({"$type": "apply_torque_to_object", "id": obj.id, "torque": t})
            obj.force.add(**f)
            obj.torque.add(**t)
        return cmds

transition_frames = 50
def main():
    try:
        c = CachedFloorPlanController(launch_build=False, port=1071)
        commands_buf = CommandsBuffer()
        tdw_agent = TDWAgent(
            True, False, True, True, False, controller=c, width=1024, height=1024
        )
        init_commands = c.get_scene_init_commands(scene="2a", layout=0, audio=False)
        print(init_commands)

        saver = StreamSaver('data', stream_name='stream_b', resize=(256, 256), agent=tdw_agent)
        builder = BStreamBuilder(tdw_agent, commands_buf)

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

        for i in tqdm(range(int(25*60*60))):
            commands_buf.push(builder.get_commands_for_dynamics(i))
            commands_buf.push(builder.get_commands_for_transforms())
            frame = tdw_agent.get_frame(commands_list=commands_buf.get())
            builder.update_info(frame)
            saver.save(i, frame=frame['main'], motion=frame['flow'])

            # flow_orig = cv2.cvtColor(frame['orig_flow'], cv2.COLOR_RGB2BGR)
            #
            # # normalize flow_orig to 0-255 with cv2
            # flow_orig = cv2.normalize(flow_orig, None, 0, 255, cv2.NORM_MINMAX)
            #
            # showInMovedWindow("Optical Flow (ORIG)", flow_orig, 100, 50)
            # cv2.waitKey(10)
            i += 1
    except Exception as e:
        print(e)
    finally:
        cv2.waitKey()
        # communicate terminate command
        c.communicate([{"$type": "terminate"}])
        saver.finalize()
        print('saved stream', saver.path)


if __name__ == "__main__":
    main()
