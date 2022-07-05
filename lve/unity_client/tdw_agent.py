from typing import Dict, List, Union
from tdw.controller import Controller
from tdw.librarian import SceneLibrarian
import random
import numpy as np
from skimage.color import rgb2hsv
from tdw.output_data import Categories, Images, OutputData, Transforms, CompositeObjects, SegmentationColors, \
    NavMeshPath, Bounds, AvatarNonKinematic, EnvironmentCollision

from tdw.tdw_utils import TDWUtils


from lve.unity_client.agent import UnityAgent

def saturate(x):
    return np.clip(x, 0, 1)

def Hue(H):
    R = abs(H * 6 - 3) - 1
    G = 2 - abs(H * 6 - 2)
    B = 2 - abs(H * 6 - 4)
    return saturate(np.dstack([R,G,B]))

def pseudohsv2rgb(HSV):
    return ((Hue(HSV[..., 0]) - 1) * np.expand_dims(HSV[..., 1], 2) + 1) * np.expand_dims(HSV[..., 2], 2)

def rgb2pseudohsv(RGB):
    HSV = np.zeros_like(RGB)

    HSV[:, :, 2] = np.max(RGB, axis=2)
    M = np.min(RGB, axis=2)
    C = HSV[..., 2] - M

    with np.errstate(divide='ignore', invalid='ignore'):
        HSV[..., 1] = C / HSV[..., 2]
        Delta = (HSV[..., 2, np.newaxis] - RGB) / C[..., np.newaxis]
        Delta -= Delta[..., [2, 0, 1]]
        Delta[..., [0, 1]] += np.array([2, 4])

        HSV[..., 0] = np.where(RGB[..., 0] >= HSV[..., 2],
                               Delta[..., 2],
                               np.where(RGB[..., 1] >= HSV[..., 2], Delta[..., 0], Delta[..., 1]))

        HSV[..., 0] = HSV[..., 0] / 6
        HSV[..., 0] = HSV[..., 0] - np.floor(HSV[..., 0])

    HSV = np.where((HSV[..., 2] == 0.0)[..., np.newaxis], np.zeros_like(HSV), HSV)
    return HSV

class TDWAgent(UnityAgent):
    def __init__(self, 
                 main_frame_active: bool = True, 
                 object_frame_active: bool = True, 
                 category_frame_active: bool = True, 
                 flow_frame_active: bool = True, 
                 depth_frame_active: bool = True, 
                 host: str = "localhost", 
                 port: int = 1071, 
                 width: int = 512, 
                 height: int = 384,
                 controller: Controller = None):
        super().__init__(
            main_frame_active=main_frame_active, 
            object_frame_active=object_frame_active, 
            category_frame_active=category_frame_active, 
            flow_frame_active=flow_frame_active, 
            depth_frame_active=depth_frame_active, 
            host=host, 
            port=port, 
            width=width, 
            height=height
        )

        if controller is None:
            self.controller = Controller()

            self.controller.start()
        else:
            self.controller = controller
            # we assume that start was already called by the owner of the controller

        # set unknown id before registering
        self.id = -1

        # retrieve scenes from SceneLibrarian
        self._scenes = [scene_record.name for scene_record in SceneLibrarian().records] + ["ProcGenScene"]
        
        # before retrieving categories, we need to load a scene
        self._categories = {}
        self._category2color = {}


    # contains the ids that were already used
    __used_ids = []

    @classmethod
    def generate_random_id(cls):
        """
        Generate a random ID for the agent.
        """

        return "".join([str(random.randint(0, 9)) for _ in range(4)])

        # while True:
        #     id = chr(random.randint(ord('a'), ord('z')))
        #     if id not in cls.__used_ids:
        #         cls.__used_ids.append(id)
        #         return id

    @property
    def scenes(self):
        return self._scenes

    @property
    def categories(self):
        return self._categories

    def __get_masks(self):
        """
        Get the masks of the agent.
        """
        return [
            key for key, flag in zip(
                ["_img", "_category", "_id", "_flow", "_depth_simple"],
                [
                    self.main_frame_active,
                    self.category_frame_active,
                    self.object_frame_active,
                    self.flow_frame_active,
                    self.depth_frame_active
                ]
            ) 
            if flag # filter the masks with flag = False
        ]

    def __mask_to_viewid(self, mask: str):
        return {
            "_img": "main",
            "_category": "category",
            "_id": "id",
            "_flow": "flow",
            "_depth_simple": "depth"
        }[mask]

    def __mask_to_shape(self, mask: str):
        return {
            "_img": (self.height, self.width, 3),
            "_category": (self.height, self.width, 3),
            "_id": (self.height, self.width, 3),
            "_flow": (self.height, self.width, 3),
            "_depth_simple": (self.height, self.width, 1)
        }[mask]

    def __update_categories_dictionaries(self, categories: Categories):
        categories_dict = {}
        categories_colors = {}

        for category_idx in range(0, categories.get_num_categories()):
            categories_dict[category_idx] = categories.get_category_name(category_idx)
            categories_colors[category_idx] = categories.get_category_color(category_idx)

        self._categories = categories_dict
        self._category2color = categories_colors

    def __convert_category_img_to_category_ids(self, category_img: np.ndarray):
        # TODO: use 255 as fill value, possibly make it configurable
        category_ids = np.full((*category_img.shape[:-1], 1), 255, dtype=np.uint8) 

        for category_idx, category_color in self._category2color.items():
            category_ids[(category_img == category_color).all(axis=-1)] = category_idx

        return category_ids

    def __convert_flow_img_to_raw_flow(self, flow_img: np.ndarray):
        flow_hsv = rgb2pseudohsv(flow_img / 255.0)
        value = flow_hsv[:, :, 2]
        hue = flow_hsv[:, :, 0]
        angle = (hue - 0.5) * 2 * np.pi
        flow_uv = np.zeros((flow_hsv.shape[0], flow_hsv.shape[1], 2))
        flow_uv[..., 0] = - value * np.cos(angle) * self.width
        flow_uv[..., 1] = - value * np.sin(angle) * self.height
        return flow_uv

    def __communicate(self, messages: Union[Dict, List[Dict]]):
        resp = self.controller.communicate(messages)
        return resp[:-1]

    def refresh_categories(self):
        # load categories in scene and update categories dictionaries
        categories_resp = self.__communicate({"$type": "send_categories", "frequency": "once"})
        categories = Categories(categories_resp[0])
        self.__update_categories_dictionaries(categories)

    def create_avatar(self, type='A_Img_Caps_Kinematic'):
        """
        Create an avatar for the agent.
        """

        self.__communicate({
            "$type": "set_screen_size",
            "width": self.width,
            "height": self.height
        })

        self.__communicate({
                "$type": "create_avatar",
                "type": type,
                "id": self.id
        })

        self.__communicate({
                "$type": "set_pass_masks",
                "avatar_id": self.id,
                "pass_masks": self.__get_masks()
        })

    def register(self):
        """
        Register the agent with the controller.
        """
        
        # generate the agent id
        self.id = self.generate_random_id()

        # set the screen size
        resp = self.__communicate({
            "$type": "set_screen_size", 
            "width": self.width, 
            "height": self.height
        }) 

    def delete(self):
        """
        Delete the agent from the controller.
        """
        self.__communicate(
            {"$type": "destroy_avatar", "avatar_id": self.id}
        )

    def change_scene(self, scene_name: str):

        # check if the scene is in the list of scenes
        if scene_name not in self._scenes:
            raise ValueError(f"Scene {scene_name} not in list of scenes")

        # TODO: if proc gen scene is selected, do not use load_streamed_scene but common load scene
        if scene_name != "ProcGenScene":
            self.controller.load_streamed_scene(scene_name)
        else:
            resp = self.__communicate({
                "$type": "load_scene",
                "scene_name": scene_name
            })        

        # we also need to recreate the avatar
        self.__create_avatar()

        self.refresh_categories()
        

    def get_frame(self, commands_list=[]):
        """
        Get the frame of the agent.
        """
        frame = {
            "sizes": {
                "main": 0,
                "category": 0,
                "object": 0,
                "flow": 0,
                "depth": 0
            },
            "deltaTime": 0,
            "main": None,
            "category": None,
            "object": None,
            "flow": None,
            "depth": None,
            "transforms": {},
            "avatars": {},
            "collisions": {}
        }



        resp = self.__communicate([
            {
                "$type": "send_images",
                "frequency": "once",
                "avatar_id": self.id
            }] + commands_list
        )

        for r in resp:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "comp":
                co = CompositeObjects(r)
                frame['objects'] = {}
                for co_idx in range(co.get_num()):
                    frame['objects'][co.get_object_id(co_idx)] = []
                    for co_sub_idx in range(co.get_num_sub_objects(co_idx)):
                        frame['objects'][co.get_object_id(co_idx)].append({
                            co.get_sub_object_id(co_idx, co_sub_idx): co.get_sub_object_machine_type(co_idx, co_sub_idx)
                        })
            if r_id == "segm":
                sc = SegmentationColors(r)
                frame['segmentation'] = {}
                for sc_idx in range(sc.get_num()):
                    frame['segmentation'][sc.get_object_id(sc_idx)] = {
                        'cat':sc.get_object_category(sc_idx),
                        'name': sc.get_object_name(sc_idx),
                        'color': sc.get_object_color(sc_idx)
                    }

            if r_id == "path":
                nav_mesh_path = NavMeshPath(r)
                frame['path'] = nav_mesh_path.get_path()

            if r_id == "boun":
                b = Bounds(r)
                for b_idx in range(b.get_num()):
                    frame['bounds'][b.get_id(b_idx)] = {'front': b.get_front(b_idx),
                                                        'back': b.get_back(b_idx),
                                                        'top': b.get_top(b_idx),
                                                        'bottom': b.get_bottom(b_idx),
                                                        'left': b.get_left(b_idx),
                                                        'right': b.get_right(b_idx),
                                                        'center': b.get_center(b_idx)
                                                        }
            if r_id == "avnk":
                a = AvatarNonKinematic(r)
                frame['avatars'][a.get_avatar_id()] = {
                    'position': a.get_position(),
                    'velocity': a.get_velocity(),
                    'angular_velocity': a.get_angular_velocity(),
                    'sleeping': a.get_sleeping()
                }
            if r_id == "enco":
                c = EnvironmentCollision(r)
                frame['collisions'] = {
                    'type': 'env',
                    'obj': c.get_object_id(),
                    'state': c.get_state()
                }
            if r_id == "tran":
                t = Transforms(r)
                for t_idx in range(t.get_num()):
                    frame['transforms'][t.get_id(t_idx)] = {'position': t.get_position(t_idx),
                                                            'rotation': t.get_rotation(t_idx)}
            if r_id == "imag":
                img = Images(r)

                for pass_idx in range(img.get_num_passes()):
                    view_id = self.__mask_to_viewid(img.get_pass_mask(pass_idx))
                    view = np.array(TDWUtils.get_pil_image(img, pass_idx))

                    # convert category view to category ids
                    if view_id == "category":
                        view = self.__convert_category_img_to_category_ids(view)
                    elif view_id == "flow":
                        frame['orig_flow'] = view
                        view = self.__convert_flow_img_to_raw_flow(view)
                        view = view

                    frame[view_id] = view
                    frame["sizes"][view_id] = view.size


                # for pass_id, mask in enumerate(self.__get_masks()):
                #     view = img.get_image(pass_id)
                #     img.get_
                #     view_id = self.__mask_to_viewid(mask)

                #     frame["sizes"][view_id] = view.shape[0]
                #     view = view.reshape(*self.__mask_to_shape(mask))
                    
                #     frame[view_id] = view
                
        frame["deltaTime"] = 0 #TODO: maybe estimate it, or leave it to zero if it is not important
        
        return frame

    def change_main_camera_flags(self, r: int, g: int, b: int):
        pass

    def toggle_follow(self):
        pass
        