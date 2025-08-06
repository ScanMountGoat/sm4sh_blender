import bpy
import time
import numpy as np

from sm4sh_blender.import_model import init_logging

from . import sm4sh_model_py

from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty


class ImportPac(bpy.types.Operator, ImportHelper):
    """Import a Smash 4 animation"""

    bl_idname = "import_scene.pac"
    bl_label = "Import Pac"

    filename_ext = ".mot"

    filter_glob: StringProperty(
        default="*.pac",
        options={"HIDDEN"},
        maxlen=255,
    )

    def execute(self, context: bpy.types.Context):
        init_logging()

        import_pac(self, context, self.filepath)
        return {"FINISHED"}


def import_pac(operator: bpy.types.Operator, context: bpy.types.Context, path: str):
    start = time.time()

    animations = sm4sh_model_py.animation.load_animations(path)

    end = time.time()
    print(f"Load {len(animations)} Animations: {end - start}")

    start = time.time()

    if context.object is None or not isinstance(
        context.object.data, bpy.types.Armature
    ):
        operator.report({"ERROR"}, "No armature selected")
        return

    armature = context.object
    if armature.animation_data is None:
        armature.animation_data_create()

    # TODO: Recreate the skeleton to avoid needing the path.
    nud_path = armature.get("original_nud")
    model = sm4sh_model_py.load_model(nud_path)
    skeleton = model.skeleton

    bone_names = {bone.hash: bone.name for bone in skeleton.bones}

    for i, bone in enumerate(skeleton.bones):
        if bone.parent_bone_index is not None and bone.parent_bone_index > i:
            print(f"invalid index {bone.parent_bone_index} > {i}")

    # TODO: Is this the best way to load all animations?
    # TODO: Optimize this.
    for name, animation in animations:
        action = import_animation(armature, skeleton, bone_names, name, animation)
        armature.animation_data.action = action

    end = time.time()
    print(f"Import Blender Animation: {end - start}")


def import_animation(
    armature, skeleton, bone_names: dict[int, str], name: str, animation
):
    action = bpy.data.actions.new(name)
    if animation.frame_count > 0:
        action.frame_end = float(animation.frame_count) - 1.0

    # Reset between each animation.
    for bone in armature.pose.bones:
        bone.matrix_basis.identity()

    fcurves = animation.fcurves(skeleton, use_blender_coordinates=True)
    locations = fcurves.translation
    rotations_xyzw = fcurves.rotation
    scales = fcurves.scale

    for hash, values in locations.items():
        if name := bone_names.get(hash):
            set_fcurves_component(action, name, "location", values[:, 0], 0)
            set_fcurves_component(action, name, "location", values[:, 1], 1)
            set_fcurves_component(action, name, "location", values[:, 2], 2)

    for hash, values in rotations_xyzw.items():
        if name := bone_names.get(hash):
            # Blender uses wxyz instead of xyzw.
            set_fcurves_component(action, name, "rotation_quaternion", values[:, 3], 0)
            set_fcurves_component(action, name, "rotation_quaternion", values[:, 0], 1)
            set_fcurves_component(action, name, "rotation_quaternion", values[:, 1], 2)
            set_fcurves_component(action, name, "rotation_quaternion", values[:, 2], 3)

    for hash, values in scales.items():
        if name := bone_names.get(hash):
            set_fcurves_component(action, name, "scale", values[:, 0], 0)
            set_fcurves_component(action, name, "scale", values[:, 1], 1)
            set_fcurves_component(action, name, "scale", values[:, 2], 2)

    return action


def set_fcurves_component(
    action, bone_name: str, value_name: str, values: np.ndarray, i: int
):
    # Values can be quickly set in the form [frame, value, frame, value, ...]
    # Assume one value at each frame index for now.
    keyframe_points = np.zeros((values.size, 2), dtype=np.float32)
    keyframe_points[:, 0] = np.arange(values.size)
    keyframe_points[:, 1] = values

    # Each coordinate of each value has its own fcurve.
    data_path = f'pose.bones["{bone_name}"].{value_name}'
    fcurve = action.fcurves.new(data_path, index=i, action_group=bone_name)
    fcurve.keyframe_points.add(count=values.size)
    fcurve.keyframe_points.foreach_set("co", keyframe_points.reshape(-1))
