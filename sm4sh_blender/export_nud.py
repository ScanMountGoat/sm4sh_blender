import bpy
import time
import numpy as np
from pathlib import Path
import re
import os
import itertools

from sm4sh_blender.import_model import init_logging

from .export_model import (
    ExportException,
    export_mesh,
)

from . import sm4sh_model_py

from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty


class ExportNud(bpy.types.Operator, ExportHelper):
    """Export a Smash 4 model"""

    bl_idname = "export_scene.nud"
    bl_label = "Export Nud"

    filename_ext = ".nud"

    filter_glob: StringProperty(
        default="*.nud",
        options={"HIDDEN"},
        maxlen=255,
    )

    def execute(self, context: bpy.types.Context):
        init_logging()

        try:
            export_nud(self, context, self.filepath)
        except ExportException as e:
            self.report({"ERROR"}, str(e))
            return {"FINISHED"}

        return {"FINISHED"}


def name_sort_index(name: str):
    # Use integer sorting for any chunks of chars that are integers.
    # This avoids unwanted behavior of alphabetical sorting like "10" coming before "2".
    return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", name)]


def export_nud(
    operator: bpy.types.Operator,
    context: bpy.types.Context,
    output_nud_path: str,
):
    start = time.time()

    armature = context.object
    if armature is None or not isinstance(armature.data, bpy.types.Armature):
        operator.report({"ERROR"}, "No armature selected")
        return

    # TODO: Will the armature preserve bone ordering for the name list?
    nud_path = armature.get("nud_path")
    original_model = sm4sh_model_py.load_model(nud_path)
    skeleton = original_model.skeleton
    bone_names = [b.name for b in skeleton.bones]

    # TODO: Export images?
    groups = []
    model = sm4sh_model_py.NudModel(groups, [], [0, 0, 0, 0], None)

    # Use a consistent ordering since Blender collections don't have one.
    sorted_objects = [o for o in armature.children if o.type == "MESH"]
    sorted_objects.sort(key=lambda o: name_sort_index(o.name))

    extract_name = lambda o: o.name.split("[")[0] if "[" in o.name else o.name

    for name, objects in itertools.groupby(sorted_objects, key=extract_name):
        meshes_parent_indices = []
        for o in objects:
            mesh_parent_index = export_mesh(context, operator, o, bone_names)
            meshes_parent_indices.append(mesh_parent_index)

        # Split since each group can only have one parent bone.
        for parent_bone_index, meshes_parents in itertools.groupby(
            meshes_parent_indices, key=lambda o: o[1]
        ):
            meshes = [mesh for mesh, _ in meshes_parents]
            bone_flags = (
                sm4sh_model_py.BoneFlags.Skinning
                if parent_bone_index is None
                else sm4sh_model_py.BoneFlags.ParentBone
            )
            group = sm4sh_model_py.NudMeshGroup(
                name,
                meshes,
                0.0,
                [0, 0, 0, 0],
                bone_flags,
                parent_bone_index,
            )
            groups.append(group)

    end = time.time()
    print(f"Create NudModel: {end - start}")

    start = time.time()

    nud = model.to_nud()
    nud.save(output_nud_path)

    end = time.time()
    print(f"Export Files: {end - start}")
