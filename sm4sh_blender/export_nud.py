import copy
import os
import re
import time
from pathlib import Path

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy_extras.io_utils import ExportHelper

from sm4sh_blender.export_material import metal_material
from sm4sh_blender.import_model import init_logging
from sm4sh_blender.utils import extract_name

from . import sm4sh_model_py
from .export_model import ExportException, export_mesh


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

    original_nud: StringProperty(
        name="Original Nud",
        description="The original .nud file to use to generate the new model. Defaults to the armature's original_nud custom property if not set",
    )

    export_nut: BoolProperty(
        name="Export Nut",
        description="Export a model.nut file with the textures used by each material",
        default=True,
    )

    export_metal: BoolProperty(
        name="Export metal.nud",
        description="Export a copy of the model.nud with metal box materials. Disable this setting when not exporting a fighter model.",
        default=True,
    )

    def execute(self, context: bpy.types.Context):
        init_logging()

        try:
            export_nud(
                self,
                context,
                self.filepath,
                self.original_nud.strip('"'),
                self.export_nut,
                self.export_metal,
            )
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
    original_nud_path: str,
    export_nut: bool,
    export_metal: bool,
):
    start = time.time()

    database_path = os.path.join(os.path.dirname(__file__), "shaders.bin")
    database = sm4sh_model_py.database.ShaderDatabase.from_file(database_path)

    sorted_objects = []
    if len(context.selected_objects) == 1:
        if original_nud_path == "":
            original_nud_path = context.selected_objects[0].get("original_nud")

        sorted_objects = [
            o for o in context.selected_objects[0].children if o.type == "MESH"
        ]
    else:
        sorted_objects = [o for o in context.selected_objects if o.type == "MESH"]

    # Use a consistent ordering since Blender collections don't have one.
    sorted_objects.sort(key=lambda o: name_sort_index(o.name))

    try:
        original_model = sm4sh_model_py.load_model(original_nud_path)
    except:
        message = f'Unable to load original nud from "{original_nud_path}". Exports may cause issues in game.'
        operator.report({"WARNING"}, message)
        original_model = None

    # Preserve the original bone order since armatures don't preserve bone order.
    bone_names = []
    if original_model is not None and original_model.skeleton is not None:
        bone_names = [b.name for b in original_model.skeleton.bones]

    image_textures = []
    if original_model is not None:
        image_textures = original_model.textures

    image_args = {}

    meshes = []
    metal_meshes = []
    for o in sorted_objects:
        name = extract_name(o.name, ".")
        # TODO: preserve sort bias?
        sort_bias = 0.0
        mesh, parent_bone_index = export_mesh(
            context, operator, o, bone_names, database, image_args
        )

        if export_metal:
            metal_mesh = make_metal_mesh(mesh, o, database, image_args)

            metal_meshes.append(
                sm4sh_model_py.model.NudMeshGroupMesh(
                    name, sort_bias, parent_bone_index, metal_mesh
                )
            )

        meshes.append(
            sm4sh_model_py.model.NudMeshGroupMesh(
                name, sort_bias, parent_bone_index, mesh
            )
        )

    groups = sm4sh_model_py.model.create_mesh_groups(meshes)
    metal_groups = sm4sh_model_py.model.create_mesh_groups(metal_meshes)

    # Preserve the order of the original groups with new groups at the end.
    if original_model is not None:
        name_pos = {g.name: i for i, g in enumerate(original_model.groups)}
        groups.sort(key=lambda g: name_pos.get(g.name, 0xFFFFFFFF))
        metal_groups.sort(key=lambda g: name_pos.get(g.name, 0xFFFFFFFF))

    # TODO: Calculate better bounding spheres
    model = sm4sh_model_py.NudModel(groups, image_textures, [0, 0, 0, 10.0], None)
    metal_model = sm4sh_model_py.NudModel(
        metal_groups, image_textures, [0, 0, 0, 10.0], None
    )

    end = time.time()
    print(f"Create NudModel: {end - start}")

    start = time.time()

    nud = model.to_nud()
    nud.save(output_nud_path)

    if export_metal:
        metal_nud = metal_model.to_nud()
        metal_nud.save(str(Path(output_nud_path).with_name("metal.nud")))

    if export_nut:
        start = time.time()
        # In game nut files sort textures by their integer hash.
        sorted_image_args = sorted(image_args.values(), key=lambda x: x.hash_id)
        model.textures = sm4sh_model_py.encode_images_rgbaf32(sorted_image_args)
        end = time.time()
        print(f"Encode Images: {end - start}")

        nut = model.to_nut()
        nut.save(output_nud_path.replace(".nud", ".nut"))

    end = time.time()
    print(f"Export Files: {end - start}")


def make_metal_mesh(
    mesh,
    o: bpy.types.Object,
    database,
    image_args,
):
    # Mesh creation is expensive, so just copy the base mesh.
    metal_mesh = copy.deepcopy(mesh)

    # TODO: error if no materials?
    blender_material = None
    if len(o.data.materials) > 0:
        blender_material = o.data.materials[0]

    # TODO: should every material be metal?
    metal_mesh.material1 = metal_material(
        mesh.material1, blender_material, database, image_args
    )

    return metal_mesh
