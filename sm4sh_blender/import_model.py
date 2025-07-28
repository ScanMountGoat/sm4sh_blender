import logging
from typing import Optional
import bpy
import numpy as np
import math

from . import sm4sh_model_py

from mathutils import Matrix, Quaternion


def init_logging():
    # Log any errors from Rust.
    log_fmt = "%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s"
    logging.basicConfig(format=log_fmt, level=logging.ERROR)


def import_nud_model(
    operator,
    context,
    model: sm4sh_model_py.nud.NudModel,
) -> Optional[bpy.types.Object]:
    armature = None
    bone_names = []
    if model.skeleton is not None:
        armature = import_armature(operator, context, model.skeleton, "Armature")
        bone_names = [b.name for b in model.skeleton.bones]

    for group in model.groups:
        for i, mesh in enumerate(group.meshes):
            import_mesh(
                operator, context.collection, group, mesh, i, armature, bone_names
            )

    return armature


def import_mesh(
    operator,
    collection: bpy.types.Collection,
    group: sm4sh_model_py.nud.NudMeshGroup,
    mesh: sm4sh_model_py.nud.NudMesh,
    i: int,
    armature: Optional[bpy.types.Object],
    bone_names: list[str],
):
    name = group.name
    blender_mesh = bpy.data.meshes.new(f"{name}[{i}]")

    indices = mesh.triangle_list_indices().astype(np.uint32)

    loop_start = np.arange(0, indices.shape[0], 3, dtype=np.uint32)
    loop_total = np.full(loop_start.shape[0], 3, dtype=np.uint32)

    blender_mesh.loops.add(indices.shape[0])
    blender_mesh.loops.foreach_set("vertex_index", indices)

    blender_mesh.polygons.add(loop_start.shape[0])
    blender_mesh.polygons.foreach_set("loop_start", loop_start)
    blender_mesh.polygons.foreach_set("loop_total", loop_total)

    # Set vertex attributes.
    positions = mesh.vertices.positions
    blender_mesh.vertices.add(positions.shape[0])
    blender_mesh.vertices.foreach_set("co", positions.reshape(-1))

    colors = mesh.vertices.colors.colors()
    if colors is not None:
        import_colors(blender_mesh, colors, "VertexColor")

    for i, uvs in enumerate(mesh.vertices.uvs):
        uvs = uvs.uvs()
        import_uvs(operator, blender_mesh, indices, uvs, f"UV{i}")

    blender_mesh.update()

    # The validate call may modify and reindex geometry.
    # Assign normals now that the mesh has been updated.
    normals = mesh.vertices.normals.normals()
    if normals is not None:
        normals_xyz = normals[:, :3]
        blender_mesh.normals_split_custom_set_from_vertices(normals_xyz)

    blender_mesh.validate()

    # Convert from Y up to Z up.
    y_up_to_z_up = Matrix.Rotation(math.radians(90), 4, "X")
    blender_mesh.transform(y_up_to_z_up)

    obj = bpy.data.objects.new(blender_mesh.name, blender_mesh)

    if parent_bone_index := group.parent_bone_index:
        # Parent the mesh to a bone using vertex weights.
        parent_bone_name = bone_names[parent_bone_index]
        vertex_group = obj.vertex_groups.new(name=parent_bone_name)
        vertex_group.add(indices.tolist(), 1.0, "REPLACE")
    else:
        weights_indices = mesh.vertices.bones.bone_indices_weights()
        if weights_indices is not None:
            indices, weights = weights_indices
            import_weight_groups(obj, indices, weights, bone_names)

        if armature is not None:
            obj.parent = armature
            modifier = obj.modifiers.new(armature.data.name, type="ARMATURE")
            modifier.object = armature

    collection.objects.link(obj)


def import_colors(
    blender_mesh: bpy.types.Mesh,
    data: np.ndarray,
    name: str,
):
    attribute = blender_mesh.color_attributes.new(
        name=name, type="FLOAT_COLOR", domain="POINT"
    )
    # TODO: remap value range?
    attribute.data.foreach_set("color", data.reshape(-1))


def import_uvs(
    operator,
    blender_mesh: bpy.types.Mesh,
    vertex_indices: np.ndarray,
    data: np.ndarray,
    name: str,
):
    uv_layer = blender_mesh.uv_layers.new(name=name)

    if uv_layer is not None:
        # This is set per loop rather than per vertex.
        loop_uvs = data[vertex_indices]
        # Flip vertically to match Blender.
        loop_uvs[:, 1] = 1.0 - loop_uvs[:, 1]
        uv_layer.data.foreach_set("uv", loop_uvs.reshape(-1))
    else:
        # Blender has a limit of 8 UV maps.
        message = f"Skipping {name} for mesh {blender_mesh.name} to avoid exceeding UV limit of 8"
        operator.report({"WARNING"}, message)


# https://github.com/ScanMountGoat/xenoblade_blender/blob/0228b8f2a8d30d1a166ea9f559fa1dd07e216f93/xenoblade_blender/import_root.py#L34
def import_armature(
    operator, context, skeleton: sm4sh_model_py.nud.VbnSkeleton, name: str
):
    armature = bpy.data.objects.new(name, bpy.data.armatures.new(name))
    bpy.context.collection.objects.link(armature)

    armature.data.display_type = "STICK"
    armature.rotation_mode = "QUATERNION"
    armature.show_in_front = True

    previous_active = context.view_layer.objects.active
    context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT", toggle=False)

    transforms = skeleton.model_space_transforms()

    for bone, transform in zip(skeleton.bones, transforms):
        new_bone = armature.data.edit_bones.new(name=bone.name)
        new_bone.head = [0, 0, 0]
        new_bone.tail = [0, 1, 0]

        y_up_to_z_up = Matrix.Rotation(math.radians(90), 4, "X")
        x_major_to_y_major = Matrix.Rotation(math.radians(-90), 4, "Z")
        new_bone.matrix = y_up_to_z_up @ Matrix(transform) @ x_major_to_y_major

    for bone in skeleton.bones:
        # TODO: Fix handling of 0xFFFFFFF (not quite 4 bytes) in sm4sh_lib.
        if bone.parent_bone_index is not None and bone.parent_bone_index < len(
            skeleton.bones
        ):
            parent_bone_name = skeleton.bones[bone.parent_bone_index].name
            parent_bone = armature.data.edit_bones.get(parent_bone_name)
            armature.data.edit_bones.get(bone.name).parent = parent_bone

    for bone in armature.data.edit_bones:
        # Prevent Blender from removing any bones.
        bone.length = 0.1

    bpy.ops.object.mode_set(mode="OBJECT")
    context.view_layer.objects.active = previous_active

    return armature


def import_weight_groups(
    blender_mesh,
    bone_indices: np.ndarray,
    bone_weights: np.ndarray,
    bone_names: list[str],
):
    if len(bone_names) == 0:
        return

    bone_vertex_weights = {name: [] for name in bone_names}

    for vertex_index, (indices, weights) in enumerate(zip(bone_indices, bone_weights)):
        for index, weight in zip(indices, weights):
            if weight > 0.0:
                name = bone_names[index]
                bone_vertex_weights[name].append((vertex_index, weight))

    for name, weights in bone_vertex_weights.items():
        # Lazily load only used vertex groups.
        if len(weights) > 0:
            group = blender_mesh.vertex_groups.get(name)
            if group is None:
                group = blender_mesh.vertex_groups.new(name=name)

                # TODO: Is there a faster way than setting weights per vertex?
                for vertex_index, weight in weights:
                    group.add([vertex_index], weight, "REPLACE")
