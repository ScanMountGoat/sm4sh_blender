import bpy
import math

from mathutils import Matrix
from . import sm4sh_model_py
import numpy as np
import bmesh


class ExportException(Exception):
    pass


# https://github.com/ScanMountGoat/xenoblade_blender/blob/0228b8f2a8d30d1a166ea9f559fa1dd07e216f93/xenoblade_blender/export_root.py#L14
def export_skeleton(armature: bpy.types.Object):
    bpy.ops.object.mode_set(mode="EDIT")

    bones = []
    for bone in armature.data.edit_bones:
        name = bone.name
        if bone.parent:
            matrix = get_bone_transform(bone.parent.matrix.inverted() @ bone.matrix)
        else:
            matrix = get_root_bone_transform(bone)

        # TODO: Find a way to make this not O(N^2)?
        parent_bone_index = None
        if bone.parent:
            for i, other in enumerate(armature.data.edit_bones):
                if other == bone.parent:
                    parent_bone_index = i
                    break

        translation, rotation, scale = matrix.decompose()
        rotation = [rotation.x, rotation.y, rotation.z, rotation.w]

        # TODO: Store the hash in the name or a custom property?
        hash = 0

        bones.append(
            sm4sh_model_py.VbnBone(
                name,
                hash,
                parent_bone_index,
                sm4sh_model_py.BoneType.Normal,
                translation,
                rotation,
                scale,
            )
        )

    bpy.ops.object.mode_set(mode="OBJECT")

    return sm4sh_model_py.Skeleton(bones)


def get_root_bone_transform(bone: bpy.types.EditBone) -> Matrix:
    bone.transform(Matrix.Rotation(math.radians(-90), 4, "X"))
    bone.transform(Matrix.Rotation(math.radians(90), 4, "Z"))
    unreoriented_matrix = get_bone_transform(bone.matrix)
    bone.transform(Matrix.Rotation(math.radians(-90), 4, "Z"))
    bone.transform(Matrix.Rotation(math.radians(90), 4, "X"))
    return unreoriented_matrix


def get_bone_transform(m: Matrix) -> Matrix:
    # This is the inverse of the get_blender_transform permutation matrix.
    # https://en.wikipedia.org/wiki/Matrix_similarity
    p = Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # Perform the transformation m in Blender's basis and convert back to Sm4sh.
    return p @ m @ p.inverted()


# Updated from the processing code written for Smash Ultimate:
# https://github.com/ssbucarlos/smash-ultimate-blender/blob/a003be92bd27e34d2a6377bb98d55d5a34e63e56/source/model/export_model.py#L956
def process_export_mesh(context: bpy.types.Context, mesh: bpy.types.Object):
    # Apply any transforms before exporting to preserve vertex positions.
    # Assume the meshes have no children that would inherit their transforms.
    mesh.data.transform(mesh.matrix_basis)
    mesh.matrix_basis.identity()

    # Apply modifiers other than armature and outlines.
    override = context.copy()
    override["object"] = mesh
    override["active_object"] = mesh
    override["selected_objects"] = [mesh]
    with context.temp_override(**override):
        for modifier in mesh.modifiers:
            if modifier.type != "ARMATURE" and modifier.type != "SOLIDIFY":
                bpy.ops.object.modifier_apply(modifier=modifier.name)

    # Get the custom normals from the original mesh.
    # We use the copy here since applying transforms alters the normals.
    loop_normals = np.zeros(len(mesh.data.loops) * 3, dtype=np.float32)
    mesh.data.loops.foreach_get("normal", loop_normals)

    # Transfer the original normals to a custom attribute.
    # This allows us to edit the mesh without affecting custom normals.
    normals_color = mesh.data.attributes.new(
        name="_custom_normals", type="FLOAT_VECTOR", domain="CORNER"
    )
    normals_color.data.foreach_set("vector", loop_normals)

    # Check if any faces are not triangles, and convert them into triangles.
    if any(len(f.vertices) != 3 for f in mesh.data.polygons):
        bm = bmesh.new()
        bm.from_mesh(mesh.data)

        bmesh.ops.triangulate(bm, faces=bm.faces[:])

        bm.to_mesh(mesh.data)
        bm.free()

    # Blender stores normals and UVs per loop rather than per vertex.
    # Edges with more than one value per vertex need to be split.
    split_duplicate_loop_attributes(mesh)
    # Rarely this will create some loose verts
    bm = bmesh.new()
    bm.from_mesh(mesh.data)

    unlinked_verts = [v for v in bm.verts if len(v.link_faces) == 0]
    bmesh.ops.delete(bm, geom=unlinked_verts, context="VERTS")

    bm.to_mesh(mesh.data)
    mesh.data.update()
    bm.clear()

    # Extract the custom normals preserved in the custom attribute.
    # Attributes should not be affected by splitting or triangulating.
    # This avoids the datatransfer modifier not handling vertices at the same position.
    loop_normals = np.zeros(len(mesh.data.loops) * 3, dtype=np.float32)
    normals = mesh.data.attributes["_custom_normals"]
    normals.data.foreach_get("vector", loop_normals)

    # Assign the preserved custom normals to the temp mesh.
    mesh.data.normals_split_custom_set(loop_normals.reshape((-1, 3)))
    mesh.data.update()


def split_duplicate_loop_attributes(mesh: bpy.types.Object):
    bm = bmesh.new()
    bm.from_mesh(mesh.data)

    edges_to_split: list[bmesh.types.BMEdge] = []

    add_duplicate_normal_edges(edges_to_split, bm)

    for layer_name in bm.loops.layers.uv.keys():
        uv_layer = bm.loops.layers.uv.get(layer_name)
        add_duplicate_uv_edges(edges_to_split, bm, uv_layer)

    # Duplicate edges cause problems with split_edges.
    edges_to_split = list(set(edges_to_split))

    # Don't modify the mesh if no edges need to be split.
    # This check also seems to prevent a potential crash.
    if len(edges_to_split) > 0:
        bmesh.ops.split_edges(bm, edges=edges_to_split)
        bm.to_mesh(mesh.data)
        mesh.data.update()

    bm.clear()

    # Check if any edges were split.
    return len(edges_to_split) > 0


def add_duplicate_normal_edges(edges_to_split, bm):
    # The original normals are preserved in a custom attribute.
    normal_layer = bm.loops.layers.float_vector.get("_custom_normals")

    # Find edges connected to vertices with more than one normal.
    # This allows converting to per vertex later by splitting edges.
    index_to_normal = {}
    for face in bm.faces:
        for loop in face.loops:
            vertex_index = loop.vert.index
            normal = loop[normal_layer]
            # Small fluctuations in normal vectors are expected during processing.
            # Check if the angle between normals is sufficiently large.
            # Assume normal vectors are normalized to have length 1.0.
            if vertex_index not in index_to_normal:
                index_to_normal[vertex_index] = normal
            elif not math.isclose(
                normal.dot(index_to_normal[vertex_index]),
                1.0,
                abs_tol=0.001,
                rel_tol=0.001,
            ):
                # Get any edges containing this vertex.
                edges_to_split.extend(loop.vert.link_edges)


def add_duplicate_uv_edges(edges_to_split, bm, uv_layer):
    # Blender stores uvs per loop rather than per vertex.
    # Find edges connected to vertices with more than one uv coord.
    # This allows converting to per vertex later by splitting edges.
    index_to_uv = {}
    for face in bm.faces:
        for loop in face.loops:
            vertex_index = loop.vert.index
            uv = loop[uv_layer].uv
            # Use strict equality since UVs are unlikely to change unintentionally.
            if vertex_index not in index_to_uv:
                index_to_uv[vertex_index] = uv
            elif uv != index_to_uv[vertex_index]:
                edges_to_split.extend(loop.vert.link_edges)


def export_mesh(
    context: bpy.types.Context,
    operator: bpy.types.Operator,
    blender_mesh: bpy.types.Object,
    bone_names: list[str],
) -> sm4sh_model_py.NudMesh:
    # Work on a copy in case we need to make any changes.
    mesh_copy = blender_mesh.copy()
    mesh_copy.data = blender_mesh.data.copy()

    try:
        process_export_mesh(context, mesh_copy)
        mesh = export_mesh_inner(operator, mesh_copy, blender_mesh.name, bone_names)
        return mesh
    finally:
        bpy.data.meshes.remove(mesh_copy.data)


# TODO: Split this into more functions.
def export_mesh_inner(
    operator: bpy.types.Operator,
    blender_mesh: bpy.types.Object,
    mesh_name: str,
    bone_names: list[str],
) -> sm4sh_model_py.NudMesh:

    mesh_data: bpy.types.Mesh = blender_mesh.data

    # This needs to be checked after processing in case there are more vertices.
    vertex_count = len(mesh_data.vertices)
    if vertex_count > 65535:
        message = f"Mesh {mesh_name} will have {vertex_count} vertices after exporting,"
        message += " which exceeds the per mesh limit of 65535."
        raise ExportException(message)

    z_up_to_y_up = np.array(Matrix.Rotation(math.radians(90), 3, "X"), dtype=np.float32)

    positions = export_positions(mesh_data, z_up_to_y_up)
    vertex_indices = export_vertex_indices(mesh_data)
    normals = export_normals(mesh_data, z_up_to_y_up, vertex_indices)
    tangents = export_tangents(mesh_data, z_up_to_y_up, vertex_indices)
    bitangents = np.zeros_like(tangents)
    bitangents[:, :3] = np.cross(normals[:, :3], tangents[:, :3]) * tangents[
        :, 3
    ].reshape((-1, 1))

    # TODO: Use a parent bone on the mesh group if single vertex group.
    influences = export_influences(operator, blender_mesh, mesh_data)
    skin_weights = sm4sh_model_py.skinning.SkinWeights.from_influences(
        influences, positions.shape[0], bone_names
    )

    uvs = []
    for uv_layer in mesh_data.uv_layers:
        uv = export_uv_layer(mesh_name, mesh_data, positions, vertex_indices, uv_layer)
        uvs.append(uv)

    byte_colors = np.ones((positions.shape[0], 4), dtype=np.uint8) * 0.5
    for color_attribute in mesh_data.color_attributes:
        if color_attribute.name == "VertexColor":
            byte_colors = export_color_attribute(
                mesh_name, mesh_data, vertex_indices, color_attribute
            )
    colors = sm4sh_model_py.vertex.Colors.from_colors_byte(byte_colors)

    normals = sm4sh_model_py.vertex.Normals.from_normals_tangents_bitangents_float32(
        normals, tangents, bitangents
    )
    bones = sm4sh_model_py.vertex.Bones.from_bone_indices_weights_float32(
        skin_weights.bone_indices, skin_weights.bone_weights
    )
    vertices = sm4sh_model_py.vertex.Vertices(positions, normals, bones, colors, uvs)

    # TODO: Set the material hash?
    # TODO: Set remaining properties?
    material = sm4sh_model_py.NudMaterial(
        0x94010161,
        sm4sh_model_py.SrcFactor.One,
        sm4sh_model_py.DstFactor.Zero,
        sm4sh_model_py.AlphaFunc.Disabled,
        sm4sh_model_py.CullMode.Inside,
        [
            sm4sh_model_py.NudTexture(
                0x10080000,
                sm4sh_model_py.MapMode.TexCoord,
                sm4sh_model_py.WrapMode.ClampToEdge,
                sm4sh_model_py.WrapMode.ClampToEdge,
                sm4sh_model_py.MinFilter.Linear,
                sm4sh_model_py.MagFilter.Linear,
                sm4sh_model_py.MipDetail.OneMipLevelAnisotropicOff,
            ),
            sm4sh_model_py.NudTexture(
                0x10080000,
                sm4sh_model_py.MapMode.TexCoord,
                sm4sh_model_py.WrapMode.ClampToEdge,
                sm4sh_model_py.WrapMode.ClampToEdge,
                sm4sh_model_py.MinFilter.Linear,
                sm4sh_model_py.MagFilter.Linear,
                sm4sh_model_py.MipDetail.OneMipLevelAnisotropicOff,
            ),
        ],
        [sm4sh_model_py.NudProperty("NU_materialHash", [0, 0, 0, 0])],
    )

    mesh = sm4sh_model_py.NudMesh(
        vertices,
        vertex_indices,
        False,
        sm4sh_model_py.PrimitiveType.TriangleList,
        material,
        None,
        None,
        None,
    )
    return mesh


def export_influences(
    operator, blender_mesh, mesh_data
) -> list[sm4sh_model_py.skinning.Influence]:
    # Export Weights
    # TODO: Reversing a vertex -> group lookup to a group -> vertex lookup is expensive.
    # TODO: Does Blender not expose this directly?
    group_to_weights = {vg.index: (vg.name, []) for vg in blender_mesh.vertex_groups}

    for vertex in mesh_data.vertices:
        # Blender doesn't enforce normalization, since it normalizes while animating.
        # Normalize on export to ensure the weights work correctly in game.
        weight_sum = sum([g.weight for g in vertex.groups])
        for group in vertex.groups:
            weight = sm4sh_model_py.skinning.VertexWeight(
                vertex.index, group.weight / weight_sum
            )
            group_to_weights[group.group][1].append(weight)

    influences = []
    for name, weights in group_to_weights.values():
        if len(weights) > 0:
            influence = sm4sh_model_py.skinning.Influence(name, weights)
            influences.append(influence)

    return influences


def export_positions(mesh_data, z_up_to_y_up):
    positions = np.zeros(len(mesh_data.vertices) * 3, dtype=np.float32)
    mesh_data.vertices.foreach_get("co", positions)
    positions = positions.reshape((-1, 3)) @ z_up_to_y_up
    return positions


def export_vertex_indices(mesh_data):
    vertex_indices = np.zeros(len(mesh_data.loops), dtype=np.uint32)
    mesh_data.loops.foreach_get("vertex_index", vertex_indices)
    return vertex_indices.astype(np.uint16)


def export_normals(mesh_data, z_up_to_y_up, vertex_indices):
    # Normals are stored per loop instead of per vertex.
    loop_normals = np.zeros(len(mesh_data.loops) * 3, dtype=np.float32)
    mesh_data.loops.foreach_get("normal", loop_normals)
    loop_normals = loop_normals.reshape((-1, 3))

    normals = np.zeros((len(mesh_data.vertices), 4), dtype=np.float32)
    normals[:, :3][vertex_indices] = loop_normals @ z_up_to_y_up

    # Some shaders use the 4th component for normal map intensity.
    if attribute := mesh_data.attributes.get("VertexNormal"):
        vertex_normals = np.zeros(len(mesh_data.vertices) * 4, dtype=np.float32)
        attribute.data.foreach_get("color", vertex_normals)

        normals[:, 3] = vertex_normals.reshape((-1, 4))[:, 3]
    else:
        normals[:, 3] = 1.0

    return normals


def export_tangents(mesh_data, z_up_to_y_up, vertex_indices):
    # Tangents are stored per loop instead of per vertex.
    loop_tangents = np.zeros(len(mesh_data.loops) * 3, dtype=np.float32)
    try:
        # TODO: Why do some meshes not have UVs for Pyra?
        mesh_data.calc_tangents()
        mesh_data.loops.foreach_get("tangent", loop_tangents)
    except:
        pass

    loop_bitangent_signs = np.zeros(len(mesh_data.loops), dtype=np.float32)
    mesh_data.loops.foreach_get("bitangent_sign", loop_bitangent_signs)

    tangents = np.zeros((len(mesh_data.vertices), 4), dtype=np.float32)
    tangents[:, :3][vertex_indices] = loop_tangents.reshape((-1, 3)) @ z_up_to_y_up
    tangents[:, 3][vertex_indices] = loop_bitangent_signs
    return tangents


def export_color_attribute(mesh_name, mesh_data, vertex_indices, color_attribute):
    if color_attribute.name != "VertexColor":
        message = f'"{color_attribute.name}" for mesh {mesh_name} is not one of the supported color attribute names.'
        message += ' Valid names are "VertexColor".'
        raise ExportException(message)

    # TODO: error for unsupported data_type.
    if color_attribute.domain == "POINT":
        colors = np.zeros(len(mesh_data.vertices) * 4, dtype=np.float32)
        color_attribute.data.foreach_get("color", colors)
    elif color_attribute.domain == "CORNER":
        loop_colors = np.zeros(len(mesh_data.loops) * 4, dtype=np.float32)
        color_attribute.data.foreach_get("color", loop_colors)
        # Convert per loop data to per vertex data.
        colors = np.zeros((len(mesh_data.vertices), 4), dtype=np.float32)
        colors[vertex_indices] = loop_colors.reshape((-1, 4))
    else:
        message = f"Unsupported color attribute domain {color_attribute.domain}"
        raise ExportException(message)

    colors = colors.reshape((-1, 4))
    byte_colors = (colors * 255.0).astype(np.uint8)
    return byte_colors


def export_uv_layer(mesh_name, mesh_data, positions, vertex_indices, uv_layer):
    uvs = np.zeros((positions.shape[0], 2), dtype=np.float32)
    loop_uvs = np.zeros(len(mesh_data.loops) * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", loop_uvs)
    uvs[vertex_indices] = loop_uvs.reshape((-1, 2))
    # Flip vertically to match in game.
    uvs[:, 1] = 1.0 - uvs[:, 1]

    # TODO: Pass the index as a parameter instead?
    match uv_layer.name:
        case "UV0":
            index = 0
        case "UV1":
            index = 1
        case "UV2":
            index = 2
        case "UV3":
            index = 3
        case "UV4":
            index = 4
        case "UV5":
            index = 5
        case "UV6":
            index = 6
        case "UV7":
            index = 7
        case "UV8":
            index = 8
        case _:
            message = f'"{uv_layer.name}" for mesh {mesh_name} is not one of the supported UV map names.'
            message += ' Valid names are "TexCoord0" to "TexCoord8".'
            raise ExportException(message)

    return sm4sh_model_py.vertex.Uvs.from_uvs_float32(uvs)
