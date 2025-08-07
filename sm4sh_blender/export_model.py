from typing import Optional, Tuple
import bpy
import math
import struct
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
) -> Tuple[sm4sh_model_py.NudMesh, Optional[int]]:
    # Work on a copy in case we need to make any changes.
    mesh_copy = blender_mesh.copy()
    mesh_copy.data = blender_mesh.data.copy()

    try:
        process_export_mesh(context, mesh_copy)
        return export_mesh_inner(operator, mesh_copy, blender_mesh.name, bone_names)
    finally:
        bpy.data.meshes.remove(mesh_copy.data)


# TODO: Split this into more functions.
def export_mesh_inner(
    operator: bpy.types.Operator,
    blender_mesh: bpy.types.Object,
    mesh_name: str,
    bone_names: list[str],
) -> Tuple[sm4sh_model_py.NudMesh, Optional[int]]:

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
    tangents[:, 3] = 1.0
    bitangents[:, 3] = 1.0

    parent_bone_index = None
    bones = None

    influences = export_influences(blender_mesh, mesh_data)
    if len(influences) > 1:
        skin_weights = sm4sh_model_py.skinning.SkinWeights.from_influences(
            influences, positions.shape[0], bone_names
        )
        bones = sm4sh_model_py.vertex.Bones(
            skin_weights.bone_indices,
            skin_weights.bone_weights,
            sm4sh_model_py.vertex.BoneElementType.Byte,
        )
    elif len(influences) == 1:
        # Avoid storing weights if there is only one influence.
        parent_name = influences[0].bone_name
        for i, name in enumerate(bone_names):
            if name == parent_name:
                parent_bone_index = i
                break

    uv_layers = []
    for uv_layer in mesh_data.uv_layers:
        uvs = export_uv_layer(mesh_data, positions, vertex_indices, uv_layer)
        uv_layers.append(uvs)

    uvs = sm4sh_model_py.vertex.Uvs.from_uvs_float16(uv_layers)

    float_colors = np.ones((positions.shape[0], 4), dtype=np.float32) * 0.5
    for color_attribute in mesh_data.color_attributes:
        if color_attribute.name == "Color":
            float_colors = export_color_attribute(
                mesh_name, mesh_data, vertex_indices, color_attribute
            )

    color_type = sm4sh_model_py.vertex.ColorElementType.Byte
    colors = sm4sh_model_py.vertex.Colors(float_colors, color_type)

    normals = sm4sh_model_py.vertex.Normals.from_normals_tangents_bitangents_float16(
        normals, tangents, bitangents
    )
    vertices = sm4sh_model_py.vertex.Vertices(positions, normals, bones, colors, uvs)

    material1 = None
    material2 = None
    material3 = None
    material4 = None
    for i, material in enumerate(mesh_data.materials):

        if i == 0:
            material1 = export_material(material)
        elif i == 1:
            material2 = export_material(material)
        elif i == 2:
            material3 = export_material(material)
        elif i == 3:
            material4 = export_material(material)
        elif i > 3:
            # TODO: Warning?
            break

    if material1 is None:
        # TODO: warning?
        material1 = default_material()

    mesh = sm4sh_model_py.NudMesh(
        vertices,
        vertex_indices,
        sm4sh_model_py.PrimitiveType.TriangleList,
        material1,
        material2,
        material3,
        material4,
    )
    return mesh, parent_bone_index


def export_influences(
    blender_mesh, mesh_data
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
    normals[:, 3] = 1.0

    return normals


def export_tangents(mesh_data, z_up_to_y_up, vertex_indices):
    # Tangents are stored per loop instead of per vertex.
    loop_tangents = np.zeros(len(mesh_data.loops) * 3, dtype=np.float32)
    try:
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
    if color_attribute.name != "Color":
        message = f'"{color_attribute.name}" for mesh {mesh_name} is not one of the supported color attribute names.'
        message += ' Valid names are "Color".'
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
    return colors


def export_uv_layer(mesh_data, positions, vertex_indices, uv_layer):
    uvs = np.zeros((positions.shape[0], 2), dtype=np.float32)
    loop_uvs = np.zeros(len(mesh_data.loops) * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", loop_uvs)
    uvs[vertex_indices] = loop_uvs.reshape((-1, 2))
    # Flip vertically to match in game.
    uvs[:, 1] = 1.0 - uvs[:, 1]

    return uvs


def export_material(material: bpy.types.Material) -> sm4sh_model_py.NudMaterial:
    # TODO: Better error handling
    flags = 0x94010161
    flags_name = extract_name(material.name, ".")
    if value := parse_int(flags_name, 16):
        flags = value

    src_factor = sm4sh_model_py.SrcFactor.One
    if value := get_enum_value(material, "src_factor", sm4sh_model_py.SrcFactor):
        src_factor = value

    dst_factor = sm4sh_model_py.DstFactor.Zero
    if value := get_enum_value(material, "dst_factor", sm4sh_model_py.DstFactor):
        dst_factor = value

    alpha_func = sm4sh_model_py.AlphaFunc.Disabled
    if value := get_enum_value(material, "alpha_func", sm4sh_model_py.AlphaFunc):
        alpha_func = value

    cull_mode = sm4sh_model_py.CullMode.Inside
    if value := get_enum_value(material, "cull_mode", sm4sh_model_py.CullMode):
        cull_mode = value

    properties = []
    texture_indices_textures = []

    # TODO: Does property order matter?
    for node in material.node_tree.nodes:
        if node.label.startswith("NU_"):
            try:
                values = node.outputs[0].default_value
            except:
                values = [0, 0, 0, 0]

            properties.append(sm4sh_model_py.NudProperty(node.label, values))
        elif node.bl_idname == "ShaderNodeTexImage":
            texture_index = parse_int(node.label)
            if texture_index is not None:
                if hash := parse_int(node.image.name, 16):
                    # TODO: Preserve wrap mode for both U and V?
                    wrap_s = sm4sh_model_py.WrapMode.ClampToEdge
                    wrap_t = sm4sh_model_py.WrapMode.ClampToEdge
                    match node.extension:
                        case "REPEAT":
                            wrap_s = sm4sh_model_py.WrapMode.Repeat
                            wrap_t = sm4sh_model_py.WrapMode.Repeat
                        case "CLIP":
                            wrap_s = sm4sh_model_py.WrapMode.ClampToEdge
                            wrap_t = sm4sh_model_py.WrapMode.ClampToEdge
                        case "MIRROR":
                            wrap_s = sm4sh_model_py.WrapMode.MirroredRepeat
                            wrap_t = sm4sh_model_py.WrapMode.MirroredRepeat

                    # TODO: investigate why texture mip settings can cause crashes.
                    texture = sm4sh_model_py.NudTexture(
                        hash,
                        sm4sh_model_py.MapMode.TexCoord,
                        wrap_s,
                        wrap_t,
                        sm4sh_model_py.MinFilter.Linear,
                        sm4sh_model_py.MagFilter.Linear,
                        sm4sh_model_py.MipDetail.OneMipLevelAnisotropicOff2,
                    )
                    texture_indices_textures.append((texture_index, texture))

    # The texture order matters.
    texture_indices_textures.sort(key=lambda x: x[0])
    textures = [texture for _, texture in texture_indices_textures]

    # The material hash property is always last.
    material_hash = 0.0
    if hash := material.get("NU_materialHash"):
        material_hash = float32_from_bits(int(hash, 16))

    properties.append(
        sm4sh_model_py.NudProperty("NU_materialHash", [material_hash, 0, 0, 0])
    )

    return sm4sh_model_py.NudMaterial(
        flags,
        src_factor,
        dst_factor,
        alpha_func,
        cull_mode,
        textures,
        properties,
    )


def get_enum_value(material, name: str, enum):
    if value := material.get(name):
        try:
            value = getattr(enum, value)
            return value
        except:
            return None

    return None


def float32_from_bits(bits: int) -> float:
    return struct.unpack("@f", struct.pack("@I", bits))[0]


def parse_int(name: str, base=10) -> Optional[int]:
    value = None
    try:
        value = int(name, base)
    except:
        value = None

    return value


def default_material() -> sm4sh_model_py.NudMaterial:
    # TODO: investigate why texture mip settings can cause crashes.
    return sm4sh_model_py.NudMaterial(
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
                sm4sh_model_py.MipDetail.OneMipLevelAnisotropicOff2,
            ),
            sm4sh_model_py.NudTexture(
                0x10080000,
                sm4sh_model_py.MapMode.TexCoord,
                sm4sh_model_py.WrapMode.ClampToEdge,
                sm4sh_model_py.WrapMode.ClampToEdge,
                sm4sh_model_py.MinFilter.Linear,
                sm4sh_model_py.MagFilter.Linear,
                sm4sh_model_py.MipDetail.OneMipLevelAnisotropicOff2,
            ),
        ],
        [
            sm4sh_model_py.NudProperty("NU_colorSamplerUV", [1, 1, 0, 0]),
            sm4sh_model_py.NudProperty("NU_fresnelColor", [1, 1, 1, 1]),
            sm4sh_model_py.NudProperty("NU_blinkColor", [0, 0, 0, 0]),
            sm4sh_model_py.NudProperty("NU_aoMinGain", [0, 0, 0, 0]),
            sm4sh_model_py.NudProperty("NU_lightMapColorOffset", [0, 0, 0, 0]),
            sm4sh_model_py.NudProperty("NU_fresnelParams", [1, 0, 0, 0]),
            sm4sh_model_py.NudProperty("NU_alphaBlendParams", [0, 0, 0, 0]),
            sm4sh_model_py.NudProperty("NU_materialHash", [0, 0, 0, 0]),
        ],
    )


def extract_name(name: str, separator: str) -> str:
    return name.split(separator)[0] if separator in name else name
