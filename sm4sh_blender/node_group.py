from typing import Tuple

import bpy

from sm4sh_blender.node_layout import layout_nodes


def create_node_group(nodes, name: str, create_node_tree):
    # Cache the node group creation.
    node_tree = bpy.data.node_groups.get(name)
    if node_tree is None:
        node_tree = create_node_tree(name)

    group = nodes.new("ShaderNodeGroup")
    group.node_tree = node_tree
    return group


def rgba_color_node_group(name: str):
    # TODO: Is it better to use XYZW naming?
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketColor", name="Color"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Red"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Green"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Blue"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Alpha"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketColor", name="Color"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Alpha"
    )

    rgb = nodes.new("ShaderNodeSeparateColor")
    links.new(input_node.outputs["Color"], rgb.inputs[0])

    output_node = nodes.new("NodeGroupOutput")
    links.new(input_node.outputs["Color"], output_node.inputs["Color"])
    links.new(rgb.outputs["Red"], output_node.inputs["Red"])
    links.new(rgb.outputs["Green"], output_node.inputs["Green"])
    links.new(rgb.outputs["Blue"], output_node.inputs["Blue"])
    links.new(input_node.outputs["Alpha"], output_node.inputs["Alpha"])

    layout_nodes(output_node, links)

    return node_tree


def dot4_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Value"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="A.x"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="A.y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="A.z"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="A.w"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="B.x"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="B.y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="B.z"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="B.w"
    )

    result_x = nodes.new("ShaderNodeMath")
    result_x.operation = "MULTIPLY"
    links.new(input_node.outputs["A.x"], result_x.inputs[0])
    links.new(input_node.outputs["B.x"], result_x.inputs[1])

    result_y = nodes.new("ShaderNodeMath")
    result_y.operation = "MULTIPLY_ADD"
    links.new(input_node.outputs["A.y"], result_y.inputs[0])
    links.new(input_node.outputs["B.y"], result_y.inputs[1])
    links.new(result_x.outputs["Value"], result_y.inputs[2])

    result_z = nodes.new("ShaderNodeMath")
    result_z.operation = "MULTIPLY_ADD"
    links.new(input_node.outputs["A.z"], result_z.inputs[0])
    links.new(input_node.outputs["B.z"], result_z.inputs[1])
    links.new(result_y.outputs["Value"], result_z.inputs[2])

    result_w = nodes.new("ShaderNodeMath")
    result_w.operation = "MULTIPLY_ADD"
    links.new(input_node.outputs["A.w"], result_w.inputs[0])
    links.new(input_node.outputs["B.w"], result_w.inputs[1])
    links.new(result_z.outputs["Value"], result_w.inputs[2])

    output_node = nodes.new("NodeGroupOutput")
    links.new(result_w.outputs["Value"], output_node.inputs["Value"])

    layout_nodes(output_node, links)

    return node_tree


def normal_map_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Vector"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="Vector"
    )

    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(input_node.outputs["Vector"], normal_map.inputs["Color"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(normal_map.outputs["Normal"], output_node.inputs["Vector"])

    layout_nodes(output_node, links)

    return node_tree


def normal_map_xyz_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Z"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Z"
    )

    normals = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["X"], normals.inputs["X"])
    links.new(input_node.outputs["Y"], normals.inputs["Y"])
    links.new(input_node.outputs["Z"], normals.inputs["Z"])

    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(normals.outputs["Vector"], normal_map.inputs["Color"])

    xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(normal_map.outputs["Normal"], xyz.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(xyz.outputs["X"], output_node.inputs["X"])
    links.new(xyz.outputs["Y"], output_node.inputs["Y"])
    links.new(xyz.outputs["Z"], output_node.inputs["Z"])

    layout_nodes(output_node, links)

    return node_tree


def normalize_xyz_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Z"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Z"
    )

    input_value = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["X"], input_value.inputs["X"])
    links.new(input_node.outputs["Y"], input_value.inputs["Y"])
    links.new(input_node.outputs["Z"], input_value.inputs["Z"])

    normalize = nodes.new("ShaderNodeVectorMath")
    normalize.operation = "NORMALIZE"
    links.new(input_value.outputs["Vector"], normalize.inputs[0])

    output_value = nodes.new("ShaderNodeSeparateXYZ")
    links.new(normalize.outputs["Vector"], output_value.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(output_value.outputs["X"], output_node.inputs["X"])
    links.new(output_value.outputs["Y"], output_node.inputs["Y"])
    links.new(output_value.outputs["Z"], output_node.inputs["Z"])

    layout_nodes(output_node, links)

    return node_tree


def sphere_map_coords_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Y"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Param"
    )

    geometry = nodes.new("ShaderNodeNewGeometry")

    adjusted_normal = nodes.new("ShaderNodeVectorMath")
    adjusted_normal.operation = "MULTIPLY_ADD"
    links.new(geometry.outputs["Position"], adjusted_normal.inputs[0])
    links.new(input_node.outputs["Param"], adjusted_normal.inputs[1])
    links.new(geometry.outputs["Normal"], adjusted_normal.inputs[2])

    transform_normal = nodes.new("ShaderNodeVectorTransform")
    transform_normal.convert_from = "OBJECT"
    transform_normal.convert_to = "CAMERA"
    links.new(adjusted_normal.outputs["Vector"], transform_normal.inputs["Vector"])

    scale = nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY_ADD"
    links.new(input_node.outputs["Param"], scale.inputs[0])
    scale.inputs[1].default_value = -0.25
    scale.inputs[2].default_value = 0.5

    map_range = nodes.new("ShaderNodeVectorMath")
    map_range.operation = "MULTIPLY_ADD"
    links.new(transform_normal.outputs["Vector"], map_range.inputs[0])
    links.new(scale.outputs["Value"], map_range.inputs[1])
    map_range.inputs[2].default_value = (0.5, 0.5, 0.5)

    xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(map_range.outputs["Vector"], xyz.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(xyz.outputs["X"], output_node.inputs["X"])
    links.new(xyz.outputs["Y"], output_node.inputs["Y"])

    layout_nodes(output_node, links)

    return node_tree


def transform_point_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Vector"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Z"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Z"
    )

    input_vector = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["X"], input_vector.inputs["X"])
    links.new(input_node.outputs["Y"], input_vector.inputs["Y"])
    links.new(input_node.outputs["Z"], input_vector.inputs["Z"])

    transform = nodes.new("ShaderNodeVectorTransform")
    transform.vector_type = "POINT"
    transform.convert_from = "OBJECT"
    transform.convert_to = "WORLD"
    links.new(input_vector.outputs["Vector"], transform.inputs["Vector"])

    output_vector = nodes.new("ShaderNodeSeparateXYZ")
    links.new(transform.outputs["Vector"], output_vector.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(transform.outputs["Vector"], output_node.inputs["Vector"])
    links.new(output_vector.outputs["X"], output_node.inputs["X"])
    links.new(output_vector.outputs["Y"], output_node.inputs["Y"])
    links.new(output_vector.outputs["Z"], output_node.inputs["Z"])

    layout_nodes(output_node, links)

    return node_tree


def transform_vector_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Vector"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Z"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Z"
    )

    input_vector = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["X"], input_vector.inputs["X"])
    links.new(input_node.outputs["Y"], input_vector.inputs["Y"])
    links.new(input_node.outputs["Z"], input_vector.inputs["Z"])

    transform = nodes.new("ShaderNodeVectorTransform")
    transform.vector_type = "VECTOR"
    transform.convert_from = "OBJECT"
    transform.convert_to = "WORLD"
    links.new(input_vector.outputs["Vector"], transform.inputs["Vector"])

    output_vector = nodes.new("ShaderNodeSeparateXYZ")
    links.new(transform.outputs["Vector"], output_vector.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(transform.outputs["Vector"], output_node.inputs["Vector"])
    links.new(output_vector.outputs["X"], output_node.inputs["X"])
    links.new(output_vector.outputs["Y"], output_node.inputs["Y"])
    links.new(output_vector.outputs["Z"], output_node.inputs["Z"])

    layout_nodes(output_node, links)

    return node_tree


def geometry_tangent_node_group(name: str):
    return geometry_tbn_node_group_inner(name, (1.0, 0.0, 0.0, 1.0))


def geometry_bitangent_node_group(name: str):
    return geometry_tbn_node_group_inner(name, (0.0, 1.0, 0.0, 1.0))


def geometry_normal_node_group(name: str):
    return geometry_tbn_node_group_inner(name, (0.0, 0.0, 1.0, 1.0))


def geometry_tbn_node_group_inner(name: str, rgba: Tuple[float, float, float, float]):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Vector"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Z"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    # The TBN basis vectors can be exposed by using values like (1,0,0), (0,1,0), or (0,0,1)
    tbn = nodes.new("ShaderNodeNormalMap")
    tbn.inputs["Color"].default_value = rgba

    output_vector = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tbn.outputs["Normal"], output_vector.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(tbn.outputs["Normal"], output_node.inputs["Vector"])
    links.new(output_vector.outputs["X"], output_node.inputs["X"])
    links.new(output_vector.outputs["Y"], output_node.inputs["Y"])
    links.new(output_vector.outputs["Z"], output_node.inputs["Z"])

    layout_nodes(output_node, links)

    return node_tree


def cube_coords_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Vector"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Z"
    )

    # XYZ cube coordinates to UV: https://en.wikipedia.org/wiki/Cube_mapping
    # The inputs are modified to convert Y-up in game to Blender's Z-up coordinates.
    neg_y = nodes.new("ShaderNodeMath")
    neg_y.operation = "MULTIPLY"
    links.new(input_node.outputs["Y"], neg_y.inputs[0])
    neg_y.inputs[1].default_value = -1.0

    coords = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["X"], coords.inputs["X"])
    links.new(input_node.outputs["Z"], coords.inputs["Y"])
    links.new(neg_y.outputs["Value"], coords.inputs["Z"])

    coords_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["Vector"], coords_xyz.inputs["Vector"])

    is_x_positive = is_positive(nodes, links, coords_xyz.outputs["X"])
    is_x_negative = invert(nodes, links, is_x_positive)

    is_y_positive = is_positive(nodes, links, coords_xyz.outputs["Y"])
    is_y_negative = invert(nodes, links, is_y_positive)

    is_z_positive = is_positive(nodes, links, coords_xyz.outputs["Z"])
    is_z_negative = invert(nodes, links, is_z_positive)

    neg_coords = nodes.new("ShaderNodeVectorMath")
    neg_coords.operation = "MULTIPLY"
    links.new(coords.outputs["Vector"], neg_coords.inputs["Vector"])
    neg_coords.inputs[1].default_value = (-1.0, -1.0, -1.0)

    neg_coords_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(neg_coords.outputs["Vector"], neg_coords_xyz.inputs["Vector"])

    abs_coords = nodes.new("ShaderNodeVectorMath")
    abs_coords.operation = "ABSOLUTE"
    links.new(coords.outputs["Vector"], abs_coords.inputs["Vector"])

    abs_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(abs_coords.outputs["Vector"], abs_xyz.inputs["Vector"])

    max_abs_xy = nodes.new("ShaderNodeMath")
    max_abs_xy.operation = "MAXIMUM"
    links.new(abs_xyz.outputs["X"], max_abs_xy.inputs[0])
    links.new(abs_xyz.outputs["Y"], max_abs_xy.inputs[1])

    max_abs_xyz = nodes.new("ShaderNodeMath")
    max_abs_xyz.operation = "MAXIMUM"
    links.new(max_abs_xy.outputs["Value"], max_abs_xyz.inputs[0])
    links.new(abs_xyz.outputs["Z"], max_abs_xyz.inputs[1])

    # Create 0.0 or 1.0 factors for the condition for each cube face.
    is_face_positive_x, is_face_negative_x = is_face_positive_negative(
        nodes, links, is_x_positive, is_x_negative, abs_xyz, max_abs_xyz, "X"
    )
    is_face_positive_y, is_face_negative_y = is_face_positive_negative(
        nodes, links, is_y_positive, is_y_negative, abs_xyz, max_abs_xyz, "Y"
    )
    is_face_positive_z, is_face_negative_z = is_face_positive_negative(
        nodes, links, is_z_positive, is_z_negative, abs_xyz, max_abs_xyz, "Z"
    )

    face_factors = [
        is_face_positive_x,
        is_face_negative_x,
        is_face_positive_y,
        is_face_negative_y,
        is_face_positive_z,
        is_face_negative_z,
    ]

    uc = chained_select(
        nodes,
        links,
        [
            neg_coords_xyz.outputs["Z"],
            coords_xyz.outputs["Z"],
            coords_xyz.outputs["X"],
            coords_xyz.outputs["X"],
            coords_xyz.outputs["X"],
            neg_coords_xyz.outputs["X"],
        ],
        face_factors,
        coords_xyz.outputs["Z"],
    )

    vc = chained_select(
        nodes,
        links,
        [
            coords_xyz.outputs["Y"],
            coords_xyz.outputs["Y"],
            neg_coords_xyz.outputs["Z"],
            coords_xyz.outputs["Z"],
            coords_xyz.outputs["Y"],
            coords_xyz.outputs["Y"],
        ],
        face_factors,
        coords_xyz.outputs["Y"],
    )

    index_offset = chained_select(
        nodes,
        links,
        [
            0.0 / 6.0,
            1.0 / 6.0,
            2.0 / 6.0,
            3.0 / 6.0,
            4.0 / 6.0,
            5.0 / 6.0,
        ],
        face_factors,
        0.0,
    )

    coords_xyz = nodes.new("ShaderNodeCombineXYZ")
    assign_float(links, coords_xyz.inputs["X"], uc)
    assign_float(links, coords_xyz.inputs["Y"], vc)

    coords_xyz_over_axis = nodes.new("ShaderNodeVectorMath")
    coords_xyz_over_axis.operation = "DIVIDE"
    links.new(coords_xyz.outputs["Vector"], coords_xyz_over_axis.inputs[0])
    links.new(max_abs_xyz.outputs["Value"], coords_xyz_over_axis.inputs[1])

    # Map range from -1 to 1 to 0 to 1
    map_range = nodes.new("ShaderNodeVectorMath")
    map_range.operation = "MULTIPLY_ADD"
    links.new(coords_xyz_over_axis.outputs["Vector"], map_range.inputs[0])
    map_range.inputs[1].default_value = (0.5, 0.5, 0.5)
    map_range.inputs[2].default_value = (0.5, 0.5, 0.0)

    # Flip each cube map face vertically to match in game.
    map_range_flip = flip_y(nodes, links, map_range.outputs["Vector"])

    # Modify the UVs based on the face index for a vertical layout.
    apply_index_scale = nodes.new("ShaderNodeCombineXYZ")
    apply_index_scale.inputs["X"].default_value = 1.0
    apply_index_scale.inputs["Y"].default_value = 1.0 / 6.0
    apply_index_scale.inputs["Z"].default_value = 1.0

    apply_index_offset = nodes.new("ShaderNodeCombineXYZ")
    apply_index_offset.inputs["X"].default_value = 0.0
    assign_float(links, apply_index_offset.inputs["Y"], index_offset)
    apply_index_offset.inputs["Z"].default_value = 0.0

    apply_index = nodes.new("ShaderNodeVectorMath")
    apply_index.operation = "MULTIPLY_ADD"
    links.new(map_range_flip.outputs["Vector"], apply_index.inputs[0])
    links.new(apply_index_scale.outputs["Vector"], apply_index.inputs[1])
    links.new(apply_index_offset.outputs["Vector"], apply_index.inputs[2])

    # Flip the entire vertically stacked cube map vertically to match in game.
    apply_index_flip = flip_y(nodes, links, apply_index.outputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(apply_index_flip.outputs["Vector"], output_node.inputs["Vector"])

    layout_nodes(output_node, links)

    return node_tree


def flip_y(nodes, links, vector_output):
    xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(vector_output, xyz.inputs["Vector"])

    flip_y = nodes.new("ShaderNodeMath")
    flip_y.operation = "SUBTRACT"
    flip_y.inputs[0].default_value = 1.0
    links.new(xyz.outputs["Y"], flip_y.inputs[1])

    result = nodes.new("ShaderNodeCombineXYZ")
    links.new(xyz.outputs["X"], result.inputs["X"])
    links.new(flip_y.outputs["Value"], result.inputs["Y"])
    links.new(xyz.outputs["Z"], result.inputs["Z"])

    return result


def is_positive(nodes, links, output):
    result = nodes.new("ShaderNodeMath")
    result.operation = "GREATER_THAN"
    links.new(output, result.inputs[0])
    result.inputs[1].default_value = 0.0
    return result


def invert(nodes, links, is_positive):
    result = nodes.new("ShaderNodeMath")
    result.operation = "SUBTRACT"
    result.inputs[0].default_value = 1.0
    links.new(is_positive.outputs["Value"], result.inputs[1])
    return result


def is_face_positive_negative(
    nodes, links, is_positive, is_negative, abs_xyz, max_abs_xyz, coord: str
):
    is_abs_coord_greatest = compare_values(
        nodes, links, abs_xyz.outputs[coord], max_abs_xyz.outputs["Value"]
    )

    is_face_positive_x = nodes.new("ShaderNodeMath")
    is_face_positive_x.operation = "MULTIPLY"
    links.new(is_positive.outputs["Value"], is_face_positive_x.inputs[0])
    links.new(is_abs_coord_greatest.outputs["Value"], is_face_positive_x.inputs[1])

    is_face_negative_x = nodes.new("ShaderNodeMath")
    is_face_negative_x.operation = "MULTIPLY"
    links.new(is_negative.outputs["Value"], is_face_negative_x.inputs[0])
    links.new(is_abs_coord_greatest.outputs["Value"], is_face_negative_x.inputs[1])
    return is_face_positive_x, is_face_negative_x


def compare_values(nodes, links, a, b):
    is_value_equal = nodes.new("ShaderNodeMath")
    is_value_equal.operation = "COMPARE"
    links.new(a, is_value_equal.inputs[0])
    links.new(b, is_value_equal.inputs[1])
    is_value_equal.inputs[2].default_value = 0.001
    return is_value_equal


def chained_select(
    nodes,
    links,
    values: list[bpy.types.NodeSocket | float],
    factors: list[bpy.types.Node],
    default_value: bpy.types.NodeSocket | float,
) -> bpy.types.NodeSocket | float:
    result = default_value

    # Chain selects together to mimic a switch statement.
    for value, factor in zip(values, factors):
        # Replace the previous result with the new value if factor is 1.0.
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "FLOAT"
        assign_float(links, mix.inputs["A"], result)
        assign_float(links, mix.inputs["B"], value)
        links.new(factor.outputs["Value"], mix.inputs["Factor"])

        result = mix.outputs["Result"]

    return result


def assign_float(
    links, output: bpy.types.NodeSocketFloat, f: bpy.types.NodeSocket | float
):
    try:
        output.default_value = f
    except:
        links.new(f, output)


def eye_vector_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Vector"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Z"
    )
    nodes = node_tree.nodes
    links = node_tree.links

    camera_data = nodes.new("ShaderNodeCameraData")

    negate_z = nodes.new("ShaderNodeVectorMath")
    negate_z.operation = "MULTIPLY"
    links.new(camera_data.outputs["View Vector"], negate_z.inputs[0])
    negate_z.inputs[1].default_value = (1.0, 1.0, -1.0)

    camera_to_object = nodes.new("ShaderNodeVectorTransform")
    camera_to_object.convert_from = "CAMERA"
    camera_to_object.convert_to = "OBJECT"
    links.new(negate_z.outputs["Vector"], camera_to_object.inputs["Vector"])

    output_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(camera_to_object.outputs["Vector"], output_xyz.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(camera_to_object.outputs["Vector"], output_node.inputs["Vector"])
    links.new(output_xyz.outputs["X"], output_node.inputs["X"])
    links.new(output_xyz.outputs["Y"], output_node.inputs["Y"])
    links.new(output_xyz.outputs["Z"], output_node.inputs["Z"])

    layout_nodes(output_node, links)

    return node_tree


def blinn_phong_spec_node_group_xyz(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="normal"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="lightDir"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="eye"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Exponent"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Value"
    )
    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")

    # TODO: Remove these inputs and just include the eye vector node here?

    h = nodes.new("ShaderNodeVectorMath")
    h.operation = "SUBTRACT"
    links.new(input_node.outputs["eye"], h.inputs[0])
    links.new(input_node.outputs["lightDir"], h.inputs[1])

    normalize_h = nodes.new("ShaderNodeVectorMath")
    normalize_h.operation = "NORMALIZE"
    links.new(h.outputs["Vector"], normalize_h.inputs[0])

    dot_n_h = nodes.new("ShaderNodeVectorMath")
    dot_n_h.operation = "DOT_PRODUCT"
    links.new(input_node.outputs["normal"], dot_n_h.inputs[0])
    links.new(normalize_h.outputs["Vector"], dot_n_h.inputs[1])

    spec = nodes.new("ShaderNodeMath")
    spec.operation = "MAXIMUM"
    links.new(dot_n_h.outputs["Value"], spec.inputs[0])
    spec.inputs[1].default_value = 0.001

    spec_pow = nodes.new("ShaderNodeMath")
    spec_pow.operation = "POWER"
    links.new(spec.outputs["Value"], spec_pow.inputs[0])
    links.new(input_node.outputs["Exponent"], spec_pow.inputs[1])

    output_node = nodes.new("NodeGroupOutput")
    links.new(spec_pow.outputs["Value"], output_node.inputs["Value"])

    layout_nodes(output_node, links)

    return node_tree


def blinn_phong_spec_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="normal.X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="normal.Y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="normal.Z"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="lightDir.X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="lightDir.Y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="lightDir.Z"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="eye.X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="eye.Y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="eye.Z"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Exponent"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Value"
    )
    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")

    n_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["normal.X"], n_xyz.inputs["X"])
    links.new(input_node.outputs["normal.Y"], n_xyz.inputs["Y"])
    links.new(input_node.outputs["normal.Z"], n_xyz.inputs["Z"])

    light_dir_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["lightDir.X"], light_dir_xyz.inputs["X"])
    links.new(input_node.outputs["lightDir.Y"], light_dir_xyz.inputs["Y"])
    links.new(input_node.outputs["lightDir.Z"], light_dir_xyz.inputs["Z"])

    # TODO: Remove these inputs and just include the eye vector node here?
    eye_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["eye.X"], eye_xyz.inputs["X"])
    links.new(input_node.outputs["eye.Y"], eye_xyz.inputs["Y"])
    links.new(input_node.outputs["eye.Z"], eye_xyz.inputs["Z"])

    spec = create_node_group(
        nodes, "BlinnPhongSpecXyz", blinn_phong_spec_node_group_xyz
    )
    links.new(n_xyz.outputs["Vector"], spec.inputs["normal"])
    links.new(light_dir_xyz.outputs["Vector"], spec.inputs["lightDir"])
    links.new(eye_xyz.outputs["Vector"], spec.inputs["eye"])
    links.new(input_node.outputs["Exponent"], spec.inputs["Exponent"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(spec.outputs["Value"], output_node.inputs["Value"])

    layout_nodes(output_node, links)

    return node_tree


def anisotropic_spec_node_group_xyz(name: str):
    """A slightly modified Ward BRDF for anisotropic specular."""
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="normal"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="tangent"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="eye"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="ParamX"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="ParamY"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Value"
    )
    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")

    # TODO: Remove these inputs and just include the tangent vector node here?
    # TODO: Remove these inputs and just include the eye vector node here?

    param_x = nodes.new("ShaderNodeMath")
    param_x.operation = "MULTIPLY"
    links.new(input_node.outputs["ParamX"], param_x.inputs[0])
    param_x.inputs[1].default_value = 3.0

    param_x2 = nodes.new("ShaderNodeMath")
    param_x2.operation = "MULTIPLY"
    links.new(param_x.outputs["Value"], param_x2.inputs[0])
    links.new(param_x.outputs["Value"], param_x2.inputs[1])

    param_y = nodes.new("ShaderNodeMath")
    param_y.operation = "MULTIPLY"
    links.new(input_node.outputs["ParamY"], param_y.inputs[0])
    param_y.inputs[1].default_value = 3.0

    param_y2 = nodes.new("ShaderNodeMath")
    param_y2.operation = "MULTIPLY"
    links.new(param_y.outputs["Value"], param_y2.inputs[0])
    links.new(param_y.outputs["Value"], param_y2.inputs[1])

    dot_eye_n = nodes.new("ShaderNodeVectorMath")
    dot_eye_n.operation = "DOT_PRODUCT"
    links.new(input_node.outputs["eye"], dot_eye_n.inputs[0])
    links.new(input_node.outputs["normal"], dot_eye_n.inputs[1])

    dot_eye_n2 = nodes.new("ShaderNodeMath")
    dot_eye_n2.operation = "MULTIPLY"
    links.new(dot_eye_n.outputs["Value"], dot_eye_n2.inputs[0])
    links.new(dot_eye_n.outputs["Value"], dot_eye_n2.inputs[1])

    neg_n = nodes.new("ShaderNodeVectorMath")
    neg_n.operation = "MULTIPLY"
    links.new(input_node.outputs["normal"], neg_n.inputs[0])
    neg_n.inputs[1].default_value = (-1.0, -1.0, -1.0)

    b = nodes.new("ShaderNodeVectorMath")
    b.operation = "MULTIPLY_ADD"
    links.new(neg_n.outputs["Vector"], b.inputs[0])
    links.new(dot_eye_n.outputs["Value"], b.inputs[1])
    links.new(input_node.outputs["eye"], b.inputs[2])

    normalize_b = nodes.new("ShaderNodeVectorMath")
    normalize_b.operation = "NORMALIZE"
    links.new(b.outputs["Vector"], normalize_b.inputs[0])

    dot_tangent_b = nodes.new("ShaderNodeVectorMath")
    dot_tangent_b.operation = "DOT_PRODUCT"
    links.new(input_node.outputs["tangent"], dot_tangent_b.inputs[0])
    links.new(normalize_b.outputs["Vector"], dot_tangent_b.inputs[1])

    dot_tangent_b2 = nodes.new("ShaderNodeMath")
    dot_tangent_b2.operation = "MULTIPLY"
    links.new(dot_tangent_b.outputs["Value"], dot_tangent_b2.inputs[0])
    links.new(dot_tangent_b.outputs["Value"], dot_tangent_b2.inputs[1])

    x_term = nodes.new("ShaderNodeMath")
    x_term.operation = "DIVIDE"
    links.new(dot_tangent_b2.outputs["Value"], x_term.inputs[0])
    links.new(param_x2.outputs["Value"], x_term.inputs[1])

    one_minus_dot_tangent_b2 = nodes.new("ShaderNodeMath")
    one_minus_dot_tangent_b2.operation = "SUBTRACT"
    one_minus_dot_tangent_b2.inputs[0].default_value = 1.0
    links.new(dot_tangent_b2.outputs["Value"], one_minus_dot_tangent_b2.inputs[1])

    y_term = nodes.new("ShaderNodeMath")
    y_term.operation = "DIVIDE"
    links.new(one_minus_dot_tangent_b2.outputs["Value"], y_term.inputs[0])
    links.new(param_y2.outputs["Value"], y_term.inputs[1])

    terms = nodes.new("ShaderNodeMath")
    terms.operation = "ADD"
    links.new(x_term.outputs["Value"], terms.inputs[0])
    links.new(y_term.outputs["Value"], terms.inputs[1])

    dot_eye_n2_minus_one = nodes.new("ShaderNodeMath")
    dot_eye_n2_minus_one.operation = "SUBTRACT"
    links.new(dot_eye_n2.outputs["Value"], dot_eye_n2_minus_one.inputs[0])
    dot_eye_n2_minus_one.inputs[1].default_value = 1.0

    spec = nodes.new("ShaderNodeMath")
    spec.operation = "DIVIDE"
    links.new(dot_eye_n2_minus_one.outputs["Value"], spec.inputs[0])
    links.new(dot_eye_n2.outputs["Value"], spec.inputs[1])

    spec_terms = nodes.new("ShaderNodeMath")
    spec_terms.operation = "MULTIPLY"
    links.new(spec.outputs["Value"], spec_terms.inputs[0])
    links.new(terms.outputs["Value"], spec_terms.inputs[1])

    spec_exp = nodes.new("ShaderNodeMath")
    spec_exp.operation = "EXPONENT"
    links.new(spec_terms.outputs["Value"], spec_exp.inputs[0])

    output_node = nodes.new("NodeGroupOutput")
    links.new(spec_exp.outputs["Value"], output_node.inputs["Value"])

    layout_nodes(output_node, links)

    return node_tree


def anisotropic_spec_node_group(name: str):
    """A slightly modified Ward BRDF for anisotropic specular."""
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="normal.X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="normal.Y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="normal.Z"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="tangent.X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="tangent.Y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="tangent.Z"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="eye.X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="eye.Y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="eye.Z"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="ParamX"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="ParamY"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Value"
    )
    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")

    n_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["normal.X"], n_xyz.inputs["X"])
    links.new(input_node.outputs["normal.Y"], n_xyz.inputs["Y"])
    links.new(input_node.outputs["normal.Z"], n_xyz.inputs["Z"])

    # TODO: Remove these inputs and just include the tangent vector node here?
    tangent_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["tangent.X"], tangent_xyz.inputs["X"])
    links.new(input_node.outputs["tangent.Y"], tangent_xyz.inputs["Y"])
    links.new(input_node.outputs["tangent.Z"], tangent_xyz.inputs["Z"])

    # TODO: Remove these inputs and just include the eye vector node here?
    eye_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["eye.X"], eye_xyz.inputs["X"])
    links.new(input_node.outputs["eye.Y"], eye_xyz.inputs["Y"])
    links.new(input_node.outputs["eye.Z"], eye_xyz.inputs["Z"])

    spec = create_node_group(
        nodes, "AnisotropicSpecularXyz", anisotropic_spec_node_group_xyz
    )
    links.new(n_xyz.outputs["Vector"], spec.inputs["normal"])
    links.new(eye_xyz.outputs["Vector"], spec.inputs["eye"])
    links.new(tangent_xyz.outputs["Vector"], spec.inputs["tangent"])
    links.new(input_node.outputs["ParamX"], spec.inputs["ParamX"])
    links.new(input_node.outputs["ParamY"], spec.inputs["ParamY"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(spec.outputs["Value"], output_node.inputs["Value"])

    layout_nodes(output_node, links)

    return node_tree


def fresnel_node_group_xyz(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="normal"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="eye"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Param"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Value"
    )
    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")

    # TODO: Remove these inputs and just include the eye vector node here?

    dot_eye_n = nodes.new("ShaderNodeVectorMath")
    dot_eye_n.operation = "DOT_PRODUCT"
    links.new(input_node.outputs["eye"], dot_eye_n.inputs[0])
    links.new(input_node.outputs["normal"], dot_eye_n.inputs[1])

    dot_eye_n_clamp = nodes.new("ShaderNodeClamp")
    dot_eye_n_clamp.clamp_type = "MINMAX"
    links.new(dot_eye_n.outputs["Value"], dot_eye_n_clamp.inputs["Value"])
    dot_eye_n_clamp.inputs["Min"].default_value = 0.0
    dot_eye_n_clamp.inputs["Max"].default_value = 1.0

    fresnel = nodes.new("ShaderNodeMath")
    fresnel.operation = "SUBTRACT"
    fresnel.inputs[0].default_value = 1.0
    links.new(dot_eye_n_clamp.outputs["Result"], fresnel.inputs[1])

    one_plus_param = nodes.new("ShaderNodeMath")
    one_plus_param.operation = "ADD"
    one_plus_param.inputs[0].default_value = 1.0
    links.new(input_node.outputs["Param"], one_plus_param.inputs[1])

    fresnel_pow = nodes.new("ShaderNodeMath")
    fresnel_pow.operation = "POWER"
    links.new(fresnel.outputs["Value"], fresnel_pow.inputs[0])
    links.new(one_plus_param.outputs["Value"], fresnel_pow.inputs[1])

    output_node = nodes.new("NodeGroupOutput")
    links.new(fresnel_pow.outputs["Value"], output_node.inputs["Value"])

    layout_nodes(output_node, links)

    return node_tree


def fresnel_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="normal.X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="normal.Y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="normal.Z"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="eye.X"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="eye.Y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="eye.Z"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Param"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Value"
    )
    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")

    n_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["normal.X"], n_xyz.inputs["X"])
    links.new(input_node.outputs["normal.Y"], n_xyz.inputs["Y"])
    links.new(input_node.outputs["normal.Z"], n_xyz.inputs["Z"])

    # TODO: Remove these inputs and just include the eye vector node here?
    eye_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["eye.X"], eye_xyz.inputs["X"])
    links.new(input_node.outputs["eye.Y"], eye_xyz.inputs["Y"])
    links.new(input_node.outputs["eye.Z"], eye_xyz.inputs["Z"])

    fresnel = create_node_group(nodes, "FresnelXyz", fresnel_node_group_xyz)
    links.new(n_xyz.outputs["Vector"], fresnel.inputs["normal"])
    links.new(eye_xyz.outputs["Vector"], fresnel.inputs["eye"])
    links.new(input_node.outputs["Param"], fresnel.inputs["Param"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(fresnel.outputs["Value"], output_node.inputs["Value"])

    layout_nodes(output_node, links)

    return node_tree


def tint_color_node_group_xyz(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="Color"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Factor"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Color"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Red"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Green"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Blue"
    )
    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")

    color_rgb = nodes.new("ShaderNodeSeparateXYZ")
    links.new(input_node.outputs["Color"], color_rgb.inputs["Vector"])

    max_xy = nodes.new("ShaderNodeMath")
    max_xy.operation = "MAXIMUM"
    links.new(color_rgb.outputs["X"], max_xy.inputs[0])
    links.new(color_rgb.outputs["Y"], max_xy.inputs[1])

    max_xyz = nodes.new("ShaderNodeMath")
    max_xyz.operation = "MAXIMUM"
    links.new(max_xy.outputs["Value"], max_xyz.inputs[0])
    links.new(color_rgb.outputs["Z"], max_xyz.inputs[1])

    color_minus_max = nodes.new("ShaderNodeVectorMath")
    color_minus_max.operation = "SUBTRACT"
    links.new(input_node.outputs["Color"], color_minus_max.inputs[0])
    links.new(max_xyz.outputs["Value"], color_minus_max.inputs[1])

    result_times_factor = nodes.new("ShaderNodeVectorMath")
    result_times_factor.operation = "MULTIPLY"
    links.new(color_minus_max.outputs["Vector"], result_times_factor.inputs[0])
    links.new(input_node.outputs["Factor"], result_times_factor.inputs[1])

    result = nodes.new("ShaderNodeVectorMath")
    result.operation = "ADD"
    links.new(result_times_factor.outputs["Vector"], result.inputs[0])
    result.inputs[1].default_value = (1.0, 1.0, 1.0)

    result_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(result.outputs["Vector"], result_xyz.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(result.outputs["Vector"], output_node.inputs["Color"])
    links.new(result_xyz.outputs["X"], output_node.inputs["Red"])
    links.new(result_xyz.outputs["Y"], output_node.inputs["Green"])
    links.new(result_xyz.outputs["Z"], output_node.inputs["Blue"])

    layout_nodes(output_node, links)

    return node_tree


def tint_color_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Red"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Green"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Blue"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="Factor"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Color"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Red"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Green"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Blue"
    )
    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")

    color_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["Red"], color_xyz.inputs["X"])
    links.new(input_node.outputs["Green"], color_xyz.inputs["Y"])
    links.new(input_node.outputs["Blue"], color_xyz.inputs["Z"])

    tint = create_node_group(nodes, "TintColorXyz", tint_color_node_group_xyz)
    links.new(color_xyz.outputs["Vector"], tint.inputs["Color"])
    links.new(input_node.outputs["Factor"], tint.inputs["Factor"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(tint.outputs["Color"], output_node.inputs["Color"])
    links.new(tint.outputs["Red"], output_node.inputs["Red"])
    links.new(tint.outputs["Green"], output_node.inputs["Green"])
    links.new(tint.outputs["Blue"], output_node.inputs["Blue"])

    layout_nodes(output_node, links)

    return node_tree


def clamp_xyz_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="Vector"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="Min"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="Max"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Vector"
    )
    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")

    in_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(input_node.outputs["Vector"], in_xyz.inputs["Vector"])

    min_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(input_node.outputs["Min"], min_xyz.inputs["Vector"])

    max_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(input_node.outputs["Max"], max_xyz.inputs["Vector"])

    clamp_x = nodes.new("ShaderNodeClamp")
    links.new(in_xyz.outputs["X"], clamp_x.inputs["Value"])
    links.new(min_xyz.outputs["X"], clamp_x.inputs["Min"])
    links.new(max_xyz.outputs["X"], clamp_x.inputs["Max"])

    clamp_y = nodes.new("ShaderNodeClamp")
    links.new(in_xyz.outputs["Y"], clamp_y.inputs["Value"])
    links.new(min_xyz.outputs["Y"], clamp_y.inputs["Min"])
    links.new(max_xyz.outputs["Y"], clamp_y.inputs["Max"])

    clamp_z = nodes.new("ShaderNodeClamp")
    links.new(in_xyz.outputs["Z"], clamp_z.inputs["Value"])
    links.new(min_xyz.outputs["Z"], clamp_z.inputs["Min"])
    links.new(max_xyz.outputs["Z"], clamp_z.inputs["Max"])

    out_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(clamp_x.outputs["Result"], out_xyz.inputs["X"])
    links.new(clamp_y.outputs["Result"], out_xyz.inputs["Y"])
    links.new(clamp_z.outputs["Result"], out_xyz.inputs["Z"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(out_xyz.outputs["Vector"], output_node.inputs["Vector"])

    layout_nodes(output_node, links)

    return node_tree


def math_xyz_node_group(name: str, op: str, inputs: list[str]):
    # Apply a scalar operation to independent XYZ components.
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Vector"
    )

    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")
    for i in inputs:
        node_tree.interface.new_socket(
            in_out="INPUT", socket_type="NodeSocketVector", name=i
        )

    input_xyz_nodes = []
    for i in inputs:
        node = nodes.new("ShaderNodeSeparateXYZ")
        links.new(input_node.outputs[i], node.inputs["Vector"])
        input_xyz_nodes.append(node)

    op_x = nodes.new("ShaderNodeMath")
    op_x.operation = op
    for i, node in enumerate(input_xyz_nodes):
        links.new(node.outputs["X"], op_x.inputs[i])

    op_y = nodes.new("ShaderNodeMath")
    op_y.operation = op
    for i, node in enumerate(input_xyz_nodes):
        links.new(node.outputs["Y"], op_y.inputs[i])

    op_z = nodes.new("ShaderNodeMath")
    op_z.operation = op
    for i, node in enumerate(input_xyz_nodes):
        links.new(node.outputs["Z"], op_z.inputs[i])

    output_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(op_x.outputs["Value"], output_xyz.inputs["X"])
    links.new(op_y.outputs["Value"], output_xyz.inputs["Y"])
    links.new(op_z.outputs["Value"], output_xyz.inputs["Z"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(output_xyz.outputs["Vector"], output_node.inputs["Vector"])

    layout_nodes(output_node, links)

    return node_tree


def sqrt_xyz_node_group(name: str):
    return math_xyz_node_group(name, "SQRT", ["Value"])


def inversesqrt_xyz_node_group(name: str):
    return math_xyz_node_group(name, "INVERSE_SQRT", ["Value"])


def less_xyz_node_group(name: str):
    return math_xyz_node_group(name, "LESS_THAN", ["Value", "Threshold"])


def greater_xyz_node_group(name: str):
    return math_xyz_node_group(name, "GREATER_THAN", ["Value", "Threshold"])


def neg_reflect_node_group_xyz(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="A"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketVector", name="B"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketVector", name="Vector"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Z"
    )
    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")

    reflect = nodes.new("ShaderNodeVectorMath")
    reflect.operation = "REFLECT"
    links.new(input_node.outputs["A"], reflect.inputs[0])
    links.new(input_node.outputs["B"], reflect.inputs[1])

    neg_reflect = nodes.new("ShaderNodeVectorMath")
    neg_reflect.operation = "MULTIPLY"
    links.new(reflect.outputs["Vector"], neg_reflect.inputs[0])
    neg_reflect.inputs[1].default_value = (-1.0, -1.0, -1.0)

    result_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(neg_reflect.outputs["Vector"], result_xyz.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(neg_reflect.outputs["Vector"], output_node.inputs["Vector"])
    links.new(result_xyz.outputs["X"], output_node.inputs["X"])
    links.new(result_xyz.outputs["Y"], output_node.inputs["Y"])
    links.new(result_xyz.outputs["Z"], output_node.inputs["Z"])

    layout_nodes(output_node, links)

    return node_tree


def neg_reflect_node_group(name: str):
    node_tree = bpy.data.node_groups.new(name, "ShaderNodeTree")

    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="A.x"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="A.y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="A.z"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="B.x"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="B.y"
    )
    node_tree.interface.new_socket(
        in_out="INPUT", socket_type="NodeSocketFloat", name="B.z"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="X"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Y"
    )
    node_tree.interface.new_socket(
        in_out="OUTPUT", socket_type="NodeSocketFloat", name="Z"
    )
    nodes = node_tree.nodes
    links = node_tree.links

    input_node = nodes.new("NodeGroupInput")

    a_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["A.x"], a_xyz.inputs["X"])
    links.new(input_node.outputs["A.y"], a_xyz.inputs["Y"])
    links.new(input_node.outputs["A.z"], a_xyz.inputs["Z"])

    b_xyz = nodes.new("ShaderNodeCombineXYZ")
    links.new(input_node.outputs["B.x"], b_xyz.inputs["X"])
    links.new(input_node.outputs["B.y"], b_xyz.inputs["Y"])
    links.new(input_node.outputs["B.z"], b_xyz.inputs["Z"])

    neg_reflect = create_node_group(nodes, "NegReflectXYZ", neg_reflect_node_group_xyz)
    links.new(a_xyz.outputs["Vector"], neg_reflect.inputs["A"])
    links.new(b_xyz.outputs["Vector"], neg_reflect.inputs["B"])

    output_node = nodes.new("NodeGroupOutput")
    links.new(neg_reflect.outputs["X"], output_node.inputs["X"])
    links.new(neg_reflect.outputs["Y"], output_node.inputs["Y"])
    links.new(neg_reflect.outputs["Z"], output_node.inputs["Z"])

    layout_nodes(output_node, links)

    return node_tree
