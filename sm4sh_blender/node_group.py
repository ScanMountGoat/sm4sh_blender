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
    links.new(output_vector.outputs["X"], output_node.inputs["X"])
    links.new(output_vector.outputs["Y"], output_node.inputs["Y"])
    links.new(output_vector.outputs["Z"], output_node.inputs["Z"])

    layout_nodes(output_node, links)

    return node_tree


def transform_vector_node_group(name: str):
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
    links.new(output_vector.outputs["X"], output_node.inputs["X"])
    links.new(output_vector.outputs["Y"], output_node.inputs["Y"])
    links.new(output_vector.outputs["Z"], output_node.inputs["Z"])

    layout_nodes(output_node, links)

    return node_tree


def geometry_bitangent_node_group(name: str):
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

    # Blender doesn't expose this directly, but we can use a vector to select the bitangent.
    tbn = nodes.new("ShaderNodeNormalMap")
    tbn.inputs["Color"].default_value = (0.0, 1.0, 0.0, 1.0)

    output_vector = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tbn.outputs["Vector"], output_vector.inputs["Vector"])

    output_node = nodes.new("NodeGroupOutput")
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
    links.new(output_vector.outputs["X"], output_node.inputs["X"])
    links.new(output_vector.outputs["Y"], output_node.inputs["Y"])
    links.new(output_vector.outputs["Z"], output_node.inputs["Z"])

    layout_nodes(output_node, links)

    return node_tree
