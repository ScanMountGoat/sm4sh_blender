import bpy

from sm4sh_blender.node_layout import layout_nodes


def create_node_group(nodes, name: str, create_node_tree):
    # Cache the node group creation.
    node_tree = bpy.data.node_groups.get(name)
    if node_tree is None:
        node_tree = create_node_tree()

    group = nodes.new("ShaderNodeGroup")
    group.node_tree = node_tree
    return group


def rgba_color_node_group():
    # TODO: Is it better to use XYZW naming?
    node_tree = bpy.data.node_groups.new("RgbaColor", "ShaderNodeTree")

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


def dot4_node_group():
    node_tree = bpy.data.node_groups.new("Dot4", "ShaderNodeTree")

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
