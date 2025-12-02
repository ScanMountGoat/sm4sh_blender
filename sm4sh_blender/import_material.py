from typing import Dict, Optional, Set, Tuple
import bpy
import typing
import struct

from sm4sh_blender.node_group import (
    create_node_group,
    dot4_node_group,
    geometry_bitangent_node_group,
    geometry_normal_node_group,
    geometry_tangent_node_group,
    normal_map_xyz_node_group,
    normalize_xyz_node_group,
    rgba_color_node_group,
    sphere_map_coords_node_group,
    transform_point_node_group,
    transform_vector_node_group,
)
from sm4sh_blender.node_layout import layout_nodes

if typing.TYPE_CHECKING:
    from ..sm4sh_model_py.sm4sh_model_py import sm4sh_model_py
else:
    from . import sm4sh_model_py


def import_material(
    material: sm4sh_model_py.NudMaterial,
    database: sm4sh_model_py.database.ShaderDatabase,
    use_advanced_nodes: bool,
) -> bpy.types.Material:
    name = f"{material.shader_id:08X}"
    shader = database.get_shader(material.shader_id)

    if use_advanced_nodes:
        # Blender doesn't support material instances that differ only by property and texture values.
        # Creating nodes that recreate the in game shader code is very expensive.
        # Copying an existing material to use as a base still gives a substantial speedup.
        base_material = bpy.data.materials.get(name)
        if base_material is not None:
            blender_material = base_material.copy()
        else:
            blender_material = create_material(material, shader)

        # Apply the material specific values for this "instance" of the shader.
        # TODO: Update values even if shader is missing to still support material editing.
        if shader is not None:
            update_material(blender_material, material, shader)
    else:
        # Use basic nodes since recreating the shaders is slow and doesn't always render properly.
        blender_material = create_material_basic(material, shader)

    return blender_material


def create_material_basic(
    material: sm4sh_model_py.NudMaterial,
    shader: Optional[sm4sh_model_py.database.ShaderProgram],
) -> bpy.types.Material:
    name = f"{material.shader_id:08X}"
    blender_material = bpy.data.materials.new(name)

    # Use custom properties to preserve values that are hard to represent in Blender.
    blender_material["src_factor"] = str(material.src_factor).removeprefix("SrcFactor.")
    blender_material["dst_factor"] = str(material.dst_factor).removeprefix("DstFactor.")
    blender_material["alpha_func"] = str(material.alpha_func).removeprefix("AlphaFunc.")
    blender_material["cull_mode"] = str(material.cull_mode).removeprefix("CullMode.")

    for prop in material.properties:
        if prop.name == "NU_materialHash":
            material_hash = float32_bits(prop.values[0])
            blender_material["NU_materialHash"] = f"{material_hash:08X}"

    blender_material.use_nodes = True
    nodes = blender_material.node_tree.nodes
    links = blender_material.node_tree.links

    # Create the nodes from scratch to ensure the required nodes are present.
    # This avoids hard coding names like "Material Output" that depend on the UI language.
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (-300, 0)

    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (0, 0)

    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    # TODO: Does preserving the property order matter?
    for i, prop in enumerate(material.properties):
        if prop.name != "NU_materialHash":
            # TODO: Custom node for xyzw values?
            node = nodes.new("ShaderNodeRGB")
            node.location = (-500, i * -200)

            node.name = prop.name
            node.label = prop.name
            node.outputs[0].default_value = prop.values[:4]

    texture_nodes = []
    for i, texture in enumerate(material.textures):
        node = nodes.new("ShaderNodeTexImage")
        node.label = str(i)
        node.location = (-800, i * -300)

        # TODO: Load global textures like color ramps.
        image_name = f"{texture.hash:08X}"
        image = bpy.data.images.get(image_name)
        if image is None:
            image = bpy.data.images.new(image_name, 4, 4, alpha=True)

        node.image = image

        # TODO: Error if U and V have the same wrap mode?
        match texture.wrap_mode_s:
            case sm4sh_model_py.WrapMode.Repeat:
                node.extension = "REPEAT"
            case sm4sh_model_py.WrapMode.MirroredRepeat:
                node.extension = "MIRROR"
            case sm4sh_model_py.WrapMode.ClampToEdge:
                node.extension = "CLIP"

        texture_nodes.append(node)

    # Texture usage is determined by the compiled shaders.
    if shader is not None:
        for name, node in zip(shader.samplers, texture_nodes):
            match name:
                case "colorSampler":
                    # TODO: Is this gamma handling part of the compiled shader code?
                    node.image.colorspace_settings.name = "sRGB"
                    links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
                case "normalSampler":
                    normal = nodes.new("ShaderNodeNormalMap")
                    normal.location = (-300, -400)
                    links.new(node.outputs["Color"], normal.inputs["Color"])
                    links.new(normal.outputs["Normal"], bsdf.inputs["Normal"])
                case "reflectionSampler":
                    pass
                case "reflectionCubeSampler":
                    pass

    return blender_material


def create_material(
    material: sm4sh_model_py.NudMaterial,
    shader: Optional[sm4sh_model_py.database.ShaderProgram],
) -> bpy.types.Material:
    name = f"{material.shader_id:08X}"
    blender_material = bpy.data.materials.new(name)

    blender_material.use_nodes = True
    nodes = blender_material.node_tree.nodes
    links = blender_material.node_tree.links

    # Create the nodes from scratch to ensure the required nodes are present.
    # This avoids hard coding names like "Material Output" that depend on the UI language.
    nodes.clear()

    output_node = nodes.new("ShaderNodeOutputMaterial")

    final_emission = nodes.new("ShaderNodeEmission")

    if shader is not None:
        # TODO: Does preserving the property order matter?
        for i, name in enumerate(shader.parameters):
            name = f"NU_{name}"
            if name != "NU_materialHash":
                node = create_node_group(nodes, "RgbaColor", rgba_color_node_group)
                node.name = name
                node.label = name

        texture_nodes = []
        for i, name in enumerate(shader.samplers):
            node = nodes.new("ShaderNodeTexImage")
            node.label = str(i)
            node.name = name
            node.location = (-800, i * -300)
            texture_nodes.append(node)

        textures = {}
        for name, node in zip(shader.samplers, texture_nodes):
            textures[name] = node

        # Create nodes for each unique assignment.
        # Storing the output name allows using a single node for values with multiple channels.
        expr_outputs = []
        for i, expr in enumerate(shader.exprs):
            # TODO: Get the values for shared uniform buffers.
            node_output = assign_output(
                shader, expr, expr_outputs, nodes, links, textures
            )
            expr_outputs.append(node_output)

        output_color = nodes.new("ShaderNodeCombineColor")

        for output, channel in [
            ("out_attr0.x", "Red"),
            ("out_attr0.y", "Green"),
            ("out_attr0.z", "Blue"),
        ]:
            if output in shader.output_dependencies:
                assign_index(
                    shader.output_dependencies[output],
                    expr_outputs,
                    links,
                    output_color.inputs[channel],
                )

        # TODO: Recreate the in game gamma correction and value range remapping from bloom.
        links.new(output_color.outputs["Color"], final_emission.inputs["Color"])

        # Recreate alpha blending.
        # TODO: Support additive and multiply blend modes.
        mix_shaders = nodes.new("ShaderNodeMixShader")
        if "out_attr0.w" in shader.output_dependencies:
            assign_index(
                shader.output_dependencies["out_attr0.w"],
                expr_outputs,
                links,
                mix_shaders.inputs["Fac"],
            )
        else:
            mix_shaders.inputs["Fac"].default_value = 1.0

        transparent_shader = nodes.new("ShaderNodeBsdfTransparent")
        links.new(transparent_shader.outputs["BSDF"], mix_shaders.inputs[1])
        links.new(final_emission.outputs["Emission"], mix_shaders.inputs[2])

        links.new(mix_shaders.outputs["Shader"], output_node.inputs["Surface"])
    else:
        # TODO: report error
        pass

    layout_nodes(output_node, links)

    return blender_material


def update_material(
    blender_material: bpy.types.Material,
    material: sm4sh_model_py.NudMaterial,
    shader: sm4sh_model_py.database.ShaderProgram,
):
    # Use custom properties to preserve values that are hard to represent in Blender.
    blender_material["src_factor"] = str(material.src_factor).removeprefix("SrcFactor.")
    blender_material["dst_factor"] = str(material.dst_factor).removeprefix("DstFactor.")
    blender_material["alpha_func"] = str(material.alpha_func).removeprefix("AlphaFunc.")
    blender_material["cull_mode"] = str(material.cull_mode).removeprefix("CullMode.")

    for prop in material.properties:
        if prop.name == "NU_materialHash":
            material_hash = float32_bits(prop.values[0])
            blender_material["NU_materialHash"] = f"{material_hash:08X}"

    nodes = blender_material.node_tree.nodes

    for prop in material.properties:
        if prop.name != "NU_materialHash":
            if node := nodes.get(prop.name):
                node.inputs["Color"].default_value = [
                    prop.values[0],
                    prop.values[1],
                    prop.values[2],
                    1.0,
                ]
                node.inputs["Alpha"].default_value = prop.values[3]

    for texture, name in zip(material.textures, shader.samplers):
        if node := nodes.get(name):
            # TODO: Load global textures like color ramps.
            image_name = f"{texture.hash:08X}"
            image = bpy.data.images.get(image_name)
            if image is None:
                image = bpy.data.images.new(image_name, 4, 4, alpha=True)

            node.image = image

            # TODO: Error if U and V have the same wrap mode?
            match texture.wrap_mode_s:
                case sm4sh_model_py.WrapMode.Repeat:
                    node.extension = "REPEAT"
                case sm4sh_model_py.WrapMode.MirroredRepeat:
                    node.extension = "MIRROR"
                case sm4sh_model_py.WrapMode.ClampToEdge:
                    node.extension = "CLIP"


def material_images_samplers(material, blender_images, samplers):
    material_textures = {}

    for i, texture in enumerate(material.textures):
        name = f"s{i}"

        image = None
        try:
            image = blender_images[texture.image_texture_index]
        except IndexError:
            pass

        # TODO: xenoblade x samplers?
        sampler = None
        if texture.sampler_index < len(samplers):
            sampler = samplers[texture.sampler_index]

        material_textures[name] = (image, sampler)

    return material_textures


def assign_normal_map(
    nodes,
    links,
    bsdf,
    x_assignment: Optional[int],
    y_assignment: Optional[int],
    intensity_assignment: Optional[int],
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
) -> Optional[bpy.types.Node]:
    if x_assignment is None or y_assignment is None:
        return None

    normals = create_node_group(
        nodes, "NormalMapXYFinal", normal_map_xy_final_node_group
    )
    normals.inputs["X"].default_value = 0.5
    normals.inputs["Y"].default_value = 0.5
    normals.inputs["Strength"].default_value = 1.0

    assign_index(
        x_assignment,
        expr_outputs,
        links,
        normals.inputs["X"],
    )
    assign_index(
        y_assignment,
        expr_outputs,
        links,
        normals.inputs["Y"],
    )

    if intensity_assignment is not None:
        assign_index(
            intensity_assignment,
            expr_outputs,
            links,
            normals.inputs["Strength"],
        )

    links.new(normals.outputs["Normal"], bsdf.inputs["Normal"])

    return normals


def assign_output(
    shader: sm4sh_model_py.database.ShaderProgram,
    expr: sm4sh_model_py.database.OutputExpr,
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    textures: Dict[str, Optional[bpy.types.Image]],
) -> Optional[Tuple[bpy.types.Node, str]]:
    if func := expr.func():
        mix_rgba_node = lambda ty: assign_mix_rgba(
            func,
            expr_outputs,
            nodes,
            links,
            ty,
        )

        math_node = lambda ty: assign_math(
            func,
            expr_outputs,
            nodes,
            links,
            ty,
        )

        group_node = lambda func, name, create_node_tree: create_cached_func_group_node(
            nodes, func, name, create_node_tree
        )

        assign_args = lambda func, node, params: assign_func_args(
            func, params, expr_outputs, links, node
        )

        assign_arg = lambda i, output: assign_index(i, expr_outputs, links, output)

        match func.op:
            case sm4sh_model_py.database.Operation.Add:
                return math_node("ADD")
            case sm4sh_model_py.database.Operation.Sub:
                return math_node("SUBTRACT")
            case sm4sh_model_py.database.Operation.Mul:
                return math_node("MULTIPLY")
            case sm4sh_model_py.database.Operation.Div:
                return math_node("DIVIDE")
            case sm4sh_model_py.database.Operation.Mix:
                return mix_rgba_node("MIX")
            case sm4sh_model_py.database.Operation.Clamp:
                node = nodes.new("ShaderNodeClamp")
                node.name = func_name(func)
                assign_args(func, node, ["Value", "Min", "Max"])
                return node, "Result"
            case sm4sh_model_py.database.Operation.Min:
                return math_node("MINIMUM")
            case sm4sh_model_py.database.Operation.Max:
                return math_node("MAXIMUM")
            case sm4sh_model_py.database.Operation.Abs:
                return math_node("ABSOLUTE")
            case sm4sh_model_py.database.Operation.Floor:
                return math_node("FLOOR")
            case sm4sh_model_py.database.Operation.Power:
                return math_node("POWER")
            case sm4sh_model_py.database.Operation.Sqrt:
                return math_node("SQRT")
            case sm4sh_model_py.database.Operation.InverseSqrt:
                return math_node("INVERSE_SQRT")
            case sm4sh_model_py.database.Operation.Fma:
                return math_node("MULTIPLY_ADD")
            case sm4sh_model_py.database.Operation.Dot:
                node = create_node_group(nodes, "Dot4", dot4_node_group)
                node.name = func_name(func)

                if len(func.args) == 6:
                    # dot3
                    assign_args(func, node, ["A.x", "A.y", "A.z", "B.x", "B.y", "B.z"])
                    node.inputs["A.w"].default_value = 0.0
                    node.inputs["B.w"].default_value = 0.0
                else:
                    # dot4
                    assign_args(
                        func,
                        node,
                        ["A.x", "A.y", "A.z", "A.w", "B.x", "B.y", "B.z", "B.w"],
                    )

                return node, "Value"
            case sm4sh_model_py.database.Operation.Sin:
                return math_node("SINE")
            case sm4sh_model_py.database.Operation.Cos:
                return math_node("COSINE")
            case sm4sh_model_py.database.Operation.Exp2:
                node = nodes.new("ShaderNodeMath")
                node.name = func_name(func)
                node.operation = "POWER"

                node.inputs[0].default_value = 2.0
                assign_arg(func.args[0], node.inputs[1])

                return node, "Value"
            case sm4sh_model_py.database.Operation.Log2:
                node = nodes.new("ShaderNodeMath")
                node.name = func_name(func)
                node.operation = "LOGARITHM"

                assign_arg(func.args[0], node.inputs[0])
                node.inputs[1].default_value = 2.0

                return node, "Value"
            case sm4sh_model_py.database.Operation.Fract:
                return mix_rgba_node("FRACT")
            # TODO: intbitstofloat, floatbitstoint

            case sm4sh_model_py.database.Operation.Select:
                return mix_rgba_node("MIX")
            case sm4sh_model_py.database.Operation.Negate:
                node = nodes.new("ShaderNodeMath")
                node.name = func_name(func)
                node.operation = "MULTIPLY"

                assign_arg(func.args[0], node.inputs[0])
                node.inputs[1].default_value = -1.0

                return node, "Value"
            case sm4sh_model_py.database.Operation.Equal:
                return math_node("COMPARE")
            case sm4sh_model_py.database.Operation.NotEqual:
                # TODO: Invert compare.
                return math_node("COMPARE")
            case sm4sh_model_py.database.Operation.Less:
                return math_node("LESS_THAN")
            case sm4sh_model_py.database.Operation.Greater:
                return math_node("GREATER_THAN")
            case sm4sh_model_py.database.Operation.LessEqual:
                # TODO: node group for leq?
                return math_node("LESS_THAN")
            case sm4sh_model_py.database.Operation.GreaterEqual:
                # TODO: node group for geq?
                return math_node("GREATER_THAN")
            case sm4sh_model_py.database.Operation.NormalMapX:
                node = group_node(func, "NormalMapXYZ", normal_map_xyz_node_group)
                assign_args(func, node, ["X", "Y", "Z"])
                return node, "X"
            case sm4sh_model_py.database.Operation.NormalMapY:
                node = group_node(func, "NormalMapXYZ", normal_map_xyz_node_group)
                assign_args(func, node, ["X", "Y", "Z"])
                return node, "Y"
            case sm4sh_model_py.database.Operation.NormalMapZ:
                node = group_node(func, "NormalMapXYZ", normal_map_xyz_node_group)
                assign_args(func, node, ["X", "Y", "Z"])
                return node, "Z"
            case sm4sh_model_py.database.Operation.NormalizeX:
                node = group_node(func, "NormalizeXYZ", normalize_xyz_node_group)
                assign_args(func, node, ["X", "Y", "Z"])
                return node, "X"
            case sm4sh_model_py.database.Operation.NormalizeY:
                node = group_node(func, "NormalizeXYZ", normalize_xyz_node_group)
                assign_args(func, node, ["X", "Y", "Z"])
                return node, "Y"
            case sm4sh_model_py.database.Operation.NormalizeZ:
                node = group_node(func, "NormalizeXYZ", normalize_xyz_node_group)
                assign_args(func, node, ["X", "Y", "Z"])
                return node, "Z"
            case sm4sh_model_py.database.Operation.SphereMapCoordX:
                node = group_node(func, "SphereMapCoords", sphere_map_coords_node_group)
                assign_args(func, node, ["Param"])
                return node, "X"
            case sm4sh_model_py.database.Operation.SphereMapCoordY:
                node = group_node(func, "SphereMapCoords", sphere_map_coords_node_group)
                assign_args(func, node, ["Param"])
                return node, "Y"
            case sm4sh_model_py.database.Operation.LocalToWorldPointX:
                node = group_node(func, "LocalToWorldPoint", transform_point_node_group)
                assign_args(func, node, ["X", "Y", "Z"])
                return node, "X"
            case sm4sh_model_py.database.Operation.LocalToWorldPointY:
                node = group_node(func, "LocalToWorldPoint", transform_point_node_group)
                assign_args(func, node, ["X", "Y", "Z"])
                return node, "Y"
            case sm4sh_model_py.database.Operation.LocalToWorldPointZ:
                node = group_node(func, "LocalToWorldPoint", transform_point_node_group)
                assign_args(func, node, ["X", "Y", "Z"])
                return node, "Z"
            case sm4sh_model_py.database.Operation.LocalToWorldVectorX:
                node = group_node(
                    func, "LocalToWorldVector", transform_vector_node_group
                )
                assign_args(func, node, ["X", "Y", "Z"])
                return node, "X"
            case sm4sh_model_py.database.Operation.LocalToWorldVectorY:
                node = group_node(
                    func, "LocalToWorldVector", transform_vector_node_group
                )
                assign_args(func, node, ["X", "Y", "Z"])
                return node, "Y"
            case sm4sh_model_py.database.Operation.LocalToWorldVectorZ:
                node = group_node(
                    func, "LocalToWorldVector", transform_vector_node_group
                )
                assign_args(func, node, ["X", "Y", "Z"])
                return node, "Z"
            case sm4sh_model_py.database.Operation.Unk:
                return None
            case _:
                # TODO: This case shouldn't happen?
                return None
    elif value := expr.value():
        return assign_value(shader, value, expr_outputs, nodes, links, textures)
    else:
        return None


def assign_index(
    i: Optional[int],
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    links,
    output,
):
    if i is not None:
        if node_output := expr_outputs[i]:
            node, output_name = node_output
            links.new(node.outputs[output_name], output)
            return
        else:
            # Set defaults to match xc3_wgpu and make debugging easier.
            assign_float(output, 0.0)


def assign_value(
    shader: sm4sh_model_py.database.ShaderProgram,
    value: sm4sh_model_py.database.Value,
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    textures,
) -> Optional[Tuple[bpy.types.Node, str]]:
    if i := value.int():
        node = nodes.new("ShaderNodeValue")
        node.outputs[0].default_value = i
        return node, "Value"
    elif f := value.float():
        node = nodes.new("ShaderNodeValue")
        node.outputs[0].default_value = f
        return node, "Value"
    elif parameter := value.parameter():
        return assign_parameter(shader, parameter, nodes, links)
    elif attribute := value.attribute():
        return assign_attribute(attribute, nodes, links)
    elif texture := value.texture():
        return assign_texture(texture, expr_outputs, nodes, links, textures)
    else:
        return None


def assign_float(output, f):
    # This may be a float, RGBA, or XYZ socket.
    try:
        output.default_value = [f] * len(output.default_value)
    except:
        output.default_value = f


def assign_parameter(
    shader: sm4sh_model_py.database.ShaderProgram,
    parameter: sm4sh_model_py.database.Parameter,
    nodes,
    links,
) -> Optional[Tuple[bpy.types.Node, str]]:
    # TODO: Is there an easy way to support all the global parameters?
    if parameter.name == "MC" or parameter.name == "MC_EFFECT":
        name = f"NU_{parameter.field}"
        if node := nodes.get(name):
            # NU_ nodes use a custom node group for RGBA channel support.
            return node, channel_rgba_name(parameter.channel)

    else:
        node = nodes.new("ShaderNodeValue")
        # TODO: add the Rust display impl to python?
        label = parameter.name
        if parameter.field:
            label += f".{parameter.field}"
        if parameter.index is not None:
            label += f"[{parameter.index}]"
        if parameter.channel:
            label += f".{parameter.channel}"

        node.label = label

        value = shader.parameter_value(parameter)
        if value is not None:
            node.outputs[0].default_value = value

        return node, "Value"


def assign_attribute(
    attribute: sm4sh_model_py.database.Attribute, nodes, links
) -> Optional[Tuple[bpy.types.Node, str]]:
    name = attribute.name
    channel = attribute.channel

    # Some attributes aren't exposed directly and require custom node groups.
    match name:
        case "a_Position":
            return assign_attribute_node(nodes, links, name, channel, "position")
        case "a_Normal":
            node = nodes.get(name)
            if node is None:
                node = create_node_group(
                    nodes, "GeometryNormal", geometry_normal_node_group
                )
                node.name = name

            return node, channel_name(channel)
        case "a_Tangent":
            node = nodes.get(name)
            if node is None:
                node = create_node_group(
                    nodes, "GeometryTangent", geometry_tangent_node_group
                )
                node.name = name

            return node, channel_name(channel)
        case "a_Binormal":
            node = nodes.get(name)
            if node is None:
                node = create_node_group(
                    nodes, "GeometryBitangent", geometry_bitangent_node_group
                )
                node.name = name

            return node, channel_name(channel)
        case "a_Color":
            return assign_attribute_node(nodes, links, name, channel, "Color")
        case "a_TexCoord0":
            return assign_attribute_node(nodes, links, name, channel, "UV0")
        case "a_TexCoord1":
            return assign_attribute_node(nodes, links, name, channel, "UV1")
        case "a_TexCoord2":
            return assign_attribute_node(nodes, links, name, channel, "UV2")
        case "bitangent_sign":
            # TODO: Is there a way to calculate this?
            return None


def assign_attribute_node(
    nodes, links, name: str, channel: Optional[str], attribute_name: str
) -> Tuple[bpy.types.Node, str]:
    node = nodes.get(name)
    if node is None:
        node = nodes.new("ShaderNodeAttribute")
        node.name = name
        node.attribute_name = attribute_name

    return assign_channel(name, channel, node, nodes, links)


def assign_channel(
    name: str, channel: Optional[str], node, nodes, links
) -> Tuple[bpy.types.Node, str]:
    output = channel_rgba_name(channel)
    if output == "Alpha":
        # Alpha isn't part of the RGB node.
        return node, "Alpha"
    else:
        # Avoid creating more than one separate RGB for each node.
        rgb_name = f"{name}.rgb"
        rgb_node = nodes.get(rgb_name)
        if rgb_node is None:
            rgb_node = nodes.new("ShaderNodeSeparateColor")
            rgb_node.name = rgb_name
            links.new(node.outputs["Color"], rgb_node.inputs["Color"])

        return rgb_node, output


def channel_rgba_name(channel: Optional[str]) -> str:
    match channel:
        case "x":
            return "Red"
        case "y":
            return "Green"
        case "z":
            return "Blue"
        case "w":
            return "Alpha"

    # TODO: How to handle the None case?
    return "Red"


def channel_name(channel: Optional[str]) -> str:
    match channel:
        case "x":
            return "X"
        case "y":
            return "Y"
        case "z":
            return "Z"
        case "w":
            return "W"

    # TODO: How to handle the None case?
    return "X"


def assign_mix_rgba(
    func: sm4sh_model_py.database.OutputExprFunc,
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    blend_type: str,
) -> Tuple[bpy.types.Node, str]:
    node = nodes.new("ShaderNodeMix")
    node.data_type = "RGBA"
    node.blend_type = blend_type
    node.name = func_name(func)

    if blend_type == "OVERLAY":
        node.inputs["Factor"].default_value = 1.0

    assign_index(func.args[0], expr_outputs, links, node.inputs["A"])
    assign_index(func.args[1], expr_outputs, links, node.inputs["B"])
    if len(func.args) == 3:
        assign_index(func.args[2], expr_outputs, links, node.inputs["Factor"])

    return node, "Result"


def assign_texture(
    texture: sm4sh_model_py.database.Texture,
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    textures: Dict[str, bpy.types.Image],
) -> Optional[Tuple[bpy.types.Node, str]]:
    name = texture_assignment_name(texture)

    # Don't use the above name for node caching for any of the texture nodes.
    # This ensures the correct channel is assigned for each assignment.
    node = textures.get(texture.name)
    if node is not None:
        assign_uvs(texture.texcoords, expr_outputs, node, nodes, links)
        return assign_channel(name, texture.channel, node, nodes, links)
    else:
        return None


def assign_uvs(texcoords: list[int], expr_outputs, node, nodes, links):
    # Texture coordinates can be made of multiple nodes.
    uv_name = f"uv{texcoords}"
    uv_node = nodes.get(uv_name)
    if uv_node is None:
        uv_node = nodes.new("ShaderNodeCombineXYZ")
        uv_node.name = uv_name

        if len(texcoords) >= 2:
            assign_index(
                texcoords[0],
                expr_outputs,
                links,
                uv_node.inputs["X"],
            )
            assign_index(
                texcoords[1],
                expr_outputs,
                links,
                uv_node.inputs["Y"],
            )

    links.new(uv_node.outputs["Vector"], node.inputs["Vector"])


def assign_math(
    func: sm4sh_model_py.database.OutputExprFunc,
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    nodes,
    links,
    op: str,
) -> Tuple[bpy.types.Node, str]:
    node = nodes.new("ShaderNodeMath")
    node.operation = op
    node.name = func_name(func)

    for arg, input in zip(func.args, node.inputs):
        assign_index(arg, expr_outputs, links, input)

    return node, "Value"


def func_name(func: sm4sh_model_py.database.OutputExprFunc):
    return func_name_inner(func.op, func.args)


def func_name_inner(op: sm4sh_model_py.database.Operation, args: list[int]):
    op_name = str(op).removeprefix("Operation.")
    # Node groups that have multiple outputs can share a node.
    replacements = [
        ("NormalizeX", "Normalize"),
        ("NormalizeY", "Normalize"),
        ("NormalizeZ", "Normalize"),
        ("NormalMapX", "NormalMap"),
        ("NormalMapY", "NormalMap"),
        ("NormalMapZ", "NormalMap"),
        ("SphereMapCoordX", "SphereMapCoord"),
        ("SphereMapCoordY", "SphereMapCoord"),
        ("LocalToWorldPointX", "LocalToWorldPoint"),
        ("LocalToWorldPointY", "LocalToWorldPoint"),
        ("LocalToWorldPointZ", "LocalToWorldPoint"),
        ("LocalToWorldVectorX", "LocalToWorldVector"),
        ("LocalToWorldVectorY", "LocalToWorldVector"),
        ("LocalToWorldVectorZ", "LocalToWorldVector"),
    ]
    for old, new in replacements:
        if op_name.startswith(old):
            op_name = op_name.replace(old, new)
            break

    func_args = ", ".join(str(a) for a in args)
    name = f"{op_name}({func_args})"
    return name


def texture_assignment_name(texture):
    coords = ", ".join(str(c) for c in texture.texcoords)
    return f"{texture.name}({coords})"


def float32_bits(f: float) -> int:
    return struct.unpack("@I", struct.pack("@f", f))[0]


def create_cached_func_group_node(
    nodes,
    func: sm4sh_model_py.database.OutputExprFunc,
    node_group_name: str,
    create_node_tree,
) -> bpy.types.Node:
    name = func_name(func)
    node = nodes.get(name)
    if node is None:
        node = create_node_group(nodes, node_group_name, create_node_tree)
        node.name = name

    return node


def assign_func_args(
    func: sm4sh_model_py.database.OutputExprFunc,
    params: list[int | str],
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str]]],
    links,
    node: bpy.types.Node,
):
    for i, param in zip(func.args, params):
        assign_index(i, assignment_outputs, links, node.inputs[param])
