import struct
import typing
from typing import Dict, Optional, Set, Tuple

import bpy

from sm4sh_blender.node_group import (
    anisotropic_spec_node_group,
    anisotropic_spec_node_group_xyz,
    blinn_phong_spec_node_group,
    blinn_phong_spec_node_group_xyz,
    clamp_xyz_node_group,
    create_node_group,
    cube_coords_node_group,
    dot4_node_group,
    eye_vector_node_group,
    fresnel_node_group,
    fresnel_node_group_xyz,
    geometry_bitangent_node_group,
    geometry_normal_node_group,
    geometry_tangent_node_group,
    greater_xyz_node_group,
    inversesqrt_xyz_node_group,
    less_xyz_node_group,
    neg_reflect_node_group,
    neg_reflect_node_group_xyz,
    normal_map_node_group,
    normal_map_xyz_node_group,
    normalize_xyz_node_group,
    rgba_color_node_group,
    sphere_map_coords_node_group,
    sqrt_xyz_node_group,
    tint_color_node_group,
    tint_color_node_group_xyz,
    transform_point_node_group,
    transform_vector_node_group,
)
from sm4sh_blender.node_layout import layout_nodes

if typing.TYPE_CHECKING:
    from sm4sh_model_py import sm4sh_model_py
else:
    from . import sm4sh_model_py


def import_material(
    material: sm4sh_model_py.NudMaterial,
    metal_material: Optional[sm4sh_model_py.NudMaterial],
    database: sm4sh_model_py.database.ShaderDatabase,
    use_advanced_nodes: bool,
) -> bpy.types.Material:
    name = f"{material.shader_id:08X}"
    shader = database.get_shader(material.shader_id)

    metal_shader = None
    if metal_material is not None:
        metal_shader = database.get_shader(metal_material.shader_id)

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

    # Preserve material settings for export later.
    update_custom_properties(blender_material, material)

    # Preserve the metal.nud material settings for export later.
    update_metal_custom_properties(blender_material, metal_material, metal_shader)

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
        used_expr_indices, used_expr_xyz_indices = used_assignments(shader)

        expr_outputs = []
        for i, expr in enumerate(shader.exprs):
            if i in used_expr_indices:
                # TODO: Get the values for shared uniform buffers.
                node_output = assign_output(
                    shader, expr, expr_outputs, nodes, links, textures
                )
                expr_outputs.append(node_output)
            else:
                expr_outputs.append(None)

        expr_outputs_xyz = []
        for i, expr in enumerate(shader.exprs_xyz):
            if i in used_expr_xyz_indices:
                node_output = assign_output_xyz(
                    shader, expr, expr_outputs, expr_outputs_xyz, nodes, links, textures
                )
                expr_outputs_xyz.append(node_output)
            else:
                expr_outputs_xyz.append(None)

        # Recreate the in game gamma correction and value range remapping from bloom.
        multiply2 = nodes.new("ShaderNodeMix")
        multiply2.data_type = "RGBA"
        multiply2.blend_type = "MULTIPLY"
        multiply2.inputs["B"].default_value = (2.0, 2.0, 2.0, 1.0)
        multiply2.inputs["Factor"].default_value = 1.0

        if "out_attr0.xyz" in shader.output_dependencies_xyz:
            assign_index(
                shader.output_dependencies_xyz["out_attr0.xyz"],
                expr_outputs_xyz,
                links,
                multiply2.inputs["A"],
            )
        else:
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

            links.new(output_color.outputs["Color"], multiply2.inputs["A"])

        gamma = nodes.new("ShaderNodeGamma")
        links.new(multiply2.outputs["Result"], gamma.inputs["Color"])
        gamma.inputs["Gamma"].default_value = 2.2

        links.new(gamma.outputs[0], final_emission.inputs["Color"])

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
            image_name = f"{texture.hash:08X}"
            image = bpy.data.images.get(image_name)
            if image is None:
                # Create a blank image to preserve the texture hash for material export.
                # TODO: report a warning that the texture is not a global texture or in the model.nut
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


def update_custom_properties(
    blender_material: bpy.types.Material, material: sm4sh_model_py.NudMaterial
):
    # Use custom properties to preserve values that are hard to represent in Blender.
    # TODO: register actual properties under blender_material.sm4sh_blender?
    blender_material["src_factor"] = str(material.src_factor).removeprefix("SrcFactor.")
    blender_material["dst_factor"] = str(material.dst_factor).removeprefix("DstFactor.")
    blender_material["alpha_func"] = str(material.alpha_func).removeprefix("AlphaFunc.")
    blender_material["cull_mode"] = str(material.cull_mode).removeprefix("CullMode.")

    # TODO: store this as a string property if blender doesn't support hex?
    for prop in material.properties:
        if prop.name == "NU_materialHash":
            material_hash = float32_bits(prop.values[0])
            blender_material["NU_materialHash"] = f"{material_hash:08X}"


def update_metal_custom_properties(
    blender_material: bpy.types.Material,
    metal_material: Optional[sm4sh_model_py.NudMaterial],
    metal_shader: Optional[sm4sh_model_py.database.ShaderProgram],
):
    if metal_material is not None:
        # Texture usage is determined by the compiled shaders.
        if metal_shader is not None:
            for texture, sampler in zip(metal_material.textures, metal_shader.samplers):
                if sampler == "colorSampler":
                    # Preserve the metal color texture in case it has an alpha channel.
                    image_name = f"{texture.hash:08X}"
                    image = bpy.data.images.get(image_name)
                    blender_material.sm4sh_blender.metal.diffuse = image
                elif sampler == "reflectionSampler":
                    # Preserve glossy vs rough reflections.
                    if texture.hash == 0x10102000:
                        blender_material.sm4sh_blender.metal.stage_cube = "10102000"
                    elif texture.hash == 0x10101000:
                        blender_material.sm4sh_blender.metal.stage_cube = "10101000"

        # Preserve material parameters for reimporting custom models that modify these values.
        # Default metal.nud models always use the same property values.
        for prop in metal_material.properties:
            match prop.name:
                case "NU_reflectionColor":
                    blender_material.sm4sh_blender.metal.reflection_color = prop.values[
                        :4
                    ]
                case "NU_fresnelColor":
                    blender_material.sm4sh_blender.metal.fresnel_color = prop.values[:4]
                case "NU_fresnelParams":
                    blender_material.sm4sh_blender.metal.fresnel_params = prop.values[
                        :4
                    ]
                case "NU_aoMinGain":
                    blender_material.sm4sh_blender.metal.ao_min_gain = prop.values[:4]


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


def assign_output(
    shader: sm4sh_model_py.database.ShaderProgram,
    expr: sm4sh_model_py.database.OutputExpr,
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str] | float]],
    nodes,
    links,
    textures: Dict[str, Optional[bpy.types.Image]],
) -> Optional[Tuple[bpy.types.Node, str] | float]:
    if func := expr.func():

        def mix_rgba_node(ty):
            return assign_mix_rgba(
                func,
                expr_outputs,
                nodes,
                links,
                ty,
            )

        def math_node(ty):
            return assign_math(
                func,
                expr_outputs,
                nodes,
                links,
                ty,
            )

        def group_node(func, name, create_node_tree):
            return create_cached_func_group_node(nodes, func, name, create_node_tree)

        def assign_args(func, node, params):
            return assign_func_args(func, params, expr_outputs, links, node)

        def assign_arg(i, output):
            return assign_index(i, expr_outputs, links, output)

        match func.op:
            case sm4sh_model_py.database.Operation.Unk:
                return None
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
            case sm4sh_model_py.database.Operation.Dot3:
                node = create_node_group(nodes, "Dot4", dot4_node_group)
                node.name = func_name(func)

                assign_args(func, node, ["A.x", "A.y", "A.z", "B.x", "B.y", "B.z"])
                node.inputs["A.w"].default_value = 0.0
                node.inputs["B.w"].default_value = 0.0

                return node, "Value"
            case sm4sh_model_py.database.Operation.Dot4:
                node = create_node_group(nodes, "Dot4", dot4_node_group)
                node.name = func_name(func)

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
                return math_node("FRACT")
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
            case sm4sh_model_py.database.Operation.VarianceShadow:
                # Shadow mapping is already handled by Blender.
                node = nodes.new("ShaderNodeValue")
                node.label = "VarianceShadow"
                node.outputs[0].default_value = 1.0
                return node, "Value"
            case sm4sh_model_py.database.Operation.BlinnPhongSpecular:
                node = group_node(
                    func, "BlinnPhongSpecular", blinn_phong_spec_node_group
                )
                assign_args(
                    func,
                    node,
                    [
                        "normal.X",
                        "normal.Y",
                        "normal.Z",
                        "lightDir.X",
                        "lightDir.Y",
                        "lightDir.Z",
                        "eye.X",
                        "eye.Y",
                        "eye.Z",
                        "Exponent",
                    ],
                )
                return node, "Value"
            case sm4sh_model_py.database.Operation.AnisotropicSpecular:
                node = group_node(
                    func, "AnisotropicSpecular", anisotropic_spec_node_group
                )
                assign_args(
                    func,
                    node,
                    [
                        "normal.X",
                        "normal.Y",
                        "normal.Z",
                        "tangent.X",
                        "tangent.Y",
                        "tangent.Z",
                        "eye.X",
                        "eye.Y",
                        "eye.Z",
                        "ParamX",
                        "ParamY",
                    ],
                )
                return node, "Value"
            case sm4sh_model_py.database.Operation.Fresnel:
                node = group_node(func, "Fresnel", fresnel_node_group)
                assign_args(
                    func,
                    node,
                    [
                        "normal.X",
                        "normal.Y",
                        "normal.Z",
                        "eye.X",
                        "eye.Y",
                        "eye.Z",
                        "Param",
                    ],
                )
                return node, "Value"
            case sm4sh_model_py.database.Operation.TintColorX:
                node = group_node(func, "TintColor", tint_color_node_group)
                assign_args(
                    func,
                    node,
                    ["Red", "Green", "Blue", "Factor"],
                )
                return node, "Red"
            case sm4sh_model_py.database.Operation.TintColorY:
                node = group_node(func, "TintColor", tint_color_node_group)
                assign_args(
                    func,
                    node,
                    ["Red", "Green", "Blue", "Factor"],
                )
                return node, "Green"
            case sm4sh_model_py.database.Operation.TintColorZ:
                node = group_node(func, "TintColor", tint_color_node_group)
                assign_args(
                    func,
                    node,
                    ["Red", "Green", "Blue", "Factor"],
                )
                return node, "Blue"
            case sm4sh_model_py.database.Operation.NegReflectX:
                node = group_node(func, "NegReflect", neg_reflect_node_group)
                assign_args(
                    func,
                    node,
                    ["A.x", "A.y", "A.z", "B.x", "B.y", "B.z"],
                )
                return node, "X"
            case sm4sh_model_py.database.Operation.NegReflectY:
                node = group_node(func, "NegReflect", neg_reflect_node_group)
                assign_args(
                    func,
                    node,
                    ["A.x", "A.y", "A.z", "B.x", "B.y", "B.z"],
                )
                return node, "Y"
            case sm4sh_model_py.database.Operation.NegReflectZ:
                node = group_node(func, "NegReflect", neg_reflect_node_group)
                assign_args(
                    func,
                    node,
                    ["A.x", "A.y", "A.z", "B.x", "B.y", "B.z"],
                )
                return node, "Z"
            case _:
                # TODO: This case shouldn't happen?
                return None
    elif value := expr.value():
        return assign_value(shader, value, expr_outputs, nodes, links, textures)
    else:
        return None


def assign_output_xyz(
    shader: sm4sh_model_py.database.ShaderProgram,
    expr: sm4sh_model_py.database.OutputExprXyz,
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str] | float]],
    expr_outputs_xyz: list[
        Optional[Tuple[bpy.types.Node, str] | tuple[float, float, float]]
    ],
    nodes,
    links,
    textures: Dict[str, Optional[bpy.types.Image]],
) -> Optional[Tuple[bpy.types.Node, str] | tuple[float, float, float]]:
    if func := expr.func():
        return assign_func_xyz(func, expr_outputs_xyz, nodes, links)
    elif value := expr.value():
        return assign_value_xyz(shader, value, expr_outputs, nodes, links, textures)
    else:
        return None


def assign_index(
    i: Optional[int],
    expr_outputs: list[
        Optional[Tuple[bpy.types.Node, str] | float | tuple[float, float, float]]
    ],
    links,
    output,
):
    if i is not None:
        node_output = expr_outputs[i]
        if node_output is not None:
            try:
                # Assign floats directly to reduce links and improve load times.
                assign_float(output, node_output)
            except:
                node, output_name = node_output
                links.new(node.outputs[output_name], output)
        else:
            # Set defaults to match sm4sh_wgpu and make debugging easier.
            assign_float(output, 0.0)


def assign_func_xyz(
    func: sm4sh_model_py.database.OutputExprFuncXyz,
    expr_outputs_xyz: list[
        Optional[Tuple[bpy.types.Node, str] | tuple[float, float, float]]
    ],
    nodes,
    links,
) -> Optional[Tuple[bpy.types.Node, str] | tuple[float, float, float]]:
    result = assign_func_xyz_inner(func, expr_outputs_xyz, nodes, links)
    if result is not None:
        node, output = result
        return assign_xyz_channel(node, output, func.channel, nodes, links)
    else:
        return None


def assign_func_xyz_inner(
    func: sm4sh_model_py.database.OutputExprFuncXyz,
    expr_outputs_xyz: list[
        Optional[Tuple[bpy.types.Node, str] | tuple[float, float, float]]
    ],
    nodes,
    links,
) -> Optional[Tuple[bpy.types.Node, str]]:
    # TODO: function that adds a separate XYZ node if needed.

    def mix_rgba_node(ty):
        return assign_mix_xyz(
            func,
            expr_outputs_xyz,
            nodes,
            links,
            ty,
        )

    def math_node(ty):
        return assign_math_xyz(
            func,
            expr_outputs_xyz,
            nodes,
            links,
            ty,
        )

    def group_node(func, name, create_node_tree):
        return create_cached_func_group_node(nodes, func, name, create_node_tree)

    def assign_args(func, node, params):
        return assign_func_args(func, params, expr_outputs_xyz, links, node)

    def assign_arg(i, output):
        return assign_index(i, expr_outputs_xyz, links, output)

    match func.op:
        case sm4sh_model_py.database.OperationXyz.Unk:
            return None
        case sm4sh_model_py.database.OperationXyz.Add:
            return math_node("ADD")
        case sm4sh_model_py.database.OperationXyz.Sub:
            return math_node("SUBTRACT")
        case sm4sh_model_py.database.OperationXyz.Mul:
            return math_node("MULTIPLY")
        case sm4sh_model_py.database.OperationXyz.Div:
            return math_node("DIVIDE")
        case sm4sh_model_py.database.OperationXyz.Mix:
            return mix_rgba_node("MIX")
        case sm4sh_model_py.database.OperationXyz.Clamp:
            node = group_node(func, "ClampXyz", clamp_xyz_node_group)
            assign_args(func, node, ["Vector", "Min", "Max"])
            return node, "Vector"
        case sm4sh_model_py.database.OperationXyz.Min:
            return math_node("MINIMUM")
        case sm4sh_model_py.database.OperationXyz.Max:
            return math_node("MAXIMUM")
        case sm4sh_model_py.database.OperationXyz.Abs:
            return math_node("ABSOLUTE")
        case sm4sh_model_py.database.OperationXyz.Floor:
            return math_node("FLOOR")
        case sm4sh_model_py.database.OperationXyz.Power:
            return math_node("POWER")
        case sm4sh_model_py.database.OperationXyz.Sqrt:
            node = group_node(func, "SqrtXyz", sqrt_xyz_node_group)
            assign_args(func, node, ["Value"])
            return node, "Vector"
        case sm4sh_model_py.database.OperationXyz.InverseSqrt:
            node = group_node(func, "InverseSqrtXyz", inversesqrt_xyz_node_group)
            assign_args(func, node, ["Value"])
            return node, "Vector"
        case sm4sh_model_py.database.OperationXyz.Fma:
            return math_node("MULTIPLY_ADD")
        case sm4sh_model_py.database.OperationXyz.Dot:
            node = nodes.new("ShaderNodeVectorMath")
            node.name = func_xyz_name(func)
            node.operation = "DOT_PRODUCT"

            assign_arg(func.args[0], node.inputs[0])
            assign_arg(func.args[1], node.inputs[1])

            return node, "Value"
        case sm4sh_model_py.database.OperationXyz.Sin:
            return math_node("SINE")
        case sm4sh_model_py.database.OperationXyz.Cos:
            return math_node("COSINE")
        case sm4sh_model_py.database.OperationXyz.Exp2:
            node = nodes.new("ShaderNodeVectorMath")
            node.name = func_xyz_name(func)
            node.operation = "POWER"

            node.inputs[0].default_value = 2.0
            assign_arg(func.args[0], node.inputs[1])

            return node, "Vector"
        case sm4sh_model_py.database.OperationXyz.Log2:
            node = nodes.new("ShaderNodeVectorMath")
            node.name = func_xyz_name(func)
            node.operation = "LOGARITHM"

            assign_arg(func.args[0], node.inputs[0])
            node.inputs[1].default_value = 2.0

            return node, "Vector"
        case sm4sh_model_py.database.OperationXyz.Fract:
            return math_node("FRACT")
        # TODO: intbitstofloat, floatbitstoint

        case sm4sh_model_py.database.OperationXyz.Select:
            return mix_rgba_node("MIX")
        case sm4sh_model_py.database.OperationXyz.Negate:
            node = nodes.new("ShaderNodeVectorMath")
            node.name = func_xyz_name(func)
            node.operation = "MULTIPLY"

            assign_arg(func.args[0], node.inputs[0])
            node.inputs[1].default_value = (-1.0, -1.0, -1.0)

            return node, "Vector"
        case sm4sh_model_py.database.OperationXyz.Equal:
            return math_node("COMPARE")
        case sm4sh_model_py.database.OperationXyz.NotEqual:
            # TODO: Invert compare.
            return math_node("COMPARE")
        case sm4sh_model_py.database.OperationXyz.Less:
            node = group_node(func, "LessXyz", less_xyz_node_group)
            assign_args(func, node, ["Value", "Threshold"])
            return node, "Vector"
        case sm4sh_model_py.database.OperationXyz.Greater:
            node = group_node(func, "GreaterXyz", greater_xyz_node_group)
            assign_args(func, node, ["Value", "Threshold"])
            return node, "Vector"
        case sm4sh_model_py.database.OperationXyz.LessEqual:
            # TODO: node group for leq?
            node = group_node(func, "LessXyz", less_xyz_node_group)
            assign_args(func, node, ["Value", "Threshold"])
            return node, "Vector"
        case sm4sh_model_py.database.OperationXyz.GreaterEqual:
            # TODO: node group for geq?
            node = group_node(func, "GreaterXyz", greater_xyz_node_group)
            assign_args(func, node, ["Value", "Threshold"])
            return node, "Vector"
        case sm4sh_model_py.database.OperationXyz.NormalMap:
            node = group_node(func, "NormalMap", normal_map_node_group)
            assign_args(func, node, ["Vector"])
            return node, "Vector"
        case sm4sh_model_py.database.OperationXyz.Normalize:
            return math_node("NORMALIZE")
        case sm4sh_model_py.database.OperationXyz.LocalToWorldPoint:
            node = group_node(func, "LocalToWorldPoint", transform_point_node_group)
            assign_args(func, node, ["X", "Y", "Z"])
            return node, "Vector"
        case sm4sh_model_py.database.OperationXyz.LocalToWorldVector:
            node = group_node(func, "LocalToWorldVector", transform_vector_node_group)
            assign_args(func, node, ["X", "Y", "Z"])
            return node, "Vector"
        case sm4sh_model_py.database.OperationXyz.VarianceShadow:
            # Shadow mapping is already handled by Blender.
            node = nodes.new("ShaderNodeValue")
            node.outputs[0].default_value = 1.0
            return node, "Value"
        case sm4sh_model_py.database.OperationXyz.BlinnPhongSpecular:
            node = group_node(
                func, "BlinnPhongSpecularXyz", blinn_phong_spec_node_group_xyz
            )
            assign_args(
                func,
                node,
                ["normal", "lightDir", "eye", "Exponent"],
            )
            return node, "Value"
        case sm4sh_model_py.database.OperationXyz.AnisotropicSpecular:
            node = group_node(
                func, "AnisotropicSpecularXyz", anisotropic_spec_node_group_xyz
            )
            assign_args(
                func,
                node,
                [
                    "normal",
                    "tangent",
                    "eye",
                    "ParamX",
                    "ParamY",
                ],
            )
            return node, "Value"
        case sm4sh_model_py.database.OperationXyz.Fresnel:
            node = group_node(func, "FresnelXyz", fresnel_node_group_xyz)
            assign_args(
                func,
                node,
                ["normal", "eye", "Param"],
            )
            return node, "Value"
        case sm4sh_model_py.database.OperationXyz.TintColor:
            node = group_node(func, "TintColorXyz", tint_color_node_group_xyz)
            assign_args(
                func,
                node,
                ["Color", "Factor"],
            )
            return node, "Color"
        case sm4sh_model_py.database.OperationXyz.NegReflect:
            node = group_node(func, "NegReflectXYZ", neg_reflect_node_group_xyz)
            assign_args(
                func,
                node,
                ["A", "B"],
            )
            return node, "Vector"
        case _:
            # TODO: This case shouldn't happen?
            return None


def assign_xyz_channel(
    node,
    output_name,
    channel: Optional[sm4sh_model_py.database.ChannelXyz],
    nodes,
    links,
) -> Tuple[bpy.types.Node, str]:
    output = channel_xyz_name(channel)
    if output == "Vector":
        # Return the original output if all XYZ channels are used.
        return node, output_name
    elif output in node.outputs:
        return node, output
    else:
        # Avoid creating more than one separate XYZ for each node.
        xyz_name = f"{node.name}.xyz"
        xyz_node = nodes.get(xyz_name)
        if xyz_node is None:
            xyz_node = nodes.new("ShaderNodeSeparateXYZ")
            xyz_node.name = xyz_name
            links.new(node.outputs[output_name], xyz_node.inputs["Vector"])

        return xyz_node, output


def assign_value(
    shader: sm4sh_model_py.database.ShaderProgram,
    value: sm4sh_model_py.database.Value,
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str] | float]],
    nodes,
    links,
    textures,
) -> Optional[Tuple[bpy.types.Node, str] | float]:
    if i := value.int():
        node = nodes.new("ShaderNodeValue")
        node.outputs[0].default_value = i
        return node, "Value"
    elif f := value.float():
        # Don't create nodes or links for constants to improve loading times.
        return f
    elif parameter := value.parameter():
        return assign_parameter(shader, parameter, nodes, links)
    elif attribute := value.attribute():
        return assign_attribute(attribute, nodes, links)
    elif texture := value.texture():
        return assign_texture(texture, expr_outputs, nodes, links, textures)
    else:
        return None


def assign_float(output, f: float | tuple[float, float, float] | bpy.types.NodeSocket):
    try:
        output.default_value = [f] * len(output.default_value)
    except:
        try:
            output.default_value = f
        except:
            try:
                # Assign XYZ to color socket.
                output.default_value = (f[0], f[1], f[2], 1.0)
            except:
                # Assign XYZ to float socket.
                output.default_value = f[0]


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
        case "eye":
            node = nodes.get(name)
            if node is None:
                node = create_node_group(nodes, "EyeVector", eye_vector_node_group)
                node.name = name

            return node, channel_name(channel)
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


def channel_xyz_name(channel: Optional[sm4sh_model_py.database.ChannelXyz]) -> str:
    match channel:
        case sm4sh_model_py.database.ChannelXyz.Xyz:
            return "Vector"
        case sm4sh_model_py.database.ChannelXyz.X:
            return "X"
        case sm4sh_model_py.database.ChannelXyz.Y:
            return "Y"
        case sm4sh_model_py.database.ChannelXyz.Z:
            return "Z"
        case sm4sh_model_py.database.ChannelXyz.W:
            return "W"

    # TODO: How to handle the None case?
    return "Vector"


def channel_xyz_rgba_name(channel: Optional[sm4sh_model_py.database.ChannelXyz]) -> str:
    match channel:
        case sm4sh_model_py.database.ChannelXyz.Xyz:
            return "Color"
        case sm4sh_model_py.database.ChannelXyz.X:
            return "Red"
        case sm4sh_model_py.database.ChannelXyz.Y:
            return "Green"
        case sm4sh_model_py.database.ChannelXyz.Z:
            return "Blue"
        case sm4sh_model_py.database.ChannelXyz.W:
            return "Alpha"

    # TODO: How to handle the None case?
    return "Color"


def assign_mix_rgba(
    func: sm4sh_model_py.database.OutputExprFunc,
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str] | float]],
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


def assign_mix_xyz(
    func: sm4sh_model_py.database.OutputExprFuncXyz,
    expr_outputs_xyz: list[
        Optional[Tuple[bpy.types.Node, str] | tuple[float, float, float]]
    ],
    nodes,
    links,
    blend_type: str,
) -> Tuple[bpy.types.Node, str]:
    # TODO: Custom nodes with vector math to support negative values?
    node = nodes.new("ShaderNodeMix")
    node.data_type = "RGBA"
    node.blend_type = blend_type
    node.name = func_xyz_name(func)

    if blend_type == "OVERLAY":
        node.inputs["Factor"].default_value = 1.0

    assign_index(func.args[0], expr_outputs_xyz, links, node.inputs["A"])
    assign_index(func.args[1], expr_outputs_xyz, links, node.inputs["B"])
    if len(func.args) == 3:
        assign_index(func.args[2], expr_outputs_xyz, links, node.inputs["Factor"])

    return node, "Result"


def assign_texture(
    texture: sm4sh_model_py.database.Texture,
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str] | float]],
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
        if len(texcoords) == 2:
            uv_node = nodes.new("ShaderNodeCombineXYZ")
            uv_node.name = uv_name

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
        elif len(texcoords) == 3:
            uv_node = create_node_group(nodes, "CubeCoords", cube_coords_node_group)
            uv_node.name = uv_name

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
            assign_index(
                texcoords[2],
                expr_outputs,
                links,
                uv_node.inputs["Z"],
            )
        else:
            # TODO: warn if texcoords do not have 2 or 3 coords?
            uv_node = nodes.new("ShaderNodeCombineXYZ")
            uv_node.name = uv_name

    links.new(uv_node.outputs["Vector"], node.inputs["Vector"])


def assign_math(
    func: sm4sh_model_py.database.OutputExprFunc,
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str] | float]],
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
    name = str(func.op).removeprefix("Operation.")
    return func_name_inner(name, func.args, "")


def func_xyz_name(func: sm4sh_model_py.database.OutputExprFuncXyz):
    name = str(func.op).removeprefix("OperationXyz.")
    return func_name_inner(name, func.args, "xyz_")


def func_name_inner(op_name: str, args: list[int], prefix: str):
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
        ("TintColorX", "TintColor"),
        ("TintColorY", "TintColor"),
        ("TintColorZ", "TintColor"),
        ("NegReflectX", "NegReflect"),
        ("NegReflectY", "NegReflect"),
        ("NegReflectZ", "NegReflect"),
    ]
    for old, new in replacements:
        if op_name.startswith(old):
            op_name = op_name.replace(old, new)
            break

    func_args = ", ".join(str(a) for a in args)
    name = f"{prefix}{op_name}({func_args})"
    return name


def texture_assignment_name(texture):
    coords = ", ".join(str(c) for c in texture.texcoords)
    return f"{texture.name}({coords})"


def float32_bits(f: float) -> int:
    return struct.unpack("@I", struct.pack("@f", f))[0]


def assign_math_xyz(
    func,
    expr_outputs_xyz: list[
        Optional[Tuple[bpy.types.Node, str] | tuple[float, float, float]]
    ],
    nodes,
    links,
    op: str,
) -> Tuple[bpy.types.Node, str]:
    node = nodes.new("ShaderNodeVectorMath")
    node.operation = op
    node.name = func_xyz_name(func)

    for arg, input in zip(func.args, node.inputs):
        assign_index(
            arg,
            expr_outputs_xyz,
            links,
            input,
        )

    return node, "Vector"


def assign_value_xyz(
    shader: sm4sh_model_py.database.ShaderProgram,
    value: sm4sh_model_py.database.ValueXyz,
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str] | float]],
    nodes,
    links,
    textures,
) -> Optional[Tuple[bpy.types.Node, str] | tuple[float, float, float]]:
    if floats := value.float():
        # Don't create nodes or links for constants to improve loading times.
        return floats
    elif parameter := value.parameter():
        return assign_parameter_xyz(shader, parameter, nodes, links)
    elif attribute := value.attribute():
        return assign_attribute_xyz(attribute, nodes, links)
    elif texture := value.texture():
        return assign_texture_xyz(texture, expr_outputs, nodes, links, textures)
    else:
        return None


def assign_parameter_xyz(
    shader: sm4sh_model_py.database.ShaderProgram,
    parameter: sm4sh_model_py.database.ParameterXyz,
    nodes,
    links,
) -> Optional[Tuple[bpy.types.Node, str]]:
    if parameter.name == "MC" or parameter.name == "MC_EFFECT":
        name = f"NU_{parameter.field}"
        if node := nodes.get(name):
            # NU_ nodes use a custom node group for RGBA channel support.
            return node, channel_xyz_rgba_name(parameter.channel)
        # TODO: create node for missing NU_ parameters with default values?
    else:
        node = nodes.new("ShaderNodeCombineXYZ")
        node.label = parameter_label_xyz(parameter)

        # Assign the vector by treating the access as 3 scalar parameters.
        channels = [None, None, None]
        if parameter.channel is not None:
            match parameter.channel:
                case sm4sh_model_py.database.ChannelXyz.Xyz:
                    channels = ["x", "y", "z"]
                case sm4sh_model_py.database.ChannelXyz.X:
                    channels = ["x", "x", "x"]
                case sm4sh_model_py.database.ChannelXyz.Y:
                    channels = ["y", "y", "y"]
                case sm4sh_model_py.database.ChannelXyz.Z:
                    channels = ["z", "z", "z"]
                case sm4sh_model_py.database.ChannelXyz.W:
                    channels = ["w", "w", "w"]

        for channel, component in zip(channels, "XYZ"):
            f = shader.parameter_value(
                sm4sh_model_py.database.Parameter(
                    parameter.name, parameter.field, parameter.index, channel
                )
            )
            if f is not None:
                node.inputs[component].default_value = f

        return node, "Vector"


def parameter_label_xyz(p: sm4sh_model_py.database.ParameterXyz) -> str:
    name = f"{p.name}.{p.field}"

    if p.index is not None:
        name += f"[{p.index}]"

    if p.channel is not None:
        match p.channel:
            case sm4sh_model_py.database.ChannelXyz.Xyz:
                name += ".xyz"
            case sm4sh_model_py.database.ChannelXyz.X:
                name += ".xxx"
            case sm4sh_model_py.database.ChannelXyz.Y:
                name += ".yyy"
            case sm4sh_model_py.database.ChannelXyz.Z:
                name += ".zzz"
            case sm4sh_model_py.database.ChannelXyz.W:
                name += ".www"

    return name


def assign_attribute_xyz(
    attribute: sm4sh_model_py.database.AttributeXyz, nodes, links
) -> Optional[Tuple[bpy.types.Node, str]]:
    name = attribute.name
    channel = attribute.channel

    # Some attributes aren't exposed directly and require custom node groups.
    match name:
        case "a_Position":
            return assign_attribute_node_xyz(nodes, links, name, channel, "position")
        case "a_Normal":
            node = nodes.get(name)
            if node is None:
                node = create_node_group(
                    nodes, "GeometryNormal", geometry_normal_node_group
                )
                node.name = name

            return node, channel_xyz_name(channel)
        case "a_Tangent":
            node = nodes.get(name)
            if node is None:
                node = create_node_group(
                    nodes, "GeometryTangent", geometry_tangent_node_group
                )
                node.name = name

            return node, channel_xyz_name(channel)
        case "a_Binormal":
            node = nodes.get(name)
            if node is None:
                node = create_node_group(
                    nodes, "GeometryBitangent", geometry_bitangent_node_group
                )
                node.name = name

            return node, channel_xyz_name(channel)
        case "a_Color":
            return assign_attribute_node_xyz(nodes, links, name, channel, "Color")
        case "a_TexCoord0":
            return assign_attribute_node_xyz(nodes, links, name, channel, "UV0")
        case "a_TexCoord1":
            return assign_attribute_node_xyz(nodes, links, name, channel, "UV1")
        case "a_TexCoord2":
            return assign_attribute_node_xyz(nodes, links, name, channel, "UV2")
        case "eye":
            node = nodes.get(name)
            if node is None:
                node = create_node_group(nodes, "EyeVector", eye_vector_node_group)
                node.name = name

            return node, channel_xyz_name(channel)
        case "bitangent_sign":
            # TODO: Is there a way to calculate this?
            return None


def assign_attribute_node_xyz(
    nodes,
    links,
    name: str,
    channel: Optional[sm4sh_model_py.database.ChannelXyz],
    attribute_name: str,
) -> Tuple[bpy.types.Node, str]:
    node = nodes.get(name)
    if node is None:
        node = nodes.new("ShaderNodeAttribute")
        node.name = name
        node.attribute_name = attribute_name

    return assign_color_channel_xyz(name, channel, node, nodes, links)


def assign_texture_xyz(
    texture: sm4sh_model_py.database.TextureXyz,
    expr_outputs: list[Optional[Tuple[bpy.types.Node, str] | float]],
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
        return assign_color_channel_xyz(name, texture.channel, node, nodes, links)
    else:
        return None


def assign_color_channel_xyz(
    name: str,
    channel: Optional[sm4sh_model_py.database.ChannelXyz],
    node,
    nodes,
    links,
) -> Tuple[bpy.types.Node, str]:
    match channel:
        case sm4sh_model_py.database.ChannelXyz.Xyz:
            return node, "Color"
        case sm4sh_model_py.database.ChannelXyz.X:
            return assign_channel(name, "x", node, nodes, links)
        case sm4sh_model_py.database.ChannelXyz.Y:
            return assign_channel(name, "y", node, nodes, links)
        case sm4sh_model_py.database.ChannelXyz.Z:
            return assign_channel(name, "z", node, nodes, links)
        case sm4sh_model_py.database.ChannelXyz.W:
            return assign_channel(name, "w", node, nodes, links)
        case _:
            return node, "Color"


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
    assignment_outputs: list[Optional[Tuple[bpy.types.Node, str] | float]],
    links,
    node: bpy.types.Node,
):
    for i, param in zip(func.args, params):
        assign_index(i, assignment_outputs, links, node.inputs[param])


def used_assignments(
    shader: sm4sh_model_py.database.ShaderProgram,
) -> Tuple[Set[int], Set[int]]:
    visited = set()
    visited_xyz = set()

    exprs = shader.exprs
    exprs_xyz = shader.exprs_xyz

    if "out_attr0.xyz" in shader.output_dependencies_xyz:
        add_used_xyz_assignments(
            visited,
            visited_xyz,
            exprs,
            exprs_xyz,
            shader.output_dependencies_xyz["out_attr0.xyz"],
        )
    else:
        for c in "xyz":
            if f"out_attr0.{c}" in shader.output_dependencies:
                add_used_assignments(
                    visited, exprs, shader.output_dependencies[f"out_attr0.{c}"]
                )

    if "out_attr0.w" in shader.output_dependencies:
        add_used_assignments(visited, exprs, shader.output_dependencies["out_attr0.w"])

    return visited, visited_xyz


def add_used_assignments(
    visited: Set[int],
    assignments: list[sm4sh_model_py.database.OutputExpr],
    i: Optional[int],
):
    if i is not None:
        if i not in visited:
            visited.add(i)

            assignment = assignments[i]
            if func := assignment.func():
                # Skip shadow map rendering for now.
                if func.op != sm4sh_model_py.database.Operation.VarianceShadow:
                    for arg in func.args:
                        add_used_assignments(visited, assignments, arg)
            elif value := assignment.value():
                if texture := value.texture():
                    for coord in texture.texcoords:
                        add_used_assignments(visited, assignments, coord)


def add_used_xyz_assignments(
    visited: Set[int],
    visited_xyz: Set[int],
    exprs: list[sm4sh_model_py.database.OutputExpr],
    exprs_xyz: list[sm4sh_model_py.database.OutputExprXyz],
    i: int,
):
    if i not in visited_xyz:
        visited_xyz.add(i)

        assignment = exprs_xyz[i]
        if func := assignment.func():
            # Skip shadow map rendering for now.
            if func.op != sm4sh_model_py.database.OperationXyz.VarianceShadow:
                for arg in func.args:
                    add_used_xyz_assignments(
                        visited, visited_xyz, exprs, exprs_xyz, arg
                    )
        elif value := assignment.value():
            # Collect the scalar assignments for texture coordinates.
            if texture := value.texture():
                for coord in texture.texcoords:
                    add_used_assignments(visited, exprs, coord)
