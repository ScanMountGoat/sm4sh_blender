from typing import Optional

import bpy
import numpy as np

from sm4sh_blender.utils import (
    extract_name,
    float32_from_bits,
    get_enum_value,
    parse_int,
)

from . import sm4sh_model_py


class SM4SH_PT_material_export_panel(bpy.types.Panel):
    bl_label = "sm4sh_blender"
    bl_idname = "SM4SH_PT_material_export_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"
    bl_category = "sm4sh_blender"

    @classmethod
    def poll(cls, context):
        if not context.object:
            return False
        if context.object.type != "MESH":
            return False
        if context.object.active_material is None:
            return False

        return True

    def draw(self, context):
        material = context.object.active_material

        layout = self.layout
        layout.use_property_split = True
        layout.prop(material.sm4sh_blender, "metal_diffuse", text="Metal Diffuse Image")
        layout.prop(
            material.sm4sh_blender,
            "metal_reflection_color",
            text="Metal Reflection Color",
        )


global_texture_hashes = {
    0x10000001,
    0x10000007,
    0x10040000,
    0x10040001,
    0x10080000,
    0x10100000,
    0x10101000,
    0x10102000,
    0x10104100,
    0x10104FFF,
}


def export_material(
    material: bpy.types.Material,
    image_args: dict[int, sm4sh_model_py.EncodeSurfaceRgba32FloatArgs],
) -> sm4sh_model_py.NudMaterial:
    # TODO: validate material textures and properties against the shader.
    # TODO: warn or error for extra properties and missing properties?

    # TODO: Better error handling
    flags = 0x94010161
    flags_name = extract_name(material.name, ".")
    if value := parse_int(flags_name, 16):
        flags = value

    src_factor = get_enum_value(
        material, "src_factor", sm4sh_model_py.SrcFactor, sm4sh_model_py.SrcFactor.One
    )

    dst_factor = get_enum_value(
        material, "dst_factor", sm4sh_model_py.DstFactor, sm4sh_model_py.DstFactor.Zero
    )

    alpha_func = get_enum_value(
        material,
        "alpha_func",
        sm4sh_model_py.AlphaFunc,
        sm4sh_model_py.AlphaFunc.Disabled,
    )

    cull_mode = get_enum_value(
        material, "cull_mode", sm4sh_model_py.CullMode, sm4sh_model_py.CullMode.Inside
    )

    properties = []
    texture_indices_textures = []

    # TODO: Does property order matter?
    for node in material.node_tree.nodes:
        if node.label.startswith("NU_"):
            # TODO: Find a more reliably way to support ShaderNodeRGB and RGBA node group.
            if "Color" in node.inputs:
                rgb = node.inputs["Color"].default_value
                alpha = node.inputs["Alpha"].default_value
                values = [rgb[0], rgb[1], rgb[2], alpha]
            else:
                try:
                    rgba = node.outputs[0].default_value
                    values = [rgba[0], rgba[1], rgba[2], rgba[3]]
                except:
                    values = [0, 0, 0, 0]

            properties.append(sm4sh_model_py.NudProperty(node.label, values))
        elif node.bl_idname == "ShaderNodeTexImage":
            texture_index = parse_int(node.label)
            if texture_index is not None and node.image is not None:
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

                    if hash not in image_args and hash not in global_texture_hashes:
                        image_args[hash] = export_image(node.image, hash)
                else:
                    # TODO: report error if the name is not valid
                    # TODO: use a default name?
                    pass

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

    alpha_ref = 0
    return sm4sh_model_py.NudMaterial(
        flags,
        src_factor,
        dst_factor,
        alpha_func,
        alpha_ref,
        cull_mode,
        textures,
        properties,
    )


def export_image(
    image: bpy.types.Image, hash: int
) -> sm4sh_model_py.EncodeSurfaceRgba32FloatArgs:
    width, height = image.size
    image_data = np.zeros(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(image_data)

    # Flip vertically to match in game.
    image_data = np.flip(
        image_data.reshape((height, width, 4)),
        axis=0,
    )

    # TODO: set mipmaps
    nut_format = get_enum_value(
        image.sm4sh_blender,
        "image_format",
        sm4sh_model_py.NutFormat,
        sm4sh_model_py.NutFormat.BC3Unorm,
    )

    # sm4sh_lib swaps channels internally to support image_dds, so we need to change formats.
    if nut_format == sm4sh_model_py.NutFormat.Rgb5A1Unorm:
        nut_format = sm4sh_model_py.NutFormat.Bgr5A1Unorm

    layers = 1
    generate_mipmaps = image.sm4sh_blender.generate_mipmaps
    if image.sm4sh_blender.image_dimension == "Cube":
        # TODO: Fix mipmaps for cube maps.
        layers = 6
        generate_mipmaps = False

    # Depth and array layers are stacked vertically when converting to 2D.
    # TODO: Error if cube dimensions are not as expected?
    return sm4sh_model_py.EncodeSurfaceRgba32FloatArgs(
        hash,
        width,
        height // layers,
        layers,
        nut_format,
        generate_mipmaps,
        image_data.reshape(-1),
    )


def default_material() -> sm4sh_model_py.NudMaterial:
    return sm4sh_model_py.NudMaterial(
        0x94010161,
        sm4sh_model_py.SrcFactor.One,
        sm4sh_model_py.DstFactor.Zero,
        sm4sh_model_py.AlphaFunc.Disabled,
        0,
        sm4sh_model_py.CullMode.Inside,
        [
            default_texture(0x10080000),
            default_texture(0x10080000),
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


def metal_material(
    material: sm4sh_model_py.NudMaterial,
    blender_material: Optional[bpy.types.Material],
    database: sm4sh_model_py.database.ShaderDatabase,
    image_args: dict[int, sm4sh_model_py.EncodeSurfaceRgba32FloatArgs],
) -> sm4sh_model_py.NudMaterial:
    shader = database.get_shader(material.shader_id)

    hash = 0.0
    for p in material.properties:
        if p.name == "NU_materialHash":
            hash = p.values[0]

    reflection_color = [3, 3, 3, 1]
    if blender_material is not None:
        reflection_color = blender_material.sm4sh_blender.metal_reflection_color

    diffuse_hash = 0x10104FFF
    if blender_material is not None:
        image = blender_material.sm4sh_blender.metal_diffuse
        if image is not None:
            if hash := parse_int(image.name, 16):
                diffuse_hash = hash
                if hash not in image_args and hash not in global_texture_hashes:
                    image_args[hash] = export_image(image, hash)

    properties = [
        sm4sh_model_py.NudProperty("NU_colorSamplerUV", [1, 1, 0, 0]),
        sm4sh_model_py.NudProperty("NU_fresnelColor", [1, 1, 1, 1]),
        sm4sh_model_py.NudProperty("NU_blinkColor", [0, 0, 0, 0]),
        sm4sh_model_py.NudProperty("NU_reflectionColor", reflection_color),
        sm4sh_model_py.NudProperty("NU_aoMinGain", [0.3, 0.3, 0.3, 1]),
        sm4sh_model_py.NudProperty("NU_lightMapColorOffset", [0, 0, 0, 0]),
        sm4sh_model_py.NudProperty("NU_fresnelParams", [3.7, 0, 0, 0]),
        sm4sh_model_py.NudProperty("NU_alphaBlendParams", [0, 0, 0, 0]),
        sm4sh_model_py.NudProperty("NU_materialHash", [hash, 0, 0, 0]),
    ]

    # Texture usage is determined by the compiled shaders.
    # Try and preserve the normal maps if present.
    normal_texture = None
    if shader is not None:
        for texture, sampler in zip(material.textures, shader.samplers):
            if sampler == "normalSampler":
                # Preserve all normal map settings for wrap modes, filtering, etc.
                normal_texture = texture
                break

    # TODO: use 0x10080000 for diffuse and find material with diffuse color parameter?
    if normal_texture is not None:
        return sm4sh_model_py.NudMaterial(
            0x9601106B,
            material.src_factor,
            material.dst_factor,
            material.alpha_func,
            material.alpha_test_ref,
            material.cull_mode,
            [
                default_texture(diffuse_hash),
                default_texture(0x10102000),
                normal_texture,
                default_texture(0x10080000),
            ],
            properties,
        )
    else:
        return sm4sh_model_py.NudMaterial(
            0x96011069,
            material.src_factor,
            material.dst_factor,
            material.alpha_func,
            material.alpha_test_ref,
            material.cull_mode,
            [
                default_texture(diffuse_hash),
                default_texture(0x10102000),
                default_texture(0x10080000),
            ],
            properties,
        )


def default_texture(hash: int) -> sm4sh_model_py.NudTexture:
    # TODO: investigate why texture mip settings can cause crashes.
    return sm4sh_model_py.NudTexture(
        hash,
        sm4sh_model_py.MapMode.TexCoord,
        sm4sh_model_py.WrapMode.ClampToEdge,
        sm4sh_model_py.WrapMode.ClampToEdge,
        sm4sh_model_py.MinFilter.Linear,
        sm4sh_model_py.MagFilter.Linear,
        sm4sh_model_py.MipDetail.OneMipLevelAnisotropicOff2,
    )
