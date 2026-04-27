import bpy
from sm4sh_blender.utils import (
    extract_name,
    float32_from_bits,
    get_enum_value,
    parse_int,
)
from . import sm4sh_model_py
import numpy as np

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


def export_image(image: bpy.types.Image, hash: int):
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
        image,
        "sm4sh_image_format",
        sm4sh_model_py.NutFormat,
        sm4sh_model_py.NutFormat.BC3Unorm,
    )

    # sm4sh_lib swaps channels internally to support image_dds, so we need to change formats.
    if nut_format == sm4sh_model_py.NutFormat.Rgb5A1Unorm:
        nut_format = sm4sh_model_py.NutFormat.Bgr5A1Unorm

    layers = 1
    generate_mipmaps = image.sm4sh_generate_mipmaps
    if image.sm4sh_image_dimension == "Cube":
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
