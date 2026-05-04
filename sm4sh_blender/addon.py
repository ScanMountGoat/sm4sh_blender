import bpy

from . import export_image, export_material, export_nud, import_animation, import_nud


def menu_import_nud(self, context):
    text = "Smash 4 Model (.nud)"
    self.layout.operator(import_nud.ImportNud.bl_idname, text=text)


def menu_import_pac(self, context):
    text = "Smash 4 Animation (.pac)"
    self.layout.operator(import_animation.ImportPac.bl_idname, text=text)


def menu_export_nud(self, context):
    text = "Smash 4 Model (.nud)"
    self.layout.operator(export_nud.ExportNud.bl_idname, text=text)


class ImageProperties(bpy.types.PropertyGroup):
    image_format: bpy.props.EnumProperty(
        name="Image Format",
        items=[
            (
                "BC1Unorm",
                "BC1Unorm",
                "DXT1 compressed RGB + 1-bit alpha (fully opaque or fully transparent)",
            ),
            ("BC2Unorm", "BC2Unorm", "DXT3 compressed RGB + alpha"),
            ("BC3Unorm", "BC3Unorm", "DXT5 compressed RGB + alpha"),
            ("BC5Unorm", "BC5Unorm", "RGTC2 compressed RG"),
            ("Rgba8Unorm", "Rgba8Unorm", "Uncompressed RGB + alpha"),
            ("Rgba82", "Rgba82", "Uncompressed RGB + alpha"),
            (
                "Bgr5A1Unorm",
                "Bgr5A1Unorm",
                "Limited precision RGB + 1-bit alpha (fully opaque or fully transparent)",
            ),
            ("B5G6R5Unorm", "B5G6R5Unorm", "Limited precision RGB"),
            (
                "Bgr5A1Unorm2",
                "Bgr5A1Unorm2",
                "Limited precision RGB + 1-bit alpha (fully opaque or fully transparent)",
            ),
            (
                "Rgb5A1Unorm",
                "Rgb5A1Unorm",
                "Limited precision RGB + 1-bit alpha (fully opaque or fully transparent)",
            ),
            ("R32Float", "R32Float", "Uncompressed single channel floating-point"),
        ],
        description="The image format to encode for the exported model.nut",
        default="BC3Unorm",
    )

    image_dimension: bpy.props.EnumProperty(
        name="Texture Dimension",
        items=[
            ("2D", "2D", "2D image"),
            ("Cube", "Cube", "2D images stacked vertically for X+, X-, Y+, Y-, Z+, Z-"),
        ],
        description="The texture dimension for the exported model.nut",
        default="2D",
    )

    generate_mipmaps: bpy.props.BoolProperty(
        name="Generate Mipmaps",
        description="Generate mipmaps for the exported model.nut",
        default=True,
    )


class MetalMaterialProperties(bpy.types.PropertyGroup):
    diffuse: bpy.props.PointerProperty(
        name="Diffuse Image",
        description="The diffuse texture image for the exported metal.nud. This should use RGB (33, 33, 33) to match in game models and include the alpha channel for transparency if needed",
        type=bpy.types.Image,
    )

    stage_cube: bpy.props.EnumProperty(
        name="Stage Reflection Cube",
        items=[
            (
                "10101000",
                "Rough (10101000)",
                "The low resolution stage cube map for rough reflections",
            ),
            (
                "10102000",
                "Glossy (10102000)",
                "The high resolution stage cube map for glossy reflections",
            ),
        ],
        description="The reflection cube map texture hash for the exported metal.nud",
        default="10102000",
    )

    reflection_color: bpy.props.FloatVectorProperty(
        name="NU_reflectionColor",
        description="The value for NU_reflectionColor for the exported metal.nud",
        subtype="COLOR",  # TODO: should this be COLOR or COLOR_GAMMA?
        size=4,
        default=(3.0, 3.0, 3.0, 1.0),
    )

    fresnel_color: bpy.props.FloatVectorProperty(
        name="NU_fresnelColor",
        description="The value for NU_fresnelColor for the exported metal.nud",
        subtype="COLOR",  # TODO: should this be COLOR or COLOR_GAMMA?
        size=4,
        default=(0.6, 0.6, 0.6, 1.0),
    )

    fresnel_params: bpy.props.FloatVectorProperty(
        name="NU_fresnelParams",
        description="The value for NU_fresnelParams for the exported metal.nud",
        subtype="XYZ",
        size=4,
        default=(3.7, 0.0, 0.0, 1.0),
    )

    ao_min_gain: bpy.props.FloatVectorProperty(
        name="NU_aoMinGain",
        description="The value for NU_fresnelColor for the exported metal.nud",
        subtype="COLOR",  # TODO: should this be COLOR or COLOR_GAMMA?
        size=4,
        default=(0.6, 0.6, 0.6, 1.0),
    )


class MaterialProperties(bpy.types.PropertyGroup):
    metal: bpy.props.PointerProperty(type=MetalMaterialProperties)


# The order here matters.
classes = [
    import_nud.ImportNud,
    import_animation.ImportPac,
    export_nud.ExportNud,
    export_image.SM4SH_PT_image_export_panel,
    export_material.SM4SH_PT_material_export_panel,
    export_material.SM4SH_PT_metal_material_export_panel,
    ImageProperties,
    MetalMaterialProperties,
    MaterialProperties,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_import_nud)
    bpy.types.TOPBAR_MT_file_import.append(menu_import_pac)

    bpy.types.TOPBAR_MT_file_export.append(menu_export_nud)

    bpy.types.Image.sm4sh_blender = bpy.props.PointerProperty(type=ImageProperties)
    bpy.types.Material.sm4sh_blender = bpy.props.PointerProperty(
        type=MaterialProperties
    )


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    bpy.types.TOPBAR_MT_file_import.remove(menu_import_nud)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import_pac)

    bpy.types.TOPBAR_MT_file_export.remove(menu_export_nud)

    del bpy.types.Image.sm4sh_blender
    del bpy.types.Material.sm4sh_blender
