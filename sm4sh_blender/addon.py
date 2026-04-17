import bpy

from . import import_nud
from . import import_animation
from . import export_nud
from . import export_image


def menu_import_nud(self, context):
    text = "Smash 4 Model (.nud)"
    self.layout.operator(import_nud.ImportNud.bl_idname, text=text)


def menu_import_pac(self, context):
    text = "Smash 4 Animation (.pac)"
    self.layout.operator(import_animation.ImportPac.bl_idname, text=text)


def menu_export_nud(self, context):
    text = "Smash 4 Model (.nud)"
    self.layout.operator(export_nud.ExportNud.bl_idname, text=text)


classes = [
    import_nud.ImportNud,
    import_animation.ImportPac,
    export_nud.ExportNud,
    export_image.SM4SH_PT_image_export_panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_import_nud)
    bpy.types.TOPBAR_MT_file_import.append(menu_import_pac)

    bpy.types.TOPBAR_MT_file_export.append(menu_export_nud)

    bpy.types.Image.sm4sh_image_format = bpy.props.EnumProperty(
        name="Image Format",
        items=[
            ("BC1Unorm", "BC1Unorm", ""),
            ("BC2Unorm", "BC2Unorm", ""),
            ("BC3Unorm", "BC3Unorm", ""),
            ("BC5Unor", "BC5Unor", ""),
            ("Rgba8Unorm", "Rgba8Unorm", ""),
            ("Rgba82", "Rgba82", ""),
            ("Bgr5A1Unorm", "Bgr5A1Unorm", ""),
            ("Bgr5A1Unorm2", "Bgr5A1Unorm2", ""),
            ("B5G6R5Unorm", "B5G6R5Unorm", ""),
            ("Rgb5A1Unorm", "Rgb5A1Unorm", ""),
            ("R32Float", "R32Float", ""),
        ],
        description="The image format to encode for the exported model.nut",
        default="BC3Unorm",
    )
    bpy.types.Image.sm4sh_image_dimension = bpy.props.EnumProperty(
        name="Texture Dimension",
        items=[
            ("2D", "2D", ""),
            ("Cube", "Cube", ""),
        ],
        description="The texture dimension for the exported model.nut",
        default="2D",
    )


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    bpy.types.TOPBAR_MT_file_import.remove(menu_import_nud)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import_pac)

    bpy.types.TOPBAR_MT_file_export.remove(menu_export_nud)

    del bpy.types.Image.sm4sh_image_format
    del bpy.types.Image.sm4sh_image_dimension
