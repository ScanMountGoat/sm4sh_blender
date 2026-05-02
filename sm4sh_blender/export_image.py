import bpy


class SM4SH_PT_image_export_panel(bpy.types.Panel):
    bl_label = "sm4sh_blender"
    bl_idname = "SM4SH_PT_image_export_panel"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "sm4sh_blender"

    def draw(self, context):
        image = context.space_data.image
        if image is not None:
            layout = self.layout
            layout.use_property_split = True
            layout.prop(image.sm4sh_blender, "image_format", text="Format")
            layout.prop(image.sm4sh_blender, "image_dimension", text="Dimension")
            layout.prop(image.sm4sh_blender, "generate_mipmaps", text="Mipmaps")
