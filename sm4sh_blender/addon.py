import bpy

from . import import_nud


def menu_import_nud(self, context):
    text = "Smash 4 Model (.nud)"
    self.layout.operator(import_nud.ImportNud.bl_idname, text=text)


classes = [
    import_nud.ImportNud,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_import_nud)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    bpy.types.TOPBAR_MT_file_import.remove(menu_import_nud)
