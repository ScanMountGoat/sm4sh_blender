import bpy

from . import import_nud
from . import import_animation


def menu_import_nud(self, context):
    text = "Smash 4 Model (.nud)"
    self.layout.operator(import_nud.ImportNud.bl_idname, text=text)


def menu_import_pac(self, context):
    text = "Smash 4 Animation (.pac)"
    self.layout.operator(import_animation.ImportPac.bl_idname, text=text)


classes = [import_nud.ImportNud, import_animation.ImportPac]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_import_nud)
    bpy.types.TOPBAR_MT_file_import.append(menu_import_pac)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    bpy.types.TOPBAR_MT_file_import.remove(menu_import_nud)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import_pac)
