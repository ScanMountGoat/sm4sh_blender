import bpy
import time

from .import_model import (
    import_nud_model,
    init_logging,
)

from . import sm4sh_model_py

from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, CollectionProperty


class ImportNud(bpy.types.Operator, ImportHelper):
    """Import a Sm4sh 4 model"""

    bl_idname = "import_scene.nud"
    bl_label = "Import Nud"

    filename_ext = ".nud"

    filter_glob: StringProperty(
        default="*.nud",
        options={"HIDDEN"},
        maxlen=255,
    )

    def execute(self, context: bpy.types.Context):
        init_logging()

        self.import_nud(context, self.filepath)

        return {"FINISHED"}

    def import_nud(
        self,
        context: bpy.types.Context,
        path: str,
    ):
        start = time.time()

        model = sm4sh_model_py.load_model(path)

        end = time.time()
        print(f"Load Model: {end - start}")

        start = time.time()

        armature = import_nud_model(self, context, model)
        if armature is not None:
            # Store the path to make exporting easier later.
            armature["original_nud"] = path

        end = time.time()
        print(f"Import Blender Scene: {end - start}")
