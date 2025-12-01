import bpy
import time
import os

from .import_model import (
    ImportException,
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

    experimental_shader_nodes: BoolProperty(
        name="Experimental Shader Nodes",
        description="Recreate shader nodes from in game compiled shaders (experimental)",
        default=False,
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

        database_path = os.path.join(os.path.dirname(__file__), "shaders.bin")
        database = sm4sh_model_py.database.ShaderDatabase.from_file(database_path)

        end = time.time()
        print(f"Load Model: {end - start}")

        start = time.time()

        try:
            armature = import_nud_model(
                self, context, model, database, self.experimental_shader_nodes
            )
            if armature is not None:
                # Store the path to make exporting easier later.
                armature["original_nud"] = path
        except ImportException as e:
            self.report({"ERROR"}, str(e))

        end = time.time()
        print(f"Import Blender Scene: {end - start}")
        return {"FINISHED"}
