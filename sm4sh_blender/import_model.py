import logging
import bpy
import numpy as np
import math

from . import sm4sh_model_py

from mathutils import Matrix


def init_logging():
    # Log any errors from Rust.
    log_fmt = "%(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s"
    logging.basicConfig(format=log_fmt, level=logging.ERROR)


def import_nud_model(
    operator,
    model: sm4sh_model_py.nud.NudModel,
):
    for group in model.groups:
        for i, mesh in enumerate(group.meshes):
            import_mesh(operator, bpy.context.collection, group, mesh, i)


def import_mesh(
    operator,
    collection: bpy.types.Collection,
    group: sm4sh_model_py.nud.NudMeshGroup,
    mesh: sm4sh_model_py.nud.NudMesh,
    i: int,
):
    name = group.name
    blender_mesh = bpy.data.meshes.new(f"{name}[{i}]")

    # TODO: Convert strips to tris
    indices = mesh.vertex_indices.astype(np.uint32)
    loop_start = np.arange(0, indices.shape[0], 3, dtype=np.uint32)
    loop_total = np.full(loop_start.shape[0], 3, dtype=np.uint32)

    blender_mesh.loops.add(indices.shape[0])
    blender_mesh.loops.foreach_set("vertex_index", indices)

    blender_mesh.polygons.add(loop_start.shape[0])
    blender_mesh.polygons.foreach_set("loop_start", loop_start)
    blender_mesh.polygons.foreach_set("loop_total", loop_total)

    # Set vertex attributes.
    positions = mesh.vertices.positions
    blender_mesh.vertices.add(positions.shape[0])
    blender_mesh.vertices.foreach_set("co", positions.reshape(-1))

    blender_mesh.update()

    # The validate call may modify and reindex geometry.
    # TODO: Assign normals now that the mesh has been updated.
    # TODO: Set remaining attributes

    blender_mesh.validate()

    # Convert from Y up to Z up.
    y_up_to_z_up = Matrix.Rotation(math.radians(90), 4, "X")
    blender_mesh.transform(y_up_to_z_up)

    obj = bpy.data.objects.new(blender_mesh.name, blender_mesh)
    collection.objects.link(obj)
