from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import types
from pathlib import Path

import bpy


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sceneweaver-repo", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--device", type=str, default="CPU")
    return parser.parse_args(argv)


def _ensure_dummy_gin() -> None:
    if "gin" in sys.modules:
        return

    module = types.ModuleType("gin")

    def configurable(fn=None, **_kwargs):
        if fn is None:
            return lambda inner: inner
        return fn

    module.configurable = configurable
    sys.modules["gin"] = module


def _load_sceneweaver_export_module(sceneweaver_repo: Path):
    export_py = sceneweaver_repo / "infinigen" / "tools" / "export.py"
    if not export_py.exists():
        raise FileNotFoundError(f"Could not find SceneWeaver export script: {export_py}")

    _ensure_dummy_gin()
    spec = importlib.util.spec_from_file_location("sceneweaver_export_tool", export_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load SceneWeaver export script: {export_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _find_node_input(node, *names: str):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return socket
    return None


def _patch_blender_compat(module) -> None:
    def create_glass_shader_compat(node_tree, export_usd):
        nodes = node_tree.nodes
        if nodes.get("Glass BSDF"):
            color = nodes["Glass BSDF"].inputs[0].default_value
            roughness = nodes["Glass BSDF"].inputs[1].default_value
            ior = nodes["Glass BSDF"].inputs[2].default_value
        else:
            color = None
            roughness = None
            ior = None

        if nodes.get("Principled BSDF"):
            nodes.remove(nodes["Principled BSDF"])

        principled_bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")

        base_color_input = _find_node_input(principled_bsdf_node, "Base Color")
        roughness_input = _find_node_input(principled_bsdf_node, "Roughness")
        ior_input = _find_node_input(principled_bsdf_node, "IOR")
        transmission_input = _find_node_input(
            principled_bsdf_node,
            "Transmission",
            "Transmission Weight",
        )
        alpha_input = _find_node_input(principled_bsdf_node, "Alpha")

        if nodes.get("Glass BSDF"):
            if base_color_input is not None:
                base_color_input.default_value = color
            if roughness_input is not None:
                roughness_input.default_value = roughness
            if ior_input is not None:
                ior_input.default_value = ior
        elif roughness_input is not None:
            roughness_input.default_value = 0

        if transmission_input is not None:
            transmission_input.default_value = 1
        if export_usd and alpha_input is not None:
            alpha_input.default_value = 0
        node_tree.links.new(
            principled_bsdf_node.outputs[0],
            nodes["Material Output"].inputs[0],
        )

    def remove_params_compat(mat, node_tree):
        nodes = node_tree.nodes
        param_dict = {}
        if nodes.get("Material Output"):
            output = nodes["Material Output"]
        elif nodes.get("Group Output"):
            output = nodes["Group Output"]
        else:
            raise ValueError("Could not find material output node")

        if (
            nodes.get("Principled BSDF")
            and output.inputs[0].links
            and output.inputs[0].links[0].from_node.bl_idname == "ShaderNodeBsdfPrincipled"
        ):
            principled_bsdf_node = nodes["Principled BSDF"]
            metallic_input = _find_node_input(principled_bsdf_node, "Metallic")
            sheen_input = _find_node_input(principled_bsdf_node, "Sheen", "Sheen Weight")
            clearcoat_input = _find_node_input(principled_bsdf_node, "Clearcoat", "Coat Weight")
            param_dict[mat.name] = {
                "Metallic": metallic_input.default_value if metallic_input is not None else 0.0,
                "Sheen": sheen_input.default_value if sheen_input is not None else 0.0,
                "Clearcoat": clearcoat_input.default_value if clearcoat_input is not None else 0.0,
            }
            if metallic_input is not None:
                metallic_input.default_value = 0.0
            if sheen_input is not None:
                sheen_input.default_value = 0.0
            if clearcoat_input is not None:
                clearcoat_input.default_value = 0.0
            return param_dict

        for node in nodes:
            if node.type == "GROUP":
                param_dict = remove_params_compat(mat, node.node_tree)
                if param_dict:
                    return param_dict

        return param_dict

    def apply_baked_tex_compat(obj, param_dict=None):
        if param_dict is None:
            param_dict = {}

        bpy.context.view_layer.objects.active = obj
        bpy.context.object.data.uv_layers["ExportUV"].active_render = True
        for uv_layer in reversed(obj.data.uv_layers):
            if "ExportUV" not in uv_layer.name:
                obj.data.uv_layers.remove(uv_layer)

        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            mat.use_nodes = True
            nodes = mat.node_tree.nodes

            excluded_nodes = [f"{bake_type}_node" for bake_type in module.ALL_BAKE]
            excluded_nodes.extend(["Material Output", "Principled BSDF"])
            for node in list(nodes):
                if node.name not in excluded_nodes:
                    nodes.remove(node)

            output = nodes["Material Output"]
            if nodes.get("Principled BSDF") is None:
                principled_bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")
            elif (
                output.inputs[0].links
                and output.inputs[0].links[0].from_node.bl_idname == "ShaderNodeBsdfPrincipled"
            ):
                principled_bsdf_node = nodes["Principled BSDF"]
            else:
                nodes.remove(nodes["Principled BSDF"])
                principled_bsdf_node = nodes.new("ShaderNodeBsdfPrincipled")

            links = mat.node_tree.links
            links.new(output.inputs[0], principled_bsdf_node.outputs[0])
            for bake_type in module.ALL_BAKE:
                tex_node = nodes.get(f"{bake_type}_node")
                if tex_node is None:
                    continue
                if bake_type == "NORMAL":
                    normal_node = nodes.new("ShaderNodeNormalMap")
                    links.new(normal_node.inputs["Color"], tex_node.outputs[0])
                    target_input = _find_node_input(principled_bsdf_node, module.ALL_BAKE[bake_type])
                    if target_input is not None:
                        links.new(target_input, normal_node.outputs[0])
                    continue
                target_input = _find_node_input(principled_bsdf_node, module.ALL_BAKE[bake_type])
                if target_input is not None:
                    links.new(target_input, tex_node.outputs[0])

            values = param_dict.get(mat.name)
            if not values:
                continue
            metallic_input = _find_node_input(principled_bsdf_node, "Metallic")
            sheen_input = _find_node_input(principled_bsdf_node, "Sheen", "Sheen Weight")
            clearcoat_input = _find_node_input(principled_bsdf_node, "Clearcoat", "Coat Weight")
            if metallic_input is not None:
                metallic_input.default_value = values["Metallic"]
            if sheen_input is not None:
                sheen_input.default_value = values["Sheen"]
            if clearcoat_input is not None:
                clearcoat_input.default_value = values["Clearcoat"]

    module.remove_params = remove_params_compat
    module.apply_baked_tex = apply_baked_tex_compat
    module.create_glass_shader = create_glass_shader_compat


def _should_drop_sceneweaver_export_obj(name: str) -> bool:
    if ".bbox_placeholder(" in name or ".spawn_placeholder(" in name:
        return True
    if name.startswith("newroom_") and not name.endswith(".floor"):
        return True
    return False


def _prune_sceneweaver_export_objects() -> None:
    for obj in list(bpy.data.objects):
        if _should_drop_sceneweaver_export_obj(obj.name):
            data = getattr(obj, "data", None)
            bpy.data.objects.remove(obj, do_unlink=True)
            if getattr(data, "users", 0) == 0:
                if isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
                elif isinstance(data, bpy.types.Curve):
                    bpy.data.curves.remove(data)


def _configure_cycles(device: str, image_res: int) -> None:
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = device.upper()
    bpy.context.scene.cycles.samples = 1
    if hasattr(bpy.context.scene.cycles, "tile_x"):
        bpy.context.scene.cycles.tile_x = image_res
    if hasattr(bpy.context.scene.cycles, "tile_y"):
        bpy.context.scene.cycles.tile_y = image_res


def _export_sceneweaver_glb(module, output_path: Path, image_res: int, device: str) -> None:
    module.remove_obj_parents()
    module.delete_objects()
    _prune_sceneweaver_export_objects()
    module.triangulate_meshes()
    module.rename_all_meshes()

    collection_views, obj_views = module.update_visibility()

    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj not in list(bpy.context.view_layer.objects):
            continue
        module.realizeInstances(obj)
        module.apply_all_modifiers(obj)

    _configure_cycles(device, image_res)
    module.format = "glb"
    textures_dir = output_path.parent / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)
    module.bake_scene(
        folderPath=textures_dir,
        image_res=image_res,
        vertex_colors=False,
        export_usd=False,
    )

    for collection, status in collection_views.items():
        collection.hide_render = status

    for obj, status in obj_views.items():
        obj.hide_render = status

    module.clean_names()
    for obj in bpy.data.objects:
        obj.hide_viewport = obj.hide_render

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        use_visible=True,
        export_yup=True,
        export_cameras=False,
        export_lights=False,
    )
    shutil.rmtree(textures_dir, ignore_errors=True)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    output_path = args.output.resolve()
    sceneweaver_repo = args.sceneweaver_repo.resolve()

    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    module = _load_sceneweaver_export_module(sceneweaver_repo)
    _patch_blender_compat(module)
    _export_sceneweaver_glb(
        module,
        output_path=output_path,
        image_res=max(args.resolution, 256),
        device=args.device,
    )
    print(f"EXPORTED {output_path}")
    bpy.ops.wm.quit_blender()
    return 0


if __name__ == "__main__":
    separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    raise SystemExit(main(sys.argv[separator + 1 :]))
