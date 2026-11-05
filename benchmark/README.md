# ReCoNav benchmark v1.0.0

192 tasks across nine HM3DSem scenes, with scene ASGs, navigation geometry and evaluation annotations. Use this benchmark to evaluate any compatible navigation method.

- tasks/hm3dsem/<scene>/<task>.json: instruction/start, task category, gold symbolic requirements, target metadata and reference route.
- scenes/hm3dsem/<scene>/asg/: final entity.json, geometry.json and topology.json.
- scenes/hm3dsem/<scene>/navigation/graph.json.gz: lossless JSON/WKB serialization of the physical navigation geometry used for route solving and collision validation.
- splits/: all.json (192 tasks) and hard44.json (44 tasks).
- vocabularies/: object and region category mappings.
- manifest.json: version, scene/family counts and coordinate conventions.

Each task's uniqueness_proof.identity_class records whether target identity is resolved by a local qualifier, direct reference, hard constraint or preference.

Set `RECONAV_DATA_ROOT` to the directory containing `benchmark/`. Use `reconav benchmark` to inspect populations. Run only the selected method with `reconav run --config configs/qwen.json --methods Qwen-ReCoNav`; choose `--split all`, `--split hard44`, or `--task-ids <ID>`. Follow the [repository instructions](../README.md) for installation, model setup and baseline commands. Use the common evaluator for physical route/constraint scoring. Prediction inputs are instruction, start and scene information; gold target IDs, contracts and route witnesses are evaluator annotations.

Coordinate conventions: route points use Habitat x/y/z with y as foot height. Task robot_position uses x and y as the horizontal coordinates, with floor_height as vertical height. ASG polygons are horizontal x/z geometry. No tasks are used for training.

The package contains scene-derived physical free-space information, not the original textured HM3D meshes. Native simulated methods need the upstream meshes. NavGPT/MapGPT and the +ASG image renderer additionally need the original JPEG-textured `.glb` files alongside `.basis.glb`; obtain both the official validation `glb` and `habitat` components. Obtain these from https://aihabitat.org/datasets/hm3d-semantics/ and follow their access terms. Check DATA_TERMS.md for the applicable restrictions and attribution.

