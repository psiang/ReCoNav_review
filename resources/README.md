# External baseline resources

| Bundle | Used by | Setup |
|---|---|---|
| [Habitat navigation](habitat-navigation-v1.0.0.tar.gz) | Simulator and image-rendering baselines | Extract the archive and set `RECONAV_HABITAT_NAVIGATION_ROOT` |
| [Lang2LTL-2 embeddings](lang2ltl2-embeddings-v1.0.0.zip) | Lang2LTL-2 | Set `RECONAV_LANG2LTL_EMBEDDINGS` to the ZIP file |

Follow the [baseline guide](../docs/baselines.md) for commands, environments and checkpoints. Original HM3D scene meshes are obtained separately through the official provider.

These resources retain their own data terms: [Habitat navigation](../reconav/baselines/resources/habitat_navigation_DATA_TERMS.md) and [Lang2LTL-2 embeddings](../reconav/baselines/resources/lang2ltl2_embeddings_DATA_TERMS.md).
