// Global world constants. The island lives in [-WORLD_SIZE/2, WORLD_SIZE/2]² on the XZ plane, sea level = 0.

export const WORLD_SIZE = 16384; // meters -> 268 km² map, ~130 km² of land
export const HEIGHTMAP_RES = 4096; // 4 m / texel
export const TEXEL = WORLD_SIZE / HEIGHTMAP_RES;
export const SEA_LEVEL = 0;
export const SEED = 20261017;

export const LAKE = { x: - 2600, z: 2300, radius: 750 }; // level is computed by the generator

export const TREE_CELL = 4; // one potential tree per 4x4 m cell -> ~16.8M cells over the map
