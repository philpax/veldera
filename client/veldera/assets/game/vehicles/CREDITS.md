# Vehicle model credits

Car models are split from the "Generic passenger car pack" by
[Comrade1280](https://sketchfab.com/Comrade1280) on Sketchfab:
<https://sketchfab.com/3d-models/generic-passenger-car-pack-20f9af9b8a404d5cb022ac6fe87f21f5>

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The per-car glbs (`compact`, `coupe`, `hatchback`, `minivan`, `offroad`,
`pickup`, `sedan`, `sport`, `suv`, `wagon`) are generated from the source pack
by `tools/split_car_pack` (see `scripts/split_car_pack.sh`), which normalizes
each car's origin and scale and re-pivots the wheels for animation. The same
attribution is embedded in each glb's `asset.copyright`.
