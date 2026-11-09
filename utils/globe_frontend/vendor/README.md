The browser loads these files locally; Node.js and a CDN are not needed at runtime.

| File | npm package | License |
| --- | --- | --- |
| `scatter-gl.min.js` | `scatter-gl@0.0.13`, `package/lib/scatter-gl.min.js` | Apache-2.0 (`SCATTER_GL_LICENSE`) |
| `three.min.js` | `three@0.125.2`, `package/build/three.min.js` | MIT (`THREE_LICENSE`) |

The ScatterGL UMD bundle expects a `THREE` browser global, so Three.js is loaded first. These are the published npm files without modification. ScatterGL's bundle also contains its OrbitControls dependency and other bundled dependencies; upstream license notices are retained in the bundle.

Upstream: https://github.com/PAIR-code/scatter-gl and https://github.com/mrdoob/three.js.

To refresh the assets, download these exact npm packages, extract only the paths above and each package's `LICENSE`, and test the compatibility adapter in `../globe.js`. It is specific to ScatterGL 0.0.13: shared point styles support the existing size slider, sprite geometry draw ranges keep helper vertices out of rendering and picking, and polyline callbacks hide reference lines during the point ID picking pass. The reset adapter restores our level camera because native `resetZoom` uses upstream camera defaults. Native text labels for helpers are empty. Frame vertices are also removed from selection callbacks before source row indices are sent to Python.

The same adapter styles/toggles native axes without recreating the camera, and applies the Front/Side/Top camera presets through the inner plot's camera setter. ScatterGL removes its native axes from the picking pass. Axis captions and pointer tooltips are DOM overlays with pointer events disabled.

`../globe_geometry.js` defines pure, deterministic Y-up reference circles and camera presets. It does not modify peptide coordinates. Rotation starts only when the user presses **Spin globe**.
