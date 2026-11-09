"use strict";

const assert = require("node:assert/strict");
const {cameraSettings, createGlobeLines} = require("../utils/globe_frontend/globe_geometry.js");

const lines = createGlobeLines();
assert.equal(lines.length, 13, "Six longitude circles and seven latitude circles");
for (const line of lines) {
  assert.equal(line.length, 97);
  assert.deepEqual(line[0], line.at(-1), "Reference circles close exactly");
  for (const point of line) {
    assert.equal(point.length, 3);
    assert.ok(point.every(Number.isFinite), "Every grid vertex is finite");
    assert.ok(Math.abs(Math.hypot(...point) - 1) < 1e-12, "Grid vertices lie on the unit sphere");
  }
}
assert.equal(lines.filter(line => line.every(point => point[1] === 0)).length, 1, "The equator is horizontal in Y-up coordinates");
for (const meridian of lines.slice(0, 6)) {
  assert.deepEqual(meridian[0], [0, 1, 0], "Meridians meet at the north pole");
  assert.deepEqual(meridian[48], [0, -1, 0], "Meridians meet at the south pole");
}

const expectedCamera = {position: [0, 0, 2.2], target: [0, 0, 0], zoom: 1};
assert.deepEqual(cameraSettings(), expectedCamera);
const changedCamera = cameraSettings();
changedCamera.position[0] = 10;
changedCamera.target[1] = 20;
assert.deepEqual(cameraSettings(), expectedCamera, "Callers cannot mutate the default camera");
assert.deepEqual(cameraSettings("side").position, [2.2, 0, 0]);
const topCamera = cameraSettings("top");
assert.equal(topCamera.position[0], 0);
assert.equal(topCamera.position[1], 2.2);
assert.ok(topCamera.position[2] > 0 && topCamera.position[2] < 0.0001, "Top view avoids the orbit-control pole singularity");
assert.deepEqual(topCamera.target, [0, 0, 0]);
assert.throws(() => cameraSettings("unknown"), RangeError);
lines[0][0][0] = 30;
assert.deepEqual(createGlobeLines()[0][0], [0, 1, 0], "Each call returns fresh geometry");

for (const invalid of [
  {segments: 7}, {segments: 8.5}, {segments: NaN},
  {meridians: 2}, {meridians: 3.5}, {meridians: Infinity},
  {latitudes: 0}, {latitudes: 1.5}, {latitudes: NaN},
]) {
  assert.throws(() => createGlobeLines(invalid), RangeError);
}

console.log("Globe geometry checks passed: closed unit circles, Y-up poles/equator, finite coordinates, and deterministic camera.");
