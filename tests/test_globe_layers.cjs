"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const geometry = require("../utils/globe_frontend/globe_geometry.js");

const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    style: {setProperty() {}}, addEventListener() {}, setAttribute() {},
    clientWidth: 600, clientHeight: 600, scrollHeight: 700, value: "front",
  });
  return elements.get(id);
}
let viewer;
const sent = [];
class Scatter {
  constructor(node, options) {
    viewer = this;
    this.options = options;
    this.styles = {point: {}, fog: {}};
    this.scatterPlot = {scene: {getObjectByName: () => null}, renderer: {setClearColor() {}}};
    this.pointVisualizer = {points: {geometry: {setDrawRange() {}}}};
  }
  render(dataset) { this.dataset = dataset; }
  updateDataset(dataset) { this.dataset = dataset; }
  select(indices) { this.selected = indices; }
  setPointColorer(fn) { this.colorer = fn; }
  setHoverPointIndex() {}
  setSequences() {}
  setPanMode() {}
  stopOrbitAnimation() {}
  resize() {}
}
Scatter.Dataset = class {
  constructor(points, metadata) { this.points = points; this.metadata = metadata; }
};
const context = vm.createContext({
  document: {getElementById: element, documentElement: element("root")},
  window: {matchMedia: () => ({addEventListener() {}}), addEventListener() {}, parent: {postMessage: msg => sent.push(msg)}},
  ScatterGL: Scatter, PeptideGlobeGeometry: geometry,
  requestAnimationFrame() {}, queueMicrotask: fn => fn(), console,
});
vm.runInContext(fs.readFileSync(require.resolve("../utils/globe_frontend/globe.js"), "utf8"), context);
const input = {
  revision: "same-projection", selection_revision: "initial", selected_indices: [], selected_only: true,
  point_size: 3, opacity: .7, show_globe: true, show_axes: false,
  points: [[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]],
  metadata: [
    {row_index: 0, id: "duplicate", is_amp: true, role: "Reference", visible: true, opacity: .2},
    {row_index: 1, id: "duplicate", is_amp: false, role: "New", visible: true, opacity: 1},
    {row_index: 2, id: "other", is_amp: null, role: "New", visible: true, opacity: 1},
  ],
};
function render() { context.input = input; vm.runInContext("render(input, null)", context); }
render();
const fullBounds = JSON.stringify(viewer.dataset.points.slice(3, 5));
assert.equal(viewer.colorer(0, new Set(), null), "rgba(0,0,255,0.2)");
assert.equal(viewer.colorer(1, new Set(), null), "rgba(255,0,0,1)");
input.metadata[0].visible = false;
render();
assert.equal(vm.runInContext("displayedMetadata.length", context), 2);
assert.equal(JSON.stringify(viewer.dataset.points.slice(2, 4)), fullBounds);
viewer.options.onSelect([0]);
assert.equal(sent.at(-1).value.selected_indices[0], 1); // Global row, not local index 0.
input.metadata.forEach(row => { row.visible = false; });
render();
assert.equal(vm.runInContext("displayedMetadata.length", context), 0);
assert.equal(JSON.stringify(viewer.dataset.points.slice(0, 2)), fullBounds);
input.metadata.forEach(row => { row.visible = true; });
input.selected_indices = [1];
render();
assert.equal(vm.runInContext("displayedMetadata.length", context), 1);
assert.equal(vm.runInContext("displayedMetadata[0].row_index", context), 1);
// Styling must update even if the displayed geometry is unchanged.
input.metadata[1].color = "#008800";
input.metadata[1].opacity = .5;
render();
assert.equal(viewer.colorer(0, new Set(), null), "rgba(0,136,0,0.5)");
console.log("Globe layer checks passed: stable bounds, hidden rows, alpha, color updates, and source selection mapping.");
