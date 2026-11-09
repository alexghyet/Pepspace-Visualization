"use strict";

// ScatterGL and Three.js are pinned, local browser bundles. See vendor/README.md.
const graph = document.getElementById("globe");
const errorBox = document.getElementById("error");
const details = document.getElementById("details");
const countLabel = document.getElementById("selection-count");
const rotateButton = document.getElementById("rotate");
const selectButton = document.getElementById("select");
const spinButton = document.getElementById("spin");
const clearButton = document.getElementById("clear");
const displayControl = document.getElementById("display");
const viewControl = document.getElementById("view");
const axesOverlay = document.getElementById("axes-overlay");
const axisLabels = ["x", "y", "z"].map(axis => document.getElementById(`axis-${axis}`));
const hoverLabel = document.getElementById("hover-label");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

let scatter = null;
let payload = null;
let displayedMetadata = [];
let localBySource = new Map();
let frameSequences = [];
let geometryKey = null;
let globeVisible = null;
let selecting = false;
let renderMode = "points";
let eventCounter = 0;
let lastFrameHeight = 0;
let background = "#ffffff";
let foreground = "#31333f";
let hoveredIndex = null;
let pointerPosition = {x: 0, y: 0};
let overlayRevision = "";
const adaptedPolylineVisualizers = new WeakSet();
const eventSession = `${Date.now()}-${Math.random().toString(36).slice(2)}`;

function send(type, data) {
  window.parent.postMessage({isStreamlitMessage: true, type, ...data}, "*");
}

function resizeFrame() {
  if (scatter) scatter.resize();
  const height = Math.ceil(document.documentElement.scrollHeight);
  if (height !== lastFrameHeight) {
    lastFrameHeight = height;
    send("streamlit:setFrameHeight", {height});
  }
}

function setSpin(enabled) {
  if (!scatter) return;
  if (enabled) scatter.startOrbitAnimation();
  else scatter.stopOrbitAnimation();
  spinButton.textContent = enabled ? "Pause spin" : "Spin globe";
  spinButton.setAttribute("aria-pressed", String(enabled));
}

function setInteractionMode(select) {
  selecting = select;
  if (scatter) {
    if (select) {
      setSpin(false);
      scatter.setSelectMode();
    } else scatter.setPanMode();
  }
  rotateButton.setAttribute("aria-pressed", String(!select));
  selectButton.setAttribute("aria-pressed", String(select));
  document.getElementById("instructions").textContent = select
    ? "Drag a rectangle to select peptides · Switch to Rotate / Pan to move the globe"
    : "Drag to rotate · Right-drag to pan · Scroll to zoom · Click a peptide or select a region";
}

function updateSelectionCount(sourceIndices) {
  const count = new Set(sourceIndices).size;
  countLabel.textContent = `${displayedMetadata.length.toLocaleString()} shown · ${count.toLocaleString()} selected`;
  clearButton.disabled = count === 0;
}

function showHover(index) {
  const peptide = Number.isInteger(index) ? displayedMetadata[index] : null;
  hoveredIndex = peptide ? index : null;
  const classLabel = peptide && peptide.is_amp === true ? "AMP (blue)"
    : peptide && peptide.is_amp === false ? "Non-AMP (red)" : "Unlabeled (gray)";
  details.textContent = peptide
    ? `ID: ${peptide.id} · ${classLabel} · Sequence: ${peptide.sequence}`
    : "Hover over a peptide to see its ID, sequence, and class.";
  hoverLabel.hidden = !peptide || payload?.show_hover_labels === false;
  if (peptide && !hoverLabel.hidden) {
    hoverLabel.textContent = `${peptide.id} · ${classLabel}`;
    positionHoverLabel();
  }
}

function positionHoverLabel() {
  if (hoverLabel.hidden) return;
  const x = Math.min(Math.max(4, graph.clientWidth - hoverLabel.offsetWidth - 4), pointerPosition.x + 14);
  const y = Math.min(Math.max(4, graph.clientHeight - hoverLabel.offsetHeight - 4), pointerPosition.y + 16);
  hoverLabel.style.left = `${Math.max(4, x)}px`;
  hoverLabel.style.top = `${Math.max(4, y)}px`;
}

function updateOverlays() {
  requestAnimationFrame(updateOverlays);
  if (!scatter || !payload || document.hidden) return;
  const camera = scatter.scatterPlot.camera;
  camera.updateMatrixWorld();
  const signature = [graph.clientWidth, graph.clientHeight, payload.show_axes,
    ...camera.matrixWorld.elements, ...camera.projectionMatrix.elements,
    ...(payload.axis_labels || [])].join(",");
  if (signature === overlayRevision) return;
  overlayRevision = signature;
  axesOverlay.hidden = payload.show_axes === false;
  if (!axesOverlay.hidden) {
    [[1.04, 0, 0], [0, 1.04, 0], [0, 0, 1.04]].forEach((point, index) => {
      const projected = new THREE.Vector3(...point).project(camera);
      const label = axisLabels[index];
      label.textContent = `${"XYZ"[index]} · ${(payload.axis_labels || ["X", "Y", "Z"])[index]}`;
      label.hidden = projected.z < -1 || projected.z > 1;
      label.style.left = `${Math.max(4, Math.min(graph.clientWidth - label.offsetWidth - 4, (projected.x + 1) * graph.clientWidth / 2 + 8))}px`;
      label.style.top = `${Math.max(4, Math.min(graph.clientHeight - label.offsetHeight - 4, (1 - projected.y) * graph.clientHeight / 2 + 8))}px`;
    });
  }
  if (viewControl.value !== "custom") {
    const preset = PeptideGlobeGeometry.cameraSettings(viewControl.value);
    const position = scatter.scatterPlot.getCameraPosition();
    const target = scatter.scatterPlot.getCameraTarget();
    if (position.some((value, i) => Math.abs(value - preset.position[i]) > 0.000001)
      || target.some(value => Math.abs(value) > 0.000001)) viewControl.value = "custom";
  }
}

function nativeLabel(text) {
  // The pinned native text renderer has a 256-glyph atlas. Keep original Unicode
  // IDs in hover/export metadata, but substitute unsupported canvas glyphs.
  return String(text).replace(/[\u0000-\u001f\u007f-\u009f]/g, " ").replace(/[^\u0000-\u00ff]/gu, "?");
}

function colorPoint(index, selected, hovered) {
  const peptide = displayedMetadata[index];
  // Helpers supply the stable globe frame and bounds, never class colors.
  if (!peptide) return "rgba(150,150,150,0)";
  const alpha = Math.max(0, Math.min(1, Number(payload.opacity) || 0));
  const focused = selected.has(index) || hovered === index;
  const opacity = selected.size && !focused ? alpha * 0.16 : alpha;
  // Keep the same class color during hover, selection, and text rendering.
  // ScatterGL's text renderer ignores alpha, so blend text against the theme.
  const rgb = peptide.is_amp === true ? [0, 0, 255]
    : peptide.is_amp === false ? [255, 0, 0] : [148, 163, 184];
  if (renderMode === "labels") {
    const base = new THREE.Color(background);
    const blend = [base.r, base.g, base.b].map((v, i) => Math.round(255 * v * (1 - opacity) + rgb[i] * opacity));
    return `rgb(${blend.join(",")})`;
  }
  return `rgba(${rgb.join(",")},${opacity})`;
}

function localSelection(sourceIndices) {
  return [...new Set(sourceIndices)].map(index => localBySource.get(index)).filter(index => index !== undefined);
}

function emitSelection(localIndices) {
  if (!payload) return;
  const selected = [...new Set(localIndices)].filter(index => Number.isInteger(index) && index >= 0 && index < displayedMetadata.length);
  const sourceIndices = selected.map(index => displayedMetadata[index].row_index);
  const revision = payload.selection_revision;
  payload = {...payload, selected_indices: sourceIndices};
  updateSelectionCount(sourceIndices);
  // Native rectangle selection also considers frame vertices. Filter those out
  // after its internal onSelect finishes; hydration never emits an event.
  queueMicrotask(() => {
    scatter.select(selected);
    send("streamlit:setComponentValue", {
      value: {revision, selected_indices: sourceIndices, event_id: `${eventSession}-${++eventCounter}`},
      dataType: "json",
    });
  });
}

function makeDataset(data) {
  const selected = new Set(data.selected_indices || []);
  const points = [];
  displayedMetadata = [];
  localBySource = new Map();
  data.points.forEach((point, index) => {
    const metadata = data.metadata[index];
    if (!metadata || (data.selected_only && selected.size > 0 && !selected.has(metadata.row_index))) return;
    localBySource.set(metadata.row_index, points.length);
    displayedMetadata.push(metadata);
    points.push(point);
  });
  const metadata = displayedMetadata.map(peptide => ({label: nativeLabel(peptide.label || peptide.id)}));
  const appendHelper = point => {
    const index = points.length;
    points.push(point);
    metadata.push({label: ""});
    return index;
  };

  // ScatterGL normalizes each dataset. Include full-data bounds even when only
  // a small selected region is visible, so filtering cannot move the peptides.
  const mins = [-1, -1, -1];
  const maxs = [1, 1, 1];
  data.points.forEach(point => point.forEach((value, axis) => {
    mins[axis] = Math.min(mins[axis], value);
    maxs[axis] = Math.max(maxs[axis], value);
  }));
  appendHelper(mins);
  appendHelper(maxs);
  // A Y-up reference grid matches the native camera without rotating peptide
  // coordinates. Closed circles avoid seams and stay identical for every dataset.
  frameSequences = PeptideGlobeGeometry.createGlobeLines().map(line => ({indices: line.map(appendHelper)}));
  return new ScatterGL.Dataset(points, metadata);
}

// Compatibility adapter for the pinned ScatterGL 0.0.13 bundle only. The public
// API has no style setter or way to exclude the globe grid from point picking.
// Keep these accesses here so a future library upgrade has one place to check.
function applyPinnedLibraryStyles() {
  if (!scatter || !scatter.styles || !scatter.styles.point) {
    throw new Error("The bundled ScatterGL version does not support the style adapter.");
  }
  const scale = Math.max(0.2, Number(payload.point_size) / 3 || 1);
  scatter.styles.point.scaleDefault = scale;
  scatter.styles.point.scaleSelected = scale * 1.65;
  scatter.styles.point.scaleHover = scale * 1.8;
  scatter.styles.backgroundColor = background;
  scatter.styles.fog.color = background;
  if (scatter.scatterPlot && scatter.scatterPlot.renderer) scatter.scatterPlot.renderer.setClearColor(background, 1);
  // Native axes are removed by ScatterGL during its ID-picking pass. Keep them
  // neutral so orientation colors cannot be mistaken for peptide class colors.
  const axes = scatter.scatterPlot.scene.getObjectByName("axes");
  if (axes) {
    axes.visible = payload.show_axes !== false;
    const color = new THREE.Color(foreground);
    const colors = axes.geometry.getAttribute("color");
    for (let index = 0; index < colors.count; index++) colors.setXYZ(index, color.r, color.g, color.b);
    colors.needsUpdate = true;
    axes.material.opacity = 0.65;
    axes.material.transparent = true;
  }
  const geometry = scatter.pointVisualizer && scatter.pointVisualizer.points && scatter.pointVisualizer.points.geometry;
  if (renderMode === "points") {
    if (!geometry || typeof geometry.setDrawRange !== "function") {
      throw new Error("The bundled ScatterGL version does not support the frame adapter.");
    }
    geometry.setDrawRange(0, displayedMetadata.length);
  }
  const polylines = scatter.polylineVisualizer;
  if (polylines && !adaptedPolylineVisualizers.has(polylines)) {
    // Upstream's polyline picking callback is empty, so visible reference lines
    // would otherwise paint gray pixels over encoded point IDs at crossings.
    const onPickingRender = polylines.onPickingRender;
    const onRender = polylines.onRender;
    polylines.onPickingRender = function (context) {
      this.polylines.forEach(line => { line.visible = false; });
      return onPickingRender.call(this, context);
    };
    polylines.onRender = function (context) {
      this.polylines.forEach(line => { line.visible = true; });
      return onRender.call(this, context);
    };
    adaptedPolylineVisualizers.add(polylines);
  }
}

function refreshStylesAndSelection() {
  applyPinnedLibraryStyles();
  scatter.setPointColorer(colorPoint);
  scatter.select(localSelection(payload.selected_indices || []));
}

function resetPinnedLibraryCamera(view = "front") {
  // ScatterGL.resetZoom ignores constructor camera settings and uses its own
  // tilted defaults. Restore our level camera through the pinned plot adapter.
  const plot = scatter && scatter.scatterPlot;
  if (!plot || typeof plot.setCameraPositionAndTarget !== "function") {
    throw new Error("The bundled ScatterGL version does not support the camera adapter.");
  }
  setSpin(false);
  scatter.resetZoom();
  const camera = PeptideGlobeGeometry.cameraSettings(view);
  plot.setCameraPositionAndTarget(camera.position, camera.target);
  setInteractionMode(selecting);
  viewControl.value = view;
  overlayRevision = "";
}

function render(data, theme) {
  if (!data || !Array.isArray(data.points) || !Array.isArray(data.metadata)) {
    throw new Error("Missing peptide coordinates or metadata.");
  }
  payload = data;
  if (theme) {
    background = theme.backgroundColor || "#ffffff";
    foreground = theme.textColor || "#31333f";
    document.documentElement.style.setProperty("--background", background);
    document.documentElement.style.setProperty("--text", foreground);
    document.documentElement.style.setProperty("--panel", theme.secondaryBackgroundColor || background);
    document.documentElement.style.setProperty("--border", theme.base === "dark" ? "#64748b" : "#94a3b8");
  }
  const nextKey = JSON.stringify([data.revision, data.selected_only && data.selected_indices.length ? [...new Set(data.selected_indices)].sort((a, b) => a - b) : null]);
  const newData = geometryKey !== nextKey;
  if (newData) {
    if (scatter) {
      // Clear stale local indices before changing the displayed row mapping.
      scatter.select([]);
      scatter.setHoverPointIndex(null);
      scatter.setSequences([]);
    }
    const dataset = makeDataset(data);
    if (!scatter) {
      scatter = new ScatterGL(graph, {
        camera: PeptideGlobeGeometry.cameraSettings(),
        rotateOnStart: false,
        selectEnabled: true,
        showLabelsOnHover: false,
        pointColorer: colorPoint,
        onHover: showHover,
        onSelect: emitSelection,
        styles: {
          backgroundColor: background,
          axesVisible: true,
          fog: {enabled: false},
          polyline: {saturation: 0, lightness: 0.6, defaultOpacity: 0.23, selectedOpacity: 0.23, deselectedOpacity: 0.23, defaultLineWidth: 1, selectedLineWidth: 1},
          // The native shader interpolates this number as a GLSL float literal.
          label3D: {color: "#ffffff", backgroundColor: "rgba(0,0,0,0)", scale: 3.2},
        },
      });
      scatter.render(dataset);
      setInteractionMode(selecting);
    } else scatter.updateDataset(dataset);
    geometryKey = nextKey;
    globeVisible = null;
    showHover(null);
  }
  if (globeVisible !== Boolean(data.show_globe)) {
    // In 0.0.13 sequences must be initialized before selected indices.
    scatter.select([]);
    scatter.setSequences(data.show_globe ? frameSequences : []);
    globeVisible = Boolean(data.show_globe);
  }
  refreshStylesAndSelection();
  const names = data.axis_labels || ["X", "Y", "Z"];
  document.getElementById("axis-description").textContent = `Display axes: ${names.map((label, index) => `${"XYZ"[index]} = ${label}`).join(" · ")}`;
  overlayRevision = "";
  showHover(hoveredIndex);
  updateSelectionCount(data.selected_indices || []);
  graph.setAttribute("aria-label", `${data.title || "Peptide globe"}. Blue AMP, red Non-AMP, gray unlabeled.`);
  errorBox.hidden = true;
  resizeFrame();
}

rotateButton.addEventListener("click", () => setInteractionMode(false));
selectButton.addEventListener("click", () => setInteractionMode(true));
spinButton.addEventListener("click", () => {
  if (!scatter) return;
  if (!scatter.isOrbiting()) setInteractionMode(false);
  setSpin(!scatter.isOrbiting());
});
document.getElementById("reset").addEventListener("click", () => {
  if (!scatter) return;
  resetPinnedLibraryCamera();
});
viewControl.addEventListener("change", () => {
  if (scatter && viewControl.value !== "custom") resetPinnedLibraryCamera(viewControl.value);
});
clearButton.addEventListener("click", () => emitSelection([]));
displayControl.addEventListener("change", () => {
  if (!scatter) return;
  renderMode = displayControl.value;
  if (renderMode === "labels") scatter.setTextRenderMode();
  else scatter.setPointRenderMode();
  refreshStylesAndSelection();
});
graph.addEventListener("pointerdown", () => setSpin(false), {capture: true});
graph.addEventListener("wheel", () => setSpin(false), {capture: true, passive: true});
graph.addEventListener("pointermove", event => {
  const rect = graph.getBoundingClientRect();
  pointerPosition = {x: event.clientX - rect.left, y: event.clientY - rect.top};
  positionHoverLabel();
}, {capture: true, passive: true});
graph.addEventListener("mouseleave", () => showHover(null));
reducedMotion.addEventListener("change", () => { if (reducedMotion.matches) setSpin(false); });
window.addEventListener("resize", resizeFrame);
window.addEventListener("message", event => {
  if (event.source !== window.parent || !event.data || event.data.type !== "streamlit:render") return;
  try {
    render(event.data.args.payload, event.data.theme);
  } catch (error) {
    setSpin(false);
    errorBox.textContent = "Could not render ScatterGL: " + error.message;
    errorBox.hidden = false;
    resizeFrame();
  }
});
send("streamlit:componentReady", {apiVersion: 1});
requestAnimationFrame(updateOverlays);
resizeFrame();
