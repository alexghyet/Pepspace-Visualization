"use strict";

// Pure geometry shared by the browser renderer and Node regression checks.
(function (root, factory) {
  const geometry = factory();
  if (typeof module === "object" && module.exports) module.exports = geometry;
  else root.PeptideGlobeGeometry = geometry;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  function cameraSettings(view = "front") {
    // Three.js uses Y-up. Looking straight toward the origin keeps the globe's
    // poles vertical and its equator horizontal on the first frame and reset.
    const positions = {front: [0, 0, 2.2], side: [2.2, 0, 0], top: [0, 2.2, 0.00001]};
    if (!Object.hasOwn(positions, view)) throw new RangeError("Unknown camera view.");
    // The tiny top-view offset avoids the Y-up orbit-control pole singularity.
    return {position: positions[view], target: [0, 0, 0], zoom: 1};
  }

  function createGlobeLines({segments = 96, meridians = 6, latitudes = 7} = {}) {
    if (!Number.isInteger(segments) || segments < 8 || !Number.isInteger(meridians) || meridians < 3 || !Number.isInteger(latitudes) || latitudes < 1) {
      throw new RangeError("The globe requires at least eight segments, three meridians, and one latitude.");
    }
    const lines = [];
    const clean = value => Math.abs(value) < 1e-12 ? 0 : value;
    const circle = pointAt => {
      const points = Array.from({length: segments}, (_, step) => pointAt(step * 2 * Math.PI / segments).map(clean));
      // Repeat the exact first vertex so floating-point roundoff cannot open a seam.
      points.push([...points[0]]);
      lines.push(points);
    };
    // Each full great circle supplies two opposite longitude meridians.
    for (let meridian = 0; meridian < meridians; meridian++) {
      const phi = meridian * Math.PI / meridians;
      circle(theta => [Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi)]);
    }
    for (let latitude = 1; latitude <= latitudes; latitude++) {
      const theta = latitude * Math.PI / (latitudes + 1);
      circle(phi => [Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi)]);
    }
    return lines;
  }

  function pointVisible(metadata, selectedOnly, selected) {
    return Boolean(metadata && metadata.visible !== false
      && (!selectedOnly || selected.size === 0 || selected.has(metadata.row_index)));
  }

  function pointRgb(metadata) {
    if (metadata.color) {
      const hex = /^#([0-9a-f]{6})$/i.exec(metadata.color);
      if (hex) return [0, 2, 4].map(start => parseInt(hex[1].slice(start, start + 2), 16));
      const rgb = /^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/i.exec(metadata.color);
      if (rgb) return rgb.slice(1).map(value => Math.max(0, Math.min(255, Number(value))));
      if (metadata.color === "blue") return [0, 0, 255];
      if (metadata.color === "red") return [255, 0, 0];
    }
    return metadata.is_amp === true ? [0, 0, 255]
      : metadata.is_amp === false ? [255, 0, 0] : [148, 163, 184];
  }

  return Object.freeze({cameraSettings, createGlobeLines, pointVisible, pointRgb});
});
