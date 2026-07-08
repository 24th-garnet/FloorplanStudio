import * as THREE from "three";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { USDZLoader } from "three/addons/loaders/USDZLoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import {
  getSource,
  readStoredSource,
  sourceLabel,
  validateFileForSource,
  bindInputSourceThumb,
  decorateInputSourcePicker,
  syncInputSourceThumb,
} from "./input-sources.js";

let sessionId = null;
let sessionKind = null;
let sessionInputSource = null;
let exportFormatDefs = [];
let objects = [];
let layerDefs = [];
const OBJECT_TAG_FALLBACK = [
  { id: "floor", label: "Floor" },
  { id: "wall", label: "Wall" },
  { id: "door", label: "Door" },
  { id: "window", label: "Window" },
  { id: "opening", label: "Opening" },
  { id: "chair", label: "Chair" },
  { id: "table", label: "Table" },
  { id: "storage", label: "Storage" },
  { id: "other", label: "Other" },
];
let tagDefinitions = [...OBJECT_TAG_FALLBACK];
let tagEditObjectId = null;
let selected = null;
let scene = null;
let camera = null;
let renderer = null;
let controls = null;
let transformControls = null;
let modelRoot = null;
let realModelRoot = null;
let replacementUsdRoot = null;
let dxfRoot = null;
let dxfGeometry = null;
let dxfPlanLayers = [];
let roomPlanFloorData = null;
let roomPlanDisplayState = null;
let dxfPlanDisplayBounds = null;
let floorPlanResizeObserver = null;
const PLAN_CANVAS_PAD_PX = 28;
const PLAN_HANDLE_HIT_PX = 11;
const PLAN_POPUP_MARGIN_PX = 16;
const PLAN_HEADER_FALLBACK_PX = 52;
const PLAN_VIEWER_SIDE_MARGIN_PX = 24;
const PLAN_VIEWER_VERT_MARGIN_PX = 22;
const PLAN_VIEWPORT_FIT = 0.9;
const PLAN_POPUP_MIN_W = 260;
const PLAN_POPUP_MIN_H = 200;
const PLAN_POPUP_DEFAULT_H = 320;
let floorPlanDragState = null;
let floorPlanResizeState = null;
let floorPlanUserSized = false;
let floorPlanLastCanvasSize = { w: 0, h: 0 };
let floorPlanPaintMapping = null;

/** Planar data uses u = world X, v = -world Z; canvas mirrors U to match default 3D orbit. */
function worldZToPlanV(z) {
  return -Number(z);
}

function planVToWorldZ(v) {
  return -Number(v);
}

function getObjectPlanAxisDirsFromState(obj) {
  const rect = obj?.world_planar_rect;
  let angleDeg = Number(rect?.long_axis?.angle_deg);
  if (!Number.isFinite(angleDeg)) {
    angleDeg = safeNumber(obj?.yaw_deg, 0);
  }
  const angleRad = (angleDeg * Math.PI) / 180;
  const longDir = [Math.cos(angleRad), Math.sin(angleRad)];
  const shortDir = [-longDir[1], longDir[0]];
  return { longDir, shortDir, angleDeg };
}

function worldXzToPlanUv(x, z) {
  return [Number(x), worldZToPlanV(z)];
}

function worldXzCornersToPlanUv(corners) {
  return corners.map((c) => worldXzToPlanUv(c[0], c[1]));
}

function planUvToCanvas(u, v, minU, maxU, minV, pad, scale) {
  return [
    pad + (maxU - u) * scale,
    pad + (v - minV) * scale,
  ];
}

function canvasToPlanUv(cx, cy, mapping) {
  if (!mapping) return [0, 0];
  const { minU, maxU, minV, pad, scale } = mapping;
  return [
    maxU - (cx - pad) / scale,
    minV + (cy - pad) / scale,
  ];
}

function planUvDeltaFromCanvasDelta(dx, dy, mapping) {
  if (!mapping?.scale) return [0, 0];
  return [-dx / mapping.scale, dy / mapping.scale];
}

function rotatePlanVector2d(vector, angleRad) {
  const [du, dv] = vector;
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  return [du * cos - dv * sin, du * sin + dv * cos];
}

function displayPlanDeltaToWorldXz(duDisplay, dvDisplay, transform) {
  const rotationRad = transform?.rotationRad || 0;
  const [duWorld, dvWorld] = rotatePlanVector2d([duDisplay, dvDisplay], -rotationRad);
  return [duWorld, -dvWorld];
}

function getObjectPlanCenterFromState(obj) {
  const pos = normalizePosition(obj?.matrix_position ?? obj?.position);
  return [pos[0], worldZToPlanV(pos[2])];
}

function getFloorPlanDisplayCenter(objectId) {
  const item = roomPlanDisplayState?.items?.find((entry) => entry.id === objectId);
  if (Array.isArray(item?.center) && item.center.length >= 2) {
    return [Number(item.center[0]), Number(item.center[1])];
  }
  const obj = objects.find((o) => o.id === objectId);
  return obj ? getObjectPlanCenterFromState(obj) : [0, 0];
}

function planAngleFromCanvas(canvasX, canvasY, centerPlan, mapping) {
  const pointerPlan = canvasToPlanUv(canvasX, canvasY, mapping);
  const [cu, cv] = centerPlan;
  return Math.atan2(pointerPlan[1] - cv, pointerPlan[0] - cu);
}

function rotatePlanPointAround2d(point, angleRad, centerPlan) {
  const [u, v] = point;
  const [cu, cv] = centerPlan;
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  const du = u - cu;
  const dv = v - cv;
  return [cu + du * cos - dv * sin, cv + du * sin + dv * cos];
}

function rotateObjectPlanarFields(obj, deltaYawRad, centerPlan) {
  const next = clonePlanarObjectState(obj);
  const center = centerPlan || getObjectPlanCenterFromState(next);

  const rect = next.world_planar_rect;
  if (rect && typeof rect === "object") {
    if (Array.isArray(rect.corners)) {
      rect.corners = rect.corners.map((corner) => rotatePlanPointAround2d(corner, deltaYawRad, center));
    }
    if (Array.isArray(rect.center)) {
      rect.center = rotatePlanPointAround2d(rect.center, deltaYawRad, center);
    }
    if (rect.long_axis && typeof rect.long_axis === "object") {
      if (Array.isArray(rect.long_axis.a)) {
        rect.long_axis.a = rotatePlanPointAround2d(rect.long_axis.a, deltaYawRad, center);
      }
      if (Array.isArray(rect.long_axis.b)) {
        rect.long_axis.b = rotatePlanPointAround2d(rect.long_axis.b, deltaYawRad, center);
      }
      if (Number.isFinite(rect.long_axis.angle_deg)) {
        rect.long_axis.angle_deg = Number(rect.long_axis.angle_deg) + THREE.MathUtils.radToDeg(deltaYawRad);
      }
      if (Array.isArray(rect.long_axis.dir)) {
        const [du, dv] = rect.long_axis.dir;
        const rotated = rotatePlanVector2d([Number(du), Number(dv)], deltaYawRad);
        rect.long_axis.dir = rotated;
      }
    }
  }

  if (Array.isArray(next.world_basis_footprint_xz)) {
    next.world_basis_footprint_xz = next.world_basis_footprint_xz.map(([x, z]) => {
      const [ru, rv] = rotatePlanPointAround2d([Number(x), worldZToPlanV(z)], deltaYawRad, center);
      return [ru, -rv];
    });
  }

  const yawDeg = THREE.MathUtils.radToDeg(deltaYawRad) + safeNumber(obj.yaw_deg, 0);
  next.yaw_deg = yawDeg;
  const quat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(yawDeg));
  next.quaternion_xyzw = [quat.x, quat.y, quat.z, quat.w];
  return next;
}

function scaleObjectPlanarFields(obj, scaleLong, scaleShort, centerPlan) {
  const next = clonePlanarObjectState(obj);
  const center = centerPlan || getObjectPlanCenterFromState(next);
  const factorLong = Math.max(Math.abs(scaleLong), 0.05);
  const factorShort = Math.max(Math.abs(scaleShort), 0.05);
  const { longDir, shortDir } = getObjectPlanAxisDirsFromState(next);

  const scalePoint = ([u, v]) => {
    const du = u - center[0];
    const dv = v - center[1];
    const alongLong = du * longDir[0] + dv * longDir[1];
    const alongShort = du * shortDir[0] + dv * shortDir[1];
    return [
      center[0] + longDir[0] * alongLong * factorLong + shortDir[0] * alongShort * factorShort,
      center[1] + longDir[1] * alongLong * factorLong + shortDir[1] * alongShort * factorShort,
    ];
  };

  const rect = next.world_planar_rect;
  if (rect && typeof rect === "object") {
    if (Array.isArray(rect.corners)) rect.corners = rect.corners.map(scalePoint);
    if (Array.isArray(rect.center)) rect.center = scalePoint(rect.center);
    if (rect.long_axis && typeof rect.long_axis === "object") {
      if (Array.isArray(rect.long_axis.a)) rect.long_axis.a = scalePoint(rect.long_axis.a);
      if (Array.isArray(rect.long_axis.b)) rect.long_axis.b = scalePoint(rect.long_axis.b);
      if (Number.isFinite(rect.long_axis.length)) rect.long_axis.length *= factorLong;
    }
    if (rect.short_axis && typeof rect.short_axis === "object") {
      if (Number.isFinite(rect.short_axis.length)) rect.short_axis.length *= factorShort;
    }
  }

  if (Array.isArray(next.world_basis_footprint_xz)) {
    next.world_basis_footprint_xz = next.world_basis_footprint_xz.map(([x, z]) => {
      const [su, sv] = scalePoint([Number(x), worldZToPlanV(z)]);
      return [su, planVToWorldZ(sv)];
    });
  }

  const baseDims = normalizeDimensions(next.dimensions);
  next.dimensions = [
    Math.max(Math.abs(baseDims[0] * factorLong), 0.05),
    Math.max(Math.abs(baseDims[1]), 0.05),
    Math.max(Math.abs(baseDims[2] * factorShort), 0.05),
  ];
  return next;
}

function syncFloorPlanEditCursor() {
  const canvas = el("dxfPlanCanvas");
  if (!canvas || sessionKind !== "usdz") return;
  const cursors = {
    translate: "move",
    rotate: "grab",
    scale: "nwse-resize",
  };
  canvas.style.cursor = selected?.id && cursors[editMode] ? cursors[editMode] : "default";
}

function pointInPolygon2d(point, corners) {
  if (!Array.isArray(corners) || corners.length < 3) return false;
  const [x, y] = point;
  let inside = false;
  for (let i = 0, j = corners.length - 1; i < corners.length; j = i++) {
    const [xi, yi] = corners[i];
    const [xj, yj] = corners[j];
    const intersect = ((yi > y) !== (yj > y))
      && (x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-12) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function getCanvasPointerPosition(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return [0, 0];
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return [
    (event.clientX - rect.left) * scaleX,
    (event.clientY - rect.top) * scaleY,
  ];
}

function pickFloorPlanItemAtCanvas(canvasX, canvasY) {
  const mapping = floorPlanPaintMapping;
  const displayState = roomPlanDisplayState;
  if (!mapping || !displayState?.items?.length) return null;
  const planPoint = canvasToPlanUv(canvasX, canvasY, mapping);
  const items = [...displayState.items].sort(
    (a, b) => roomPlanLayerOrder(b.category) - roomPlanLayerOrder(a.category),
  );
  for (const item of items) {
    if (pointInPolygon2d(planPoint, item.corners)) {
      return objects.find((o) => o.id === item.id) || null;
    }
  }
  return null;
}

let wallExtractData = null;
let wallHighlightRoot = null;
let wallExtrudeData = null;
let wallExtrudeRoot = null;
let extrudeCompareReport = null;
let extrudeCompareActiveSource = null;
let loadingModel = false;
let modelLoadChain = Promise.resolve();
let transformTimer = null;
let lastAlignedWallIds = new Set();
let warnedUsdLoaderFailure = false;
let lastLoadedUsdZUrl = "";
let editMode = "translate";
const TRANSFORM_MODE_STORAGE_KEY = "floorplan-studio.transform-mode";
const UNDO_LIMIT = 10;
const undoStack = [];
const TRANSFORM_UNDO_LIMIT = 10;
const transformUndoStack = [];
let isUndoing = false;
let isTransformUndoing = false;
let transformDragSnapshot = null;
let replacementAssets = [];
let textureAssets = [];
const replacementProxyModelCache = new Map();
const textureThreeCache = new Map();
const textureVersionByKey = new Map();
const textureLoader = new THREE.TextureLoader();

const objectMeshes = new Map();
const objectPickMeshes = new Map();
const usdzPickMeshes = [];
let selectionHighlightRoot = null;
let selectionHighlightGroup = null;
const selectionLineMaterials = [];
const _pickRaycaster = new THREE.Raycaster();
const _pickMouse = new THREE.Vector2();
const _movePlane = new THREE.Plane();
const _movePlaneHit = new THREE.Vector3();
let planeEditDrag = null;
const SELECTION_EDGE_THRESHOLD = 52;
let selectionCornerSphereGeo = null;
const usdzLoader = new USDZLoader();
const gltfLoader = new GLTFLoader();
const DXF_PLAN_THEME = {
  background: "#0c0609",
  gridMinor: "rgba(200, 120, 140, 0.08)",
  gridMajor: "rgba(200, 120, 140, 0.16)",
  line: "#6eb5e8",
  emptyText: "#9a7884",
  scale: "#d4b8c0",
};
const ROOM_PLAN_THEME = {
  floor: { fill: "rgba(245, 246, 248, 0.1)", stroke: "rgba(217, 221, 227, 0.55)" },
  wall: { fill: "rgba(111, 119, 130, 0.32)", stroke: "rgba(154, 163, 173, 0.85)" },
  door: { fill: "rgba(209, 123, 40, 0.22)", stroke: "#d17b28" },
  window: { fill: "rgba(78, 121, 167, 0.22)", stroke: "#6eb5e8" },
  opening: { fill: "rgba(154, 163, 173, 0.18)", stroke: "#9aa3ad" },
  chair: { fill: "rgba(107, 142, 193, 0.24)", stroke: "#6b8ec1" },
  table: { fill: "rgba(138, 114, 181, 0.24)", stroke: "#8a72b5" },
  storage: { fill: "rgba(106, 163, 126, 0.24)", stroke: "#6aa37e" },
  object: { fill: "rgba(196, 154, 78, 0.22)", stroke: "#c49a4e" },
  selected: {
    fill: "rgba(196, 77, 106, 0.16)",
    stroke: "#fff0f4",
    glow: "rgba(232, 160, 176, 0.42)",
    lineWidth: 2.4,
  },
};
let hasFittedCamera = false;

const WALK_UP = new THREE.Vector3(0, 1, 0);
const _walkDown = new THREE.Vector3(0, -1, 0);
const WALK_EYE_HEIGHT = 1.6;
const WALK_SPEED = 1.4;
const WALK_JUMP_HEIGHT = 0.3;
const WALK_PLAYER_RADIUS = 0.3;
const WALK_GRAVITY = 9.8;
const WALK_GROUND_EPS = 0.08;
const WALK_FEET_PROBE_LIFT = 0.12;
const WALK_GROUND_PROBE_DEPTH = 15;
const WALK_FOV_DEFAULT = 100;
const WALK_FOV_MIN = 40;
const WALK_FOV_MAX = 120;
const WALK_FOV_WHEEL_STEP = 3;
let walkWheelCaptureBound = false;
let walkModeActive = false;
let walkSpawnPickActive = false;
let walkSpawn = null;
let walkSpawnIsAuto = false;
let walkVelocityY = 0;
let walkOnGround = false;
let walkCollisionMeshes = [];
let pointerLockControls = null;
let walkCamera = null;
let walkPlayerMarker = null;
let planarTransformAccent = null;
let planarTransformAccentPhase = 0;

const GIZMO_AXIS = {
  x: new THREE.Color(0xf0a0b4),
  y: new THREE.Color(0xc44d6a),
  z: new THREE.Color(0xa888e8),
  idle: new THREE.Color(0x8a7078),
  hub: new THREE.Color(0xffe8ee),
};

function replaceGizmoMaterial(material, color, { active = true, opacity = 0.92 } = {}) {
  if (material?.userData?.floorplanGizmoStyled) {
    material.color.copy(color);
    material.emissive.copy(color);
    material.emissiveIntensity = active ? 0.82 : 0.18;
    material.opacity = active ? opacity : 0.38;
    material.needsUpdate = true;
    return material;
  }
  const next = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: active ? 0.82 : 0.18,
    metalness: 0.55,
    roughness: 0.22,
    transparent: true,
    opacity: active ? opacity : 0.38,
    depthTest: false,
    depthWrite: false,
  });
  next.userData.floorplanGizmoStyled = true;
  if (material && typeof material.dispose === "function") material.dispose();
  return next;
}

function gizmoAxisColor(token, mode) {
  if (mode === "rotate") return GIZMO_AXIS.y;
  if (token === "X") return GIZMO_AXIS.x;
  if (token === "Z") return GIZMO_AXIS.z;
  return GIZMO_AXIS.idle;
}

function ensurePlanarTransformAccent() {
  if (planarTransformAccent) return planarTransformAccent;

  const group = new THREE.Group();
  group.name = "planar_transform_accent";
  group.renderOrder = 10050;

  const ringMat = new THREE.MeshStandardMaterial({
    color: GIZMO_AXIS.y,
    emissive: GIZMO_AXIS.y,
    emissiveIntensity: 0.42,
    metalness: 0.35,
    roughness: 0.3,
    transparent: true,
    opacity: 0.5,
    depthTest: false,
    depthWrite: false,
  });

  const ring = new THREE.Mesh(new THREE.TorusGeometry(1, 0.028, 12, 72), ringMat);
  ring.rotation.x = Math.PI / 2;
  ring.name = "accent_ring";
  group.add(ring);

  const hub = new THREE.Mesh(new THREE.SphereGeometry(0.055, 20, 20), ringMat.clone());
  hub.name = "accent_hub";
  group.add(hub);

  const axisMat = ringMat.clone();
  axisMat.opacity = 0.72;
  axisMat.emissiveIntensity = 0.55;

  const xAxis = new THREE.Mesh(new THREE.BoxGeometry(1.35, 0.03, 0.03), axisMat.clone());
  xAxis.material.color.copy(GIZMO_AXIS.x);
  xAxis.material.emissive.copy(GIZMO_AXIS.x);
  xAxis.name = "accent_axis_x";
  group.add(xAxis);

  const yAxis = new THREE.Mesh(new THREE.BoxGeometry(0.03, 1.35, 0.03), axisMat.clone());
  yAxis.material.color.copy(GIZMO_AXIS.y);
  yAxis.material.emissive.copy(GIZMO_AXIS.y);
  yAxis.name = "accent_axis_y";
  group.add(yAxis);

  const xCap = new THREE.Mesh(new THREE.SphereGeometry(0.05, 14, 14), axisMat.clone());
  xCap.material.color.copy(GIZMO_AXIS.x);
  xCap.material.emissive.copy(GIZMO_AXIS.x);
  xCap.position.set(0.68, 0, 0);
  xCap.name = "accent_cap_x";
  group.add(xCap);

  const yCap = new THREE.Mesh(new THREE.SphereGeometry(0.05, 14, 14), axisMat.clone());
  yCap.material.color.copy(GIZMO_AXIS.y);
  yCap.material.emissive.copy(GIZMO_AXIS.y);
  yCap.position.set(0, 0.68, 0);
  yCap.name = "accent_cap_y";
  group.add(yCap);

  planarTransformAccent = group;
  planarTransformAccent.visible = false;
  if (scene) scene.add(planarTransformAccent);
  return planarTransformAccent;
}

function syncPlanarTransformAccent() {
  const accent = ensurePlanarTransformAccent();
  accent.visible = false;
}

function getTransformControlsGizmo() {
  return transformControls?._gizmo || null;
}

function transformGizmoPartName(node) {
  let cur = node;
  while (cur && cur !== transformControls) {
    const name = String(cur.name || "").trim().toUpperCase();
    if (["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ", "E", "XYZE"].includes(name)) return name;
    cur = cur.parent;
  }
  return "";
}

function enforcePlanarTransformHandleVisibility() {
  const gizmoRoot = getTransformControlsGizmo();
  if (!transformControls?.visible || !gizmoRoot) return;

  const mode = gizmoRoot.mode || transformControls.getMode?.() || "scale";
  const allowed = mode === "rotate"
    ? new Set(["Y"])
    : mode === "scale"
      ? new Set(["X", "Z"])
      : new Set();
  const alwaysHide = new Set(["XZ", "YZ", "XYZ", "XYZE", "XY", "E"]);
  if (mode === "scale") alwaysHide.add("Y");
  if (mode === "rotate") {
    alwaysHide.add("X");
    alwaysHide.add("Z");
  }

  const hideHandle = (handle) => {
    if (!handle) return;
    handle.visible = false;
  };

  const enforceHandleVisibility = (group) => {
    if (!group?.children) return;
    for (const handle of group.children) {
      const part = String(handle.name || "").trim().toUpperCase();
      if (!part || alwaysHide.has(part) || !allowed.has(part)) {
        hideHandle(handle);
      }
    }
  };

  if (mode === "scale") {
    if (gizmoRoot.helper?.scale) gizmoRoot.helper.scale.visible = false;
    enforceHandleVisibility(gizmoRoot.gizmo?.scale);
    enforceHandleVisibility(gizmoRoot.picker?.scale);
  } else if (mode === "rotate") {
    if (gizmoRoot.helper?.rotate) gizmoRoot.helper.rotate.visible = false;
    enforceHandleVisibility(gizmoRoot.gizmo?.rotate);
    enforceHandleVisibility(gizmoRoot.picker?.rotate);
  }

  for (const bucket of ["gizmo", "picker", "helper"]) {
    const groups = gizmoRoot[bucket];
    if (!groups) continue;
    for (const group of Object.values(groups)) {
      if (!group?.traverse) continue;
      group.traverse((node) => {
        const part = transformGizmoPartName(node);
        if (!part) return;
        if (alwaysHide.has(part) || !allowed.has(part)) {
          node.visible = false;
        }
      });
      for (const partName of [...alwaysHide, "X", "Y", "Z", "E", "XYZE"]) {
        if (allowed.has(partName)) continue;
        hideHandle(group.getObjectByName?.(partName));
      }
    }
  }
}

function enforcePlanarTransformAxes() {
  if (!transformControls) return;

  const gizmoRoot = getTransformControlsGizmo();

  transformControls.showZ = editMode === "scale";
  if (editMode === "scale") {
    transformControls.showX = true;
    transformControls.showY = false;
  } else if (editMode === "rotate") {
    transformControls.showX = false;
    transformControls.showY = true;
    transformControls.showZ = false;
  } else {
    transformControls.showX = false;
    transformControls.showY = false;
  }

  if (gizmoRoot) {
    gizmoRoot.showX = transformControls.showX;
    gizmoRoot.showY = transformControls.showY;
    gizmoRoot.showZ = transformControls.showZ;
  }

  enforcePlanarTransformHandleVisibility();
}

function hookTransformGizmoVisibility() {
  const gizmoRoot = getTransformControlsGizmo();
  if (!gizmoRoot || gizmoRoot.userData.planarAxisHooked) return;
  gizmoRoot.userData.planarAxisHooked = true;
  const original = gizmoRoot.updateMatrixWorld.bind(gizmoRoot);
  gizmoRoot.updateMatrixWorld = (force) => {
    original(force);
    enforcePlanarTransformHandleVisibility();
  };
}

function restyleTransformControlsGizmo() {
  if (!transformControls?.traverse) return;
  enforcePlanarTransformAxes();
  const mode = transformControls?.getMode?.() || "translate";
  const relevant = mode === "rotate"
    ? new Set(["Y"])
    : mode === "scale"
      ? new Set(["X", "Z"])
      : new Set(["X", "Y"]);
  const planes = new Set(["XY", "XZ", "YZ", "XYZ", "XYZE"]);
  const alwaysHide = mode === "rotate"
    ? new Set(["X", "Z", "E", "XYZE", "XZ", "YZ", "XYZ", "XY"])
    : mode === "scale"
      ? new Set(["Y", "XZ", "YZ", "XYZ", "XYZE", "XY"])
      : new Set(["Z", "XZ", "YZ", "XYZ", "XYZE", "XY"]);

  const nameToken = (node) => {
    const n = String(node?.name || "").trim().toUpperCase();
    if (!n) return "";
    if (n === "E") return "E";
    if (n.includes("XYZ")) return "XYZ";
    if (n.includes("XY")) return "XY";
    if (n.includes("YZ")) return "YZ";
    if (n.includes("XZ")) return "XZ";
    if (n.endsWith("X") || n === "X") return "X";
    if (n.endsWith("Y") || n === "Y") return "Y";
    if (n.endsWith("Z") || n === "Z") return "Z";
    return "";
  };

  const ancestorNameIncludes = (node, needle) => {
    let cur = node?.parent;
    const upperNeedle = String(needle || "").toUpperCase();
    while (cur) {
      const n = String(cur?.name || "").toUpperCase();
      if (n && n.includes(upperNeedle)) return true;
      if (cur === transformControls) break;
      cur = cur.parent;
    }
    return false;
  };

  transformControls.traverse((node) => {
    const mat = node?.material;
    const token = nameToken(node);
    const rawName = String(node?.name || "").toUpperCase();
    const geoType = String(node?.geometry?.type || "");

    const isPlaneLike = geoType === "PlaneGeometry"
      || planes.has(token)
      || planes.has(rawName)
      || rawName.includes("XY")
      || rawName.includes("YZ")
      || rawName.includes("XZ")
      || rawName.includes("XYZ")
      || rawName.includes("XYZE")
      || rawName.includes("PLANE");
    const isCenterScale = mode === "scale" && (
      token === "XYZ"
      || rawName.includes("XYZ")
      || ancestorNameIncludes(node, "XYZ")
    );
    const isHiddenAxis = token && (!relevant.has(token) || alwaysHide.has(token));

    if (isPlaneLike || isCenterScale || isHiddenAxis) {
      node.visible = false;
      return;
    }

    if (!mat || !node.isMesh) return;

    const isActive = token ? relevant.has(token) : false;
    const axisColor = gizmoAxisColor(token, mode);
    const apply = (m) => {
      if (!m) return;
      const upgraded = replaceGizmoMaterial(m, axisColor, { active: isActive });
      if (upgraded !== m && node) node.material = upgraded;
    };
    if (Array.isArray(mat)) {
      node.material = mat.map((m) => replaceGizmoMaterial(m, axisColor, { active: isActive }));
    } else {
      node.material = replaceGizmoMaterial(mat, axisColor, { active: isActive });
    }
    node.renderOrder = 10001;
  });

  syncPlanarTransformAccent();
}
let walkClock = null;
const walkKeys = {
  forward: false,
  back: false,
  left: false,
  right: false,
  jump: false,
};
const _walkForward = new THREE.Vector3();
const _walkRight = new THREE.Vector3();
const _walkWish = new THREE.Vector3();
const _walkRaycaster = new THREE.Raycaster();

const el = (id) => document.getElementById(id);

function setOverlayStatus(_message) {
  const status = el("overlayStatus");
  if (!status) return;
  status.textContent = "";
  status.hidden = true;
}

document.addEventListener("DOMContentLoaded", () => {
  try {
    initViewer();
  } catch (err) {
    console.error("initViewer failed:", err);
    showWalkFeedback(`3Dビュー初期化エラー: ${err?.message || err}`);
  }
  try {
    initWalkViewer();
  } catch (err) {
    console.error("initWalkViewer failed:", err);
    showWalkFeedback(`1人称ビュー初期化エラー: ${err?.message || err}`);
  }
  bindUI();
  try {
    bindWalkModeUI();
  } catch (err) {
    console.error("bindWalkModeUI failed:", err);
    showWalkFeedback(`Walk mode UI error: ${err?.message || err}`);
  }
  try {
    syncFloorPlanPanel();
  } catch (err) {
    console.error("syncFloorPlanPanel failed:", err);
  }
});

function bindUI() {
  initStudioInputSourcePicker();
  initImportDropZone();
  el("extractWalls")?.addEventListener("click", onExtractWalls);
  el("extrudeWalls")?.addEventListener("click", onExtrudeWalls);
  el("runWallExtrusion")?.addEventListener("click", onRunWallExtrusion);
  el("mlPocCompare")?.addEventListener("click", onMlPocCompare);
  el("mlRenderCompare")?.addEventListener("click", onMlRenderCompare);
  el("extrudeCompare")?.addEventListener("click", onExtrudeCompare);
  el("wallRefineCompare")?.addEventListener("click", onWallRefineCompare);
  el("extrudeCompareSourceBtns")?.addEventListener("click", onExtrudeCompareSourceClick);
  el("apply")?.addEventListener("click", onApply);
  el("deleteSelected")?.addEventListener("click", onDeleteSelected);
  el("applyReplacement")?.addEventListener("click", onApplyReplacement);
  el("removeReplacement")?.addEventListener("click", onRemoveReplacement);
  el("applyTexture")?.addEventListener("click", onApplyTexture);
  el("removeTexture")?.addEventListener("click", onRemoveTexture);
  el("alignWalls")?.addEventListener("click", onAlignWalls);
  el("snapBottomsToFloor")?.addEventListener("click", onSnapBottomsToFloor);
  el("undo")?.addEventListener("click", onUndo);
  el("transformUndo")?.addEventListener("click", onTransformUndo);
  initTransformModePicker();
  updateTransformUndoUI();
  el("showRealModel")?.addEventListener("change", () => {
    syncObjectDisplayTargets();
  });
  el("showProxy")?.addEventListener("change", () => {
    syncObjectDisplayTargets();
  });
  el("showDxf")?.addEventListener("change", () => {
    if (dxfRoot) dxfRoot.visible = el("showDxf").checked;
  });
  el("showWall")?.addEventListener("change", syncWallVisibility);
  el("showFloorPlan")?.addEventListener("click", (event) => {
    event.preventDefault();
    const btn = el("showFloorPlan");
    if (!(btn instanceof HTMLButtonElement)) return;
    setFloorPlanVisible(btn.getAttribute("aria-pressed") !== "true");
  });
  el("closeFloorPlan")?.addEventListener("click", (event) => {
    event.preventDefault();
    setFloorPlanVisible(false);
  });
  initFloorPlanPopupDrag();
  initFloorPlanPopupResize();
  initFloorPlanCanvasInteraction();
  el("focus3dViewer")?.addEventListener("change", () => {
    if (walkModeActive) return;
    syncFocus3dViewer();
  });
  document.addEventListener("keydown", onWalkKeyDown);
  document.addEventListener("keyup", onWalkKeyUp);
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !isFloorPlanVisible()) return;
    if (walkModeActive) return;
    event.preventDefault();
    setFloorPlanVisible(false);
  });
  window.addEventListener("resize", () => {
    resizeViewer();
    resizeWalkViewer();
    drawFloorPlanView();
  });
  updateUndoUI();
  syncEditMode();
  syncWallExtractButton();
  syncExtrudeWallsButton();
  syncWallExtrusionButton();
  syncMlPocButton();
  syncMlRenderCompareButton();
  syncFocus3dViewer({ fitCamera: false });
  syncWalkModeUI();
  void loadReplacementAssets();
  void loadTagDefinitions();
  void loadExportFormats();
  void refreshTextureLibrary();
  el("tagEditorAddBtn")?.addEventListener("click", (e) => {
    e.preventDefault();
    void onTagEditorAdd();
  });
  el("tagEditorReset")?.addEventListener("click", (e) => {
    e.preventDefault();
    void onTagEditorReset();
  });
  el("tagEditorDialog")?.addEventListener("close", () => {
    tagEditObjectId = null;
  });
  el("tagEditorClose")?.addEventListener("click", () => closeTagEditor());
  el("tagBulkTag")?.addEventListener("change", () => refreshTagBulkUI());
  el("tagBulkApplyTexture")?.addEventListener("click", (e) => {
    e.preventDefault();
    void onTagBulkApplyTexture();
  });
  el("tagBulkRemoveTexture")?.addEventListener("click", (e) => {
    e.preventDefault();
    void onTagBulkRemoveTexture();
  });
  el("tagBulkApplyReplacement")?.addEventListener("click", (e) => {
    e.preventDefault();
    void onTagBulkApplyReplacement();
  });
  el("tagBulkRemoveReplacement")?.addEventListener("click", (e) => {
    e.preventDefault();
    void onTagBulkRemoveReplacement();
  });
}

async function loadTagDefinitions() {
  try {
    const res = await fetch("/api/tag-definitions");
    if (!res.ok) return;
    const data = await res.json();
    tagDefinitions = Array.isArray(data.tags) && data.tags.length ? data.tags : [...OBJECT_TAG_FALLBACK];
  } catch {
    tagDefinitions = [...OBJECT_TAG_FALLBACK];
  } finally {
    populateTagEditorSelect();
    populateTagBulkSelects();
    refreshTagBulkUI();
  }
}

function collectTagsInSession() {
  const seen = new Set();
  const ids = [];
  for (const obj of objects) {
    for (const tagId of getObjectTags(obj)) {
      const id = String(tagId || "").toLowerCase();
      if (!id || seen.has(id)) continue;
      seen.add(id);
      ids.push(id);
    }
  }
  if (!ids.length) return [];

  const ordered = [];
  for (const def of tagDefinitions) {
    const id = String(def.id || "").toLowerCase();
    if (seen.has(id)) {
      ordered.push({ id, label: def.label || def.id });
      seen.delete(id);
    }
  }
  for (const id of seen) {
    ordered.push({ id, label: tagLabel(id) });
  }
  return ordered;
}

function populateTagBulkTagSelect() {
  const tagSelect = el("tagBulkTag");
  if (!(tagSelect instanceof HTMLSelectElement)) return;
  const prev = tagSelect.value;
  const sessionTags = collectTagsInSession();
  tagSelect.innerHTML = "";
  if (!sessionTags.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = sessionId ? "No tags in file" : "Import USDZ";
    option.disabled = true;
    option.selected = true;
    tagSelect.appendChild(option);
    tagSelect.value = "";
    return;
  }
  for (const tag of sessionTags) {
    const option = document.createElement("option");
    option.value = tag.id;
    option.textContent = tag.label;
    tagSelect.appendChild(option);
  }
  if (prev && [...tagSelect.options].some((o) => o.value === prev)) {
    tagSelect.value = prev;
  }
}

function populateTagBulkSelects() {
  populateTagBulkTagSelect();
  const texSelect = el("tagBulkTexture");
  if (texSelect instanceof HTMLSelectElement) {
    texSelect.innerHTML = "";
    for (const key of ["floor", "wall"]) {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = key.charAt(0).toUpperCase() + key.slice(1);
      texSelect.appendChild(option);
    }
  }
  const repSelect = el("tagBulkReplacement");
  if (repSelect instanceof HTMLSelectElement && replacementAssets.length) {
    repSelect.innerHTML = "";
    for (const asset of replacementAssets) {
      const option = document.createElement("option");
      option.value = asset.key;
      option.textContent = asset.label || asset.key;
      repSelect.appendChild(option);
    }
  }
}

function isSurfaceTag(tagId) {
  const t = String(tagId || "").toLowerCase();
  return t === "floor" || t === "wall";
}

function countObjectsWithTag(tagId) {
  const tag = String(tagId || "").toLowerCase();
  return objects.filter((o) => getObjectTags(o).includes(tag)).length;
}

function getTagBulkTagId() {
  const select = el("tagBulkTag");
  return select instanceof HTMLSelectElement ? select.value : "";
}

function refreshTagBulkUI() {
  populateTagBulkTagSelect();
  const panel = el("tagBulkControls");
  const texBlock = el("tagBulkTextureBlock");
  const repBlock = el("tagBulkReplacementBlock");
  const canUse = Boolean(sessionId) && objects.length > 0;
  if (panel) panel.classList.toggle("disabled", !canUse);

  const tagId = getTagBulkTagId();
  const matchCount = canUse && tagId ? countObjectsWithTag(tagId) : 0;

  const surface = isSurfaceTag(tagId);
  if (texBlock) {
    texBlock.hidden = false;
    texBlock.classList.toggle("is-inactive", !canUse || !surface);
  }
  if (repBlock) {
    repBlock.hidden = false;
    repBlock.classList.toggle("is-inactive", !canUse || surface);
  }

  const texDisabled = !canUse || !surface || matchCount <= 0;
  const repDisabled = !canUse || surface || matchCount <= 0;
  for (const id of ["tagBulkApplyTexture", "tagBulkRemoveTexture"]) {
    const btn = el(id);
    if (btn) btn.classList.toggle("disabled", texDisabled);
  }
  for (const id of ["tagBulkApplyReplacement", "tagBulkRemoveReplacement"]) {
    const btn = el(id);
    if (btn) btn.classList.toggle("disabled", repDisabled);
  }
  const tagSelect = el("tagBulkTag");
  if (tagSelect) tagSelect.disabled = !canUse;
  const texSelect = el("tagBulkTexture");
  if (texSelect) texSelect.disabled = texDisabled;
  const repSelect = el("tagBulkReplacement");
  if (repSelect) repSelect.disabled = repDisabled;
}

async function applyTagBulkResponse(data, options = {}) {
  objects = Array.isArray(data.objects) ? data.objects : objects;
  if (Array.isArray(data.layers)) layerDefs = data.layers;
  if (selected?.id) selected = objects.find((o) => o.id === selected.id) || null;
  renderLayerPanel();
  renderObjectList();
  refreshReplacementUI();
  refreshTagBulkUI();
  if (options.reloadModel && data.usdz_url) {
    await reloadUsdOverlayAndProxy(data.usdz_url, { fitCamera: false });
  } else {
    buildProxyScene(objects, { fitCamera: false });
    if (realModelRoot) await syncReplacementUsdOverlays(realModelRoot, objects);
  }
  if (options.reloadModel) {
    await ensureReplacementVisualsApplied(options);
  }
  if (selected) {
    fillEditor(selected);
    if (!isSurfaceTextureOnlyObject(selected)) attachTransform(selected.id);
  }
  const updated = Number(data.updated_count ?? data.removed_count ?? 0);
  if (options.showSummary && updated > 0) {
    const shown = replacementUsdRoot?.children?.length || 0;
    if (shown > 0) alert(`${updated} objects were replaced`);
  }
}

async function ensureReplacementVisualsApplied(options = {}) {
  const expected = getAssetsObjects(objects).filter((obj) => getReplacementAssetKey(obj));
  if (!expected.length) return;
  let shown = replacementUsdRoot?.children?.length || 0;
  if (shown <= 0 && realModelRoot) {
    await syncReplacementUsdOverlays(realModelRoot, objects);
    shown = replacementUsdRoot?.children?.length || 0;
  }
  const showReal = el("showRealModel");
  if (showReal instanceof HTMLInputElement && shown > 0 && !showReal.checked) {
    showReal.checked = true;
    if (realModelRoot) realModelRoot.visible = true;
    if (replacementUsdRoot) replacementUsdRoot.visible = true;
  }
  document.body.dataset.replaceOverlayCount = String(shown);
  document.body.dataset.replaceExpectedCount = String(expected.length);
  if (shown <= 0) {
    const msg = `Replacement display failed (${expected.length} object${expected.length === 1 ? "" : "s"}). Keep USDZ ON and reload the page.`;
    setOverlayStatus(msg);
    console.error("[replace]", msg);
    if (options.showSummary) {
      alert(`置換モデルを表示できませんでした（${expected.length}件）。USDZがONか確認し、ページを再読み込みしてください。`);
    }
  }
}

async function onTagBulkApplyTexture() {
  if (!sessionId) return;
  const tag = getTagBulkTagId();
  const select = el("tagBulkTexture");
  const textureKey = select instanceof HTMLSelectElement ? select.value : "";
  if (!tag || !textureKey) return;
  const res = await fetch(`/api/apply-texture-by-tag/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tag, texture_key: textureKey }),
  });
  let data = {};
  try {
    data = await res.json();
  } catch {
    alert(`一括テクスチャ適用に失敗しました (HTTP ${res.status})`);
    return;
  }
  if (!res.ok) {
    alert(data.error || "bulk texture apply failed");
    return;
  }
  await applyTagBulkResponse(data, { showSummary: true });
}

async function onTagBulkRemoveTexture() {
  if (!sessionId) return;
  const tag = getTagBulkTagId();
  if (!tag) return;
  const res = await fetch(`/api/remove-texture-by-tag/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tag }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "bulk texture remove failed");
    return;
  }
  await applyTagBulkResponse(data, { showSummary: true });
}

async function onTagBulkApplyReplacement() {
  if (!sessionId) {
    alert("先にUSDZをインポートしてください");
    return;
  }
  const tag = getTagBulkTagId();
  const select = el("tagBulkReplacement");
  const assetKey = select instanceof HTMLSelectElement ? select.value : "";
  if (!tag) {
    alert("Target tag を選択してください");
    return;
  }
  if (!assetKey) {
    alert("置換アセットを選択してください");
    return;
  }
  setOverlayStatus("Replacement: applying...");
  const res = await fetch(`/api/replace-object-by-tag/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tag, asset_key: assetKey }),
  });
  let data = {};
  try {
    data = await res.json();
  } catch {
    alert(`一括置換に失敗しました (HTTP ${res.status})`);
    return;
  }
  if (!res.ok) {
    alert(data.error || "bulk replace failed");
    return;
  }
  await applyTagBulkResponse(data, { showSummary: true, reloadModel: true });
}

async function onTagBulkRemoveReplacement() {
  if (!sessionId) return;
  const tag = getTagBulkTagId();
  if (!tag) return;
  const res = await fetch(`/api/unreplace-object-by-tag/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tag }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "bulk unreplace failed");
    return;
  }
  await applyTagBulkResponse(data, { showSummary: true, reloadModel: true });
}

function populateTagEditorSelect() {
  const select = el("tagEditorAdd");
  if (!(select instanceof HTMLSelectElement)) return;
  select.innerHTML = "";
  for (const tag of tagDefinitions) {
    const option = document.createElement("option");
    option.value = tag.id;
    option.textContent = tag.label || tag.id;
    select.appendChild(option);
  }
}

function tagLabel(tagId) {
  const id = String(tagId || "").toLowerCase();
  return tagDefinitions.find((tag) => tag.id === id)?.label || id;
}

function getObjectTags(obj) {
  if (!obj) return ["other"];
  if (Array.isArray(obj.tags)) {
    const tags = obj.tags.map((t) => String(t || "").toLowerCase()).filter(Boolean);
    return tags.length ? tags : ["other"];
  }
  const layerId = String(obj.layer || "other").toLowerCase();
  return layerId ? [layerId] : ["other"];
}

function getPrimaryObjectTag(obj) {
  return getObjectTags(obj)[0] || "other";
}

function isAutoTag(obj, tagId) {
  if (obj?.tags_overridden) return false;
  const inferred = Array.isArray(obj?.inferred_tags)
    ? obj.inferred_tags.map((t) => String(t || "").toLowerCase())
    : [String(obj?.inferred_layer || "").toLowerCase()].filter(Boolean);
  return inferred.includes(String(tagId || "").toLowerCase());
}

function initTransformModePicker() {
  const picker = el("transformModePicker");
  if (!picker) return;

  // Always start in Move mode (do not restore previous mode on load).
  editMode = "translate";
  localStorage.setItem(TRANSFORM_MODE_STORAGE_KEY, editMode);

  const buttons = picker.querySelectorAll?.("button[data-mode]") || [];
  if (buttons.length) {
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const value = btn.getAttribute("data-mode");
        if (value === "translate" || value === "scale" || value === "rotate") {
          editMode = value;
          localStorage.setItem(TRANSFORM_MODE_STORAGE_KEY, value);
          syncEditMode();
        }
      });
    });
  } else {
    // Back-compat: legacy radio segmented control.
    bindInputSourceThumb(picker, (value) => {
      if (value === "translate" || value === "scale" || value === "rotate") {
        editMode = value;
        localStorage.setItem(TRANSFORM_MODE_STORAGE_KEY, value);
        syncEditMode();
      }
    }, { persistSelection: false });

    const radio = picker.querySelector(`input[value="${editMode}"]`);
    if (radio instanceof HTMLInputElement) radio.checked = true;
  }
  syncEditMode();
}

function syncEditMode() {
  const picker = el("transformModePicker");
  if (picker) {
    const buttons = picker.querySelectorAll?.("button[data-mode]") || [];
    if (buttons.length) {
      buttons.forEach((btn) => {
        const mode = btn.getAttribute("data-mode");
        const active = mode === editMode;
        btn.classList.toggle("is-active", active);
        btn.setAttribute("aria-pressed", active ? "true" : "false");
      });
    } else {
      const radio = picker.querySelector(`input[value="${editMode}"]`);
      if (radio instanceof HTMLInputElement) radio.checked = true;
      syncInputSourceThumb(picker);
    }
  }

  if (!transformControls) {
    syncFloorPlanEditCursor();
    if (isFloorPlanVisible()) drawFloorPlanView();
    return;
  }
  const mode = editMode === "scale" ? "scale" : editMode === "rotate" ? "rotate" : "translate";
  transformControls.setMode(mode);
  transformControls.setSpace?.("local");
  // Transform: floor-plane X/Z scale only. Rotate: Y (yaw) only.
  transformControls.showX = editMode === "scale";
  transformControls.showY = editMode === "rotate";
  transformControls.showZ = editMode === "scale";
  transformControls.visible = editMode !== "translate";
  transformControls.setSize?.(1.22);
  restyleTransformControlsGizmo();
  requestAnimationFrame(() => restyleTransformControlsGizmo());

  if (editMode === "translate") {
    transformControls.detach();
    syncPlanarTransformAccent();
    updateSelectionHighlight3D();
    syncFloorPlanEditCursor();
    if (isFloorPlanVisible()) drawFloorPlanView();
    return;
  }

  if (selected?.id && !isSurfaceTextureOnlyObject(selected)) {
    attachTransform(selected.id);
  } else {
    transformControls.detach();
  }
  updateSelectionHighlight3D();
  syncFloorPlanEditCursor();
  if (isFloorPlanVisible()) drawFloorPlanView();
}

function setPickMouseFromEvent(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  _pickMouse.set(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );
}

function intersectMovePlaneFromEvent(event, planeY) {
  if (!renderer || !camera) return null;
  setPickMouseFromEvent(event);
  _pickRaycaster.setFromCamera(_pickMouse, camera);
  _movePlane.set(_movePlaneNormal, -planeY);
  return _pickRaycaster.ray.intersectPlane(_movePlane, _movePlaneHit) ? _movePlaneHit.clone() : null;
}

const _movePlaneNormal = new THREE.Vector3(0, 1, 0);

function shiftPlanarPoint2d(point, du, dv) {
  return [Number(point[0]) + du, Number(point[1]) + dv];
}

function clonePlanarObjectState(obj) {
  if (!obj) return null;
  const pos = normalizePosition(obj.matrix_position ?? obj.position);
  const clone = {
    ...obj,
    matrix_position: [...pos],
    position: [...pos],
  };
  if (Array.isArray(obj.quaternion_xyzw)) clone.quaternion_xyzw = [...obj.quaternion_xyzw];

  if (obj.world_planar_rect && typeof obj.world_planar_rect === "object") {
    const rect = obj.world_planar_rect;
    clone.world_planar_rect = {
      ...rect,
      center: Array.isArray(rect.center) ? [...rect.center] : rect.center,
      corners: Array.isArray(rect.corners) ? rect.corners.map((c) => [...c]) : rect.corners,
      long_axis: rect.long_axis && typeof rect.long_axis === "object"
        ? {
            ...rect.long_axis,
            a: Array.isArray(rect.long_axis.a) ? [...rect.long_axis.a] : rect.long_axis.a,
            b: Array.isArray(rect.long_axis.b) ? [...rect.long_axis.b] : rect.long_axis.b,
            dir: Array.isArray(rect.long_axis.dir) ? [...rect.long_axis.dir] : rect.long_axis.dir,
          }
        : rect.long_axis,
      short_axis: rect.short_axis && typeof rect.short_axis === "object"
        ? { ...rect.short_axis }
        : rect.short_axis,
    };
  }

  if (Array.isArray(obj.world_basis_footprint_xz)) {
    clone.world_basis_footprint_xz = obj.world_basis_footprint_xz.map((c) => [...c]);
  }
  return clone;
}

function translateObjectPlanarFields(obj, deltaX, deltaZ) {
  const du = deltaX;
  const dv = -deltaZ;
  const next = clonePlanarObjectState(obj);
  const pos = normalizePosition(next.matrix_position ?? next.position);
  next.matrix_position = [pos[0] + deltaX, pos[1], pos[2] + deltaZ];
  next.position = [...next.matrix_position];

  const rect = next.world_planar_rect;
  if (rect && typeof rect === "object") {
    if (Array.isArray(rect.corners)) {
      rect.corners = rect.corners.map((c) => shiftPlanarPoint2d(c, du, dv));
    }
    if (Array.isArray(rect.center)) {
      rect.center = shiftPlanarPoint2d(rect.center, du, dv);
    }
    if (rect.long_axis && typeof rect.long_axis === "object") {
      if (Array.isArray(rect.long_axis.a)) rect.long_axis.a = shiftPlanarPoint2d(rect.long_axis.a, du, dv);
      if (Array.isArray(rect.long_axis.b)) rect.long_axis.b = shiftPlanarPoint2d(rect.long_axis.b, du, dv);
    }
  }

  if (Array.isArray(next.world_basis_footprint_xz)) {
    next.world_basis_footprint_xz = next.world_basis_footprint_xz.map(
      (c) => [Number(c[0]) + deltaX, Number(c[1]) + deltaZ],
    );
  }
  return next;
}

function findReplacementWrapperForObject(objectId) {
  if (!replacementUsdRoot || !objectId) return null;
  let wrapper = null;
  replacementUsdRoot.traverse((node) => {
    if (!wrapper && node.userData?.objectId === objectId) wrapper = node;
  });
  return wrapper;
}

function syncUsdOverlaysForObject(obj) {
  if (!obj) return;
  if (realModelRoot) {
    const node = findUsdNodeForObject(realModelRoot, obj);
    if (node) applyManifestTransformToNode(node, obj);
  }
  const wrapper = findReplacementWrapperForObject(obj.id);
  if (wrapper) applyManifestTransformToNode(wrapper, obj);
}

function getLiveObjectStateForPlanEditDrag() {
  if (!planeEditDrag) return null;
  const pivot = objectMeshes.get(planeEditDrag.id);
  const base = planeEditDrag.baseObject;
  if (!pivot || !base) return null;

  if (planeEditDrag.mode === "rotate") {
    const deltaYaw = pivot.rotation.y - planeEditDrag.startYaw;
    const live = rotateObjectPlanarFields(base, deltaYaw, planeEditDrag.worldCenterPlan);
    live.yaw_deg = THREE.MathUtils.radToDeg(pivot.rotation.y);
    live.quaternion_xyzw = [pivot.quaternion.x, pivot.quaternion.y, pivot.quaternion.z, pivot.quaternion.w];
    return live;
  }
  if (planeEditDrag.mode === "scale") {
    const sx = Math.abs(pivot.scale.x) / Math.max(Math.abs(planeEditDrag.startScale[0]), 1e-6);
    const sz = Math.abs(pivot.scale.z) / Math.max(Math.abs(planeEditDrag.startScale[2]), 1e-6);
    return scaleObjectPlanarFields(base, sx, sz, planeEditDrag.worldCenterPlan);
  }

  const deltaX = pivot.position.x - planeEditDrag.startPosition[0];
  const deltaZ = pivot.position.z - planeEditDrag.startPosition[2];
  return translateObjectPlanarFields(base, deltaX, deltaZ);
}

function refreshFloorPlanLiveDuringEdit() {
  if (!planeEditDrag || sessionKind !== "usdz") return;
  const live = getLiveObjectStateForPlanEditDrag();
  if (!live) return;
  const liveObjects = getAssetsObjects(objects).map((obj) => (
    obj.id === planeEditDrag.id ? live : obj
  ));
  roomPlanFloorData = buildRoomPlanFloorPayloadFromSceneObjects(liveObjects);
  updateRoomPlanDisplayState();
  drawFloorPlanView();
}

function syncPlanEditDragVisuals() {
  const live = getLiveObjectStateForPlanEditDrag();
  if (!live) return;
  syncUsdOverlaysForObject(live);
  refreshFloorPlanLiveDuringEdit();
  updateSelectionHighlight3D();
}

function restoreObjectVisualFromSnapshot(id, snapshot) {
  const pivot = objectMeshes.get(id);
  if (!pivot || !snapshot) return;
  pivot.position.set(snapshot.position[0], snapshot.position[1], snapshot.position[2]);
  pivot.quaternion.set(
    snapshot.quaternion_xyzw[0],
    snapshot.quaternion_xyzw[1],
    snapshot.quaternion_xyzw[2],
    snapshot.quaternion_xyzw[3],
  );
  pivot.scale.set(1, 1, 1);
  const obj = objects.find((o) => o.id === id);
  if (obj) syncUsdOverlaysForObject(obj);
  if (sessionKind === "usdz") {
    roomPlanFloorData = buildRoomPlanFloorPayloadFromSceneObjects(getAssetsObjects(objects));
    updateRoomPlanDisplayState();
    drawFloorPlanView();
  }
  updateSelectionHighlight3D();
}

function applyPlaneMovePivotPosition(pivot, x, z, planeY) {
  pivot.position.set(x, planeY, z);
  el("x").value = pivot.position.x.toFixed(4);
  el("y").value = pivot.position.y.toFixed(4);
  el("z").value = pivot.position.z.toFixed(4);
  syncPlanEditDragVisuals();
}

function startPlaneEditDrag(event, obj, options = {}) {
  const pivot = objectMeshes.get(obj.id);
  if (!pivot || isSurfaceTextureOnlyObject(obj)) return;

  const mode = options.mode || editMode;
  const source = options.source === "floorplan" ? "floorplan" : "viewer";
  let offsetX = 0;
  let offsetZ = 0;
  if (source === "viewer" && mode === "translate") {
    const hit = intersectMovePlaneFromEvent(event, pivot.position.y);
    if (!hit) return;
    offsetX = pivot.position.x - hit.x;
    offsetZ = pivot.position.z - hit.z;
  }

  const drag = {
    id: obj.id,
    mode,
    source,
    planeY: pivot.position.y,
    offsetX,
    offsetZ,
    pointerId: event.pointerId,
    before: snapshotObject(obj.id),
    startPosition: [pivot.position.x, pivot.position.y, pivot.position.z],
    startClientX: event.clientX,
    startClientY: event.clientY,
    baseObject: clonePlanarObjectState(obj),
    floorPlanMapping: source === "floorplan"
      ? (options.floorPlanMapping || floorPlanPaintMapping)
      : null,
    worldCenterPlan: getObjectPlanCenterFromState(obj),
  };

  if (mode === "rotate" && source === "floorplan") {
    const canvas = el("dxfPlanCanvas");
    if (!canvas || !drag.floorPlanMapping) return;
    const [cx, cy] = getCanvasPointerPosition(event, canvas);
    const centerPlan = getFloorPlanDisplayCenter(obj.id);
    drag.centerPlan = centerPlan;
    drag.startYaw = pivot.rotation.y;
    drag.startAnglePlan = planAngleFromCanvas(cx, cy, centerPlan, drag.floorPlanMapping);
  } else if (mode === "scale" && source === "floorplan") {
    const canvas = el("dxfPlanCanvas");
    if (!canvas || !drag.floorPlanMapping) return;
    const [cx, cy] = getCanvasPointerPosition(event, canvas);
    const centerPlan = getFloorPlanDisplayCenter(obj.id);
    const pointerPlan = canvasToPlanUv(cx, cy, drag.floorPlanMapping);
    drag.centerPlan = centerPlan;
    drag.startScale = [pivot.scale.x, pivot.scale.y, pivot.scale.z];
    const displayItem = roomPlanDisplayState?.items?.find((entry) => entry.id === obj.id);
    const { longDir, shortDir } = getPlanItemAxisDirs(displayItem || { long_angle_deg: obj.yaw_deg });
    drag.longDir = longDir;
    drag.shortDir = shortDir;
    const du = pointerPlan[0] - centerPlan[0];
    const dv = pointerPlan[1] - centerPlan[1];
    drag.startDistLong = Math.max(Math.abs(du * longDir[0] + dv * longDir[1]), 0.08);
    drag.startDistShort = Math.max(Math.abs(du * shortDir[0] + dv * shortDir[1]), 0.08);
    drag.startDistPlan = Math.max(Math.hypot(du, dv), 0.08);
    const handle = options.handle;
    if (handle?.type === "scale-axis") {
      drag.scaleHandle = handle.axis === "short" ? "short" : "long";
    } else if (handle?.type === "scale-corner") {
      drag.scaleHandle = "uniform";
    } else {
      drag.scaleHandle = "uniform";
    }
  } else if (mode !== "translate") {
    return;
  }

  planeEditDrag = drag;
  if (controls) controls.enabled = false;
  const captureEl = source === "floorplan" ? el("dxfPlanCanvas") : renderer?.domElement;
  captureEl?.setPointerCapture?.(event.pointerId);
  event.preventDefault();
}

function startPlaneMoveDrag(event, obj, options = {}) {
  startPlaneEditDrag(event, obj, { ...options, mode: "translate" });
}

function updatePlaneEditDrag(event) {
  if (!planeEditDrag || event.pointerId !== planeEditDrag.pointerId) return;
  const pivot = objectMeshes.get(planeEditDrag.id);
  if (!pivot) return;

  if (planeEditDrag.mode === "rotate" && planeEditDrag.source === "floorplan") {
    const canvas = el("dxfPlanCanvas");
    const mapping = planeEditDrag.floorPlanMapping;
    if (!canvas || !mapping) return;
    const [cx, cy] = getCanvasPointerPosition(event, canvas);
    const angle = planAngleFromCanvas(cx, cy, planeEditDrag.centerPlan, mapping);
    let delta = angle - planeEditDrag.startAnglePlan;
    delta = Math.atan2(Math.sin(delta), Math.cos(delta));
    pivot.rotation.set(0, planeEditDrag.startYaw + delta, 0);
    pivot.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), pivot.rotation.y);
    el("yaw").value = THREE.MathUtils.radToDeg(pivot.rotation.y).toFixed(2);
    syncPlanEditDragVisuals();
    return;
  }

  if (planeEditDrag.mode === "scale" && planeEditDrag.source === "floorplan") {
    const canvas = el("dxfPlanCanvas");
    const mapping = planeEditDrag.floorPlanMapping;
    if (!canvas || !mapping) return;
    const [cx, cy] = getCanvasPointerPosition(event, canvas);
    const pointerPlan = canvasToPlanUv(cx, cy, mapping);
    const [cu, cv] = planeEditDrag.centerPlan;
    const du = pointerPlan[0] - cu;
    const dv = pointerPlan[1] - cv;
    const longDir = planeEditDrag.longDir || [1, 0];
    const shortDir = planeEditDrag.shortDir || [0, 1];
    const distLong = Math.max(Math.abs(du * longDir[0] + dv * longDir[1]), 0.05);
    const distShort = Math.max(Math.abs(du * shortDir[0] + dv * shortDir[1]), 0.05);
    const distRadial = Math.max(Math.hypot(du, dv), 0.05);
    const factorLong = clamp(distLong / planeEditDrag.startDistLong, 0.05, 20);
    const factorShort = clamp(distShort / planeEditDrag.startDistShort, 0.05, 20);
    const factorUniform = clamp(distRadial / planeEditDrag.startDistPlan, 0.05, 20);
    if (planeEditDrag.scaleHandle === "long") {
      pivot.scale.set(
        planeEditDrag.startScale[0] * factorLong,
        1,
        planeEditDrag.startScale[2],
      );
    } else if (planeEditDrag.scaleHandle === "short") {
      pivot.scale.set(
        planeEditDrag.startScale[0],
        1,
        planeEditDrag.startScale[2] * factorShort,
      );
    } else {
      pivot.scale.set(
        planeEditDrag.startScale[0] * factorUniform,
        1,
        planeEditDrag.startScale[2] * factorUniform,
      );
    }
    syncPlanEditDragVisuals();
    return;
  }

  if (planeEditDrag.source === "floorplan") {
    const mapping = planeEditDrag.floorPlanMapping;
    if (!mapping) return;
    const dx = event.clientX - planeEditDrag.startClientX;
    const dy = event.clientY - planeEditDrag.startClientY;
    const [duDisplay, dvDisplay] = planUvDeltaFromCanvasDelta(dx, dy, mapping);
    const [deltaX, deltaZ] = displayPlanDeltaToWorldXz(duDisplay, dvDisplay, mapping.transform);
    applyPlaneMovePivotPosition(
      pivot,
      planeEditDrag.startPosition[0] + deltaX,
      planeEditDrag.startPosition[2] + deltaZ,
      planeEditDrag.planeY,
    );
    return;
  }

  const hit = intersectMovePlaneFromEvent(event, planeEditDrag.planeY);
  if (!hit) return;

  applyPlaneMovePivotPosition(
    pivot,
    hit.x + planeEditDrag.offsetX,
    hit.z + planeEditDrag.offsetZ,
    planeEditDrag.planeY,
  );
}

function updatePlaneMoveDrag(event) {
  updatePlaneEditDrag(event);
}

async function finishPlaneTranslateDrag(drag) {
  const pivot = objectMeshes.get(drag.id);
  if (!pivot || !drag.before) return;

  const after = snapshotObject(drag.id);
  const res = await fetch(`/api/object/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      id: drag.id,
      position: [pivot.position.x, pivot.position.y, pivot.position.z],
      yaw_deg: THREE.MathUtils.radToDeg(pivot.rotation.y),
      quaternion_xyzw: [pivot.quaternion.x, pivot.quaternion.y, pivot.quaternion.z, pivot.quaternion.w],
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "update failed");
    restoreObjectVisualFromSnapshot(drag.id, drag.before);
    scheduleRoomPlanRefresh();
    return;
  }

  objects = Array.isArray(data.objects) ? data.objects : [];
  selected = objects.find((o) => o.id === drag.id) || null;
  renderObjectList();
  fillEditor(selected);
  buildProxyScene(objects, { fitCamera: false });
  if (realModelRoot) syncUsdOverlayTransforms(realModelRoot, objects);
  if (selected) attachTransform(selected.id);
  scheduleRoomPlanRefresh();

  if (after) {
    pushTransformUndo({
      label: drag.source === "floorplan" ? "Move (plan)" : "Move (3D)",
      changes: [{ id: drag.id, before: drag.before, after }],
    });
  }
}

async function finishPlaneRotateDrag(drag) {
  const pivot = objectMeshes.get(drag.id);
  if (!pivot || !drag.before) return;

  const after = snapshotObject(drag.id);
  const res = await fetch(`/api/object/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      id: drag.id,
      position: [pivot.position.x, pivot.position.y, pivot.position.z],
      yaw_deg: THREE.MathUtils.radToDeg(pivot.rotation.y),
      quaternion_xyzw: [pivot.quaternion.x, pivot.quaternion.y, pivot.quaternion.z, pivot.quaternion.w],
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "update failed");
    restoreObjectVisualFromSnapshot(drag.id, drag.before);
    scheduleRoomPlanRefresh();
    return;
  }

  objects = Array.isArray(data.objects) ? data.objects : [];
  selected = objects.find((o) => o.id === drag.id) || null;
  renderObjectList();
  fillEditor(selected);
  buildProxyScene(objects, { fitCamera: false });
  if (realModelRoot) syncUsdOverlayTransforms(realModelRoot, objects);
  if (selected) attachTransform(selected.id);
  scheduleRoomPlanRefresh();

  if (after) {
    pushTransformUndo({
      label: drag.source === "floorplan" ? "Rotate (plan)" : "Rotate (3D)",
      changes: [{ id: drag.id, before: drag.before, after }],
    });
  }
}

async function finishPlaneScaleDrag(drag) {
  const pivot = objectMeshes.get(drag.id);
  if (!pivot || !drag.before) return;
  const sx = Math.abs(pivot.scale.x);
  const sz = Math.abs(pivot.scale.z);
  if (Math.abs(sx - 1) < 1e-4 && Math.abs(sz - 1) < 1e-4) {
    pivot.scale.set(1, 1, 1);
    scheduleRoomPlanRefresh();
    return;
  }
  await commitScaleTransform(drag.id, drag.before, {
    undoLabel: drag.source === "floorplan" ? "Transform (plan)" : "Transform (3D)",
  });
}

async function endPlaneEditDrag(event) {
  if (!planeEditDrag || event.pointerId !== planeEditDrag.pointerId) return;

  const drag = planeEditDrag;
  planeEditDrag = null;
  if (controls) controls.enabled = true;
  const captureEl = drag.source === "floorplan" ? el("dxfPlanCanvas") : renderer?.domElement;
  captureEl?.releasePointerCapture?.(event.pointerId);

  if (drag.mode === "scale") {
    await finishPlaneScaleDrag(drag);
    return;
  }
  if (drag.mode === "rotate") {
    await finishPlaneRotateDrag(drag);
    return;
  }
  await finishPlaneTranslateDrag(drag);
}

async function endPlaneMoveDrag(event) {
  await endPlaneEditDrag(event);
}

async function onSnapBottomsToFloor(event) {
  event.preventDefault();
  if (!sessionId) {
    alert("先にUSDZを読み込んでください");
    return;
  }
  const res = await fetch(`/api/snap-bottoms-to-floor/${sessionId}`, { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "snap bottoms to floor failed");
    return;
  }
  const undoItems = Array.isArray(data.undo_items) ? data.undo_items : [];
  if (undoItems.length) {
    pushUndo({ label: "Snap bottoms to floor", kind: "geometry", items: undoItems });
  }
  objects = Array.isArray(data.objects) ? data.objects : [];
  selected = selected?.id ? objects.find((o) => o.id === selected.id) || null : null;
  renderObjectList();
  fillEditor(selected);
  await reloadUsdOverlayAndProxy(data.usdz_url, { fitCamera: false });
  scheduleRoomPlanRefresh();
  const moved = Number(data.moved ?? 0);
  const floorTopY = Number(data.floor_top_y);
  if (moved <= 0) {
    const skip = data.skip_counts || {};
    alert(
      [
        "底面を伸ばす対象がありませんでした（すでに床に接している、またはメッシュが読めない等）。",
        `床上面の参照 Y: ${Number.isFinite(floorTopY) ? floorTopY.toFixed(4) : "?"} m`,
        `skip: already_seated=${skip.already_seated ?? 0} missing_file=${skip.missing_file ?? 0}`,
      ].join("\n")
    );
    return;
  }
  alert(
    [
      `${moved} 件のオブジェクトの底面を床まで伸ばしました（位置は変更していません）。`,
      `床上面の参照 Y: ${floorTopY.toFixed(4)} m`,
    ].join("\n")
  );
}

async function onAlignWalls(event) {
  event.preventDefault();
  if (!sessionId) {
    alert("先にUSDZを読み込んでください");
    return;
  }
  const beforeById = snapshotById(objects);
  const res = await fetch(`/api/align-walls/${sessionId}`, { method: "POST" });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "wall alignment failed");
    return;
  }
  objects = Array.isArray(data.objects) ? data.objects : [];
  selected = null;
  lastAlignedWallIds = new Set(Array.isArray(data.moved_ids) ? data.moved_ids : []);
  renderObjectList();
  await reloadUsdOverlayAndProxy(data.usdz_url, { fitCamera: false });
  scheduleRoomPlanRefresh();
  const movedIds = Array.isArray(data.moved_ids) ? data.moved_ids : [];
  if (movedIds.length) {
    const afterById = snapshotById(objects);
    const changes = movedIds
      .map((id) => ({ id, before: beforeById.get(id), after: afterById.get(id) }))
      .filter((c) => c.before && c.after);
    if (changes.length) pushUndo({ label: "Align walls", changes });
  }
  setTimeout(() => {
    lastAlignedWallIds = new Set();
  }, 2500);
  const moved = Number(data.moved ?? 0);
  const totalShift = Number(data.total_shift ?? 0);
  const maxShift = Number(data.max_shift ?? 0);
  if (moved <= 0) {
    const hint =
      typeof data.align_hint === "string" && data.align_hint
        ? data.align_hint
        : "壁は動きませんでした。床の検出や床─壁の距離を確認してください。";
    alert(hint);
    return;
  }
  alert(
    [
      `Aligned ${moved} wall object(s).`,
      `Total shift: ${totalShift.toFixed(4)} m`,
      `Max shift: ${maxShift.toFixed(4)} m`,
    ].join("\n")
  );
}

function applyLayersFromResponse(layers) {
  layerDefs = Array.isArray(layers) ? layers : [];
  renderLayerPanel();
}

async function loadExportFormats() {
  try {
    const res = await fetch("/api/export-formats");
    const data = await res.json();
    exportFormatDefs = Array.isArray(data.formats) ? data.formats : [];
  } catch (err) {
    console.warn("loadExportFormats failed:", err);
    exportFormatDefs = [];
  }
  renderExportPanel();
}

async function exportSessionFormat(formatId, filename) {
  if (!sessionId) return;
  const url = `/api/export/${encodeURIComponent(sessionId)}/${encodeURIComponent(formatId)}`;
  const res = await fetch(url);
  if (!res.ok) {
    let message = `Export failed (${res.status})`;
    try {
      const data = await res.json();
      if (data?.error) message = data.error;
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(message);
  }
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename || "export.dxf";
  anchor.rel = "noopener";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(objectUrl);
}

function renderExportPanel() {
  const box = el("exportFormats");
  if (!box) return;
  if (!sessionId || sessionKind !== "usdz") {
    box.innerHTML = '<div class="meta">Import USDZ to export as DXF.</div>';
    return;
  }
  const dxfFormats = exportFormatDefs.filter((fmt) => fmt.id === "dxf");
  if (!dxfFormats.length) {
    box.innerHTML = '<div class="meta">Loading DXF export…</div>';
    return;
  }
  const list = document.createElement("div");
  list.className = "export-format-list";
  for (const fmt of dxfFormats) {
    const row = document.createElement("div");
    row.className = `export-format-row${fmt.available ? "" : " is-unavailable"}`;
    const head = document.createElement("div");
    head.className = "export-format-head";
    if (fmt.available) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "button export-format-btn";
      btn.textContent = `.${fmt.extension}`;
      btn.addEventListener("click", async () => {
        const original = btn.textContent;
        btn.disabled = true;
        btn.textContent = "Exporting…";
        try {
          await exportSessionFormat(fmt.id, fmt.filename || `edited_room.${fmt.extension}`);
        } catch (err) {
          console.error("export failed:", err);
          alert(err?.message || "Export failed");
        } finally {
          btn.disabled = false;
          btn.textContent = original;
        }
      });
      head.appendChild(btn);
    } else {
      const ext = document.createElement("span");
      ext.className = "export-ext-muted";
      ext.textContent = `.${fmt.extension}`;
      head.appendChild(ext);
    }
    const title = document.createElement("strong");
    title.textContent = fmt.label;
    head.appendChild(title);
    row.appendChild(head);
    const desc = document.createElement("div");
    desc.className = "meta export-format-desc";
    desc.textContent = fmt.available
      ? fmt.description
      : `${fmt.description}（${fmt.unavailable_reason || "利用不可"}）`;
    row.appendChild(desc);
    list.appendChild(row);
  }
  box.innerHTML = "";
  box.appendChild(list);
}

async function loadLayers() {
  if (!sessionId) {
    layerDefs = [];
    renderLayerPanel();
    return;
  }
  const res = await fetch(`/api/layers/${sessionId}`);
  const data = await res.json();
  if (!res.ok) {
    console.warn("load layers failed:", data.error || res.status);
    return;
  }
  applyLayersFromResponse(data.layers);
}

function getObjectLayerId(obj) {
  return getPrimaryObjectTag(obj);
}

function getLayerDef(layerId) {
  return layerDefs.find((layer) => layer.id === layerId) || null;
}

function isLayerVisible(layerId) {
  const layer = getLayerDef(layerId);
  if (layer) return Boolean(layer.visible);
  const defaults = {
    floor: true,
    wall: true,
    door: true,
    window: true,
    opening: true,
    chair: true,
    table: true,
    storage: true,
    other: true,
  };
  return defaults[layerId] ?? true;
}

function isObjectLayerLocked(obj) {
  const layer = getLayerDef(getObjectLayerId(obj));
  return Boolean(layer?.locked);
}

function getLayerZOrder(layerId) {
  const layer = getLayerDef(layerId);
  if (layer && Number.isFinite(layer.z_order)) return layer.z_order;
  const defaults = { floor: 0, wall: 10, door: 20, window: 20, opening: 20, chair: 30, table: 31, storage: 32, other: 99 };
  return defaults[layerId] ?? 50;
}

function sortByLayerZOrder(items) {
  return [...items].sort(
    (a, b) => getLayerZOrder(getObjectLayerId(a.raw || a)) - getLayerZOrder(getObjectLayerId(b.raw || b))
  );
}

function isAssetsObject(obj) {
  const rel = String(obj?.path || obj?.id || "").replace(/\\/g, "/").replace(/^\/+/, "");
  return rel.startsWith("assets/");
}

function getAssetsObjects(list = objects) {
  return (list || []).filter((obj) => isAssetsObject(obj));
}

function renderLayerPanel() {
  const box = el("layers");
  if (!box) return;
  box.innerHTML = "";
  if (sessionKind === "dxf") {
    box.innerHTML = `<div class="meta">DXF sessions have no object layers (3D line view only).</div>`;
    return;
  }
  if (!layerDefs.length) {
    box.innerHTML = `<div class="meta">Import a RoomPlan file to populate layers.</div>`;
    return;
  }

  const list = document.createElement("div");
  list.className = "layer-list";

  for (const layer of layerDefs.filter((l) => l.id !== "other")) {
    const label = escapeHtml(layer.label || layer.id);
    const row = document.createElement("div");
    row.className = "layer-item" + (layer.visible ? "" : " is-hidden");
    row.innerHTML = `
      <label class="layer-check" title="Show / hide layer">
        <input type="checkbox" data-layer-visible="${escapeHtml(layer.id)}" ${layer.visible ? "checked" : ""} aria-label="Show ${label}">
        <span class="layer-check-box" aria-hidden="true"></span>
      </label>
      <span class="layer-name">${label}</span>
      ${
        layer.locked
          ? `<span class="layer-lock" title="Locked" aria-label="Locked">
              <svg class="layer-lock-icon" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                <path d="M4.5 7V5a3.5 3.5 0 1 1 7 0v2" stroke="currentColor" stroke-width="1.25" stroke-linecap="round"/>
                <rect x="3.25" y="7" width="9.5" height="6.5" rx="1.2" stroke="currentColor" stroke-width="1.25"/>
              </svg>
            </span>`
          : ""
      }
    `;
    list.appendChild(row);
  }

  box.appendChild(list);

  box.querySelectorAll("input[data-layer-visible]").forEach((input) => {
    input.addEventListener("change", () => {
      const layerId = input.getAttribute("data-layer-visible");
      if (!layerId) return;
      void setLayerVisibility(layerId, input.checked);
    });
  });
}

async function setLayerVisibility(layerId, visible) {
  if (!sessionId) return;
  const res = await fetch(`/api/layers/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ layer_id: layerId, visible }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "layer update failed");
    await loadLayers();
    return;
  }
  layerDefs = Array.isArray(data.layers) ? data.layers : layerDefs;
  objects = Array.isArray(data.objects) ? data.objects : objects;
  if (selected && !shouldRenderObjectInScene(selected)) {
    selected = null;
    transformControls?.detach();
    el("editor")?.classList.add("disabled");
  }
  renderLayerPanel();
  renderObjectList();
  applyInteriorVisibility();
}


function clearWallExtrusion() {
  wallExtrudeData = null;
  if (!wallExtrudeRoot || !scene) {
    wallExtrudeRoot = null;
    updateWallExtrudeInfo();
    updateWallExtrusionInfo();
    return;
  }
  scene.remove(wallExtrudeRoot);
  disposeObject(wallExtrudeRoot);
  wallExtrudeRoot = null;
  updateWallExtrudeInfo();
  updateWallExtrusionInfo();
}

function isFocus3dViewerEnabled() {
  const input = el("focus3dViewer");
  return input instanceof HTMLInputElement && input.checked;
}

function getPrimaryViewerRoot() {
  if (wallExtrudeRoot?.visible) return wallExtrudeRoot;
  if (modelRoot?.visible) return modelRoot;
  if (realModelRoot?.visible) return realModelRoot;
  if (dxfRoot?.visible) return dxfRoot;
  return wallExtrudeRoot || modelRoot || realModelRoot || dxfRoot || null;
}

function syncFocus3dViewer({ fitCamera: shouldFit = true } = {}) {
  const focused = isFocus3dViewerEnabled() && !walkModeActive;
  document.querySelector("main")?.classList.toggle("main-viewer-focus", focused);
  document.querySelector(".panel-3d")?.classList.toggle("panel-3d-immersive", focused);
  resizeViewer();
  if (shouldFit && !walkModeActive) {
    const root = getPrimaryViewerRoot();
    if (root) fitCamera(root);
  }
  updateViewerOverlayHint();
}

function canUseWalkMode() {
  return Boolean(realModelRoot && walkCollisionMeshes.length);
}

function rebuildWalkCollisionMeshes() {
  walkCollisionMeshes = [];
  if (!realModelRoot) return;
  realModelRoot.traverse((node) => {
    if (node?.isMesh && node.geometry) walkCollisionMeshes.push(node);
  });
}

function getWalkFeetPosition() {
  const cam = walkCamera || camera;
  return cam.position.clone().addScaledVector(WALK_UP, -WALK_EYE_HEIGHT);
}

function setWalkFeetPosition(feet) {
  const cam = walkCamera || camera;
  cam.position.copy(feet).addScaledVector(WALK_UP, WALK_EYE_HEIGHT);
}

function syncWalkPanelLayout() {
  const active = walkModeActive;
  document.querySelector("main")?.classList.toggle("main-viewer-focus", active);
  document.querySelector(".panel-3d")?.classList.toggle("panel-3d-immersive", active);
  document.body.classList.toggle("walk-mode-active", active);
  if (active) bindWalkWheelHandlers();
  else unbindWalkWheelHandlers();
  resizeViewer();
}

function ensureWalkPlayerMarker() {
  if (!scene || walkPlayerMarker) return;
  walkPlayerMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.18, 14, 14),
    new THREE.MeshBasicMaterial({ color: 0x2ecc71 })
  );
  walkPlayerMarker.visible = false;
  scene.add(walkPlayerMarker);
}

function updateWalkPlayerMarker() {
  ensureWalkPlayerMarker();
  if (!walkPlayerMarker || !walkCamera) return;
  walkPlayerMarker.visible = walkModeActive;
  if (!walkModeActive) return;
  const feet = getWalkFeetPosition();
  walkPlayerMarker.position.copy(feet).addScaledVector(WALK_UP, 0.9);
}

function probeWalkFloorColumn(x, z, topY, bottomY) {
  if (!walkCollisionMeshes.length) return null;
  const origin = new THREE.Vector3(x, topY + 2, z);
  _walkRaycaster.set(origin, _walkDown);
  _walkRaycaster.far = Math.max(topY - bottomY + 10, 5);
  const hits = _walkRaycaster.intersectObjects(walkCollisionMeshes, true);
  if (!hits.length) return null;
  let floorY = hits[0].point.y;
  for (const hit of hits) floorY = Math.min(floorY, hit.point.y);
  return floorY;
}

function pickWalkGroundY(hits, referenceY) {
  const maxStandY = referenceY + WALK_GROUND_EPS;
  let groundY = null;
  for (const hit of hits) {
    const y = hit.point.y;
    if (y > maxStandY) continue;
    groundY = groundY === null ? y : Math.max(groundY, y);
  }
  return groundY;
}

function probeWalkGround(x, z, referenceY) {
  if (!walkCollisionMeshes.length) return null;
  const origin = new THREE.Vector3(x, referenceY + WALK_FEET_PROBE_LIFT, z);
  _walkRaycaster.set(origin, _walkDown);
  _walkRaycaster.far = WALK_GROUND_PROBE_DEPTH;
  const hits = _walkRaycaster.intersectObjects(walkCollisionMeshes, true);
  if (!hits.length) return null;
  return pickWalkGroundY(hits, referenceY);
}

function autoSetWalkSpawnDefault() {
  if (!canUseWalkMode()) return false;
  const box = new THREE.Box3().setFromObject(realModelRoot);
  if (box.isEmpty()) return false;
  const center = box.getCenter(new THREE.Vector3());
  const inset = 0.2;
  const sampleXZ = [
    [center.x, center.z],
    [THREE.MathUtils.lerp(box.min.x, box.max.x, inset), THREE.MathUtils.lerp(box.min.z, box.max.z, inset)],
    [THREE.MathUtils.lerp(box.min.x, box.max.x, 1 - inset), THREE.MathUtils.lerp(box.min.z, box.max.z, inset)],
    [THREE.MathUtils.lerp(box.min.x, box.max.x, inset), THREE.MathUtils.lerp(box.min.z, box.max.z, 1 - inset)],
    [THREE.MathUtils.lerp(box.min.x, box.max.x, 1 - inset), THREE.MathUtils.lerp(box.min.z, box.max.z, 1 - inset)],
  ];
  let bestFeet = null;
  let bestScore = Infinity;
  for (const [x, z] of sampleXZ) {
    const floorY = probeWalkFloorColumn(x, z, box.max.y, box.min.y);
    const y = floorY !== null ? floorY : box.min.y;
    const score = Math.abs(y - box.min.y);
    if (score < bestScore) {
      bestScore = score;
      bestFeet = new THREE.Vector3(x, y, z);
    }
  }
  if (!bestFeet) return false;
  const size = box.getSize(new THREE.Vector3());
  const yaw = size.x >= size.z ? 0 : Math.PI / 2;
  walkSpawn = { position: bestFeet, yaw };
  walkSpawnIsAuto = true;
  walkSpawnPickActive = false;
  return true;
}

function isWalkHorizontalBlocked(nextFeet, currentFeet) {
  const move = nextFeet.clone().sub(currentFeet);
  move.y = 0;
  const dist = move.length();
  if (dist < 1e-6) return false;
  move.normalize();
  const heights = [0.35, 1.0, 1.45];
  for (const h of heights) {
    const origin = currentFeet.clone().addScaledVector(WALK_UP, h);
    _walkRaycaster.set(origin, move);
    _walkRaycaster.far = dist + WALK_PLAYER_RADIUS;
    const hits = _walkRaycaster.intersectObjects(walkCollisionMeshes, true);
    if (hits.length && hits[0].distance < dist + WALK_PLAYER_RADIUS - 0.02) return true;
  }
  return false;
}

function tryWalkMoveHorizontal(delta) {
  const feet = getWalkFeetPosition();
  for (const axis of ["x", "z"]) {
    if (Math.abs(delta[axis]) < 1e-8) continue;
    const next = feet.clone();
    next[axis] += delta[axis];
    if (!isWalkHorizontalBlocked(next, feet)) feet[axis] = next[axis];
  }
  setWalkFeetPosition(feet);
}

function showWalkFeedback(message) {
  const status = el("walkModeStatus");
  if (!status) return;
  const text = message || "";
  status.textContent = text;
  status.hidden = !text;
}

function bindWalkModeUI() {
  publishWalkAPI();
  const root = el("walkModeControls");
  if (!root || root.dataset.walkBound === "1") return;
  root.dataset.walkBound = "1";
  root.addEventListener("click", onWalkControlsClick);
}

function publishWalkAPI() {
  window.__topViewerWalk = {
    toggleWalkMode,
    toggleWalkSpawnPick,
    enterWalkMode,
    exitWalkMode,
    canUseWalkMode,
  };
}

function onWalkControlsClick(event) {
  const btn = event.target instanceof Element ? event.target.closest("button") : null;
  if (!btn || !el("walkModeControls")?.contains(btn)) return;
  event.preventDefault();
  event.stopPropagation();
  try {
    switch (btn.id) {
      case "walkPickSpawn":
        toggleWalkSpawnPick();
        break;
      case "walkModeToggle":
        showWalkFeedback(walkModeActive ? "Walkthrough を終了中…" : "Walkthrough を開始中…");
        toggleWalkMode();
        break;
      default:
        break;
    }
  } catch (err) {
    console.error("walk control failed:", err);
    showWalkFeedback(`操作エラー: ${err?.message || err}`);
    alert(`Walk mode error: ${err?.message || err}`);
  }
}

function ensureWalkViewerReady() {
  if (walkCamera && pointerLockControls) return true;
  try {
    initWalkViewer();
  } catch (err) {
    console.error("ensureWalkViewerReady failed:", err);
    showWalkFeedback(`1人称ビュー初期化エラー: ${err?.message || err}`);
    return false;
  }
  return Boolean(walkCamera && pointerLockControls);
}

function formatWalkSpawnLabel() {
  if (!walkSpawn) return "未設定";
  const p = walkSpawn.position;
  const yawDeg = ((walkSpawn.yaw * 180) / Math.PI).toFixed(0);
  const source = walkSpawnIsAuto ? "自動" : "手動";
  return `${source} (${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)}) yaw ${yawDeg}°`;
}

function syncWalkModeUI() {
  const walkReady = canUseWalkMode();
  const pickBtn = el("walkPickSpawn");
  const toggleBtn = el("walkModeToggle");
  const status = el("walkModeStatus");

  if (pickBtn) {
    pickBtn.classList.toggle("is-armed", walkSpawnPickActive);
    const pickLabel = pickBtn.querySelector(".walkthrough-btn-label");
    const labelText = walkSpawnPickActive ? "Waiting…" : "Click mesh";
    if (pickLabel) pickLabel.textContent = labelText;
    else pickBtn.textContent = labelText;
    pickBtn.classList.toggle("is-muted", walkModeActive);
  }
  if (toggleBtn) {
    toggleBtn.textContent = walkModeActive ? "Exit Walkthrough" : "Start Walkthrough";
    toggleBtn.classList.toggle("is-muted", !walkModeActive && !walkReady);
  }

  if (!status) return;

  if (walkSpawnPickActive) {
    if (walkReady) {
      status.textContent = "";
      status.hidden = true;
    } else {
      status.textContent =
        "クリック待ちですが、USDZオーバーレイが未読込のためメッシュを選択できません。下記を確認してください。";
      status.hidden = false;
    }
    return;
  }

  if (loadingModel) {
    status.textContent = "USDZオーバーレイを読み込み中…";
    status.hidden = false;
    return;
  }

  if (!walkReady) {
    if (sessionKind === "usdz" && lastLoadedUsdZUrl && !realModelRoot) {
      status.textContent =
        "USDZオーバーレイの読込に失敗しています。「Show USDZ model overlay」にチェックを入れ、USDZを再インポートしてください。";
    } else if (sessionKind === "usdz") {
      status.textContent = "USDZオーバーレイの読込完了をお待ちください。";
    } else {
      status.textContent = "USDZインポート後に利用できます。";
    }
    status.hidden = false;
    return;
  }

  if (!walkSpawn) {
    status.textContent =
      "スポーン未設定 — オーバーレイ読込時に自動設定されます。変更する場合は下のボタンを使ってください。";
    status.hidden = false;
    return;
  }

  status.textContent = "";
  status.hidden = true;
}

function updateViewerOverlayHint() {
  const overlay = el("viewerOverlay");
  if (!overlay) return;
  if (walkModeActive) {
    overlay.innerHTML = "";
    overlay.hidden = true;
    return;
  }
  if (walkSpawnPickActive) {
    overlay.innerHTML = "";
    overlay.hidden = true;
    return;
  }
  overlay.innerHTML = "";
  overlay.hidden = true;
}

function materialIndexForIntersect(hit) {
  const mesh = hit.object;
  if (!mesh?.isMesh) return 0;
  if (hit.face?.materialIndex != null) return hit.face.materialIndex;
  if (hit.faceIndex != null && mesh.geometry?.groups?.length) {
    const triStart = hit.faceIndex * 3;
    for (const group of mesh.geometry.groups) {
      if (triStart >= group.start && triStart < group.start + group.count) {
        return group.materialIndex ?? 0;
      }
    }
  }
  return 0;
}

function materialForIntersect(hit) {
  const mesh = hit.object;
  if (!mesh?.isMesh) return null;
  const materials = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
  if (!materials.length) return null;
  const index = materialIndexForIntersect(hit);
  return materials[index] ?? materials[0] ?? null;
}

function isTexturedWalkSpawnHit(hit) {
  return Boolean(materialForIntersect(hit)?.map);
}

function collectWalkSpawnPickTargets() {
  const targets = collectScenePickTargets();
  if (targets.length) return targets;
  return walkCollisionMeshes.filter((mesh) => mesh?.visible !== false);
}

function raycastWalkMeshes(event) {
  if (!renderer || !camera) return null;
  const targets = collectWalkSpawnPickTargets();
  if (!targets.length) return null;
  const rect = renderer.domElement.getBoundingClientRect();
  const mouse = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1
  );
  _walkRaycaster.setFromCamera(mouse, camera);
  const hits = _walkRaycaster.intersectObjects(targets, true);
  for (const hit of hits) {
    if (isTexturedWalkSpawnHit(hit)) return hit;
  }
  return null;
}

function setWalkSpawnFromClick(event) {
  event.preventDefault();
  event.stopPropagation();
  const hit = raycastWalkMeshes(event);
  if (!hit) {
    showWalkFeedback("メッシュに当たりませんでした。左の3Dビューでモデル上をクリックしてください。");
    return;
  }
  const feet = hit.point.clone();
  const euler = new THREE.Euler(0, 0, 0, "YXZ");
  euler.setFromQuaternion(camera.quaternion);
  walkSpawn = { position: feet.clone(), yaw: euler.y };
  walkSpawnIsAuto = false;
  walkSpawnPickActive = false;
  if (controls && !walkModeActive) controls.enabled = true;
  syncWalkModeUI();
  updateViewerOverlayHint();
}

function toggleWalkSpawnPick() {
  if (walkModeActive) {
    showWalkFeedback("Walkthrough 中はスポーンを変更できません。先に終了してください。");
    return;
  }
  if (walkSpawnPickActive) {
    walkSpawnPickActive = false;
    if (controls && !walkModeActive) controls.enabled = true;
    syncWalkModeUI();
    updateViewerOverlayHint();
    return;
  }
  if (!canUseWalkMode()) {
    const msg = loadingModel
      ? "USDZオーバーレイを読み込み中です。完了後に再度お試しください。"
      : "USDZオーバーレイを読み込んでからスポーンを設定してください。";
    showWalkFeedback(msg);
    return;
  }
  walkSpawnPickActive = true;
  if (controls) controls.enabled = false;
  syncWalkModeUI();
  updateViewerOverlayHint();
}

function applyWalkSpawnToCamera() {
  if (!walkSpawn) return;
  const cam = walkCamera || camera;
  cam.up.copy(WALK_UP);
  cam.position.copy(walkSpawn.position).addScaledVector(WALK_UP, WALK_EYE_HEIGHT);
  cam.rotation.set(0, walkSpawn.yaw, 0);
  walkVelocityY = 0;
  walkOnGround = true;
}

function enterWalkMode() {
  if (!canUseWalkMode()) {
    showWalkFeedback("USDZオーバーレイを読み込んでから Walkthrough を開始してください。");
    return;
  }
  if (!walkSpawn && !autoSetWalkSpawnDefault()) {
    showWalkFeedback("スポーン位置を自動設定できませんでした。USDZオーバーレイを確認してください。");
    return;
  }
  if (!ensureWalkViewerReady()) {
    showWalkFeedback("1人称ビューの初期化に失敗しました。ページを再読み込みしてください。");
    return;
  }
  walkModeActive = true;
  walkSpawnPickActive = false;
  const focus = el("focus3dViewer");
  if (focus instanceof HTMLInputElement && focus.checked) focus.checked = false;
  syncFocus3dViewer({ fitCamera: false });
  syncWalkPanelLayout();
  if (controls) controls.enabled = false;
  transformControls?.detach();
  selected = null;
  el("editor")?.classList.add("disabled");
  const showReal = el("showRealModel");
  if (showReal instanceof HTMLInputElement && !showReal.checked) {
    showReal.checked = true;
    if (realModelRoot) realModelRoot.visible = true;
  }
  applyWalkSpawnToCamera();
  resetWalkCameraFov();
  updateWalkPlayerMarker();
  if (!walkClock) walkClock = new THREE.Clock();
  else walkClock.start();
  syncWalkModeUI();
  updateViewerOverlayHint();
  showWalkFeedback("Walkthrough 開始 — 画面をクリックして視点をロックしてください。");
  resizeViewer();
}

function exitWalkMode() {
  walkModeActive = false;
  walkSpawnPickActive = false;
  walkVelocityY = 0;
  walkOnGround = false;
  resetWalkKeys();
  pointerLockControls?.unlock();
  if (controls) controls.enabled = true;
  el("editor")?.classList.toggle("disabled", !selected?.id);
  unbindWalkWheelHandlers();
  document.body.classList.remove("walk-mode-active");
  syncWalkPanelLayout();
  updateWalkPlayerMarker();
  syncWalkModeUI();
  updateViewerOverlayHint();
}

function toggleWalkMode() {
  if (walkModeActive) {
    exitWalkMode();
    return;
  }
  enterWalkMode();
  if (!walkModeActive) return;
  requestAnimationFrame(() => {
    resizeWalkViewer();
    try {
      pointerLockControls?.lock();
    } catch (err) {
      console.warn("pointer lock failed:", err);
      showWalkFeedback("Walkthrough 開始 — 画面をクリックして視点をロックしてください。");
    }
  });
}

function resetWalkKeys() {
  walkKeys.forward = false;
  walkKeys.back = false;
  walkKeys.left = false;
  walkKeys.right = false;
  walkKeys.jump = false;
}

function onWalkKeyDown(event) {
  if (!walkModeActive) return;
  switch (event.code) {
    case "KeyW":
    case "ArrowUp":
      walkKeys.forward = true;
      event.preventDefault();
      break;
    case "KeyS":
    case "ArrowDown":
      walkKeys.back = true;
      event.preventDefault();
      break;
    case "KeyA":
    case "ArrowLeft":
      walkKeys.left = true;
      event.preventDefault();
      break;
    case "KeyD":
    case "ArrowRight":
      walkKeys.right = true;
      event.preventDefault();
      break;
    case "Space":
      walkKeys.jump = true;
      event.preventDefault();
      break;
    default:
      break;
  }
}

function onWalkKeyUp(event) {
  switch (event.code) {
    case "KeyW":
    case "ArrowUp":
      walkKeys.forward = false;
      break;
    case "KeyS":
    case "ArrowDown":
      walkKeys.back = false;
      break;
    case "KeyA":
    case "ArrowLeft":
      walkKeys.left = false;
      break;
    case "KeyD":
    case "ArrowRight":
      walkKeys.right = false;
      break;
    case "Space":
      walkKeys.jump = false;
      break;
    default:
      break;
  }
}

function updateWalkMode(delta) {
  if (!walkModeActive || !walkCamera) return;

  if (walkKeys.jump && walkOnGround) {
    walkVelocityY = Math.sqrt(2 * WALK_GRAVITY * WALK_JUMP_HEIGHT);
    walkOnGround = false;
  }

  if (!walkOnGround) walkVelocityY -= WALK_GRAVITY * delta;

  walkCamera.getWorldDirection(_walkForward);
  _walkForward.y = 0;
  if (_walkForward.lengthSq() < 1e-8) _walkForward.set(0, 0, -1);
  else _walkForward.normalize();
  _walkRight.crossVectors(_walkForward, WALK_UP).normalize();

  _walkWish.set(0, 0, 0);
  if (walkKeys.forward) _walkWish.add(_walkForward);
  if (walkKeys.back) _walkWish.sub(_walkForward);
  if (walkKeys.left) _walkWish.sub(_walkRight);
  if (walkKeys.right) _walkWish.add(_walkRight);
  if (_walkWish.lengthSq() > 0) {
    _walkWish.normalize().multiplyScalar(WALK_SPEED * delta);
    tryWalkMoveHorizontal(_walkWish);
  }

  const feet = getWalkFeetPosition();
  feet.y += walkVelocityY * delta;
  const groundY = probeWalkGround(feet.x, feet.z, feet.y);
  if (groundY !== null && feet.y <= groundY + WALK_GROUND_EPS) {
    feet.y = groundY;
    walkVelocityY = 0;
    walkOnGround = true;
  } else if (groundY === null && feet.y < walkSpawn.position.y - 5) {
    applyWalkSpawnToCamera();
  }
  setWalkFeetPosition(feet);
}

function clearWallHighlight() {
  wallExtractData = null;
  if (!wallHighlightRoot || !scene) {
    wallHighlightRoot = null;
    updateWallExtractInfo();
    updateWallExtrusionInfo();
    drawFloorPlanView();
    return;
  }
  scene.remove(wallHighlightRoot);
  disposeObject(wallHighlightRoot);
  wallHighlightRoot = null;
  updateWallExtractInfo();
  updateWallExtrusionInfo();
  drawFloorPlanView();
}

function clearDxfPlanLayers() {
  dxfPlanLayers = [];
  renderDxfPlanLayerPanel();
}

function applyDxfPlanLayers(layers) {
  if (!Array.isArray(layers) || !layers.length) {
    clearDxfPlanLayers();
    return;
  }
  dxfPlanLayers = layers.map((layer) => ({
    id: layer.id,
    label: layer.label || layer.id,
    segment_count: layer.segment_count ?? 0,
    visible: layer.visible !== false,
    positions: Array.isArray(layer.positions) ? layer.positions : [],
  }));
  renderDxfPlanLayerPanel();
}

function isDxfPlanLayerVisible(layerId) {
  const layer = dxfPlanLayers.find((item) => item.id === layerId);
  return layer ? Boolean(layer.visible) : true;
}

function renderDxfPlanLayerPanel() {
  const panel = el("dxfPlanLayers");
  if (!panel) return;
  if (sessionKind === "usdz") {
    panel.hidden = true;
    panel.innerHTML = "";
    return;
  }
  if (!dxfPlanLayers.length) {
    panel.hidden = true;
    panel.innerHTML = "";
    return;
  }
  panel.hidden = false;
  panel.innerHTML = dxfPlanLayers.map((layer) => `
    <div class="layer-row${layer.visible ? "" : " is-hidden"}">
      <label class="layer-toggle" title="Show / hide layer on plan view">
        <input type="checkbox" data-dxf-layer-visible="${escapeHtml(layer.id)}" ${layer.visible ? "checked" : ""}>
        <span>表示</span>
      </label>
      <div>
        <strong>${escapeHtml(layer.label || layer.id)}</strong>
        <span class="layer-count"> (${layer.segment_count ?? 0})</span>
      </div>
    </div>
  `).join("");
  panel.querySelectorAll("input[data-dxf-layer-visible]").forEach((input) => {
    input.addEventListener("change", () => {
      const layerId = input.getAttribute("data-dxf-layer-visible");
      if (!layerId) return;
      void setDxfPlanLayerVisibility(layerId, input.checked);
    });
  });
}

async function setDxfPlanLayerVisibility(layerId, visible) {
  const layer = dxfPlanLayers.find((item) => item.id === layerId);
  if (layer) layer.visible = visible;
  renderDxfPlanLayerPanel();
  drawFloorPlanView();
  if (!sessionId || sessionKind !== "dxf") return;
  try {
    const res = await fetch(`/api/dxf-layers/${sessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ layer_id: layerId, visible }),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || "DXF layer update failed");
      return;
    }
    if (Array.isArray(data.layers)) {
      const positionsById = new Map(dxfPlanLayers.map((item) => [item.id, item.positions]));
      applyDxfPlanLayers(
        data.layers.map((item) => ({
          ...item,
          positions: positionsById.get(item.id) || [],
        })),
      );
      drawFloorPlanView();
    }
  } catch (err) {
    console.error("setDxfPlanLayerVisibility failed:", err);
  }
}

function strokeDxfPlanPositions(ctx, positions, toCanvas) {
  if (!positions?.length) return;
  ctx.beginPath();
  for (let i = 0; i < positions.length; i += 6) {
    const [cx0, cy0] = toCanvas(positions[i], positions[i + 2]);
    const [cx1, cy1] = toCanvas(positions[i + 3], positions[i + 5]);
    ctx.moveTo(cx0, cy0);
    ctx.lineTo(cx1, cy1);
  }
  ctx.stroke();
}

function clearDxfScene() {
  clearDxfPlanLayers();
  clearWallExtrusion();
  clearWallHighlight();
  dxfGeometry = null;
  if (!dxfRoot || !scene) {
    dxfRoot = null;
    drawFloorPlanView();
    return;
  }
  scene.remove(dxfRoot);
  disposeObject(dxfRoot);
  dxfRoot = null;
  drawFloorPlanView();
}

function updateWallExtractInfo(meta = wallExtractData) {
  const info = el("wallExtractInfo");
  if (!info) return;
  if (!meta?.wall_count) {
    info.textContent = sessionKind === "dxf"
      ? "DXFインポート後に「壁抽出」（平行ペア）または「ML壁比較」で検出します。"
      : "DXFインポート後に平行線分から壁を検出します。";
    return;
  }
  if (meta.source === "ml") {
    info.textContent = `ML壁: ${meta.wall_count} ポリゴン（オレンジでハイライト）`;
    return;
  }
  info.textContent = `壁: ${meta.wall_count} 本を検出（オレンジでハイライト）`;
}

function syncExtrudeWallsButton() {
  const btn = el("extrudeWalls");
  if (!(btn instanceof HTMLButtonElement)) return;
  const ready = sessionKind === "dxf" && !!sessionId;
  btn.classList.toggle("disabled", !ready);
}

function updateWallExtrudeInfo(meta = wallExtrudeData) {
  const info = el("wallExtrudeInfo");
  if (!info) return;
  if (!meta?.mesh_wall_count) {
    info.textContent = sessionKind === "dxf"
      ? "DXFインポート後、ML壁マスクから3Dへ押し出します（高さ 2.4 m）。押出比較で3経路を切り替えられます。"
      : "壁抽出後に3Dへ押し出します（高さ 2.4 m）。";
    return;
  }
  const compareLabel = meta.compare_source || meta.variant_id
    ? (meta.label || ({ heuristic: "平行ペア", ml_dxf: "ML(DXF)", ml_svg: "ML(SVG)" }[meta.compare_source]))
    : null;
  const src = compareLabel || (meta.source === "ml" ? "ML" : "平行ペア");
  info.textContent = `押出: ${meta.mesh_wall_count} 面 (${src}) · 高さ ${meta.height_m} m`;
}

function updateWallExtrusionInfo() {
  const info = el("wallExtrusionInfo");
  if (!info) return;
  const extract = wallExtractData;
  const extrude = wallExtrudeData;
  if (!extract?.wall_count && !extrude?.mesh_wall_count) {
    info.textContent = sessionKind === "dxf"
      ? "DXFインポート後、平行線分から壁を検出して3Dへ押し出します（高さ 2.4 m）。"
      : "DXFインポート後に利用できます。";
    return;
  }
  const parts = [];
  if (extract?.wall_count) {
    parts.push(`検出: ${extract.wall_count} 本（オレンジ）`);
  }
  if (extrude?.mesh_wall_count) {
    parts.push(`押出: ${extrude.mesh_wall_count} 面 · 高さ ${extrude.height_m} m`);
  }
  info.textContent = parts.join(" · ");
}

function syncWallExtrusionButton() {
  const btn = el("runWallExtrusion");
  if (!(btn instanceof HTMLButtonElement)) return;
  btn.classList.toggle("disabled", sessionKind !== "dxf" || !sessionId);
}

function syncWallVisibility() {
  const visible = el("showWall")?.checked ?? true;
  if (wallExtrudeRoot) wallExtrudeRoot.visible = visible;
  if (wallHighlightRoot) wallHighlightRoot.visible = visible;
}

function buildWallExtrusion(payload, options = {}) {
  if (!scene || !payload?.positions?.length || !payload?.indices?.length) {
    clearWallExtrusion();
    return;
  }
  clearWallExtrusion();
  wallExtrudeData = payload;
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(payload.positions, 3));
  geom.setIndex(payload.indices);
  geom.computeVertexNormals();
  const color = options.color ?? 0xc9ced6;
  const mat = new THREE.MeshLambertMaterial({ color });
  wallExtrudeRoot = new THREE.Mesh(geom, mat);
  wallExtrudeRoot.renderOrder = 1;
  scene.add(wallExtrudeRoot);
  syncWallVisibility();
  if (dxfRoot) dxfRoot.visible = false;
  if (options.fitCamera !== false) {
    fitCamera(wallExtrudeRoot);
    hasFittedCamera = true;
  }
  updateWallExtrudeInfo(payload);
  updateWallExtrusionInfo();
}

async function onRunWallExtrusion(event) {
  event.preventDefault();
  if (!sessionId || sessionKind !== "dxf") {
    alert("先にDXFをインポートしてください");
    return;
  }
  const btn = el("runWallExtrusion");
  if (btn instanceof HTMLButtonElement) btn.disabled = true;
  const info = el("wallExtrusionInfo");
  if (info) info.textContent = "平行線分から壁を検出して押し出し中…";
  try {
    const res = await fetch(`/api/wall-extrusion/${sessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    let data = {};
    try {
      data = await res.json();
    } catch {
      const hint = res.status === 404
        ? " API が見つかりません。app.py 変更後は Flask を再起動してください。"
        : "";
      throw new Error(`サーバー応答を読み取れませんでした (HTTP ${res.status})${hint}`);
    }
    if (!res.ok) {
      alert(data.error || `壁押出に失敗しました (HTTP ${res.status})`);
      if (info) updateWallExtrusionInfo();
      return;
    }
    buildWallHighlight({
      wall_count: data.wall_count,
      highlight_positions: data.highlight_positions,
      source: data.source || "heuristic",
    });
    buildWallExtrusion(data, { fitCamera: true });
  } catch (err) {
    console.error("onRunWallExtrusion failed:", err);
    alert(err?.message || "壁押出に失敗しました");
    if (info) updateWallExtrusionInfo();
  } finally {
    if (btn instanceof HTMLButtonElement) btn.disabled = false;
    syncWallExtrusionButton();
  }
}

async function onExtrudeWalls(event) {
  event.preventDefault();
  if (!sessionId || sessionKind !== "dxf") {
    alert("先にDXFをインポートしてください");
    return;
  }
  const btn = el("extrudeWalls");
  if (btn instanceof HTMLButtonElement) btn.disabled = true;
  const info = el("wallExtrudeInfo");
  if (info) info.textContent = "ML壁マスクから押し出し中…";
  try {
    const res = await fetch(`/api/extrude-walls/${sessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: "ml" }),
    });
    let data = {};
    try {
      data = await res.json();
    } catch {
      throw new Error(`サーバー応答を読み取れませんでした (HTTP ${res.status})`);
    }
    if (!res.ok) {
      alert(data.error || `壁押出に失敗しました (HTTP ${res.status})`);
      return;
    }
    buildWallExtrusion(data);
    try {
      const mlRes = await fetch(`/api/wall-extract-ml/${sessionId}`);
      if (mlRes.ok) {
        buildWallHighlight(await mlRes.json());
      }
    } catch (err) {
      console.warn("ML wall highlight load failed:", err);
    }
  } catch (err) {
    console.error("onExtrudeWalls failed:", err);
    alert(err?.message || "壁押出に失敗しました");
  } finally {
    if (btn instanceof HTMLButtonElement) btn.disabled = false;
    syncExtrudeWallsButton();
  }
}

function syncWallExtractButton() {
  const btn = el("extractWalls");
  if (!(btn instanceof HTMLButtonElement)) return;
  btn.classList.toggle("disabled", sessionKind !== "dxf" || !sessionId);
}

function buildWallHighlight(payload, options = {}) {
  if (!scene || !payload?.highlight_positions?.length) {
    clearWallHighlight();
    return;
  }
  clearWallHighlight();
  wallExtractData = payload;
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(payload.highlight_positions, 3));
  const color = options.color ?? 0xff7a00;
  const mat = new THREE.LineBasicMaterial({ color });
  wallHighlightRoot = new THREE.LineSegments(geom, mat);
  wallHighlightRoot.renderOrder = 2;
  scene.add(wallHighlightRoot);
  syncWallVisibility();
  updateWallExtractInfo(payload);
  updateWallExtrusionInfo();
  drawFloorPlanView();
}

async function onExtractWalls(event) {
  event.preventDefault();
  if (!sessionId || sessionKind !== "dxf") {
    alert("先にDXFをインポートしてください");
    return;
  }
  const btn = el("extractWalls");
  if (btn instanceof HTMLButtonElement) btn.disabled = true;
  try {
    const res = await fetch(`/api/extract-walls/${sessionId}`, { method: "POST" });
    let data = {};
    try {
      data = await res.json();
    } catch {
      throw new Error(`サーバー応答を読み取れませんでした (HTTP ${res.status})`);
    }
    if (!res.ok) {
      alert(data.error || `壁抽出に失敗しました (HTTP ${res.status})`);
      return;
    }
    buildWallHighlight(data);
    clearWallExtrusion();
    syncExtrudeWallsButton();
    updateWallExtrudeInfo(null);
  } catch (err) {
    console.error("onExtractWalls failed:", err);
    alert(err?.message || "壁抽出に失敗しました");
  } finally {
    if (btn instanceof HTMLButtonElement) btn.disabled = false;
    syncWallExtractButton();
  }
}

function syncMlPocButton() {
  const btn = el("mlPocCompare");
  if (!(btn instanceof HTMLButtonElement)) return;
  btn.classList.toggle("disabled", sessionKind !== "dxf" || !sessionId);
}

function syncMlRenderCompareButton() {
  const btn = el("mlRenderCompare");
  if (!(btn instanceof HTMLButtonElement)) return;
  btn.classList.toggle("disabled", sessionKind !== "dxf" || !sessionId);
}

function syncExtrudeCompareButton() {
  const btn = el("extrudeCompare");
  if (!(btn instanceof HTMLButtonElement)) return;
  btn.classList.toggle("disabled", sessionKind !== "dxf" || !sessionId);
}

function syncWallRefineCompareButton() {
  const btn = el("wallRefineCompare");
  if (!(btn instanceof HTMLButtonElement)) return;
  btn.classList.toggle("disabled", sessionKind !== "dxf" || !sessionId);
}

function clearExtrudeCompareUI() {
  extrudeCompareReport = null;
  extrudeCompareActiveSource = null;
  const bar = el("extrudeCompareBar");
  const btns = el("extrudeCompareSourceBtns");
  const stats = el("extrudeCompareStats");
  const info = el("extrudeCompareInfo");
  if (bar) bar.hidden = true;
  if (btns) btns.innerHTML = "";
  if (stats) {
    stats.textContent = "";
    stats.innerHTML = "";
  }
  if (info) {
    info.textContent = sessionKind === "dxf"
      ? "全手法（平行ペア / 精修 / DXF直接 / SVG）× 全ONNXモデルで壁を押し出し、3Dで切り替え比較します。"
      : "DXFセッションでのみ利用できます。";
  }
  const refineInfo = el("wallRefineCompareInfo");
  if (refineInfo) {
    refineInfo.textContent = sessionKind === "dxf"
      ? "平行ペア / 精修 / 構造マージの3段階を押し出して3Dで比較します。"
      : "DXFセッションでのみ利用できます。";
  }
}

function getExtrudeCompareVariants(report) {
  if (Array.isArray(report?.variants) && report.variants.length) {
    return report.variants;
  }
  const sources = report?.sources || {};
  return Object.entries(sources).map(([id, src]) => ({ id, ...src }));
}

function findExtrudeCompareVariant(report, variantId) {
  return getExtrudeCompareVariants(report).find((v) => v.id === variantId) || null;
}

function hexColorToNumber(hex) {
  if (!hex || typeof hex !== "string") return 0xc9ced6;
  const cleaned = hex.replace("#", "");
  const n = Number.parseInt(cleaned, 16);
  return Number.isFinite(n) ? n : 0xc9ced6;
}

function formatExtrudeVariantCell(variant) {
  if (!variant) return '<td class="is-na">—</td>';
  if (!variant.available) {
    return `<td class="is-na" title="${variant.error || "押出なし"}">—</td>`;
  }
  return `<td title="${variant.label || ""}">${variant.mesh_wall_count} 面</td>`;
}

function formatRefineStatsNote(cell) {
  const stats = cell?.refine_stats;
  if (!stats) return "";
  const input = stats.input_wall_count;
  const refined = stats.output_wall_count;
  const structural = stats.structural_output_wall_count;
  if (structural != null && stats.structural_input_wall_count != null) {
    return ` (${input}→${refined}→${structural})`;
  }
  if (input != null && refined != null) {
    return ` (${input}→${refined})`;
  }
  return "";
}

function renderExtrudeCompareMatrix(report) {
  const matrix = report?.matrix;
  const models = report?.models || [];
  if (!matrix) return "";

  const heuristicMethods = [
    { id: "heuristic", label: "平行ペア" },
    { id: "heuristic_refined", label: "平行ペア(精修)" },
    { id: "heuristic_structural", label: "平行ペア(構造)" },
  ];
  const mlMethods = [
    { id: "dxf_pil", label: "DXF直接" },
    { id: "dxf_svg_cairosvg", label: "SVG" },
  ];

  const formatHeuristicRow = (hm) => {
    const cell = matrix[hm.id]?._;
    const colspan = Math.max(models.length, 1);
    if (!cell) {
      return `<tr><th>${hm.label}</th><td class="is-na" colspan="${colspan}">—</td></tr>`;
    }
    if (!cell.available) {
      return `<tr><th>${cell.label || hm.label}</th><td class="is-na" colspan="${colspan}" title="${cell.error || "押出なし"}">—</td></tr>`;
    }
    const refineNote = formatRefineStatsNote(cell);
    return `<tr><th>${cell.label || hm.label}</th><td colspan="${colspan}">${cell.mesh_wall_count} 面${refineNote}</td></tr>`;
  };

  if (!models.length) {
    return `<table class="extrude-compare-matrix"><tbody>${heuristicMethods.map(formatHeuristicRow).join("")}</tbody></table>`;
  }

  const header = `<tr><th>手法 \\ モデル</th>${models.map((m) => `<th>${m.label || m.id}</th>`).join("")}</tr>`;
  const heuristicRows = heuristicMethods.map(formatHeuristicRow).join("");
  const mlRows = mlMethods.map((method) => {
    const row = matrix[method.id] || {};
    return `<tr><th>${method.label}</th>${models.map((m) => formatExtrudeVariantCell(row[m.id])).join("")}</tr>`;
  }).join("");
  return `<table class="extrude-compare-matrix"><thead>${header}</thead><tbody>${heuristicRows}${mlRows}</tbody></table>`;
}

function formatExtrudeCompareStats(report) {
  const variants = getExtrudeCompareVariants(report);
  if (!variants.length) return "";
  const parts = variants.map((variant) => {
    if (!variant.available) {
      return `${variant.label || variant.id}: —`;
    }
    return `${variant.label || variant.id}: ${variant.mesh_wall_count} 面`;
  });
  const summary = `高さ ${report.height_m ?? "?"} m · ${report.available_count ?? "?"} / ${report.variant_count ?? variants.length} 件`;
  return `${summary} — ${parts.join(" | ")}`;
}

function renderExtrudeCompareBar(report) {
  const bar = el("extrudeCompareBar");
  const btns = el("extrudeCompareSourceBtns");
  const stats = el("extrudeCompareStats");
  if (!bar || !btns) return;

  extrudeCompareReport = report;
  btns.innerHTML = "";
  const variants = getExtrudeCompareVariants(report);

  for (const variant of variants) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.source = variant.id;
    button.textContent = variant.label || variant.id;
    if (variant.color) {
      button.style.borderLeft = `4px solid ${variant.color}`;
    }
    if (!variant.available) {
      button.classList.add("unavailable");
      button.disabled = true;
      button.title = variant.error || "押出なし";
    } else if (variant.id === extrudeCompareActiveSource) {
      button.classList.add("active");
    }
    btns.appendChild(button);
  }

  if (stats) {
    const matrixHtml = renderExtrudeCompareMatrix(report);
    stats.innerHTML = `${formatExtrudeCompareStats(report)}${matrixHtml}`;
  }
  bar.hidden = !variants.length;
}

async function selectExtrudeCompareSource(source, { fitCamera = false } = {}) {
  if (!sessionId || !source) return;
  const srcMeta = findExtrudeCompareVariant(extrudeCompareReport, source)
    || extrudeCompareReport?.sources?.[source];
  if (!srcMeta?.available) return;

  extrudeCompareActiveSource = source;
  renderExtrudeCompareBar(extrudeCompareReport);

  const color = hexColorToNumber(srcMeta.color);
  try {
    const [meshRes, extractRes] = await Promise.all([
      fetch(`/api/extrude-compare/${sessionId}/${encodeURIComponent(source)}`),
      fetch(`/api/extrude-compare/${sessionId}/${encodeURIComponent(source)}/extract`),
    ]);
    if (meshRes.ok) {
      buildWallExtrusion(await meshRes.json(), { color, fitCamera });
    }
    if (extractRes.ok) {
      buildWallHighlight(await extractRes.json(), { color });
    }
  } catch (err) {
    console.error("selectExtrudeCompareSource failed:", err);
  }
}

function firstAvailableExtrudeVariant(report) {
  const variants = getExtrudeCompareVariants(report);
  return variants.find((v) => v.available)?.id || null;
}

function isExtrudeVariantAvailable(report, variantId) {
  const variant = findExtrudeCompareVariant(report, variantId)
    || report?.sources?.[variantId];
  return !!variant?.available;
}

function onExtrudeCompareSourceClick(event) {
  const button = event.target?.closest?.("button[data-source]");
  if (!(button instanceof HTMLButtonElement) || button.disabled) return;
  const source = button.dataset.source;
  if (!source) return;
  void selectExtrudeCompareSource(source);
}

function pickExtrudeCompareReport(batchData, refineData) {
  const batchVariants = getExtrudeCompareVariants(batchData);
  const refineVariants = getExtrudeCompareVariants(refineData);
  if (batchData && batchVariants.length > 3) {
    return batchData;
  }
  const refineReady =
    refineData?.compare_kind === "refine"
    && refineVariants.some((v) => v.id === "heuristic_structural");
  if (refineReady) {
    return refineData;
  }
  if (batchData) {
    return batchData;
  }
  if (refineData) {
    return refineData;
  }
  return null;
}

async function loadExtrudeCompareReport() {
  if (!sessionId || sessionKind !== "dxf") {
    clearExtrudeCompareUI();
    return;
  }
  try {
    const [batchRes, refineRes] = await Promise.all([
      fetch(`/api/extrude-compare/${sessionId}`),
      fetch(`/api/wall-refine-compare/${sessionId}`),
    ]);
    let batchData = null;
    let refineData = null;
    if (batchRes.ok) {
      batchData = await batchRes.json();
    }
    if (refineRes.ok) {
      refineData = await refineRes.json();
    }
    const data = pickExtrudeCompareReport(batchData, refineData);
    if (!data) {
      clearExtrudeCompareUI();
      return;
    }
    renderExtrudeCompareBar(data);
    const firstAvailable = firstAvailableExtrudeVariant(data);
    if (firstAvailable && !extrudeCompareActiveSource) {
      await selectExtrudeCompareSource(firstAvailable, { fitCamera: false });
    } else if (extrudeCompareActiveSource && isExtrudeVariantAvailable(data, extrudeCompareActiveSource)) {
      await selectExtrudeCompareSource(extrudeCompareActiveSource, { fitCamera: false });
    }
  } catch {
    clearExtrudeCompareUI();
  }
}

async function onExtrudeCompare(event) {
  event.preventDefault();
  if (!sessionId || sessionKind !== "dxf") {
    alert("先にDXFをインポートしてください");
    return;
  }
  const btn = el("extrudeCompare");
  if (btn instanceof HTMLButtonElement) btn.disabled = true;
  const info = el("extrudeCompareInfo");
  if (info) info.textContent = "全手法×全モデルの一括押出を実行中…";
  try {
    const res = await fetch(`/api/extrude-compare/${sessionId}`, { method: "POST" });
    let data = {};
    try {
      data = await res.json();
    } catch {
      const hint = res.status === 404
        ? " API が見つかりません。app.py 変更後は Flask を再起動してください。"
        : "";
      throw new Error(`サーバー応答を読み取れませんでした (HTTP ${res.status})${hint}`);
    }
    if (!res.ok) {
      alert(data.error || `押出比較に失敗しました (HTTP ${res.status})`);
      if (info) info.textContent = data.error || "押出比較に失敗しました";
      return;
    }
    renderExtrudeCompareBar(data);
    const firstAvailable = firstAvailableExtrudeVariant(data);
    if (!firstAvailable) {
      alert("押し出せる壁がありませんでした");
      if (info) info.textContent = "押し出せる壁がありませんでした";
      return;
    }
    const modelCount = data.models?.length ?? 1;
    if (info) {
      info.textContent = `一括押出完了（${data.variant_count ?? "?"} 件 / モデル ${modelCount}）— 3Dパネルで切り替え`;
    }
    await selectExtrudeCompareSource(firstAvailable, { fitCamera: true });
  } catch (err) {
    console.error("onExtrudeCompare failed:", err);
    alert(err?.message || "押出比較に失敗しました");
    if (info) info.textContent = err?.message || "押出比較に失敗しました";
  } finally {
    if (btn instanceof HTMLButtonElement) btn.disabled = false;
    syncExtrudeCompareButton();
  }
}

async function onWallRefineCompare(event) {
  event.preventDefault();
  if (!sessionId || sessionKind !== "dxf") {
    alert("先にDXFをインポートしてください");
    return;
  }
  const btn = el("wallRefineCompare");
  if (btn instanceof HTMLButtonElement) btn.disabled = true;
  const info = el("wallRefineCompareInfo");
  if (info) info.textContent = "平行ペア / 精修 / 構造マージの押出比較を実行中…";
  try {
    const res = await fetch(`/api/wall-refine-compare/${sessionId}`, { method: "POST" });
    let data = {};
    try {
      data = await res.json();
    } catch {
      const hint = res.status === 404
        ? " API が見つかりません。app.py 変更後は Flask を再起動してください。"
        : "";
      throw new Error(`サーバー応答を読み取れませんでした (HTTP ${res.status})${hint}`);
    }
    if (!res.ok) {
      alert(data.error || `精修壁比較に失敗しました (HTTP ${res.status})`);
      if (info) info.textContent = data.error || "精修壁比較に失敗しました";
      return;
    }
    renderExtrudeCompareBar(data);
    const firstAvailable = firstAvailableExtrudeVariant(data);
    if (!firstAvailable) {
      alert("押し出せる壁がありませんでした");
      if (info) info.textContent = "押し出せる壁がありませんでした";
      return;
    }
    const structural = findExtrudeCompareVariant(data, "heuristic_structural");
    const stats = structural?.refine_stats || findExtrudeCompareVariant(data, "heuristic_refined")?.refine_stats;
    const refineNote = stats?.structural_output_wall_count != null
      ? `（壁 ${stats.input_wall_count} → ${stats.output_wall_count} → ${stats.structural_output_wall_count}）`
      : stats
        ? `（壁 ${stats.input_wall_count} → ${stats.output_wall_count}）`
        : "";
    if (info) {
      info.textContent = `精修壁比較完了 ${refineNote} — 3Dパネルで切り替え`;
    }
    await selectExtrudeCompareSource(firstAvailable, { fitCamera: true });
  } catch (err) {
    console.error("onWallRefineCompare failed:", err);
    alert(err?.message || "精修壁比較に失敗しました");
    if (info) info.textContent = err?.message || "精修壁比較に失敗しました";
  } finally {
    if (btn instanceof HTMLButtonElement) btn.disabled = false;
    syncWallRefineCompareButton();
  }
}

function syncMlComparePanelsVisibility() {
  const stack = el("mlComparePanels");
  const pocPanel = el("mlPocPanel");
  const renderPanel = el("mlRenderComparePanel");
  if (!stack) return;
  const anyVisible = !(pocPanel?.hidden ?? true) || !(renderPanel?.hidden ?? true);
  stack.hidden = !anyVisible;
}

function clearMlPocPanel() {
  const panel = el("mlPocPanel");
  const metrics = el("mlPocMetrics");
  const preview = el("mlPocPreview");
  const info = el("mlPocInfo");
  if (panel) panel.hidden = true;
  if (metrics) metrics.textContent = "";
  if (preview) preview.removeAttribute("src");
  if (info) {
    info.textContent = sessionKind === "dxf"
      ? "DXFインポート後に ONNX 壁セグメンテーションと平行ペア結果を比較します。"
      : "DXFセッションでのみ利用できます。";
  }
  syncMlComparePanelsVisibility();
}

function clearMlRenderComparePanel() {
  const panel = el("mlRenderComparePanel");
  const metrics = el("mlRenderCompareMetrics");
  const preview = el("mlRenderComparePreview");
  const info = el("mlRenderCompareInfo");
  if (panel) panel.hidden = true;
  if (metrics) metrics.textContent = "";
  if (preview) preview.removeAttribute("src");
  if (info) {
    info.textContent = sessionKind === "dxf"
      ? "DXF直接ラスタと SVG+cairosvg 経路の ML 結果を比較します。"
      : "DXFセッションでのみ利用できます。";
  }
  syncMlComparePanelsVisibility();
}

function clearMlComparePanels() {
  clearMlPocPanel();
  clearMlRenderComparePanel();
  const stack = el("mlComparePanels");
  if (stack) stack.hidden = true;
}

function updateMlPocPanel(report) {
  const panel = el("mlPocPanel");
  const metrics = el("mlPocMetrics");
  const preview = el("mlPocPreview");
  const info = el("mlPocInfo");
  if (!report || !sessionId) {
    clearMlPocPanel();
    return;
  }
  const iou = report.wall_iou ?? "?";
  const prec = report.wall_precision ?? "?";
  const rec = report.wall_recall ?? "?";
  const ms = report.inference_ms_cpu ?? "?";
  const walls = report.heuristic_wall_count ?? "?";
  if (info) {
    info.textContent = `ML PoC 完了 — IoU ${iou}, P ${prec}, R ${rec}, ${ms} ms`;
  }
  if (metrics) {
    metrics.textContent = `平行ペア壁 ${walls} 本 | ML ${report.ml_wall_pixels ?? "?"} px | ヒューリスティック ${report.heuristic_wall_pixels ?? "?"} px`;
  }
  if (preview) {
    preview.src = `/api/ml-poc/${sessionId}/overlay_compare.png?ts=${Date.now()}`;
  }
  if (panel) panel.hidden = false;
  syncMlComparePanelsVisibility();
}

async function loadMlPocReport() {
  if (!sessionId || sessionKind !== "dxf") {
    clearMlPocPanel();
    return;
  }
  try {
    const res = await fetch(`/api/ml-poc/${sessionId}`);
    if (!res.ok) {
      clearMlPocPanel();
      return;
    }
    const data = await res.json();
    updateMlPocPanel(data);
  } catch {
    clearMlPocPanel();
  }
}

function updateMlRenderComparePanel(report) {
  const panel = el("mlRenderComparePanel");
  const metrics = el("mlRenderCompareMetrics");
  const preview = el("mlRenderComparePreview");
  const info = el("mlRenderCompareInfo");
  if (!report || !sessionId) {
    clearMlRenderComparePanel();
    return;
  }
  const dxf = report.render_paths?.dxf_pil ?? {};
  const svg = report.render_paths?.dxf_svg_cairosvg ?? {};
  if (info) {
    info.textContent = [
      "レンダ比較完了",
      `DXF IoU ${dxf.wall_iou ?? "?"}`,
      `SVG IoU ${svg.wall_iou ?? "?"}`,
    ].join(" — ");
  }
  if (metrics) {
    metrics.textContent = [
      `DXF: P ${dxf.wall_precision ?? "?"} R ${dxf.wall_recall ?? "?"} (${dxf.ml_wall_pixels ?? "?"} px)`,
      `SVG: P ${svg.wall_precision ?? "?"} R ${svg.wall_recall ?? "?"} (${svg.ml_wall_pixels ?? "?"} px)`,
      `ML同士 IoU ${report.ml_dxf_vs_ml_svg?.wall_iou ?? "?"}`,
    ].join(" | ");
  }
  if (preview) {
    preview.src = `/api/ml-poc-render-compare/${sessionId}/overlay_compare.png?ts=${Date.now()}`;
  }
  if (panel) panel.hidden = false;
  syncMlComparePanelsVisibility();
}

async function loadMlRenderCompareReport() {
  if (!sessionId || sessionKind !== "dxf") {
    clearMlRenderComparePanel();
    return;
  }
  try {
    const res = await fetch(`/api/ml-poc-render-compare/${sessionId}`);
    if (!res.ok) {
      clearMlRenderComparePanel();
      return;
    }
    const data = await res.json();
    updateMlRenderComparePanel(data);
  } catch {
    clearMlRenderComparePanel();
  }
}

async function loadMlCompareReports() {
  await Promise.all([loadMlPocReport(), loadMlRenderCompareReport()]);
}

async function onMlPocCompare(event) {
  event.preventDefault();
  if (!sessionId || sessionKind !== "dxf") {
    alert("先にDXFをインポートしてください");
    return;
  }
  const btn = el("mlPocCompare");
  if (btn instanceof HTMLButtonElement) btn.disabled = true;
  const info = el("mlPocInfo");
  if (info) info.textContent = "ML推論と比較を実行中…（初回は数十秒かかることがあります）";
  try {
    const res = await fetch(`/api/ml-poc/${sessionId}`, { method: "POST" });
    let data = {};
    try {
      data = await res.json();
    } catch {
      const hint = res.status === 404
        ? " API が見つかりません。app.py 変更後は Flask を再起動してください。"
        : "";
      throw new Error(`サーバー応答を読み取れませんでした (HTTP ${res.status})${hint}`);
    }
    if (!res.ok) {
      alert(data.error || `ML PoC に失敗しました (HTTP ${res.status})`);
      if (info) info.textContent = data.error || "ML PoC に失敗しました";
      return;
    }
    updateMlPocPanel(data);
    try {
      const mlRes = await fetch(`/api/wall-extract-ml/${sessionId}`);
      if (mlRes.ok) {
        buildWallHighlight(await mlRes.json());
      }
    } catch (err) {
      console.warn("ML wall highlight load failed:", err);
    }
  } catch (err) {
    console.error("onMlPocCompare failed:", err);
    alert(err?.message || "ML PoC に失敗しました");
    if (info) info.textContent = err?.message || "ML PoC に失敗しました";
  } finally {
    if (btn instanceof HTMLButtonElement) btn.disabled = false;
    syncMlPocButton();
    syncMlRenderCompareButton();
  }
}

async function onMlRenderCompare(event) {
  event.preventDefault();
  if (!sessionId || sessionKind !== "dxf") {
    alert("先にDXFをインポートしてください");
    return;
  }
  const btn = el("mlRenderCompare");
  if (btn instanceof HTMLButtonElement) btn.disabled = true;
  const info = el("mlRenderCompareInfo");
  if (info) info.textContent = "レンダ比較を実行中…（DXF + SVG の2回推論）";
  try {
    const res = await fetch(`/api/ml-poc-render-compare/${sessionId}`, { method: "POST" });
    let data = {};
    try {
      data = await res.json();
    } catch {
      const hint = res.status === 404
        ? " API が見つかりません。app.py 変更後は Flask を再起動してください。"
        : "";
      throw new Error(`サーバー応答を読み取れませんでした (HTTP ${res.status})${hint}`);
    }
    if (!res.ok) {
      alert(data.error || `レンダ比較に失敗しました (HTTP ${res.status})`);
      if (info) info.textContent = data.error || "レンダ比較に失敗しました";
      return;
    }
    updateMlRenderComparePanel(data);
  } catch (err) {
    console.error("onMlRenderCompare failed:", err);
    alert(err?.message || "レンダ比較に失敗しました");
    if (info) info.textContent = err?.message || "レンダ比較に失敗しました";
  } finally {
    if (btn instanceof HTMLButtonElement) btn.disabled = false;
    syncMlRenderCompareButton();
  }
}

function setDxfGeometry(payload) {
  if (!payload?.positions?.length) {
    dxfGeometry = null;
    return;
  }
  dxfGeometry = {
    positions: payload.positions,
    bounds: payload.bounds || payload.dxf?.bounds || null,
    segmentCount: payload.segment_count ?? payload.dxf?.segment_count ?? 0,
  };
}

function paintFloorPlanIdleScene(ctx, w, h) {
  const vignette = ctx.createRadialGradient(w * 0.5, h * 0.16, 0, w * 0.5, h * 0.5, Math.max(w, h) * 0.75);
  vignette.addColorStop(0, "rgba(196, 77, 106, 0.14)");
  vignette.addColorStop(0.42, "rgba(58, 18, 34, 0.1)");
  vignette.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, w, h);

  const pad = 22;
  const gridStep = 18;
  for (let x = pad; x <= w - pad; x += gridStep) {
    const major = Math.round((x - pad) / gridStep) % 5 === 0;
    ctx.strokeStyle = major ? "rgba(200, 120, 140, 0.16)" : "rgba(200, 120, 140, 0.06)";
    ctx.lineWidth = major ? 0.75 : 0.5;
    ctx.beginPath();
    ctx.moveTo(x, pad);
    ctx.lineTo(x, h - pad);
    ctx.stroke();
  }
  for (let y = pad; y <= h - pad; y += gridStep) {
    const major = Math.round((y - pad) / gridStep) % 5 === 0;
    ctx.strokeStyle = major ? "rgba(200, 120, 140, 0.16)" : "rgba(200, 120, 140, 0.06)";
    ctx.lineWidth = major ? 0.75 : 0.5;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
    ctx.stroke();
  }

  const margin = 16;
  const tick = 12;
  ctx.strokeStyle = "rgba(232, 160, 176, 0.3)";
  ctx.lineWidth = 1;
  for (const [cx, cy, sx, sy] of [
    [margin, margin, 1, 1],
    [w - margin, margin, -1, 1],
    [margin, h - margin, 1, -1],
    [w - margin, h - margin, -1, -1],
  ]) {
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + sx * tick, cy);
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx, cy + sy * tick);
    ctx.stroke();
  }
}

function drawDxfPlanGrid(ctx, minX, maxX, minZ, maxZ, toCanvas) {
  const startX = Math.floor(minX);
  const endX = Math.ceil(maxX);
  const startZ = Math.floor(minZ);
  const endZ = Math.ceil(maxZ);
  for (let x = startX; x <= endX + 1e-6; x += 0.5) {
    const major = Math.abs(x - Math.round(x)) < 1e-6;
    const [cx0, cy0] = toCanvas(x, minZ);
    const [cx1, cy1] = toCanvas(x, maxZ);
    ctx.beginPath();
    ctx.moveTo(cx0, cy0);
    ctx.lineTo(cx1, cy1);
    ctx.strokeStyle = major ? DXF_PLAN_THEME.gridMajor : DXF_PLAN_THEME.gridMinor;
    ctx.lineWidth = major ? 1 : 0.5;
    ctx.stroke();
  }
  for (let z = startZ; z <= endZ + 1e-6; z += 0.5) {
    const major = Math.abs(z - Math.round(z)) < 1e-6;
    const [cx0, cy0] = toCanvas(minX, z);
    const [cx1, cy1] = toCanvas(maxX, z);
    ctx.beginPath();
    ctx.moveTo(cx0, cy0);
    ctx.lineTo(cx1, cy1);
    ctx.strokeStyle = major ? DXF_PLAN_THEME.gridMajor : DXF_PLAN_THEME.gridMinor;
    ctx.lineWidth = major ? 1 : 0.5;
    ctx.stroke();
  }
}

function drawDxfPlanScaleBar(ctx, w, h, pxPerMeter) {
  const barPx = Math.min(80, Math.max(pxPerMeter, 24));
  const barM = barPx / Math.max(pxPerMeter, 1e-9);
  const label = barM >= 0.95 && barM <= 1.05 ? "1 m" : `${barM.toFixed(1)} m`;
  const x0 = w - 24 - barPx;
  const y0 = h - 18;
  ctx.strokeStyle = DXF_PLAN_THEME.scale;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x0 + barPx, y0);
  ctx.stroke();
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = DXF_PLAN_THEME.scale;
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(label, x0 + barPx / 2, y0 - 8);
}

function isFloorPlanVisible() {
  const btn = el("showFloorPlan");
  return btn?.getAttribute("aria-pressed") === "true";
}

function syncFloorPlanToggleState(visible) {
  const btn = el("showFloorPlan");
  if (!(btn instanceof HTMLButtonElement)) return;
  btn.setAttribute("aria-pressed", visible ? "true" : "false");
  btn.classList.toggle("is-active", visible);
}

function setFloorPlanVisible(visible) {
  syncFloorPlanToggleState(visible);
  syncFloorPlanPanel();
}

function syncFloorPlanPanel() {
  const popup = el("floorPlanPopup");
  const visible = isFloorPlanVisible();
  syncFloorPlanToggleState(visible);
  if (popup) {
    popup.hidden = !visible;
    popup.classList.toggle("is-open", visible);
  }
  if (visible) {
    setupFloorPlanResizeObserver();
    renderDxfPlanLayerPanel();
    void refreshRoomPlanFloorData().then(() => drawFloorPlanView());
    requestAnimationFrame(() => drawFloorPlanView());
  } else {
    teardownFloorPlanResizeObserver();
    floorPlanUserSized = false;
    floorPlanLastCanvasSize = { w: 0, h: 0 };
  }
}

function setupFloorPlanResizeObserver() {
  const canvas = el("dxfPlanCanvas");
  const viewer = canvas?.closest(".floorplan-viewer");
  if (!viewer || floorPlanResizeObserver) return;
  floorPlanResizeObserver = new ResizeObserver(() => {
    if (!isFloorPlanVisible()) return;
    const targetCanvas = el("dxfPlanCanvas");
    const targetViewer = targetCanvas?.closest(".floorplan-viewer");
    if (!targetViewer) return;
    const rect = targetViewer.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    if (w === floorPlanLastCanvasSize.w && h === floorPlanLastCanvasSize.h) return;
    drawFloorPlanView({ afterLayout: true });
  });
  floorPlanResizeObserver.observe(viewer);
}

function teardownFloorPlanResizeObserver() {
  floorPlanResizeObserver?.disconnect();
  floorPlanResizeObserver = null;
}

function clampFloorPlanPopupPosition(popup) {
  if (!popup) return;
  const popupRect = popup.getBoundingClientRect();
  const margin = 8;
  let left = popupRect.left;
  let top = popupRect.top;
  const maxLeft = Math.max(margin, window.innerWidth - popupRect.width - margin);
  const maxTop = Math.max(margin, window.innerHeight - popupRect.height - margin);
  left = Math.min(Math.max(left, margin), maxLeft);
  top = Math.min(Math.max(top, margin), maxTop);
  popup.style.right = "auto";
  popup.style.left = `${left}px`;
  popup.style.top = `${top}px`;
}

function initFloorPlanPopupDrag() {
  const popup = el("floorPlanPopup");
  const handle = popup?.querySelector(".floorplan-popup-drag-handle");
  if (!popup || !handle) return;

  const endDrag = (event) => {
    if (!floorPlanDragState) return;
    floorPlanDragState = null;
    popup.classList.remove("is-dragging");
    try {
      handle.releasePointerCapture(event.pointerId);
    } catch {
      // ignore if capture was already released
    }
  };

  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    if (event.target.closest(".floorplan-popup-close")) return;

    const popupRect = popup.getBoundingClientRect();
    popup.style.right = "auto";
    popup.style.left = `${popupRect.left}px`;
    popup.style.top = `${popupRect.top}px`;

    floorPlanDragState = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startLeft: popupRect.left,
      startTop: popupRect.top,
    };
    popup.classList.add("is-dragging");
    handle.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  handle.addEventListener("pointermove", (event) => {
    if (!floorPlanDragState || event.pointerId !== floorPlanDragState.pointerId) return;
    const popupRect = popup.getBoundingClientRect();
    const dx = event.clientX - floorPlanDragState.startX;
    const dy = event.clientY - floorPlanDragState.startY;
    const margin = 8;
    const maxLeft = Math.max(margin, window.innerWidth - popupRect.width - margin);
    const maxTop = Math.max(margin, window.innerHeight - popupRect.height - margin);
    const left = Math.min(Math.max(floorPlanDragState.startLeft + dx, margin), maxLeft);
    const top = Math.min(Math.max(floorPlanDragState.startTop + dy, margin), maxTop);
    popup.style.left = `${left}px`;
    popup.style.top = `${top}px`;
    event.preventDefault();
  });

  handle.addEventListener("pointerup", endDrag);
  handle.addEventListener("pointercancel", endDrag);

  window.addEventListener("resize", () => {
    if (!popup || popup.hidden) return;
    clampFloorPlanPopupPosition(popup);
    if (isFloorPlanVisible()) drawFloorPlanView();
  });
}

function initFloorPlanPopupResize() {
  const popup = el("floorPlanPopup");
  const handle = popup?.querySelector(".floorplan-popup-resize-handle");
  if (!popup || !handle) return;

  const endResize = (event) => {
    if (!floorPlanResizeState) return;
    floorPlanResizeState = null;
    popup.classList.remove("is-resizing");
    try {
      handle.releasePointerCapture(event.pointerId);
    } catch {
      // ignore if capture was already released
    }
    drawFloorPlanView({ afterLayout: true });
  };

  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.stopPropagation();

    const popupRect = popup.getBoundingClientRect();
    popup.style.right = "auto";
    popup.style.left = `${popupRect.left}px`;
    popup.style.top = `${popupRect.top}px`;

    floorPlanUserSized = true;
    floorPlanResizeState = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startW: popupRect.width,
      startH: popupRect.height,
      anchorLeft: popupRect.left,
      anchorTop: popupRect.top,
    };
    popup.classList.add("is-resizing");
    handle.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  handle.addEventListener("pointermove", (event) => {
    if (!floorPlanResizeState || event.pointerId !== floorPlanResizeState.pointerId) return;
    const margin = PLAN_POPUP_MARGIN_PX;
    const dx = event.clientX - floorPlanResizeState.startX;
    const dy = event.clientY - floorPlanResizeState.startY;
    const maxW = window.innerWidth - margin - floorPlanResizeState.anchorLeft;
    const maxH = window.innerHeight - margin - floorPlanResizeState.anchorTop;
    const width = Math.min(maxW, Math.max(PLAN_POPUP_MIN_W, floorPlanResizeState.startW + dx));
    const height = Math.min(maxH, Math.max(PLAN_POPUP_MIN_H, floorPlanResizeState.startH + dy));
    popup.style.width = `${Math.round(width)}px`;
    popup.style.height = `${Math.round(height)}px`;
    drawFloorPlanView({ afterLayout: true });
    event.preventDefault();
  });

  handle.addEventListener("pointerup", endResize);
  handle.addEventListener("pointercancel", endResize);
}

function initFloorPlanCanvasInteraction() {
  const canvas = el("dxfPlanCanvas");
  if (!canvas) return;

  canvas.addEventListener("pointerdown", onFloorPlanCanvasPointerDown);
  canvas.addEventListener("pointermove", onFloorPlanCanvasPointerMove);
  canvas.addEventListener("pointerup", onFloorPlanCanvasPointerUp);
  canvas.addEventListener("pointercancel", onFloorPlanCanvasPointerUp);
}

function onFloorPlanCanvasPointerDown(event) {
  if (sessionKind !== "usdz" || !isFloorPlanVisible()) return;
  if (walkModeActive || planeEditDrag) return;

  const canvas = el("dxfPlanCanvas");
  if (!canvas) return;
  const [cx, cy] = getCanvasPointerPosition(event, canvas);

  if (
    selected?.id
    && (editMode === "scale" || editMode === "rotate")
    && !isSurfaceTextureOnlyObject(selected)
  ) {
    const selectedItem = roomPlanDisplayState?.items?.find((item) => item.id === selected.id);
    const selectedObj = objects.find((o) => o.id === selected.id);
    const handle = selectedItem ? pickFloorPlanHandleAtCanvas(cx, cy, selectedItem, editMode) : null;
    if (handle && selectedObj && !isObjectLayerLocked(selectedObj)) {
      startPlaneEditDrag(event, selectedObj, {
        source: "floorplan",
        floorPlanMapping: floorPlanPaintMapping,
        mode: editMode,
        handle,
      });
      event.preventDefault();
      event.stopPropagation();
      return;
    }
  }

  const obj = pickFloorPlanItemAtCanvas(cx, cy);

  if (obj && !isObjectLayerLocked(obj)) {
    selectObject(obj);
    if (!isSurfaceTextureOnlyObject(obj) && (editMode === "translate" || editMode === "scale" || editMode === "rotate")) {
      startPlaneEditDrag(event, obj, {
        source: "floorplan",
        floorPlanMapping: floorPlanPaintMapping,
        mode: editMode,
      });
    }
    event.preventDefault();
    event.stopPropagation();
    return;
  }

  if (selected) clearObjectSelection();
}

function onFloorPlanCanvasPointerMove(event) {
  if (!planeEditDrag || planeEditDrag.source !== "floorplan") return;
  if (event.pointerId !== planeEditDrag.pointerId) return;
  updatePlaneEditDrag(event);
  event.preventDefault();
}

function onFloorPlanCanvasPointerUp(event) {
  if (!planeEditDrag || planeEditDrag.source !== "floorplan") return;
  if (event.pointerId !== planeEditDrag.pointerId) return;
  void endPlaneEditDrag(event);
  event.preventDefault();
}

function planItemFromSceneObject(obj) {
  if (!obj || typeof obj !== "object") return null;
  const rect = obj.world_planar_rect;
  let corners = [];
  if (Array.isArray(rect?.corners) && rect.corners.length >= 3) {
    corners = rect.corners.map((c) => [Number(c[0]), Number(c[1])]);
  } else if (Array.isArray(obj.world_basis_footprint_xz) && obj.world_basis_footprint_xz.length >= 3) {
    corners = worldXzCornersToPlanUv(obj.world_basis_footprint_xz);
  }
  if (corners.length < 3) return null;

  const usedXzFootprintFallback = !(
    Array.isArray(rect?.corners) && rect.corners.length >= 3
  );
  const longAxis = (rect?.long_axis && typeof rect.long_axis === "object")
    ? rect.long_axis
    : (obj.world_basis_long_axis_xz && typeof obj.world_basis_long_axis_xz === "object")
      ? obj.world_basis_long_axis_xz
      : {};
  const shortAxis = (rect?.short_axis && typeof rect.short_axis === "object") ? rect.short_axis : {};
  const center = Array.isArray(rect?.center) && rect.center.length >= 2
    ? [Number(rect.center[0]), Number(rect.center[1])]
    : [
        corners.reduce((sum, c) => sum + c[0], 0) / corners.length,
        corners.reduce((sum, c) => sum + c[1], 0) / corners.length,
      ];
  const dims = Array.isArray(obj.dimensions) ? obj.dimensions : [0.5, 0.5, 0.5];
  let longM = Number(longAxis.length) || Math.max(Math.abs(Number(dims[0])), 0.05);
  let shortM = Number(shortAxis.length) || Math.max(Math.abs(Number(dims[2])), 0.05);
  if (shortM > longM) {
    const swap = longM;
    longM = shortM;
    shortM = swap;
  }

  let longAngleDeg = Number(longAxis.angle_deg ?? obj.yaw_deg) || 0;
  if (usedXzFootprintFallback && longAxis === obj.world_basis_long_axis_xz) {
    longAngleDeg = -longAngleDeg;
  }

  return {
    id: obj.id,
    name: obj.name || obj.id,
    category: obj.category,
    corners,
    center,
    long_m: longM,
    short_m: shortM,
    long_angle_deg: longAngleDeg,
  };
}

function buildRoomPlanFloorPayloadFromSceneObjects(sceneObjects) {
  if (!Array.isArray(sceneObjects) || !sceneObjects.length) return null;
  const items = [];
  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (const obj of sceneObjects) {
    const item = planItemFromSceneObject(obj);
    if (!item) continue;
    items.push(item);
    for (const [u, v] of item.corners) {
      minU = Math.min(minU, u);
      maxU = Math.max(maxU, u);
      minV = Math.min(minV, v);
      maxV = Math.max(maxV, v);
    }
  }
  if (!items.length) return null;
  return {
    objects: items,
    bounds: { min_u: minU, max_u: maxU, min_v: minV, max_v: maxV },
    coordinate_space: "world_planar",
  };
}

function floorPlanPayloadHasGeometry(payload) {
  const list = payload?.objects;
  if (!Array.isArray(list) || !list.length) return false;
  return list.some((item) => Array.isArray(item.corners) && item.corners.length >= 3);
}

async function refreshRoomPlanFloorData() {
  if (sessionKind !== "usdz" || !sessionId) {
    roomPlanFloorData = null;
    return;
  }
  try {
    const res = await fetch(`/api/floorplan/${encodeURIComponent(sessionId)}`);
    if (!res.ok) {
      roomPlanFloorData = buildRoomPlanFloorPayloadFromSceneObjects(objects);
      return;
    }
    const payload = await res.json();
    roomPlanFloorData = floorPlanPayloadHasGeometry(payload)
      ? payload
      : (buildRoomPlanFloorPayloadFromSceneObjects(objects) || payload);
  } catch (err) {
    console.error("refreshRoomPlanFloorData failed:", err);
    roomPlanFloorData = buildRoomPlanFloorPayloadFromSceneObjects(objects);
  }
}

function scheduleRoomPlanRefresh() {
  if (sessionKind !== "usdz") return;
  void refreshRoomPlanFloorData().then(() => drawFloorPlanView());
}

function roomPlanStyleForCategory(category) {
  const key = normalizeCategory(category);
  return ROOM_PLAN_THEME[key] || ROOM_PLAN_THEME.object;
}

function roomPlanLayerOrder(category) {
  const key = normalizeCategory(category);
  const order = { floor: 0, wall: 1, opening: 2, window: 3, door: 4, chair: 5, table: 6, storage: 7, object: 8 };
  return order[key] ?? 8;
}

function planLineWidthForCategory(category, selected) {
  if (selected) return 2.5;
  const key = normalizeCategory(category);
  if (key === "wall") return 2;
  if (key === "opening" || key === "window") return 1.2;
  return 1;
}

function truncatePlanLabel(text, maxLen = 16) {
  const value = String(text || "").trim();
  if (value.length <= maxLen) return value;
  return `${value.slice(0, maxLen - 1)}…`;
}

function formatPlanMeters(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "?";
  if (n >= 0.95 && n <= 1.05) return "1.00 m";
  return `${n.toFixed(2)} m`;
}

function polygonArea2d(corners) {
  if (!Array.isArray(corners) || corners.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < corners.length; i += 1) {
    const [x1, y1] = corners[i];
    const [x2, y2] = corners[(i + 1) % corners.length];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) * 0.5;
}

function rotatePlanPoint2d(point, angleRad, pivot) {
  const [u, v] = point;
  const [pu, pv] = pivot;
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  const du = u - pu;
  const dv = v - pv;
  return [pu + du * cos - dv * sin, pv + du * sin + dv * cos];
}

function getBoundsFromItemCorners(items) {
  let minU = Infinity;
  let maxU = -Infinity;
  let minV = Infinity;
  let maxV = -Infinity;
  for (const item of items) {
    for (const [u, v] of item.corners || []) {
      minU = Math.min(minU, u);
      maxU = Math.max(maxU, u);
      minV = Math.min(minV, v);
      maxV = Math.max(maxV, v);
    }
  }
  if (!Number.isFinite(minU)) {
    return { minU: -1, maxU: 1, minV: -1, maxV: 1 };
  }
  return { minU, maxU, minV, maxV };
}

function expandPlanBounds(bounds, marginRatio = 0.04) {
  const spanU = Math.max(bounds.maxU - bounds.minU, 0.01);
  const spanV = Math.max(bounds.maxV - bounds.minV, 0.01);
  const padU = spanU * marginRatio;
  const padV = spanV * marginRatio;
  return {
    minU: bounds.minU - padU,
    maxU: bounds.maxU + padU,
    minV: bounds.minV - padV,
    maxV: bounds.maxV + padV,
  };
}

function computeDxfPlanDisplayBounds() {
  if (!dxfGeometry?.positions?.length) return null;
  const { positions } = dxfGeometry;
  let minX;
  let maxX;
  let minZ;
  let maxZ;
  if (dxfGeometry.bounds?.min && dxfGeometry.bounds?.max) {
    minX = dxfGeometry.bounds.min[0];
    maxX = dxfGeometry.bounds.max[0];
    minZ = dxfGeometry.bounds.min[2];
    maxZ = dxfGeometry.bounds.max[2];
  } else {
    minX = Infinity;
    maxX = -Infinity;
    minZ = Infinity;
    maxZ = -Infinity;
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i];
      const z = positions[i + 2];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
    }
  }

  const rawBounds = {
    minU: minX,
    maxU: maxX,
    minV: worldZToPlanV(maxZ),
    maxV: worldZToPlanV(minZ),
  };
  const padded = expandPlanBounds(rawBounds, 0.06);
  const centerU = (padded.minU + padded.maxU) / 2;
  const centerV = (padded.minV + padded.maxV) / 2;
  const spanU = Math.max(padded.maxU - padded.minU, 0.01);
  const spanV = Math.max(padded.maxV - padded.minV, 0.01);
  const rotate90 = spanU > spanV;

  const applyPlanRotation = (u, v) => {
    const du = u - centerU;
    const dv = v - centerV;
    if (!rotate90) return { u, v };
    return { u: centerU + dv, v: centerV - du };
  };

  let fitMinU = Infinity;
  let fitMaxU = -Infinity;
  let fitMinV = Infinity;
  let fitMaxV = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const u = positions[i];
    const v = worldZToPlanV(positions[i + 2]);
    const p = applyPlanRotation(u, v);
    fitMinU = Math.min(fitMinU, p.u);
    fitMaxU = Math.max(fitMaxU, p.u);
    fitMinV = Math.min(fitMinV, p.v);
    fitMaxV = Math.max(fitMaxV, p.v);
  }
  return expandPlanBounds({ minU: fitMinU, maxU: fitMaxU, minV: fitMinV, maxV: fitMaxV }, 0.04);
}

function computePlanDisplayTransform(items) {
  const floors = items.filter((item) => normalizeCategory(item.category) === "floor");
  const candidates = floors.length ? floors : items;
  let best = null;
  let bestArea = -1;
  for (const item of candidates) {
    const area = polygonArea2d(item.corners);
    if (area > bestArea) {
      bestArea = area;
      best = item;
    }
  }
  if (!best) {
    return { rotationRad: 0, pivot: [0, 0] };
  }

  const pivot = Array.isArray(best.center) && best.center.length >= 2
    ? [Number(best.center[0]), Number(best.center[1])]
    : [
        best.corners.reduce((sum, corner) => sum + corner[0], 0) / best.corners.length,
        best.corners.reduce((sum, corner) => sum + corner[1], 0) / best.corners.length,
      ];
  const longAngleDeg = Number(best.long_angle_deg) || 0;
  const rotationRad = ((90 - longAngleDeg) * Math.PI) / 180;
  return { rotationRad, pivot };
}

function applyPlanDisplayTransform(items, transform) {
  const { rotationRad, pivot } = transform;
  if (!rotationRad) return items;
  return items.map((item) => ({
    ...item,
    corners: item.corners.map((corner) => rotatePlanPoint2d(corner, rotationRad, pivot)),
    center: Array.isArray(item.center)
      ? rotatePlanPoint2d(item.center, rotationRad, pivot)
      : item.center,
    long_angle_deg: (Number(item.long_angle_deg) || 0) + (rotationRad * 180) / Math.PI,
  }));
}

function updateRoomPlanDisplayState() {
  if (!roomPlanFloorData) {
    roomPlanDisplayState = null;
    return;
  }
  const rawItems = Array.isArray(roomPlanFloorData.objects) ? roomPlanFloorData.objects : [];
  const items = rawItems.map(normalizePlanItem);
  if (!items.length) {
    roomPlanDisplayState = null;
    return;
  }
  const transform = computePlanDisplayTransform(items);
  const transformed = applyPlanDisplayTransform(items, transform);
  const floorItems = transformed.filter((item) => normalizeCategory(item.category) === "floor");
  const bounds = getBoundsFromItemCorners(floorItems.length ? floorItems : transformed);
  roomPlanDisplayState = { items: transformed, bounds, transform };
}

function resizeFloorPlanPopupForBounds(bounds) {
  if (floorPlanUserSized) return;
  const popup = el("floorPlanPopup");
  if (!popup || popup.hidden || !bounds) return;

  const panel = el("floorPlanPanel");
  const header = panel?.querySelector(".extrusion-panel-head");
  const spanU = Math.max(bounds.maxU - bounds.minU, 0.5);
  const spanV = Math.max(bounds.maxV - bounds.minV, 0.5);

  const maxPopupW = window.innerWidth * PLAN_VIEWPORT_FIT;
  const maxPopupH = window.innerHeight * PLAN_VIEWPORT_FIT;
  const headerH = header?.offsetHeight || PLAN_HEADER_FALLBACK_PX;
  const chromeW = PLAN_VIEWER_SIDE_MARGIN_PX * 2;
  const chromeH = headerH + PLAN_VIEWER_VERT_MARGIN_PX;
  const availContentW = Math.max(160, maxPopupW - chromeW);
  const availContentH = Math.max(160, maxPopupH - chromeH);
  const scale = Math.min(availContentW / spanU, availContentH / spanV);
  const contentW = spanU * scale;
  const contentH = spanV * scale;

  popup.style.width = `${Math.round(contentW + chromeW)}px`;
  popup.style.height = `${Math.round(contentH + chromeH)}px`;
  clampFloorPlanPopupPosition(popup);
}

function applyFloorPlanPopupDefaultSize() {
  if (floorPlanUserSized) return;
  const popup = el("floorPlanPopup");
  if (!popup || popup.hidden) return;
  const width = Math.min(520, Math.max(PLAN_POPUP_MIN_W, Math.floor(window.innerWidth * 0.42)));
  popup.style.width = `${width}px`;
  popup.style.height = `${PLAN_POPUP_DEFAULT_H}px`;
  clampFloorPlanPopupPosition(popup);
}

function shouldAutoFitFloorPlanPopup() {
  if (sessionKind === "usdz") return Boolean(roomPlanDisplayState?.bounds);
  if (sessionKind === "dxf") return Boolean(dxfPlanDisplayBounds || dxfGeometry?.positions?.length);
  return false;
}

function measureFloorPlanCanvasBox() {
  const canvas = el("dxfPlanCanvas");
  const viewer = canvas?.closest(".floorplan-viewer");
  if (viewer) {
    const viewerRect = viewer.getBoundingClientRect();
    if (viewerRect.width > 1 && viewerRect.height > 1) {
      return {
        w: Math.max(120, Math.floor(viewerRect.width)),
        h: Math.max(120, Math.floor(viewerRect.height)),
      };
    }
  }

  const popup = el("floorPlanPopup");
  const panel = el("floorPlanPanel");
  const header = panel?.querySelector(".extrusion-panel-head");
  if (popup && panel) {
    const popupRect = popup.getBoundingClientRect();
    const headerH = header?.getBoundingClientRect().height || PLAN_HEADER_FALLBACK_PX;
    const w = Math.max(120, Math.floor(popupRect.width - PLAN_VIEWER_SIDE_MARGIN_PX * 2));
    const h = Math.max(
      120,
      Math.floor(popupRect.height - headerH - PLAN_VIEWER_VERT_MARGIN_PX),
    );
    if (w > 1 && h > 1) {
      return { w, h };
    }
  }

  return { w: 400, h: 240 };
}

function getPlanBounds(payload) {
  const bounds = payload?.bounds || {};
  let minU = bounds.min_u ?? bounds.minU ?? bounds.min_x;
  let maxU = bounds.max_u ?? bounds.maxU ?? bounds.max_x;
  let minV = bounds.min_v ?? bounds.minV ?? bounds.min_z;
  let maxV = bounds.max_v ?? bounds.maxV ?? bounds.max_z;
  if ([minU, maxU, minV, maxV].every((v) => Number.isFinite(v))) {
    return { minU, maxU, minV, maxV };
  }
  const items = Array.isArray(payload?.objects) ? payload.objects.map(normalizePlanItem) : [];
  minU = Infinity;
  maxU = -Infinity;
  minV = Infinity;
  maxV = -Infinity;
  for (const item of items) {
    for (const [u, v] of item.corners) {
      minU = Math.min(minU, u);
      maxU = Math.max(maxU, u);
      minV = Math.min(minV, v);
      maxV = Math.max(maxV, v);
    }
  }
  if (!Number.isFinite(minU)) {
    return { minU: -1, maxU: 1, minV: -1, maxV: 1 };
  }
  return { minU, maxU, minV, maxV };
}

function normalizePlanItem(item) {
  if (!item || typeof item !== "object") {
    return { corners: [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]], center: [0, 0], long_m: 0.1, short_m: 0.1, long_angle_deg: 0 };
  }
  if (Array.isArray(item.corners) && item.corners.length >= 3) {
    const center = Array.isArray(item.center) && item.center.length >= 2
      ? [Number(item.center[0]), Number(item.center[1])]
      : [
          item.corners.reduce((s, c) => s + c[0], 0) / item.corners.length,
          item.corners.reduce((s, c) => s + c[1], 0) / item.corners.length,
        ];
    return {
      ...item,
      center,
      long_m: Number(item.long_m) || 0.5,
      short_m: Number(item.short_m) || 0.5,
      long_angle_deg: Number(item.long_angle_deg) || 0,
      corners: item.corners.map((c) => [Number(c[0]), Number(c[1])]),
    };
  }

  const u = Number(item.x ?? item.center?.[0] ?? 0);
  const v = worldZToPlanV(item.z ?? item.center?.[1] ?? 0);
  const width = Math.max(Number(item.width ?? item.long_m) || 0.5, 0.05);
  const depth = Math.max(Number(item.depth ?? item.short_m) || 0.5, 0.05);
  const yaw = ((Number(item.yaw_deg ?? item.long_angle_deg) || 0) * Math.PI) / 180;
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  const hw = width * 0.5;
  const hd = depth * 0.5;
  const corners = [
    [-hw, -hd],
    [hw, -hd],
    [hw, hd],
    [-hw, hd],
  ].map(([lx, ly]) => [u + lx * cos - ly * sin, v + lx * sin + ly * cos]);

  return {
    ...item,
    center: [u, v],
    long_m: Math.max(width, depth),
    short_m: Math.min(width, depth),
    long_angle_deg: Number(item.yaw_deg ?? item.long_angle_deg) || 0,
    corners,
  };
}

function strokePlanPolygon(ctx, corners, toCanvas, style, lineWidth = 1) {
  if (!Array.isArray(corners) || corners.length < 3) return;
  const pts = corners.map(([u, v]) => toCanvas(u, v));
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i += 1) {
    ctx.lineTo(pts[i][0], pts[i][1]);
  }
  ctx.closePath();
  if (style?.fill) {
    ctx.fillStyle = style.fill;
    ctx.fill();
  }
  ctx.strokeStyle = style?.stroke || DXF_PLAN_THEME.line;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

function tracePlanPolygonPath(ctx, corners, toCanvas) {
  if (!Array.isArray(corners) || corners.length < 3) return;
  const pts = corners.map(([u, v]) => toCanvas(u, v));
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i += 1) {
    ctx.lineTo(pts[i][0], pts[i][1]);
  }
  ctx.closePath();
}

function getPlanItemAxisDirs(item) {
  const obj = item?.id ? objects.find((o) => o.id === item.id) : null;
  if (obj) return getObjectPlanAxisDirsFromState(obj);
  const angleRad = ((Number(item?.long_angle_deg) || 0) * Math.PI) / 180;
  const longDir = [Math.cos(angleRad), Math.sin(angleRad)];
  const shortDir = [-longDir[1], longDir[0]];
  const halfLong = Math.max(Number(item?.long_m) || 0.1, 0.05) * 0.5;
  const halfShort = Math.max(Number(item?.short_m) || 0.1, 0.05) * 0.5;
  return { longDir, shortDir, halfLong, halfShort };
}

function buildPlanItemEditHandles(item, toCanvas, mapping, mode = editMode) {
  if (!item?.center || !Array.isArray(item.corners) || item.corners.length < 3) return [];
  const [cu, cv] = item.center;
  const { longDir, shortDir, halfLong, halfShort } = getPlanItemAxisDirs(item);

  if (mode === "rotate") {
    const radiusPlan = Math.max(halfLong, halfShort) * 1.18 + 0.14;
    const radiusCanvas = radiusPlan * mapping.scale;
    const [ccx, ccy] = toCanvas(cu, cv);
    const hx = cu + longDir[0] * radiusPlan;
    const hy = cv + longDir[1] * radiusPlan;
    return [
      { id: "rotate-center", type: "rotate-center", plan: [cu, cv], canvas: [ccx, ccy] },
      {
        id: "rotate-ring",
        type: "rotate-ring",
        centerCanvas: [ccx, ccy],
        radiusCanvas,
        radiusPlan,
      },
      { id: "rotate-knob", type: "rotate-knob", plan: [hx, hy], canvas: toCanvas(hx, hy) },
    ];
  }

  if (mode === "scale") {
    const handles = item.corners.map((corner, index) => ({
      id: `corner-${index}`,
      type: "scale-corner",
      plan: [Number(corner[0]), Number(corner[1])],
      canvas: toCanvas(Number(corner[0]), Number(corner[1])),
    }));
    const axisDefs = [
      { id: "axis-long-pos", axis: "long", dir: longDir, half: halfLong },
      { id: "axis-long-neg", axis: "long", dir: [-longDir[0], -longDir[1]], half: halfLong },
      { id: "axis-short-pos", axis: "short", dir: shortDir, half: halfShort },
      { id: "axis-short-neg", axis: "short", dir: [-shortDir[0], -shortDir[1]], half: halfShort },
    ];
    for (const ax of axisDefs) {
      const pu = cu + ax.dir[0] * ax.half;
      const pv = cv + ax.dir[1] * ax.half;
      handles.push({
        id: ax.id,
        type: "scale-axis",
        axis: ax.axis,
        plan: [pu, pv],
        canvas: toCanvas(pu, pv),
      });
    }
    return handles;
  }

  return [];
}

function pickFloorPlanHandleAtCanvas(canvasX, canvasY, item, mode = editMode) {
  if (!item || !floorPlanPaintMapping) return null;
  const mapping = floorPlanPaintMapping;
  const toCanvas = (u, v) => planUvToCanvas(
    u,
    v,
    mapping.minU,
    mapping.maxU,
    mapping.minV,
    mapping.pad,
    mapping.scale,
  );
  const handles = buildPlanItemEditHandles(item, toCanvas, mapping, mode);
  let best = null;
  let bestDist = PLAN_HANDLE_HIT_PX;

  for (const handle of handles) {
    if (handle.type === "rotate-ring") {
      const dist = Math.hypot(canvasX - handle.centerCanvas[0], canvasY - handle.centerCanvas[1]);
      const ringDist = Math.abs(dist - handle.radiusCanvas);
      if (ringDist <= PLAN_HANDLE_HIT_PX * 1.35 && ringDist < bestDist) {
        best = handle;
        bestDist = ringDist;
      }
      continue;
    }
    if (!handle.canvas) continue;
    const dist = Math.hypot(canvasX - handle.canvas[0], canvasY - handle.canvas[1]);
    if (dist <= PLAN_HANDLE_HIT_PX && dist < bestDist) {
      best = handle;
      bestDist = dist;
    }
  }
  return best;
}

function paintPlanHandleSquare(ctx, canvasPoint, { fill, stroke, size = 8 } = {}) {
  const [x, y] = canvasPoint;
  const half = size * 0.5;
  ctx.fillStyle = fill || "rgba(255, 240, 244, 0.95)";
  ctx.strokeStyle = stroke || "rgba(196, 77, 106, 0.95)";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x - half, y - half, size, size, 2);
  } else {
    ctx.rect(x - half, y - half, size, size);
  }
  ctx.fill();
  ctx.stroke();
}

function paintPlanHandleDot(ctx, canvasPoint, { fill, stroke, radius = 6 } = {}) {
  const [x, y] = canvasPoint;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = fill || "rgba(255, 240, 244, 0.95)";
  ctx.fill();
  ctx.strokeStyle = stroke || "rgba(196, 77, 106, 0.95)";
  ctx.lineWidth = 1.6;
  ctx.stroke();
}

function paintPlanRotateEditOverlay(ctx, item, toCanvas, mapping) {
  const handles = buildPlanItemEditHandles(item, toCanvas, mapping, "rotate");
  const ring = handles.find((h) => h.type === "rotate-ring");
  const knob = handles.find((h) => h.type === "rotate-knob");
  const center = handles.find((h) => h.type === "rotate-center");
  if (!ring || !knob || !center) return;

  ctx.save();
  ctx.beginPath();
  ctx.arc(ring.centerCanvas[0], ring.centerCanvas[1], ring.radiusCanvas, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(196, 77, 106, 0.88)";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 5]);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.beginPath();
  ctx.moveTo(center.canvas[0], center.canvas[1]);
  ctx.lineTo(knob.canvas[0], knob.canvas[1]);
  ctx.strokeStyle = "rgba(240, 160, 176, 0.92)";
  ctx.lineWidth = 1.6;
  ctx.stroke();

  paintPlanHandleDot(ctx, center.canvas, { radius: 5, fill: "rgba(196, 77, 106, 0.35)" });
  paintPlanHandleDot(ctx, knob.canvas, { radius: 7 });
  ctx.restore();
}

function paintPlanScaleEditOverlay(ctx, item, toCanvas) {
  const handles = buildPlanItemEditHandles(item, toCanvas, floorPlanPaintMapping, "scale");
  const longHandles = handles.filter((h) => h.type === "scale-axis" && h.axis === "long");
  const shortHandles = handles.filter((h) => h.type === "scale-axis" && h.axis === "short");

  ctx.save();
  if (longHandles.length === 2) {
    ctx.beginPath();
    ctx.moveTo(longHandles[0].canvas[0], longHandles[0].canvas[1]);
    ctx.lineTo(longHandles[1].canvas[0], longHandles[1].canvas[1]);
    ctx.strokeStyle = "rgba(240, 160, 176, 0.72)";
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
  }
  if (shortHandles.length === 2) {
    ctx.beginPath();
    ctx.moveTo(shortHandles[0].canvas[0], shortHandles[0].canvas[1]);
    ctx.lineTo(shortHandles[1].canvas[0], shortHandles[1].canvas[1]);
    ctx.strokeStyle = "rgba(168, 136, 232, 0.72)";
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  for (const handle of handles) {
    if (handle.type === "scale-corner") {
      paintPlanHandleSquare(ctx, handle.canvas, { size: 9 });
    } else if (handle.type === "scale-axis") {
      paintPlanHandleSquare(ctx, handle.canvas, {
        size: 8,
        fill: handle.axis === "long" ? "rgba(240, 160, 176, 0.95)" : "rgba(200, 180, 240, 0.95)",
        stroke: handle.axis === "long" ? "rgba(196, 77, 106, 0.95)" : "rgba(168, 136, 232, 0.95)",
      });
    }
  }
  ctx.restore();
}

function paintPlanSelectionHighlight(ctx, corners, toCanvas) {
  if (!Array.isArray(corners) || corners.length < 3) return;
  const pts = corners.map(([u, v]) => toCanvas(u, v));
  const theme = ROOM_PLAN_THEME.selected;

  tracePlanPolygonPath(ctx, corners, toCanvas);
  ctx.fillStyle = theme.fill;
  ctx.fill();

  ctx.save();
  ctx.shadowColor = "rgba(196, 77, 106, 0.55)";
  ctx.shadowBlur = 16;
  ctx.strokeStyle = theme.glow;
  ctx.lineWidth = 4.5;
  tracePlanPolygonPath(ctx, corners, toCanvas);
  ctx.stroke();
  ctx.restore();

  ctx.strokeStyle = theme.stroke;
  ctx.lineWidth = theme.lineWidth;
  tracePlanPolygonPath(ctx, corners, toCanvas);
  ctx.stroke();

  const accent = 7;
  ctx.strokeStyle = "rgba(255, 240, 244, 0.9)";
  ctx.lineWidth = 1.4;
  ctx.lineCap = "round";
  for (let i = 0; i < pts.length; i += 1) {
    const [cx, cy] = pts[i];
    const [px, py] = pts[(i - 1 + pts.length) % pts.length];
    const [nx, ny] = pts[(i + 1) % pts.length];
    const v1x = cx - px;
    const v1y = cy - py;
    const v2x = nx - cx;
    const v2y = ny - cy;
    const len1 = Math.hypot(v1x, v1y) || 1;
    const len2 = Math.hypot(v2x, v2y) || 1;
    const d1x = (v1x / len1) * accent;
    const d1y = (v1y / len1) * accent;
    const d2x = (v2x / len2) * accent;
    const d2y = (v2y / len2) * accent;
    ctx.beginPath();
    ctx.moveTo(cx - d1x, cy - d1y);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx + d2x, cy + d2y);
    ctx.stroke();
  }
}

function drawPlanLabel(ctx, text, u, v, toCanvas, { selected = false } = {}) {
  const label = truncatePlanLabel(text);
  if (!label) return;
  const [cx, cy] = toCanvas(u, v);
  ctx.font = `${selected ? 600 : 500} 11px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const metrics = ctx.measureText(label);
  const padX = 5;
  const padY = 3;
  const boxW = metrics.width + padX * 2;
  const boxH = 14 + padY;
  ctx.fillStyle = "rgba(12, 6, 9, 0.9)";
  ctx.strokeStyle = selected ? "rgba(232, 160, 176, 0.55)" : "rgba(200, 120, 140, 0.24)";
  ctx.lineWidth = 1;
  const x0 = cx - boxW / 2;
  const y0 = cy - boxH / 2;
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(x0, y0, boxW, boxH, 4);
    ctx.fill();
    ctx.stroke();
  } else {
    ctx.fillRect(x0, y0, boxW, boxH);
    ctx.strokeRect(x0, y0, boxW, boxH);
  }
  ctx.fillStyle = selected ? "#f0e8ec" : "#d4b8c0";
  ctx.fillText(label, cx, cy + 0.5);
}

function drawPlanAlignedDimension(ctx, center, angleDeg, lengthM, toCanvas, scale, offsetM, label) {
  if (!center || !Number.isFinite(lengthM) || lengthM <= 0) return;
  const rad = (angleDeg * Math.PI) / 180;
  const dirU = Math.cos(rad);
  const dirV = Math.sin(rad);
  const perpU = -dirV;
  const perpV = dirU;
  const half = lengthM / 2;
  const cu = center[0];
  const cv = center[1];
  const aU = cu - dirU * half;
  const aV = cv - dirV * half;
  const bU = cu + dirU * half;
  const bV = cv + dirV * half;
  const off = offsetM;
  const dimAU = aU + perpU * off;
  const dimAV = aV + perpV * off;
  const dimBU = bU + perpU * off;
  const dimBV = bV + perpV * off;
  const [ax, ay] = toCanvas(aU, aV);
  const [bx, by] = toCanvas(bU, bV);
  const [dax, day] = toCanvas(dimAU, dimAV);
  const [dbx, dby] = toCanvas(dimBU, dimBV);
  const tick = Math.max(3, Math.min(8, 4 + scale * 0.02));

  ctx.strokeStyle = "rgba(212, 184, 192, 0.72)";
  ctx.fillStyle = "rgba(232, 220, 228, 0.92)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(dax, day);
  ctx.moveTo(bx, by);
  ctx.lineTo(dbx, dby);
  ctx.moveTo(dax, day);
  ctx.lineTo(dbx, dby);
  ctx.stroke();

  const tdx = dbx - dax;
  const tdy = dby - day;
  const tLen = Math.hypot(tdx, tdy) || 1;
  const nx = -tdy / tLen;
  const ny = tdx / tLen;
  ctx.beginPath();
  ctx.moveTo(dax + nx * tick, day + ny * tick);
  ctx.lineTo(dax - nx * tick, day - ny * tick);
  ctx.moveTo(dbx + nx * tick, dby + ny * tick);
  ctx.lineTo(dbx - nx * tick, dby - ny * tick);
  ctx.stroke();

  ctx.font = "10px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(label, (dax + dbx) / 2, (day + dby) / 2 - 2);
}

function paintDxfFloorPlan(ctx, w, h) {
  if (!dxfGeometry?.positions?.length) {
    paintFloorPlanIdleScene(ctx, w, h);
    return;
  }

  const { positions } = dxfGeometry;
  let minX;
  let maxX;
  let minZ;
  let maxZ;
  if (dxfGeometry.bounds?.min && dxfGeometry.bounds?.max) {
    minX = dxfGeometry.bounds.min[0];
    maxX = dxfGeometry.bounds.max[0];
    minZ = dxfGeometry.bounds.min[2];
    maxZ = dxfGeometry.bounds.max[2];
  } else {
    minX = Infinity;
    maxX = -Infinity;
    minZ = Infinity;
    maxZ = -Infinity;
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i];
      const z = positions[i + 2];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
    }
  }

  // Plan-space bounds are computed from ALL vectors (all layers), regardless of visibility.
  // Then rotate the plan 90° if needed so the bounding box long side is vertical.
  const rawBounds = {
    minU: minX,
    maxU: maxX,
    minV: worldZToPlanV(maxZ),
    maxV: worldZToPlanV(minZ),
  };
  const padded = expandPlanBounds(rawBounds, 0.06);
  const centerU = (padded.minU + padded.maxU) / 2;
  const centerV = (padded.minV + padded.maxV) / 2;
  const spanU = Math.max(padded.maxU - padded.minU, 0.01);
  const spanV = Math.max(padded.maxV - padded.minV, 0.01);
  const rotate90 = spanU > spanV;

  const applyPlanRotation = (u, v) => {
    const du = u - centerU;
    const dv = v - centerV;
    if (!rotate90) return { u, v };
    // Clockwise 90° around center (u,v) -> (v, -u)
    return { u: centerU + dv, v: centerV - du };
  };

  // Compute bounds in the rotated space (so we can fit precisely).
  let fitMinU = Infinity;
  let fitMaxU = -Infinity;
  let fitMinV = Infinity;
  let fitMaxV = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const u = positions[i];
    const v = worldZToPlanV(positions[i + 2]);
    const p = applyPlanRotation(u, v);
    fitMinU = Math.min(fitMinU, p.u);
    fitMaxU = Math.max(fitMaxU, p.u);
    fitMinV = Math.min(fitMinV, p.v);
    fitMaxV = Math.max(fitMaxV, p.v);
  }
  const fitBounds = expandPlanBounds({ minU: fitMinU, maxU: fitMaxU, minV: fitMinV, maxV: fitMaxV }, 0.04);
  dxfPlanDisplayBounds = fitBounds;
  const fitSpanU = Math.max(fitBounds.maxU - fitBounds.minU, 0.01);
  const fitSpanV = Math.max(fitBounds.maxV - fitBounds.minV, 0.01);
  const basePad = PLAN_CANVAS_PAD_PX;
  const scale = Math.min((w - basePad * 2) / fitSpanU, (h - basePad * 2) / fitSpanV);
  // Center the fitted bounds so leftover margin is split evenly.
  const padX = Math.max(basePad, (w - fitSpanU * scale) / 2);
  const padY = Math.max(basePad, (h - fitSpanV * scale) / 2);
  const toCanvasUv = (u, v) => [
    padX + (fitBounds.maxU - Number(u)) * scale,
    padY + (Number(v) - fitBounds.minV) * scale,
  ];
  const toCanvas = (x, z) => {
    const u = Number(x);
    const v = worldZToPlanV(Number(z));
    const p = applyPlanRotation(u, v);
    return toCanvasUv(p.u, p.v);
  };

  floorPlanPaintMapping = {
    minU: fitBounds.minU,
    maxU: fitBounds.maxU,
    minV: fitBounds.minV,
    maxV: fitBounds.maxV,
    padX,
    padY,
    scale,
    transform: { rotationRad: rotate90 ? Math.PI / 2 : 0, pivot: [centerU, centerV] },
  };

  // Draw grid in plan space, post-rotation.
  drawDxfPlanGrid(ctx, fitBounds.minU, fitBounds.maxU, fitBounds.minV, fitBounds.maxV, (u, v) => toCanvasUv(u, v));

  ctx.strokeStyle = DXF_PLAN_THEME.line;
  ctx.lineWidth = 1;
  const visibleLayers = dxfPlanLayers.filter((layer) => layer.visible && layer.positions?.length);
  if (dxfPlanLayers.length) {
    for (const layer of visibleLayers) {
      strokeDxfPlanPositions(ctx, layer.positions, toCanvas);
    }
  } else {
    strokeDxfPlanPositions(ctx, positions, toCanvas);
  }

  drawDxfPlanScaleBar(ctx, w, h, scale);
}

function paintRoomPlanFootprints(ctx, w, h, displayState = roomPlanDisplayState) {
  const items = Array.isArray(displayState?.items) ? displayState.items : [];
  if (!items.length) {
    paintFloorPlanIdleScene(ctx, w, h);
    return;
  }

  items.sort((a, b) => roomPlanLayerOrder(a.category) - roomPlanLayerOrder(b.category));

  const floorItems = items.filter((item) => normalizeCategory(item.category) === "floor");
  const fitBounds = expandPlanBounds(
    displayState?.bounds || getBoundsFromItemCorners(floorItems.length ? floorItems : items),
  );
  const { minU, maxU, minV, maxV } = fitBounds;
  const pad = PLAN_CANVAS_PAD_PX;
  const spanU = Math.max(maxU - minU, 0.01);
  const spanV = Math.max(maxV - minV, 0.01);
  const scale = Math.min((w - pad * 2) / spanU, (h - pad * 2) / spanV);
  const toCanvas = (u, v) => planUvToCanvas(u, v, minU, maxU, minV, pad, scale);

  floorPlanPaintMapping = {
    minU,
    maxU,
    minV,
    maxV,
    pad,
    scale,
    transform: displayState?.transform || { rotationRad: 0, pivot: [0, 0] },
  };

  drawDxfPlanGrid(ctx, minU, maxU, minV, maxV, toCanvas);

  for (const item of items) {
    const style = roomPlanStyleForCategory(item.category);
    const lineWidth = planLineWidthForCategory(item.category, false);
    strokePlanPolygon(ctx, item.corners, toCanvas, style, lineWidth);
  }

  const selectedItem = items.find((item) => selected?.id && item.id === selected.id);
  if (selectedItem) {
    paintPlanSelectionHighlight(ctx, selectedItem.corners, toCanvas);
    if (editMode === "rotate" && !isSurfaceTextureOnlyObject(selected)) {
      paintPlanRotateEditOverlay(ctx, selectedItem, toCanvas, floorPlanPaintMapping);
    } else if (editMode === "scale" && !isSurfaceTextureOnlyObject(selected)) {
      paintPlanScaleEditOverlay(ctx, selectedItem, toCanvas);
    }
  }

  drawDxfPlanScaleBar(ctx, w, h, scale);
}

function drawFloorPlanView({ afterLayout = false } = {}) {
  const canvas = el("dxfPlanCanvas");
  if (!canvas || !isFloorPlanVisible()) return;

  if (sessionKind === "usdz") {
    updateRoomPlanDisplayState();
  }
  if (sessionKind === "dxf") {
    dxfPlanDisplayBounds = computeDxfPlanDisplayBounds();
  }
  const autoFit = shouldAutoFitFloorPlanPopup();
  if (!floorPlanUserSized) {
    if (autoFit) {
      const bounds = sessionKind === "usdz" ? roomPlanDisplayState?.bounds : dxfPlanDisplayBounds;
      resizeFloorPlanPopupForBounds(bounds);
    } else {
      applyFloorPlanPopupDefaultSize();
    }
  }

  if (autoFit && !afterLayout) {
    requestAnimationFrame(() => drawFloorPlanView({ afterLayout: true }));
    return;
  }

  const { w, h } = measureFloorPlanCanvasBox();
  floorPlanLastCanvasSize = { w, h };
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = DXF_PLAN_THEME.background;
  ctx.fillRect(0, 0, w, h);

  if (sessionKind === "dxf") {
    paintDxfFloorPlan(ctx, w, h);
    return;
  }
  if (sessionKind === "usdz") {
    paintRoomPlanFootprints(ctx, w, h, roomPlanDisplayState);
    return;
  }

  paintFloorPlanIdleScene(ctx, w, h);
}

function clearUsdZScene() {
  if (walkModeActive) exitWalkMode();
  walkSpawn = null;
  walkSpawnIsAuto = false;
  walkSpawnPickActive = false;
  walkCollisionMeshes = [];
  transformControls?.detach();
  objectMeshes.clear();
  objectPickMeshes.clear();
  usdzPickMeshes.length = 0;
  clearSelectionHighlight3D();
  if (modelRoot) {
    scene?.remove(modelRoot);
    disposeObject(modelRoot);
    modelRoot = null;
  }
  if (realModelRoot) {
    scene?.remove(realModelRoot);
    disposeObject(realModelRoot);
    realModelRoot = null;
  }
  clearReplacementUsdOverlays();
  setOverlayStatus("");
  roomPlanFloorData = null;
  roomPlanDisplayState = null;
  syncWalkModeUI();
}

function buildDxfScene(payload, options = {}) {
  if (!scene || !payload?.positions?.length) return;
  clearDxfScene();
  setDxfGeometry(payload);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(payload.positions, 3));
  const mat = new THREE.LineBasicMaterial({ color: 0x2b4c6f });
  dxfRoot = new THREE.LineSegments(geom, mat);
  dxfRoot.visible = el("showDxf")?.checked ?? true;
  scene.add(dxfRoot);
  const shouldFit = options.fitCamera === true || !hasFittedCamera;
  if (shouldFit) {
    fitCamera(dxfRoot);
    hasFittedCamera = true;
  }
  drawFloorPlanView();
}

function getStudioInputSource() {
  const checked = document.querySelector('input[name="studioInputSource"]:checked');
  return checked?.value || readStoredSource("roomplan_usdz");
}

function initStudioInputSourcePicker() {
  const picker = el("studioInputSourcePicker");
  if (!picker) return;

  const stored = readStoredSource("roomplan_usdz");
  const radio = picker.querySelector(`input[value="${stored}"]`);
  if (radio instanceof HTMLInputElement) radio.checked = true;

  decorateInputSourcePicker(picker);
  bindInputSourceThumb(picker, (sourceId) => {
    updateStudioImportPanels(sourceId);
  });
  updateStudioImportPanels(getStudioInputSource());
}

let pendingImportFile = null;

function updateStudioImportPanels(sourceId) {
  const source = getSource(sourceId);
  const fileInput = el("importFileInput");

  if (fileInput && source) fileInput.accept = source.extensions.join(",");

  if (pendingImportFile) {
    const err = validateFileForSource(pendingImportFile, sourceId);
    if (err) clearImportStaging();
  }
}

function syncImportStagingUI() {
  const importBtn = el("importBtn");
  const label = el("importDropLabel");
  const zone = el("importDropZone");
  const busy = zone?.classList.contains("is-uploading");

  if (importBtn instanceof HTMLButtonElement) {
    importBtn.disabled = busy || !pendingImportFile;
  }
  if (label) {
    label.textContent = pendingImportFile
      ? pendingImportFile.name
      : "Drop file here";
    label.classList.toggle("is-staged", Boolean(pendingImportFile));
  }
}

function stageImportFile(file) {
  if (!file) return;
  const sourceId = getStudioInputSource();
  const validationError = validateFileForSource(file, sourceId);
  if (validationError) {
    alert(validationError);
    return;
  }
  pendingImportFile = file;
  syncImportStagingUI();
}

function clearImportStaging() {
  pendingImportFile = null;
  const fileInput = el("importFileInput");
  if (fileInput instanceof HTMLInputElement) fileInput.value = "";
  syncImportStagingUI();
}

function initImportDropZone() {
  const zone = el("importDropZone");
  const fileInput = el("importFileInput");
  const importBtn = el("importBtn");
  if (!zone || !fileInput) return;

  let dragDepth = 0;

  const setDragover = (active) => {
    zone.classList.toggle("is-dragover", active);
  };

  zone.addEventListener("click", () => {
    if (zone.classList.contains("is-uploading")) return;
    fileInput.click();
  });

  importBtn?.addEventListener("click", async (event) => {
    event.preventDefault();
    if (!pendingImportFile || zone.classList.contains("is-uploading")) return;
    const file = pendingImportFile;
    clearImportStaging();
    await importStudioFile(file);
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) stageImportFile(file);
  });

  zone.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dragDepth += 1;
    setDragover(true);
  });

  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    setDragover(true);
  });

  zone.addEventListener("dragleave", (event) => {
    event.preventDefault();
    dragDepth -= 1;
    if (dragDepth <= 0) {
      dragDepth = 0;
      setDragover(false);
    }
  });

  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    setDragover(false);
    if (zone.classList.contains("is-uploading")) return;
    const file = event.dataTransfer?.files?.[0];
    if (file) stageImportFile(file);
  });

  syncImportStagingUI();
}

async function importStudioFile(file) {
  const sourceId = getStudioInputSource();
  const validationError = validateFileForSource(file, sourceId);
  if (validationError) {
    alert(validationError);
    return;
  }
  resetStudioStateForNewImport();
  const kind = getSource(sourceId)?.kind;
  if (kind === "dxf") await uploadDxfFile(file, sourceId);
  else await uploadUsdzFile(file, sourceId);
}

function resetStudioStateForNewImport() {
  // Clear any previously imported layers/objects immediately so the UI
  // doesn't show stale layer state while a new import is uploading.
  sessionId = null;
  sessionKind = null;
  sessionInputSource = null;
  selected = null;
  objects = [];
  layerDefs = [];
  lastAlignedWallIds = new Set();
  undoStack.length = 0;
  transformUndoStack.length = 0;

  // Clear both scenes; the next upload will rebuild the appropriate one.
  clearUsdZScene();
  clearDxfScene();
  updateDxfInfo(null);

  // Refresh panels that depend on layers/selection.
  renderObjectList();
  renderLayerPanel();
  renderExportPanel();
  refreshReplacementUI();
  refreshTagBulkUI();
  updateUndoUI();
  updateTransformUndoUI();
  syncWallExtractButton();
  syncExtrudeWallsButton();
  syncWallExtrusionButton();
  syncMlPocButton();
  syncMlRenderCompareButton();
  syncExtrudeCompareButton();
  syncWallRefineCompareButton();
  drawFloorPlanView();
}

function setImportDropZoneBusy(busy) {
  el("importDropZone")?.classList.toggle("is-uploading", busy);
  syncImportStagingUI();
}

function updateDxfInfo(meta) {
  const info = el("dxfInfo");
  if (!info) return;
  if (!meta) {
    info.textContent = "DXF: (none)";
    return;
  }
  const name = meta.source_filename || "drawing.dxf";
  const segs = meta.segment_count ?? 0;
  const layerCount = meta.layer_count;
  const layerNote = Number.isFinite(layerCount) && layerCount > 0 ? ` · ${layerCount} layers` : "";
  info.textContent = `DXF: ${name} (${segs.toLocaleString()} segments${layerNote})`;
}

async function uploadDxfFile(file, sourceId) {
  setImportDropZoneBusy(true);
  try {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("source", sourceId);
    const res = await fetch("/api/upload-dxf", { method: "POST", body: fd });
    let data = {};
    try {
      data = await res.json();
    } catch {
      throw new Error(`サーバー応答を読み取れませんでした (HTTP ${res.status})`);
    }
    if (!res.ok) {
      alert(data.error || `DXF import failed (HTTP ${res.status})`);
      return;
    }

    sessionId = data.session_id;
    sessionKind = "dxf";
    sessionInputSource = data.input_source || sourceId;

    clearUsdZScene();
    objects = [];
    layerDefs = [];
    selected = null;
    lastAlignedWallIds = new Set();
    renderObjectList();
    renderLayerPanel();
    hasFittedCamera = false;
    buildDxfScene(data, { fitCamera: true });
    applyDxfPlanLayers(data.layers || []);
    updateDxfInfo(data.dxf);
    syncWallExtractButton();
    syncExtrudeWallsButton();
    syncWallExtrusionButton();
    syncMlPocButton();
    syncMlRenderCompareButton();
    syncExtrudeCompareButton();
    syncWallRefineCompareButton();
    updateWallExtractInfo(null);
    updateWallExtrudeInfo(null);
    updateWallExtrusionInfo();
    clearMlComparePanels();
    clearExtrudeCompareUI();
    void loadMlCompareReports();
    void loadExtrudeCompareReport();

    renderExportPanel();
    refreshReplacementUI();
    refreshTagBulkUI();
  } catch (err) {
    console.error("uploadDxfFile failed:", err);
    alert(err?.message || "DXF import に失敗しました");
  } finally {
    setImportDropZoneBusy(false);
  }
}

async function uploadUsdzFile(file, sourceId) {
  setImportDropZoneBusy(true);
  try {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("source", sourceId);
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    let data = {};
    try {
      data = await res.json();
    } catch {
      throw new Error(`サーバー応答を読み取れませんでした (HTTP ${res.status})`);
    }
    if (!res.ok) {
      alert(data.error || `Import failed (HTTP ${res.status})`);
      return;
    }

    sessionId = data.session_id;
    sessionKind = "usdz";
    sessionInputSource = data.input_source || sourceId;
    clearDxfScene();
    updateDxfInfo(null);
    syncWallExtractButton();
    syncExtrudeWallsButton();
    syncWallExtrusionButton();
    updateWallExtractInfo(null);
    updateWallExtrudeInfo(null);
    updateWallExtrusionInfo();
    objects = Array.isArray(data.objects) ? data.objects : [];
    applyLayersFromResponse(data.layers);
    selected = null;
    lastAlignedWallIds = new Set();
    renderObjectList();
    hasFittedCamera = false;
    if (data.usdz_url) {
      await reloadUsdOverlayAndProxy(data.usdz_url, { fitCamera: true });
    } else {
      buildProxyScene(objects, { fitCamera: true });
    }

    renderExportPanel();
    refreshReplacementUI();
    refreshTagBulkUI();
    scheduleRoomPlanRefresh();
  } catch (err) {
    console.error("uploadUsdzFile failed:", err);
    alert(err?.message || "Import に失敗しました");
  } finally {
    setImportDropZoneBusy(false);
  }
}

function isFloorOrWallObject(obj) {
  const cat = normalizeCategory(obj?.category);
  if (cat === "floor" || cat === "wall") return true;
  const name = String(obj?.name || "").trim().toLowerCase();
  return name === "floor" || name === "wall";
}

/** floor/wall: テクスチャ適用のための選択のみ（変形・削除不可）。 */
function isSurfaceTextureOnlyObject(obj) {
  return isFloorOrWallObject(obj);
}

function syncTransformUiForSelection() {
  const surfaceOnly = isSurfaceTextureOnlyObject(selected);
  const canEdit = Boolean(sessionId && selected?.id);
  const picker = el("transformModePicker");
  picker?.querySelectorAll('input[type="radio"]').forEach((input) => {
    if (input instanceof HTMLInputElement) {
      input.disabled = !canEdit || surfaceOnly;
    }
  });
  const deleteBtn = el("deleteSelected");
  if (deleteBtn) deleteBtn.classList.toggle("disabled", !canEdit || surfaceOnly);
}

function getTextureAssetKey(obj) {
  const key = String(obj?.texture_asset_key || "").toLowerCase();
  if (key === "floor" || key === "wall") return key;
  return "";
}

function refreshReplacementUI() {
  const canEdit = Boolean(sessionId && selected?.id);
  const isSurface = isFloorOrWallObject(selected);
  const replacementControls = el("replacementControls");
  const textureControls = el("textureControls");
  if (replacementControls) replacementControls.hidden = isSurface;
  if (textureControls) textureControls.hidden = !isSurface;

  const applyReplacementBtn = el("applyReplacement");
  const removeReplacementBtn = el("removeReplacement");
  const replacementSelect = el("replacementAsset");
  if (applyReplacementBtn) applyReplacementBtn.classList.toggle("disabled", !canEdit || isSurface);
  if (removeReplacementBtn) removeReplacementBtn.classList.toggle("disabled", !canEdit || isSurface);
  if (replacementSelect) replacementSelect.disabled = !canEdit || isSurface;

  const applyTextureBtn = el("applyTexture");
  const removeTextureBtn = el("removeTexture");
  const textureSelect = el("textureAsset");
  if (applyTextureBtn) applyTextureBtn.classList.toggle("disabled", !canEdit || !isSurface);
  if (removeTextureBtn) removeTextureBtn.classList.toggle("disabled", !canEdit || !isSurface);
  if (textureSelect) textureSelect.disabled = !canEdit || !isSurface;

  if (textureSelect instanceof HTMLSelectElement && selected && isSurface) {
    const current = getTextureAssetKey(selected);
    if (current) textureSelect.value = current;
    else {
      const cat = normalizeCategory(selected.category);
      if (cat === "floor" || cat === "wall") textureSelect.value = cat;
    }
  }

  syncTransformUiForSelection();
}

function textureAssetUrl(textureKey) {
  const key = String(textureKey || "").toLowerCase();
  const version = textureVersionByKey.get(key) ?? 0;
  return `/api/texture-asset/${encodeURIComponent(key)}?v=${version}`;
}

function invalidateThreeTextureCache(textureKey) {
  const key = String(textureKey || "").toLowerCase();
  if (!key) return;
  for (const cacheKey of [...textureThreeCache.keys()]) {
    if (cacheKey === key || cacheKey.startsWith(`${key}@`)) {
      textureThreeCache.delete(cacheKey);
    }
  }
}

function syncTextureVersionsFromAssets(assets) {
  let changed = false;
  for (const asset of assets) {
    const key = String(asset?.key || "").toLowerCase();
    if (!key) continue;
    const nextVersion = Number(asset.mtime) || 0;
    const prevVersion = textureVersionByKey.get(key);
    if (prevVersion !== undefined && prevVersion !== nextVersion) {
      invalidateThreeTextureCache(key);
      changed = true;
    }
    textureVersionByKey.set(key, nextVersion);
  }
  return changed;
}

async function loadReplacementAssets() {
  try {
    const res = await fetch("/api/replacement-assets");
    const data = await res.json();
    if (!res.ok) return;
    replacementAssets = Array.isArray(data.assets) ? data.assets : [];
    const select = el("replacementAsset");
    if (!(select instanceof HTMLSelectElement)) return;
    if (!replacementAssets.length) return;
    select.innerHTML = "";
    for (const asset of replacementAssets) {
      const option = document.createElement("option");
      option.value = asset.key;
      option.textContent = asset.label || asset.key;
      select.appendChild(option);
    }
  } finally {
    populateTagBulkSelects();
    refreshReplacementUI();
    refreshTagBulkUI();
  }
}

async function loadTextureAssets() {
  try {
    const res = await fetch("/api/texture-assets");
    const data = await res.json();
    if (!res.ok) return false;
    const assets = Array.isArray(data.assets) ? data.assets : [];
    syncTextureVersionsFromAssets(assets);
    textureAssets = assets;
    const select = el("textureAsset");
    if (!(select instanceof HTMLSelectElement)) return true;
    if (!textureAssets.length) return true;
    select.innerHTML = "";
    for (const asset of textureAssets) {
      const option = document.createElement("option");
      option.value = asset.key;
      option.textContent = asset.label || asset.key;
      select.appendChild(option);
    }
    return true;
  } finally {
    refreshReplacementUI();
  }
}

async function preloadThreeTextures() {
  const keys = textureAssets.length
    ? textureAssets.map((a) => a.key).filter(Boolean)
    : ["floor", "wall"];
  for (const key of keys) {
    try {
      await loadThreeTexture(key);
    } catch (err) {
      console.warn("texture preload failed:", key, err);
    }
  }
}

async function refreshTextureLibrary() {
  const ok = await loadTextureAssets();
  if (!ok) return;
  await preloadThreeTextures();
  if (objects.length) buildProxyScene(objects, { fitCamera: false });
}

function loadThreeTexture(textureKey) {
  const key = String(textureKey || "").toLowerCase();
  if (!key) return Promise.resolve(null);
  const version = textureVersionByKey.get(key) ?? 0;
  const cacheKey = `${key}@${version}`;
  if (textureThreeCache.has(cacheKey)) return textureThreeCache.get(cacheKey);
  const p = textureLoader
    .loadAsync(textureAssetUrl(key))
    .then((tex) => {
      tex.wrapS = THREE.RepeatWrapping;
      tex.wrapT = THREE.RepeatWrapping;
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.userData.sharedBase = true;
      return tex;
    })
    .catch((err) => {
      console.error("three texture load failed:", key, err);
      textureThreeCache.delete(cacheKey);
      return null;
    });
  textureThreeCache.set(cacheKey, p);
  return p;
}

async function onApply(event) {
  event.preventDefault();
  if (!selected || !sessionId) return;
  if (isSurfaceTextureOnlyObject(selected)) return;
  const pivot = objectMeshes.get(selected.id);
  if (!pivot) return;

  const yawInput = el("yaw");
  const targetYaw = Number.parseFloat(yawInput?.value ?? "");
  if (Number.isFinite(targetYaw)) {
    applyYawInputToPivot(pivot, targetYaw);
  }

  await sendUpdate(
    selected.id,
    [parseFloat(el("x").value), parseFloat(el("y").value), parseFloat(el("z").value)],
    parseFloat(el("yaw").value),
    [pivot.quaternion.x, pivot.quaternion.y, pivot.quaternion.z, pivot.quaternion.w],
    { reloadModel: true }
  );
}

async function onDeleteSelected(event) {
  event.preventDefault();
  if (!selected?.id || !sessionId) return;
  if (isSurfaceTextureOnlyObject(selected)) return;
  const id = selected.id;
  if (!confirm(`Delete selected object?\n\n${selected.name || id}`)) return;

  const res = await fetch(`/api/delete-object/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "delete failed");
    return;
  }
  if (data.deleted?.id && typeof data.deleted?.usda_text === "string") {
    pushUndo({ label: "Delete", kind: "delete", deleted: { id: data.deleted.id, usda_text: data.deleted.usda_text } });
  }
  objects = Array.isArray(data.objects) ? data.objects : [];
  selected = null;
  renderObjectList();
  refreshReplacementUI();
  scheduleRoomPlanRefresh();
  await reloadUsdOverlayAndProxy(data.usdz_url, { fitCamera: false });
}

async function onApplyTexture(event) {
  event.preventDefault();
  if (!sessionId || !selected?.id) return;
  if (!isFloorOrWallObject(selected)) {
    alert("テクスチャは floor または wall のみ適用できます");
    return;
  }
  const select = el("textureAsset");
  const textureKey = select instanceof HTMLSelectElement ? select.value : "";
  if (!textureKey) {
    alert("テクスチャを選択してください");
    return;
  }
  const res = await fetch(`/api/apply-texture/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: selected.id, texture_key: textureKey }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "apply texture failed");
    return;
  }
  objects = Array.isArray(data.objects) ? data.objects : [];
  selected = objects.find((o) => o.id === selected.id) || null;
  renderObjectList();
  fillEditor(selected);
  refreshReplacementUI();
  buildProxyScene(objects, { fitCamera: false });
  if (selected) attachTransform(selected.id);
}

async function onRemoveTexture(event) {
  event.preventDefault();
  if (!sessionId || !selected?.id) return;
  const res = await fetch(`/api/remove-texture/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: selected.id }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "remove texture failed");
    return;
  }
  objects = Array.isArray(data.objects) ? data.objects : [];
  selected = objects.find((o) => o.id === selected.id) || null;
  renderObjectList();
  fillEditor(selected);
  refreshReplacementUI();
  buildProxyScene(objects, { fitCamera: false });
  if (selected) attachTransform(selected.id);
}

async function onApplyReplacement(event) {
  event.preventDefault();
  if (!sessionId || !selected?.id) return;
  const select = el("replacementAsset");
  const assetKey = select instanceof HTMLSelectElement ? select.value : "";
  if (!assetKey) {
    alert("置換アセットを選択してください");
    return;
  }
  const res = await fetch(`/api/replace-object/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: selected.id, asset_key: assetKey }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "replace failed");
    return;
  }
  objects = Array.isArray(data.objects) ? data.objects : [];
  selected = objects.find((o) => o.id === selected.id) || null;
  renderObjectList();
  fillEditor(selected);
  refreshReplacementUI();
  await reloadUsdOverlayAndProxy(data.usdz_url, { fitCamera: false });
  if (selected) attachTransform(selected.id);
}

async function onRemoveReplacement(event) {
  event.preventDefault();
  if (!sessionId || !selected?.id) return;
  const res = await fetch(`/api/unreplace-object/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: selected.id }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "remove replacement failed");
    return;
  }
  objects = Array.isArray(data.objects) ? data.objects : [];
  selected = objects.find((o) => o.id === selected.id) || null;
  renderObjectList();
  fillEditor(selected);
  refreshReplacementUI();
  await reloadUsdOverlayAndProxy(data.usdz_url, { fitCamera: false });
  if (selected) attachTransform(selected.id);
}

function initViewer() {
  const viewer = el("viewer");
  if (!viewer) throw new Error("#viewer element not found");
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x121014);
  camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
  camera.position.set(5, 4, 7);
  renderer = new THREE.WebGLRenderer({ antialias: true });
  // Ensure consistent color management across loaders/materials.
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setPixelRatio(window.devicePixelRatio);
  viewer.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.update();

  transformControls = new TransformControls(camera, renderer.domElement);
  transformControls.setMode("translate");
  transformControls.setSpace("local");
  transformControls.visible = false;
  hookTransformGizmoVisibility();
  restyleTransformControlsGizmo();
  transformControls.addEventListener("dragging-changed", onTransformDraggingChanged);
  transformControls.addEventListener("objectChange", onTransformChange);
  transformControls.addEventListener("change", enforcePlanarTransformAxes);
  scene.add(transformControls);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.0));
  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(5, 8, 5);
  scene.add(dir);
  scene.add(new THREE.GridHelper(10, 10));

  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointermove", onViewerPointerMove);
  renderer.domElement.addEventListener("pointerup", onViewerPointerUp);
  renderer.domElement.addEventListener("pointercancel", onViewerPointerUp);
  resizeViewer();
  animate();
}

function initWalkViewer() {
  const dom = renderer?.domElement;
  if (!dom) throw new Error("main renderer not ready");
  if (!walkCamera) {
    walkCamera = new THREE.PerspectiveCamera(WALK_FOV_DEFAULT, 1, 0.05, 500);
    walkCamera.up.copy(WALK_UP);
  }
  if (!pointerLockControls) {
    pointerLockControls = new PointerLockControls(walkCamera, dom);
    pointerLockControls.addEventListener("lock", () => syncWalkModeUI());
    pointerLockControls.addEventListener("unlock", () => syncWalkModeUI());
    dom.addEventListener("pointerdown", onWalkViewerPointerDown);
  }
  bindWalkWheelHandlers();
  resizeWalkViewer();
}

function isPointerOverWalkViewer(event) {
  if (!walkModeActive) return false;
  const viewer = el("viewer");
  if (!viewer) return false;
  const rect = viewer.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return false;
  const x = event.clientX;
  const y = event.clientY;
  return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
}

function bindWalkWheelHandlers() {
  if (walkWheelCaptureBound) return;
  walkWheelCaptureBound = true;
  document.addEventListener("wheel", onWalkWheelCapture, { capture: true, passive: false });
}

function unbindWalkWheelHandlers() {
  if (!walkWheelCaptureBound) return;
  walkWheelCaptureBound = false;
  document.removeEventListener("wheel", onWalkWheelCapture, { capture: true, passive: false });
}

function adjustWalkFovFromWheel(event) {
  if (!walkCamera) return;
  const direction = event.deltaY > 0 ? 1 : -1;
  walkCamera.fov = THREE.MathUtils.clamp(
    walkCamera.fov + direction * WALK_FOV_WHEEL_STEP,
    WALK_FOV_MIN,
    WALK_FOV_MAX
  );
  walkCamera.updateProjectionMatrix();
  syncWalkModeUI();
}

function onWalkWheelCapture(event) {
  if (!walkModeActive || !walkCamera) return;
  if (!isPointerOverWalkViewer(event)) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  adjustWalkFovFromWheel(event);
}

function resetWalkCameraFov() {
  if (!walkCamera) return;
  walkCamera.fov = WALK_FOV_DEFAULT;
  walkCamera.updateProjectionMatrix();
}

function onWalkViewerPointerDown(event) {
  if (!walkModeActive) return;
  if (pointerLockControls && !pointerLockControls.isLocked) {
    event.preventDefault();
    pointerLockControls.lock();
  }
}

function resizeWalkViewer() {
  const viewer = el("viewer");
  if (!viewer || !walkCamera) return;
  const rect = viewer.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  walkCamera.aspect = rect.width / rect.height;
  walkCamera.updateProjectionMatrix();
}

function onTransformDraggingChanged(e) {
  if (controls) controls.enabled = !e.value;
  if (!selected?.id) return;
  if (e.value) {
    transformDragSnapshot = { id: selected.id, before: snapshotObject(selected.id) };
  } else if (transformDragSnapshot && transformDragSnapshot.id === selected.id) {
    const before = transformDragSnapshot.before;
    const mode = transformControls?.getMode?.() || "translate";
    transformDragSnapshot = null;
    if (mode === "scale") {
      void commitScaleTransform(selected.id, before);
      return;
    }
    const after = snapshotObject(selected.id);
    if (before && after) {
      pushTransformUndo({
        label: mode === "rotate" ? "Rotate (3D)" : "Move (3D)",
        changes: [{ id: selected.id, before, after }],
      });
    }
  }
}

function resizeViewer() {
  const viewer = el("viewer");
  if (!viewer || !renderer || !camera) return;
  const rect = viewer.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
  renderer.setSize(rect.width, rect.height);
  refreshSelectionLineResolution(rect.width, rect.height);
  if (walkCamera) {
    walkCamera.aspect = rect.width / rect.height;
    walkCamera.updateProjectionMatrix();
  }
}

function animate() {
  requestAnimationFrame(animate);
  planarTransformAccentPhase += 0.045;
  enforcePlanarTransformAxes();
  syncPlanarTransformAccent();
  if (walkModeActive && walkClock) {
    const delta = Math.min(walkClock.getDelta(), 0.05);
    updateWalkMode(delta);
    updateWalkPlayerMarker();
  }
  if (walkModeActive && walkCamera) {
    renderer?.render(scene, walkCamera);
  } else {
    renderer?.render(scene, camera);
  }
}

async function reloadUsdOverlayAndProxy(usdzUrl, options = {}) {
  if (usdzUrl) {
    const sep = usdzUrl.includes("?") ? "&" : "?";
    await loadRealModel(`${usdzUrl}${sep}t=${Date.now()}`, options);
    return;
  }
  buildProxyScene(objects, options);
}

async function loadRealModel(usdzUrl, options = {}) {
  if (!scene || !usdzUrl) return;
  const run = () => loadRealModelNow(usdzUrl, options);
  const task = modelLoadChain.then(run, run);
  modelLoadChain = task.catch(() => {});
  return task;
}

async function loadRealModelNow(usdzUrl, options = {}) {
  loadingModel = true;
  syncWalkModeUI();
  lastLoadedUsdZUrl = usdzUrl;
  try {
    if (realModelRoot) {
      scene.remove(realModelRoot);
      disposeObject(realModelRoot);
      realModelRoot = null;
    }
    try {
      realModelRoot = await usdzLoader.loadAsync(usdzUrl);
      normalizeUsdZTextures(realModelRoot);
      pruneUsdZOverlayToAssets(realModelRoot);
      syncUsdOverlayTransforms(realModelRoot, objects);
      await syncReplacementUsdOverlays(realModelRoot, objects);
      await applyUsdZTextureFallbackIfMissing(realModelRoot);
      configureUsdZOverlayLayer(realModelRoot);
      scene.add(realModelRoot);
      rebuildWalkCollisionMeshes();
      autoSetWalkSpawnDefault();
      warnedUsdLoaderFailure = false;
      syncWalkModeUI();
      if (objects.length) {
        buildProxyScene(objects, { ...options, usdRoot: realModelRoot });
        if (selected?.id) {
          fillEditor(selected);
          if (!isSurfaceTextureOnlyObject(selected)) attachTransform(selected.id);
        }
      } else {
        syncObjectDisplayTargets();
      }
    } catch (err) {
      realModelRoot = null;
      walkCollisionMeshes = [];
      if (walkModeActive) exitWalkMode();
      walkSpawn = null;
      walkSpawnIsAuto = false;
      const showReal = el("showRealModel");
      if (showReal) showReal.checked = false;
      console.error("USDZ overlay load failed:", err);
      if (!warnedUsdLoaderFailure) {
        alert("USDZ overlay could not be loaded by THREE USDZLoader. Proxy editing remains available.");
        warnedUsdLoaderFailure = true;
      }
    }
  } finally {
    loadingModel = false;
    syncWalkModeUI();
  }
}

const PROXY_LABEL_RENDER_ORDER = 10;
const PROXY_LABEL_EDGE_RENDER_ORDER = 11;
const USDZ_OVERLAY_RENDER_ORDER = 20;
const REPLACEMENT_USD_RENDER_ORDER = USDZ_OVERLAY_RENDER_ORDER + 5;

function configureProxyLabelLayer(root) {
  if (!root?.traverse) return;
  root.renderOrder = PROXY_LABEL_RENDER_ORDER;
  root.traverse((node) => {
    if (node.isLineSegments || node.isLine) {
      node.renderOrder = PROXY_LABEL_EDGE_RENDER_ORDER;
      const mat = node.material;
      if (mat) {
        mat.transparent = true;
        mat.depthWrite = false;
        mat.depthTest = true;
      }
      return;
    }
    if (!node.isMesh) return;
    node.renderOrder = PROXY_LABEL_RENDER_ORDER;
    const mats = Array.isArray(node.material) ? node.material : node.material ? [node.material] : [];
    for (const m of mats) {
      if (!m) continue;
      m.transparent = true;
      m.depthWrite = false;
      m.depthTest = true;
    }
  });
}

function pruneUsdZOverlayToAssets(root) {
  if (!root?.traverse) return;
  // Hide the raw scan hull only. Asset meshes live under Mesh_grp (or as separate USD roots).
  root.traverse((node) => {
    const name = String(node.name || "");
    if (name === "Section_grp") {
      node.visible = false;
      node.traverse((child) => {
        child.visible = false;
      });
    }
  });
}

function configureUsdZOverlayLayer(root) {
  if (!root?.traverse) return;
  root.renderOrder = USDZ_OVERLAY_RENDER_ORDER;
  root.traverse((node) => {
    if (!node.isMesh) return;
    node.renderOrder = USDZ_OVERLAY_RENDER_ORDER;
    const mats = Array.isArray(node.material) ? node.material : node.material ? [node.material] : [];
    for (const m of mats) {
      if (!m) continue;
      m.depthTest = true;
      const isTransparent = Boolean(m.transparent) || (Number.isFinite(m.opacity) && m.opacity < 0.999);
      m.depthWrite = !isTransparent;
      if (isTransparent) m.transparent = true;
    }
  });
}

function normalizeUsdZTextures(root) {
  if (!root?.traverse) return;
  root.traverse((node) => {
    if (!node?.isMesh) return;
    const mats = Array.isArray(node.material) ? node.material : node.material ? [node.material] : [];
    for (const m of mats) {
      if (!m) continue;
      // USDZLoader sometimes leaves maps in linear space; enforce sRGB for color maps.
      if (m.map) {
        m.map.colorSpace = THREE.SRGBColorSpace;
        m.map.needsUpdate = true;
      }
      // Prefer visible defaults when USD authored values are missing.
      if (typeof m.metalness === "number" && !Number.isFinite(m.metalness)) m.metalness = 0;
      if (typeof m.roughness === "number" && !Number.isFinite(m.roughness)) m.roughness = 1;
      m.needsUpdate = true;
    }
  });
}

function inferSessionIdFromUsdZUrl(usdzUrl) {
  const m = String(usdzUrl || "").match(/\/api\/usdz\/([a-f0-9\-]{36})\//i);
  return m ? m[1] : "";
}

async function applyUsdZTextureFallbackIfMissing(root) {
  // If USDZLoader produced no texture maps at all, try applying the first extracted
  // texture from the session to any mesh that has UVs. This is a diagnostic fallback
  // to confirm textures are accessible and UVs exist.
  if (!root?.traverse) return;
  let matsWithMap = 0;
  let meshesWithUv = 0;
  root.traverse((node) => {
    if (!node?.isMesh) return;
    if (node.geometry?.attributes?.uv) meshesWithUv += 1;
    const mats = Array.isArray(node.material) ? node.material : node.material ? [node.material] : [];
    for (const m of mats) if (m?.map) matsWithMap += 1;
  });
  if (matsWithMap > 0) return; // already has textures

  const sid = inferSessionIdFromUsdZUrl(lastLoadedUsdZUrl) || inferSessionIdFromUsdZUrl(location?.href || "");
  if (!sid) {
    return;
  }
  if (meshesWithUv <= 0) {
    return;
  }

  try {
    const res = await fetch(`/api/session-assets/${sid}`);
    const data = await res.json();
    const assets = Array.isArray(data.assets) ? data.assets : [];
    const first = assets.find((p) => /\.(png|jpe?g|webp)$/i.test(String(p)));
    if (!first) {
      return;
    }
    const tex = await textureLoader.loadAsync(`/api/session-asset/${sid}?path=${encodeURIComponent(first)}`);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;

    let applied = 0;
    root.traverse((node) => {
      if (!node?.isMesh) return;
      if (!node.geometry?.attributes?.uv) return;
      const mats = Array.isArray(node.material) ? node.material : node.material ? [node.material] : [];
      for (const m of mats) {
        if (!m || m.map) continue;
        m.map = tex;
        m.needsUpdate = true;
        applied += 1;
      }
    });
  } catch (err) {
    console.warn("USDZ texture fallback failed:", err);
  }
}

function expectedUsdReferenceNodeName(obj) {
  const name = String(obj?.name || "").trim();
  const idxMatch = name.match(/(\d+)$/);
  const idx = idxMatch ? idxMatch[1] : "";
  if (name === "Floor0" || /^Floor/i.test(name)) return "Floor_grp";
  if (name === "Table0" || /^Table/i.test(name)) return "Table_grp";
  if (name === "Wall0" || /^Wall/i.test(name)) return "Wall_0_grp";
  if (/^Chair/i.test(name)) return idx !== "" ? `Chair_grp__r${idx}` : "Chair_grp";
  if (/^Storage/i.test(name)) return idx !== "" ? `Storage_grp__r${idx}` : "Storage_grp";
  const cat = normalizeCategory(obj?.category);
  if (cat === "floor") return "Floor_grp";
  if (cat === "table") return "Table_grp";
  if (cat === "wall") return "Wall_0_grp";
  if (cat === "chair" && idx !== "") return `Chair_grp__r${idx}`;
  if (cat === "storage" && idx !== "") return `Storage_grp__r${idx}`;
  return name;
}

function findUsdNodeForObject(usdRoot, obj) {
  if (!usdRoot?.traverse || !obj) return null;
  const candidates = [expectedUsdReferenceNodeName(obj), String(obj?.name || "").trim()].filter(Boolean);
  for (const candidate of candidates) {
    let found = null;
    usdRoot.traverse((node) => {
      if (found || node.name !== candidate) return;
      if (node.isMesh || node.isGroup || node.isObject3D) found = node;
    });
    if (found) return found;
  }
  return null;
}

function applyManifestTransformToNode(node, obj) {
  if (!node || !obj) return;
  const pos = normalizePosition(obj.matrix_position ?? obj.position);
  const quat = normalizeQuaternion(obj.quaternion_xyzw);
  node.position.set(pos[0], pos[1], pos[2]);
  node.rotation.set(0, 0, 0);
  node.scale.set(1, 1, 1);
  if (quat) node.quaternion.set(quat[0], quat[1], quat[2], quat[3]);
  else node.rotation.y = THREE.MathUtils.degToRad(safeNumber(obj.yaw_deg, 0));
  node.updateMatrix();
}

/** USDZLoader ignores mesh-level transforms inside referenced assets; re-apply manifest transforms. */
function syncUsdOverlayTransforms(usdRoot, items) {
  if (!usdRoot?.traverse || !Array.isArray(items)) return;
  for (const obj of getAssetsObjects(items)) {
    const node = findUsdNodeForObject(usdRoot, obj);
    if (node) applyManifestTransformToNode(node, obj);
  }
}

const replacementUsdSourceCache = new Map();

function clearReplacementUsdOverlays() {
  if (replacementUsdRoot && scene) {
    scene.remove(replacementUsdRoot);
    disposeObject(replacementUsdRoot);
  }
  replacementUsdRoot = null;
  rebuildUsdPickMeshes();
}

function usdOverlayHasMeshes(root) {
  if (!root?.traverse) return false;
  let found = false;
  root.traverse((node) => {
    if (node.isMesh) found = true;
  });
  return found;
}

function hideUsdObjectNodes(nodes) {
  if (!Array.isArray(nodes)) return;
  for (const node of nodes) {
    if (!node) continue;
    node.visible = false;
    node.traverse((child) => {
      child.visible = false;
    });
  }
}

function setUsdNodeTreeVisible(node, visible) {
  if (!node) return;
  node.visible = visible;
  node.traverse((child) => {
    child.visible = visible;
  });
}

function setObjectUsdNodesVisible(obj, visible) {
  if (!obj?.id) return;
  if (replacementUsdRoot) {
    let wrapper = null;
    replacementUsdRoot.traverse((node) => {
      if (!wrapper && node.userData?.objectId === obj.id) wrapper = node;
    });
    if (wrapper) {
      setUsdNodeTreeVisible(wrapper, visible);
      return;
    }
  }
  if (!realModelRoot) return;
  for (const node of collectUsdNodesForObject(realModelRoot, obj)) {
    setUsdNodeTreeVisible(node, visible);
  }
}

/** Keep proxy + USDZ visibility aligned for every object (selected or not). */
function syncObjectDisplayTargets() {
  if (sessionKind !== "usdz") return;
  const showProxyGlobal = el("showProxy")?.checked ?? true;
  const showUsdGlobal = el("showRealModel")?.checked ?? true;

  if (modelRoot) modelRoot.visible = showProxyGlobal;
  if (realModelRoot) realModelRoot.visible = showUsdGlobal;
  if (replacementUsdRoot) replacementUsdRoot.visible = showUsdGlobal;

  for (const [id, pivot] of objectMeshes.entries()) {
    const obj = objects.find((o) => o.id === id);
    if (!obj) continue;
    const inScene = shouldRenderObjectInScene(obj);
    pivot.visible = inScene && showProxyGlobal;
    setObjectUsdNodesVisible(obj, inScene && showUsdGlobal);
  }
  rebuildUsdPickMeshes();
}

function collectUsdNodesForObject(usdRoot, obj) {
  if (!usdRoot?.traverse || !obj) return [];
  const found = new Set();
  const candidates = new Set(
    [expectedUsdReferenceNodeName(obj), String(obj?.name || "").trim()].filter(Boolean)
  );
  usdRoot.traverse((node) => {
    const name = String(node.name || "").trim();
    if (!name || !candidates.has(name)) return;
    if (node.isMesh || node.isGroup || node.isObject3D) found.add(node);
  });
  return [...found];
}

function normalizeReplacementUsdOverlayMaterials(root) {
  if (!root?.traverse) return;
  root.traverse((node) => {
    if (!node.isMesh || !node.material) return;
    if (Array.isArray(node.material)) {
      node.material = node.material.map((m) => normalizeReplacementMaterial(m));
    } else {
      node.material = normalizeReplacementMaterial(node.material);
    }
  });
  normalizeUsdZTextures(root);
}

async function loadReplacementUsdSource(assetKey) {
  const key = String(assetKey || "").toLowerCase();
  if (!key) return null;
  if (replacementUsdSourceCache.has(key)) return replacementUsdSourceCache.get(key);

  const p = (async () => {
    // Replacement library USDZ files are USDC-only; THREE USDZLoader returns an empty Group
    // instead of throwing, so GLB must be tried first.
    try {
      const gltf = await gltfLoader.loadAsync(`/api/replacement-asset/${encodeURIComponent(key)}?t=${Date.now()}`);
      const root = gltf?.scene || null;
      if (root && usdOverlayHasMeshes(root)) {
        normalizeReplacementUsdOverlayMaterials(root);
        configureUsdZOverlayLayer(root);
        return root;
      }
    } catch (err) {
      console.warn("replacement GLB load failed:", key, err);
    }

    const usdzUrl = `/api/replacement-asset/${encodeURIComponent(key)}?format=usdz&t=${Date.now()}`;
    try {
      const root = await usdzLoader.loadAsync(usdzUrl);
      if (root && usdOverlayHasMeshes(root)) {
        normalizeReplacementUsdOverlayMaterials(root);
        configureUsdZOverlayLayer(root);
        return root;
      }
    } catch (err) {
      console.warn("replacement USDZ load failed:", key, err);
    }
    replacementUsdSourceCache.delete(key);
    return null;
  })();

  replacementUsdSourceCache.set(key, p);
  p.catch(() => replacementUsdSourceCache.delete(key));
  return p;
}

function cloneUsdOverlayModel(root) {
  const cloned = root.clone(true);
  normalizeReplacementUsdOverlayMaterials(cloned);
  return cloned;
}

function measureReplacementModelBounds(model) {
  const bbox = new THREE.Box3().setFromObject(model);
  if (bbox.isEmpty()) return null;
  return {
    size: bbox.getSize(new THREE.Vector3()),
    center: bbox.getCenter(new THREE.Vector3()),
  };
}

function manifestObjectQuaternion(obj) {
  const quatArr = normalizeQuaternion(obj?.quaternion_xyzw);
  if (quatArr) return new THREE.Quaternion(quatArr[0], quatArr[1], quatArr[2], quatArr[3]);
  return new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(0, 1, 0),
    THREE.MathUtils.degToRad(safeNumber(obj?.yaw_deg, 0))
  );
}

function measureObjectLocalBounds(root, objectQuat) {
  const invQuat = objectQuat.clone().invert();
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  root.updateWorldMatrix(true, false);
  root.traverse((node) => {
    if (!node.isMesh || !node.geometry?.attributes?.position) return;
    const attr = node.geometry.attributes.position;
    for (let i = 0; i < attr.count; i += 1) {
      v.fromBufferAttribute(attr, i);
      v.applyMatrix4(node.matrixWorld);
      v.applyQuaternion(invQuat);
      box.expandByPoint(v);
    }
  });
  if (box.isEmpty()) return null;
  return {
    size: box.getSize(new THREE.Vector3()),
    center: box.getCenter(new THREE.Vector3()),
  };
}

function manifestProxyDimensions(obj) {
  const dim = normalizeDimensions(obj.dimensions);
  return new THREE.Vector3(
    Math.max(Math.abs(dim[0]), 0.1),
    Math.max(Math.abs(dim[1]), 0.1),
    Math.max(Math.abs(dim[2]), 0.1)
  );
}

function measureInnerLocalBounds(root, inner) {
  const invInner = new THREE.Matrix4().copy(inner.matrixWorld).invert();
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  root.updateWorldMatrix(true, true);
  root.traverse((node) => {
    if (!node.isMesh || !node.geometry?.attributes?.position) return;
    const attr = node.geometry.attributes.position;
    for (let i = 0; i < attr.count; i += 1) {
      v.fromBufferAttribute(attr, i);
      v.applyMatrix4(node.matrixWorld);
      v.applyMatrix4(invInner);
      box.expandByPoint(v);
    }
  });
  if (box.isEmpty()) return null;
  return {
    size: box.getSize(new THREE.Vector3()),
    center: box.getCenter(new THREE.Vector3()),
    min: box.min.clone(),
    max: box.max.clone(),
  };
}

function refineReplacementToProxyBox(wrapper, inner, offsetGroup, scaleGroup, proxySize) {
  wrapper.updateMatrixWorld(true, true);
  let fitted = measureInnerLocalBounds(wrapper, inner);
  if (!fitted) return;

  scaleGroup.scale.multiply(
    new THREE.Vector3(
      proxySize.x / Math.max(fitted.size.x, 1e-6),
      proxySize.y / Math.max(fitted.size.y, 1e-6),
      proxySize.z / Math.max(fitted.size.z, 1e-6)
    )
  );

  wrapper.updateMatrixWorld(true, true);
  fitted = measureInnerLocalBounds(wrapper, inner);
  if (!fitted) return;

  offsetGroup.position.x -= fitted.center.x / scaleGroup.scale.x;
  offsetGroup.position.y -= fitted.center.y / scaleGroup.scale.y;
  offsetGroup.position.z -= fitted.center.z / scaleGroup.scale.z;

  wrapper.updateMatrixWorld(true, true);
  fitted = measureInnerLocalBounds(wrapper, inner);
  if (!fitted) return;
  const half = proxySize.clone().multiplyScalar(0.5);
  for (const axis of ["x", "y", "z"]) {
    const minTarget = -half[axis];
    const maxTarget = half[axis];
    const shift = (minTarget - fitted.min[axis] + maxTarget - fitted.max[axis]) * 0.5;
    if (Math.abs(shift) > 1e-7) {
      offsetGroup.position[axis] += shift / scaleGroup.scale[axis];
    }
  }
  wrapper.updateMatrixWorld(true);
}

/**
 * Match replacement mesh to manifest label (proxy) via affine transform:
 * M = T(matrix_position) · R(quaternion) · T(local_bbox_center) · S(proxy_dim / local_size) · T(−local_center)
 * Scale S uses the same dimensions as createManifestProxyPivot BoxGeometry, in label/object-local axes.
 */
function mountReplacementWithLabelAffine(model, obj) {
  const objectQuat = manifestObjectQuaternion(obj);
  model.position.set(0, 0, 0);
  model.scale.set(1, 1, 1);
  model.updateWorldMatrix(true, false);

  const localBounds = measureObjectLocalBounds(model, objectQuat);
  if (!localBounds) return null;

  const proxySize = manifestProxyDimensions(obj);
  const labelScale = new THREE.Vector3(
    proxySize.x / Math.max(localBounds.size.x, 1e-6),
    proxySize.y / Math.max(localBounds.size.y, 1e-6),
    proxySize.z / Math.max(localBounds.size.z, 1e-6)
  );

  const offsetGroup = new THREE.Group();
  offsetGroup.position.copy(localBounds.center).multiplyScalar(-1);

  const scaleGroup = new THREE.Group();
  scaleGroup.scale.copy(labelScale);
  scaleGroup.add(offsetGroup);
  offsetGroup.add(model);

  const bboxCenter = normalizeLocalBBoxCenter(obj.local_bbox_center);
  const inner = new THREE.Group();
  inner.position.set(bboxCenter[0], bboxCenter[1], bboxCenter[2]);
  inner.add(scaleGroup);

  const pos = normalizePosition(obj.matrix_position ?? obj.position);
  const wrapper = new THREE.Group();
  wrapper.add(inner);
  wrapper.position.set(pos[0], pos[1], pos[2]);
  wrapper.quaternion.copy(objectQuat);
  wrapper.scale.set(1, 1, 1);

  refineReplacementToProxyBox(wrapper, inner, offsetGroup, scaleGroup, proxySize);
  wrapper.updateMatrix();

  return wrapper;
}

async function syncReplacementUsdOverlays(usdRoot, items) {
  clearReplacementUsdOverlays();
  if (!scene || !Array.isArray(items)) return;
  const replaced = getAssetsObjects(items).filter((obj) => getReplacementAssetKey(obj));
  if (!replaced.length) return;

  replacementUsdRoot = new THREE.Group();
  replacementUsdRoot.name = "replacement_usd_overlays";
  let loadedCount = 0;
  let failedCount = 0;

  for (const obj of replaced) {
    const assetKey = getReplacementAssetKey(obj);
    const source = await loadReplacementUsdSource(assetKey);
    if (!source || !usdOverlayHasMeshes(source)) {
      failedCount += 1;
      continue;
    }

    const originalNodes = collectUsdNodesForObject(usdRoot, obj);

    const model = cloneUsdOverlayModel(source);
    const wrapper = mountReplacementWithLabelAffine(model, obj);
    if (!wrapper) {
      failedCount += 1;
      disposeObject(model);
      continue;
    }
    wrapper.userData.objectId = obj.id;
    wrapper.userData.replacementAssetKey = assetKey;

    hideUsdObjectNodes(originalNodes);
    wrapper.renderOrder = REPLACEMENT_USD_RENDER_ORDER;
    wrapper.traverse((node) => {
      if (node.isMesh) node.renderOrder = REPLACEMENT_USD_RENDER_ORDER;
    });
    configureUsdZOverlayLayer(wrapper);
    replacementUsdRoot.add(wrapper);
    loadedCount += 1;
  }

  if (!replacementUsdRoot.children.length) {
    clearReplacementUsdOverlays();
    document.body.dataset.replaceOverlayCount = "0";
    document.body.dataset.replaceFailedCount = String(failedCount);
    if (failedCount > 0) {
      const msg = `Replacement load failed (${failedCount})`;
      console.error("[replace]", msg);
    }
    return;
  }

  scene.add(replacementUsdRoot);
  syncObjectDisplayTargets();
  document.body.dataset.replaceOverlayCount = String(loadedCount);
  document.body.dataset.replaceFailedCount = String(failedCount);
}

function createManifestProxyPivot(o) {
  const dim = normalizeDimensions(o.dimensions);
  const pos = normalizePosition(o.matrix_position ?? o.position);
  const geom = new THREE.BoxGeometry(
    Math.max(Math.abs(dim[0]), 0.1),
    Math.max(Math.abs(dim[1]), 0.1),
    Math.max(Math.abs(dim[2]), 0.1)
  );
  const mat = new THREE.MeshStandardMaterial({
    color: colorForCategory(o.category),
    transparent: true,
    opacity: isFlatCategory(o.category) ? 0.5 : 0.8,
  });
  const pivot = new THREE.Group();
  pivot.userData.objectId = o.id;
  pivot.position.set(pos[0], pos[1], pos[2]);
  const quat = normalizeQuaternion(o.quaternion_xyzw);
  if (quat) pivot.quaternion.set(quat[0], quat[1], quat[2], quat[3]);
  else pivot.rotation.y = THREE.MathUtils.degToRad(safeNumber(o.yaw_deg, 0));
  const mesh = new THREE.Mesh(geom, mat);
  mesh.userData.objectId = o.id;
  const bboxCenter = normalizeLocalBBoxCenter(o.local_bbox_center);
  mesh.position.set(bboxCenter[0], bboxCenter[1], bboxCenter[2]);
  pivot.add(mesh);
  return { pivot, mesh, dim };
}

function buildProxyScene(items, options = {}) {
  if (!scene || !Array.isArray(items)) return;
  const sceneObjects = getAssetsObjects(items);
  if (modelRoot) {
    scene.remove(modelRoot);
    disposeObject(modelRoot);
  }
  transformControls?.detach();
  objectMeshes.clear();
  objectPickMeshes.clear();
  usdzPickMeshes.length = 0;

  modelRoot = new THREE.Group();
  for (const o of sceneObjects) {
    const built = createManifestProxyPivot(o);
    if (!built) continue;
    const { pivot, mesh, dim } = built;
    const edgeMat = new THREE.LineBasicMaterial({ color: 0, transparent: true, depthWrite: false });
    mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry), edgeMat));
    pivot.visible = shouldRenderObjectInScene(o);
    modelRoot.add(pivot);
    const textureKey = getTextureAssetKey(o);
    if (textureKey && isFloorOrWallObject(o)) {
      void applySurfaceTexture(mesh, textureKey, dim);
    }
    objectMeshes.set(o.id, pivot);
    objectPickMeshes.set(o.id, mesh);
  }
  scene.add(modelRoot);
  configureProxyLabelLayer(modelRoot);
  syncObjectDisplayTargets();
  const shouldFit = options.fitCamera === true || !hasFittedCamera;
  if (shouldFit) {
    fitCamera(modelRoot);
    hasFittedCamera = true;
  }
}

function cloneReplacementModel(root) {
  const cloned = root.clone(true);
  cloned.traverse((node) => {
    if (node.isMesh && node.material) {
      if (Array.isArray(node.material)) {
        node.material = node.material.map((m) => normalizeReplacementMaterial(m.clone()));
      } else {
        node.material = normalizeReplacementMaterial(node.material.clone());
      }
    }
  });
  return cloned;
}

function normalizeReplacementMaterial(material) {
  if (!material) return material;
  material.side = THREE.DoubleSide;
  material.depthWrite = true;
  material.depthTest = true;
  if (material.map) {
    material.map.colorSpace = THREE.SRGBColorSpace;
    material.map.needsUpdate = true;
    material.color?.set?.(0xffffff);
  }
  // Replacement proxies should stay visible even if source has alpha-heavy settings.
  material.transparent = false;
  material.opacity = 1.0;
  material.needsUpdate = true;
  return material;
}

async function loadReplacementProxyModel(assetKey) {
  const key = String(assetKey || "").toLowerCase();
  if (!key) return null;
  if (replacementProxyModelCache.has(key)) return replacementProxyModelCache.get(key);
  const p = gltfLoader.loadAsync(`/api/replacement-asset/${encodeURIComponent(key)}?t=${Date.now()}`).then((gltf) => gltf?.scene || null).catch((err) => {
    console.error("replacement proxy load failed:", key, err);
    // Do not cache failures forever; allow retry on next replacement/render.
    replacementProxyModelCache.delete(key);
    return null;
  });
  replacementProxyModelCache.set(key, p);
  return p;
}

function getReplacementAssetKey(obj) {
  const fromState = String(obj?.replacement_asset_key || "").toLowerCase();
  if (fromState === "chair" || fromState === "table" || fromState === "storage") return fromState;
  return "";
}

/**
 * Tile a face without stretching: texture long edge → face short edge, repeat to fill.
 * @returns {{ repeatU: number, repeatV: number }}
 */
function computeAspectPreservingRepeat(faceW, faceH, texW, texH) {
  const safeTexW = Math.max(texW, 1);
  const safeTexH = Math.max(texH, 1);
  const safeFaceW = Math.max(faceW, 1e-6);
  const safeFaceH = Math.max(faceH, 1e-6);
  const texLong = Math.max(safeTexW, safeTexH);
  const texShort = Math.min(safeTexW, safeTexH);
  const faceShort = Math.min(safeFaceW, safeFaceH);
  const tileLong = faceShort;
  const tileShort = faceShort * (texShort / texLong);
  const texLandscape = safeTexW >= safeTexH;
  const faceWIsShort = safeFaceW <= safeFaceH;
  if (texLandscape === faceWIsShort) {
    return {
      repeatU: safeFaceW / tileLong,
      repeatV: safeFaceH / tileShort,
    };
  }
  return {
    repeatU: safeFaceW / tileShort,
    repeatV: safeFaceH / tileLong,
  };
}

function disposeMeshMaterials(mesh) {
  if (!mesh?.material) return;
  const baseTex = mesh.userData.baseTexture;
  const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  for (const mat of mats) {
    if (mat.map && mat.map !== baseTex) mat.map.dispose();
    mat.dispose();
  }
}

async function applySurfaceTexture(mesh, textureKey, dim) {
  const baseTex = await loadThreeTexture(textureKey);
  if (!baseTex || !mesh) return;
  const img = baseTex.image;
  const texW = img?.width || 1;
  const texH = img?.height || 1;
  const w = Math.max(Math.abs(dim[0]), 0.1);
  const h = Math.max(Math.abs(dim[1]), 0.1);
  const d = Math.max(Math.abs(dim[2]), 0.1);
  // BoxGeometry face order: +x, -x, +y, -y, +z, -z (each [width along U, height along V] in meters).
  const faceSizes = [
    [d, h],
    [d, h],
    [w, d],
    [w, d],
    [w, h],
    [w, h],
  ];

  disposeMeshMaterials(mesh);
  mesh.userData.baseTexture = baseTex;

  mesh.material = faceSizes.map(([faceW, faceH]) => {
    const map = baseTex.clone();
    map.wrapS = THREE.RepeatWrapping;
    map.wrapT = THREE.RepeatWrapping;
    map.colorSpace = THREE.SRGBColorSpace;
    const { repeatU, repeatV } = computeAspectPreservingRepeat(faceW, faceH, texW, texH);
    map.repeat.set(repeatU, repeatV);
    map.needsUpdate = true;
    return new THREE.MeshStandardMaterial({
      map,
      transparent: true,
      opacity: 0.88,
      depthWrite: false,
    });
  });
  configureProxyLabelLayer(mesh);
}

async function attachReplacementProxyVisual(pivot, obj, dim, placeholderMesh = null, forcedAssetKey = "") {
  const assetKey = String(forcedAssetKey || getReplacementAssetKey(obj) || "").toLowerCase();
  if (!assetKey) return;
  const src = await loadReplacementProxyModel(assetKey);
  if (!src || !pivot?.parent || pivot.userData.objectId !== obj.id) return;

  const cloned = cloneReplacementModel(src);
  const visualGroup = new THREE.Group();
  visualGroup.userData.isReplacementVisual = true;
  visualGroup.add(cloned);

  const bbox = new THREE.Box3().setFromObject(visualGroup);
  if (bbox.isEmpty()) return;
  const size = bbox.getSize(new THREE.Vector3());
  const center = bbox.getCenter(new THREE.Vector3());
  cloned.position.sub(center);
  const sx = Math.max(Math.abs(dim[0]), 0.1) / Math.max(size.x, 1e-6);
  const sy = Math.max(Math.abs(dim[1]), 0.1) / Math.max(size.y, 1e-6);
  const sz = Math.max(Math.abs(dim[2]), 0.1) / Math.max(size.z, 1e-6);
  visualGroup.scale.set(sx, sy, sz);
  if (placeholderMesh?.position) {
    visualGroup.position.copy(placeholderMesh.position);
  } else {
    const bboxCenter = normalizeLocalBBoxCenter(obj.local_bbox_center);
    visualGroup.position.set(bboxCenter[0], bboxCenter[1], bboxCenter[2]);
  }
  pivot.add(visualGroup);
  configureProxyLabelLayer(visualGroup);
  if (placeholderMesh?.material) {
    if (Array.isArray(placeholderMesh.material)) {
      placeholderMesh.material.forEach((m) => {
        m.transparent = true;
        m.opacity = 0.12;
      });
    } else {
      placeholderMesh.material.transparent = true;
      placeholderMesh.material.opacity = 0.12;
    }
  }
}

function applyInteriorVisibility() {
  syncObjectDisplayTargets();
}

function tagUsdPickMeshesForObject(rootNode, objectId) {
  if (!rootNode || !objectId) return;
  rootNode.traverse((node) => {
    if (!node.isMesh) return;
    node.userData.objectId = objectId;
    usdzPickMeshes.push(node);
  });
}

function rebuildUsdPickMeshes() {
  usdzPickMeshes.length = 0;
  if (sessionKind !== "usdz") return;
  const sceneObjects = getAssetsObjects(objects);
  for (const obj of sceneObjects) {
    if (!shouldRenderObjectInScene(obj)) continue;
    if (replacementUsdRoot) {
      let wrapper = null;
      replacementUsdRoot.traverse((node) => {
        if (!wrapper && node.userData?.objectId === obj.id) wrapper = node;
      });
      if (wrapper) {
        tagUsdPickMeshesForObject(wrapper, obj.id);
        continue;
      }
    }
    if (realModelRoot) {
      for (const node of collectUsdNodesForObject(realModelRoot, obj)) {
        tagUsdPickMeshesForObject(node, obj.id);
      }
    }
  }
}

function getSelectionVisualRoot(obj) {
  if (!obj?.id) return null;
  if (replacementUsdRoot) {
    let wrapper = null;
    replacementUsdRoot.traverse((node) => {
      if (!wrapper && node.userData?.objectId === obj.id) wrapper = node;
    });
    if (wrapper) return wrapper;
  }
  if (realModelRoot) {
    const nodes = collectUsdNodesForObject(realModelRoot, obj);
    if (nodes.length) return nodes[0];
  }
  return objectMeshes.get(obj.id) || null;
}

function ensureSelectionHighlightRoot() {
  if (!selectionHighlightRoot) {
    selectionHighlightRoot = new THREE.Group();
    selectionHighlightRoot.name = "selection_highlight";
    selectionHighlightRoot.renderOrder = 1200;
  }
  if (scene && !scene.children.includes(selectionHighlightRoot)) {
    scene.add(selectionHighlightRoot);
  }
  return selectionHighlightRoot;
}

function clearSelectionHighlight3D() {
  if (selectionHighlightGroup && selectionHighlightRoot) {
    selectionHighlightRoot.remove(selectionHighlightGroup);
    disposeObject(selectionHighlightGroup);
    selectionHighlightGroup = null;
  }
  for (const mat of selectionLineMaterials) mat.dispose();
  selectionLineMaterials.length = 0;
}

function refreshSelectionLineResolution(width, height) {
  for (const mat of selectionLineMaterials) {
    mat.resolution.set(width, height);
  }
}

function createSelectionLineMaterial({ color, linewidth, opacity }) {
  const viewer = el("viewer");
  const rect = viewer?.getBoundingClientRect();
  const mat = new LineMaterial({
    color,
    linewidth,
    transparent: opacity < 1,
    opacity,
    depthTest: true,
    depthWrite: false,
    alphaToCoverage: true,
  });
  mat.resolution.set(rect?.width || 800, rect?.height || 600);
  selectionLineMaterials.push(mat);
  return mat;
}

function addSelectionLineLayer(group, node, { linewidth, color, opacity, order }) {
  const edges = createSilhouetteEdgesGeometry(node.geometry);
  const positions = [];
  const pos = edges.attributes.position;
  for (let i = 0; i < pos.count; i += 1) {
    positions.push(pos.getX(i), pos.getY(i), pos.getZ(i));
  }
  edges.dispose();
  if (positions.length < 6) return;

  const lineGeo = new LineGeometry();
  lineGeo.setPositions(positions);
  const line = new Line2(
    lineGeo,
    createSelectionLineMaterial({ color, linewidth, opacity }),
  );
  line.computeLineDistances();
  line.userData.isSelectionEdge = true;
  line.renderOrder = (node.renderOrder || 0) + order;
  node.updateWorldMatrix(true, false);
  line.matrix.copy(node.matrixWorld);
  line.matrixAutoUpdate = false;
  group.add(line);
}

function collectSilhouetteCornerPoints(geometry) {
  const edges = createSilhouetteEdgesGeometry(geometry);
  const pos = edges.attributes.position;
  const seen = new Set();
  const points = [];
  for (let i = 0; i < pos.count; i += 1) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const key = `${x.toFixed(5)}|${y.toFixed(5)}|${z.toFixed(5)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    points.push([x, y, z]);
  }
  edges.dispose();
  return points;
}

function getSelectionCornerSphereGeo() {
  if (!selectionCornerSphereGeo) {
    selectionCornerSphereGeo = new THREE.SphereGeometry(1, 16, 16);
  }
  return selectionCornerSphereGeo;
}

function addSelectionCornerMarkers(group, node) {
  const corners = collectSilhouetteCornerPoints(node.geometry);
  if (!corners.length) return;

  node.geometry.computeBoundingBox();
  const size = new THREE.Vector3();
  node.geometry.boundingBox.getSize(size);
  const radius = Math.max(Math.min(size.x, size.y, size.z) * 0.034, 0.018);
  const baseOrder = (node.renderOrder || 0) + 24;
  const sphereGeo = getSelectionCornerSphereGeo();

  const markerRoot = new THREE.Group();
  markerRoot.userData.isSelectionEdge = true;
  node.updateWorldMatrix(true, false);
  markerRoot.matrix.copy(node.matrixWorld);
  markerRoot.matrixAutoUpdate = false;

  for (const [x, y, z] of corners) {
    const glow = new THREE.Mesh(
      sphereGeo,
      new THREE.MeshBasicMaterial({
        color: 0xc44d6a,
        transparent: true,
        opacity: 0.5,
        depthTest: true,
        depthWrite: false,
      }),
    );
    glow.userData.isSelectionEdge = true;
    glow.position.set(x, y, z);
    glow.scale.setScalar(radius * 1.75);
    glow.renderOrder = baseOrder;
    markerRoot.add(glow);

    const core = new THREE.Mesh(
      sphereGeo,
      new THREE.MeshBasicMaterial({
        color: 0xfff8fa,
        transparent: true,
        opacity: 0.98,
        depthTest: true,
        depthWrite: false,
      }),
    );
    core.userData.isSelectionEdge = true;
    core.position.set(x, y, z);
    core.scale.setScalar(radius);
    core.renderOrder = baseOrder + 1;
    markerRoot.add(core);
  }

  group.add(markerRoot);
}

function buildSelectionHighlightGroup(root, options = {}) {
  const showEdges = options.showEdges !== false;
  const showCorners = options.showCorners !== false;
  const group = new THREE.Group();
  root.updateMatrixWorld(true);
  root.traverse((node) => {
    if (!node.isMesh || !node.geometry || node.userData?.isSelectionEdge) return;
    if (showEdges) {
      const layers = [
        { linewidth: 10, color: 0x6a1f32, opacity: 0.34, order: 1 },
        { linewidth: 7, color: 0x9e334f, opacity: 0.6, order: 2 },
        { linewidth: 4.8, color: 0xc44d6a, opacity: 0.9, order: 3 },
        { linewidth: 2.6, color: 0xfff8fa, opacity: 1, order: 4 },
      ];
      for (const layer of layers) {
        addSelectionLineLayer(group, node, layer);
      }
    }
    if (showCorners) {
      addSelectionCornerMarkers(group, node);
    }
  });
  return group.children.length ? group : null;
}

function createSilhouetteEdgesGeometry(geometry) {
  return new THREE.EdgesGeometry(geometry, SELECTION_EDGE_THRESHOLD);
}

function updateSelectionHighlight3D() {
  clearSelectionHighlight3D();
  if (!selected?.id || sessionKind !== "usdz") return;
  const root = getSelectionVisualRoot(selected);
  if (!root) return;
  selectionHighlightGroup = buildSelectionHighlightGroup(root, {
    showEdges: editMode !== "rotate",
    showCorners: editMode !== "rotate",
  });
  if (!selectionHighlightGroup) return;
  ensureSelectionHighlightRoot().add(selectionHighlightGroup);
}

function collectScenePickTargets() {
  const targets = [];
  const showReal = el("showRealModel")?.checked ?? true;
  const showProxy = el("showProxy")?.checked ?? true;
  if (showReal && (realModelRoot?.visible || replacementUsdRoot?.visible)) {
    for (const mesh of usdzPickMeshes) {
      if (mesh?.visible !== false) targets.push(mesh);
    }
  }
  if (showProxy && modelRoot?.visible) {
    for (const mesh of objectPickMeshes.values()) {
      if (mesh?.visible !== false) targets.push(mesh);
    }
  }
  return targets;
}

function resolvePickObjectId(hitObject) {
  let node = hitObject;
  while (node) {
    if (node.userData?.objectId) return node.userData.objectId;
    node = node.parent;
  }
  return null;
}

function clearObjectSelection() {
  selected = null;
  renderObjectList();
  el("editor")?.classList.add("disabled");
  transformControls?.detach();
  clearSelectionHighlight3D();
  syncObjectDisplayTargets();
  syncFloorPlanEditCursor();
  refreshReplacementUI();
  drawFloorPlanView();
  syncPlanarTransformAccent();
}

function fitCamera(root) {
  if (!root || !camera || !controls) return;
  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = Math.max(box.getSize(new THREE.Vector3()).length(), 1.0);
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(size * 0.8, size * 0.6, size * 0.9));
  camera.near = Math.max(size / 1000, 0.001);
  camera.far = size * 100;
  camera.updateProjectionMatrix();
  controls.update();
}

function onPointerDown(event) {
  if (walkSpawnPickActive) {
    setWalkSpawnFromClick(event);
    return;
  }
  if (walkModeActive) return;
  if (transformControls?.dragging) return;
  if (planeEditDrag) return;
  if (!renderer || !camera || sessionKind !== "usdz") return;
  if (!modelRoot && !realModelRoot && !replacementUsdRoot) return;

  setPickMouseFromEvent(event);
  _pickRaycaster.setFromCamera(_pickMouse, camera);
  const targets = collectScenePickTargets();
  if (!targets.length) return;
  const hits = _pickRaycaster.intersectObjects(targets, false);
  if (!hits.length) {
    if (selected) clearObjectSelection();
    return;
  }
  const id = resolvePickObjectId(hits[0].object);
  const obj = objects.find((o) => o.id === id);
  if (obj && !isObjectLayerLocked(obj)) {
    selectObject(obj);
    if (editMode === "translate" && !isSurfaceTextureOnlyObject(obj)) {
      startPlaneMoveDrag(event, obj);
    }
  }
}

function onViewerPointerMove(event) {
  if (!planeEditDrag || planeEditDrag.source === "floorplan") return;
  updatePlaneEditDrag(event);
  event.preventDefault();
}

function onViewerPointerUp(event) {
  if (!planeEditDrag || planeEditDrag.source === "floorplan") return;
  void endPlaneEditDrag(event);
  event.preventDefault();
}

function onTransformChange() {
  if (!selected || isSurfaceTextureOnlyObject(selected)) return;
  const mode = transformControls?.getMode?.() || "translate";
  const pivot = objectMeshes.get(selected.id);
  if (!pivot) return;
  if (mode === "scale") {
    // Floor-plane X/Z scale only; height (Y) stays fixed.
    pivot.scale.y = 1;
    updateSelectionHighlight3D();
    syncPlanarTransformAccent();
    return;
  }
  if (mode === "rotate") {
    // Planar-only rotate: keep yaw (Y axis) only.
    pivot.rotation.set(0, pivot.rotation.y, 0);
  }
  el("x").value = pivot.position.x.toFixed(4);
  el("y").value = pivot.position.y.toFixed(4);
  el("z").value = pivot.position.z.toFixed(4);
  el("yaw").value = THREE.MathUtils.radToDeg(pivot.rotation.y).toFixed(2);
  updateSelectionHighlight3D();
  syncPlanarTransformAccent();
  clearTimeout(transformTimer);
  transformTimer = setTimeout(() => {
    sendUpdate(
      selected.id,
      [pivot.position.x, pivot.position.y, pivot.position.z],
      THREE.MathUtils.radToDeg(pivot.rotation.y),
      [pivot.quaternion.x, pivot.quaternion.y, pivot.quaternion.z, pivot.quaternion.w],
      { reloadModel: false }
    );
  }, 250);
}

async function commitScaleTransform(id, beforeSnapshot, options = {}) {
  const pivot = objectMeshes.get(id);
  const obj = objects.find((o) => o.id === id);
  if (!pivot || !obj || !sessionId) return;

  const sx = Math.abs(pivot.scale.x);
  const sz = Math.abs(pivot.scale.z);
  if (Math.abs(sx - 1) < 1e-4 && Math.abs(sz - 1) < 1e-4) return;

  const base = normalizeDimensions(obj.dimensions);
  const beforeDims = [...base];
  const newDims = [
    Math.max(Math.abs(base[0] * sx), 0.05),
    Math.max(Math.abs(base[1]), 0.05),
    Math.max(Math.abs(base[2] * sz), 0.05),
  ];

  const res = await fetch(`/api/resize-object/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, dimensions: newDims }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "transform (scale) failed");
    if (beforeSnapshot) {
      pivot.position.set(beforeSnapshot.position[0], beforeSnapshot.position[1], beforeSnapshot.position[2]);
      pivot.quaternion.set(
        beforeSnapshot.quaternion_xyzw[0],
        beforeSnapshot.quaternion_xyzw[1],
        beforeSnapshot.quaternion_xyzw[2],
        beforeSnapshot.quaternion_xyzw[3],
      );
    }
    pivot.scale.set(1, 1, 1);
    scheduleRoomPlanRefresh();
    return;
  }

  pushTransformUndo({
    label: options.undoLabel || "Transform (scale)",
    kind: "resize",
    resize: { id, dimensions: beforeDims },
  });
  pivot.scale.set(1, 1, 1);
  objects = Array.isArray(data.objects) ? data.objects : objects;
  selected = objects.find((o) => o.id === id) || selected;
  renderObjectList();
  fillEditor(selected);
  await reloadUsdOverlayAndProxy(data.usdz_url, { fitCamera: false });
  scheduleRoomPlanRefresh();
  if (selected) attachTransform(selected.id);
}

function snapshotObject(id) {
  const pivot = objectMeshes.get(id);
  if (!pivot) return null;
  return {
    position: [pivot.position.x, pivot.position.y, pivot.position.z],
    quaternion_xyzw: [pivot.quaternion.x, pivot.quaternion.y, pivot.quaternion.z, pivot.quaternion.w],
  };
}

function snapshotById(items) {
  const m = new Map();
  for (const o of items || []) {
    if (!o?.id) continue;
    const pos = normalizePosition(o.matrix_position ?? o.position);
    const quat = normalizeQuaternion(o.quaternion_xyzw) || [0, 0, 0, 1];
    m.set(o.id, { position: [pos[0], pos[1], pos[2]], quaternion_xyzw: quat });
  }
  return m;
}

function pushUndo(entry) {
  if (isUndoing) return;
  undoStack.push({ ...entry, at: Date.now() });
  while (undoStack.length > UNDO_LIMIT) undoStack.shift();
  updateUndoUI();
}

function pushTransformUndo(entry) {
  if (isTransformUndoing || isUndoing) return;
  transformUndoStack.push({ ...entry, at: Date.now() });
  while (transformUndoStack.length > TRANSFORM_UNDO_LIMIT) transformUndoStack.shift();
  updateTransformUndoUI();
}

function updateUndoUI() {
  const btn = el("undo");
  if (!btn) return;
  if (undoStack.length) btn.classList.remove("disabled");
  else btn.classList.add("disabled");
}

function updateTransformUndoUI() {
  const btn = el("transformUndo");
  if (!btn) return;
  const hasUndo = transformUndoStack.length > 0;
  btn.classList.toggle("disabled", !hasUndo);
  btn.disabled = !hasUndo;
  btn.setAttribute("aria-disabled", hasUndo ? "false" : "true");
}

async function undoTransformEntry(entry) {
  if (entry.kind === "resize" && entry.resize?.id && Array.isArray(entry.resize?.dimensions)) {
    const res = await fetch(`/api/resize-object/${sessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: entry.resize.id, dimensions: entry.resize.dimensions }),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.error || "undo resize failed");
      return false;
    }
    objects = Array.isArray(data.objects) ? data.objects : [];
    selected = objects.find((o) => o.id === entry.resize.id) || null;
    renderObjectList();
    fillEditor(selected);
    await reloadUsdOverlayAndProxy(data.usdz_url, { fitCamera: false });
    scheduleRoomPlanRefresh();
    if (selected) attachTransform(selected.id);
    return true;
  }

  if (!entry?.changes?.length) return false;
  const preserveSelectedId = selected?.id || "";
  for (const ch of entry.changes) {
    if (!ch?.id || !ch.before) continue;
    await sendUpdate(ch.id, ch.before.position, 0, ch.before.quaternion_xyzw, {
      reloadModel: false,
      preserveSelectedId,
    });
  }
  return true;
}

async function onTransformUndo(event) {
  event.preventDefault();
  const btn = el("transformUndo");
  if (btn?.classList.contains("disabled")) return;
  const entry = transformUndoStack.pop();
  updateTransformUndoUI();
  if (!entry) return;

  isTransformUndoing = true;
  try {
    await undoTransformEntry(entry);
  } finally {
    isTransformUndoing = false;
  }
}

async function onUndo(event) {
  event.preventDefault();
  const btn = el("undo");
  if (btn?.classList.contains("disabled")) return;
  const entry = undoStack.pop();
  updateUndoUI();
  if (!entry) return;

  isUndoing = true;
  try {
    if (entry.kind === "delete" && entry.deleted?.id && typeof entry.deleted?.usda_text === "string") {
      const res = await fetch(`/api/restore-object/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: entry.deleted.id, usda_text: entry.deleted.usda_text }),
      });
      const data = await res.json();
      if (!res.ok) {
        alert(data.error || "undo restore failed");
        return;
      }
      objects = Array.isArray(data.objects) ? data.objects : [];
      selected = null;
      renderObjectList();
      await reloadUsdOverlayAndProxy(data.usdz_url, { fitCamera: false });
      return;
    }
    if (entry.kind === "geometry" && Array.isArray(entry.items) && entry.items.length) {
      let usdzUrl = null;
      for (const item of entry.items) {
        if (!item?.id || typeof item.usda_text !== "string") continue;
        const res = await fetch(`/api/restore-object-geometry/${sessionId}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: item.id, usda_text: item.usda_text }),
        });
        const data = await res.json();
        if (!res.ok) {
          alert(data.error || "undo geometry failed");
          return;
        }
        objects = Array.isArray(data.objects) ? data.objects : [];
        usdzUrl = data.usdz_url || usdzUrl;
      }
      selected = selected?.id ? objects.find((o) => o.id === selected.id) || null : null;
      renderObjectList();
      fillEditor(selected);
      await reloadUsdOverlayAndProxy(usdzUrl, { fitCamera: false });
      return;
    }
    if (!entry?.changes?.length) return;
    const preserveSelectedId = selected?.id || "";
    for (const ch of entry.changes) {
      if (!ch?.id || !ch.before) continue;
      await sendUpdate(ch.id, ch.before.position, 0, ch.before.quaternion_xyzw, {
        reloadModel: false,
        preserveSelectedId,
      });
    }
  } finally {
    isUndoing = false;
  }
}

async function sendUpdate(id, position, yawDeg, quaternionXyzw, options = {}) {
  if (!sessionId) return;
  const res = await fetch(`/api/object/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, position, yaw_deg: yawDeg, quaternion_xyzw: quaternionXyzw }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "update failed");
    return;
  }
  objects = Array.isArray(data.objects) ? data.objects : [];
  if (options.preserveSelectedId) {
    const keep = objects.find((o) => o.id === options.preserveSelectedId) || null;
    selected = keep;
  } else {
    selected = objects.find((o) => o.id === id) || null;
  }
  renderObjectList();
  fillEditor(selected);
  if (options.reloadModel && data.usdz_url) {
    await reloadUsdOverlayAndProxy(data.usdz_url, { fitCamera: false });
  } else {
    buildProxyScene(objects, { fitCamera: false });
    if (realModelRoot) syncUsdOverlayTransforms(realModelRoot, objects);
    if (selected) attachTransform(selected.id);
  }
  scheduleRoomPlanRefresh();
}

async function applyObjectTagsResponse(data, options = {}) {
  objects = Array.isArray(data.objects) ? data.objects : objects;
  layerDefs = Array.isArray(data.layers) ? data.layers : layerDefs;
  const editId = options.preserveEditId || tagEditObjectId;
  if (options.preserveSelectedId) {
    selected = objects.find((o) => o.id === options.preserveSelectedId) || null;
  } else if (selected?.id) {
    selected = objects.find((o) => o.id === selected.id) || null;
  }
  if (selected && !shouldRenderObjectInScene(selected)) {
    selected = null;
    transformControls?.detach();
  }
  renderLayerPanel();
  renderObjectList();
  refreshReplacementUI();
  refreshTagBulkUI();
  if (options.reloadModel && data.usdz_url) {
    await reloadUsdOverlayAndProxy(data.usdz_url, { fitCamera: false });
  } else {
    buildProxyScene(objects, { fitCamera: false });
    if (selected) {
      fillEditor(selected);
      if (!isSurfaceTextureOnlyObject(selected)) attachTransform(selected.id);
    }
  }
  if (editId) {
    const edited = objects.find((o) => o.id === editId);
    if (edited) openTagEditor(edited, { reopen: true });
    else closeTagEditor();
  }
}

function openTagEditor(obj, options = {}) {
  if (!obj?.id || !sessionId) return;
  tagEditObjectId = obj.id;
  const dialog = el("tagEditorDialog");
  const title = el("tagEditorTitle");
  if (title) title.textContent = obj.name || obj.id;
  renderTagEditorChips(obj);
  populateTagEditorSelect();
  const resetBtn = el("tagEditorReset");
  if (resetBtn) resetBtn.classList.toggle("disabled", !obj.tags_overridden);
  if (dialog instanceof HTMLDialogElement) {
    if (!options.reopen || !dialog.open) dialog.showModal();
  }
}

function closeTagEditor() {
  tagEditObjectId = null;
  const dialog = el("tagEditorDialog");
  if (dialog instanceof HTMLDialogElement && dialog.open) dialog.close();
}

function renderTagEditorChips(obj) {
  const box = el("tagEditorChips");
  if (!box) return;
  box.innerHTML = "";
  const tags = getObjectTags(obj);
  if (!tags.length) {
    box.innerHTML = '<div class="meta">No tags</div>';
    return;
  }
  for (const tagId of tags) {
    const chip = document.createElement("span");
    chip.className = "tag-chip" + (isAutoTag(obj, tagId) ? " is-auto" : "");
    const label = document.createElement("span");
    label.textContent = tagLabel(tagId);
    chip.appendChild(label);
    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.title = "Remove tag";
    removeBtn.textContent = "×";
    removeBtn.addEventListener("click", () => void updateObjectTag("remove", tagId));
    chip.appendChild(removeBtn);
    box.appendChild(chip);
  }
}

async function updateObjectTag(action, tag) {
  if (!sessionId || !tagEditObjectId) return;
  const res = await fetch(`/api/object-tags/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: tagEditObjectId, action, tag }),
  });
  let data = {};
  try {
    data = await res.json();
  } catch {
    alert(`タグ更新に失敗しました (HTTP ${res.status})。サーバーを再起動してページをリロードしてください。`);
    return;
  }
  if (!res.ok) {
    alert(data.error || `タグ更新に失敗しました (HTTP ${res.status})`);
    return;
  }
  await applyObjectTagsResponse(data, { preserveEditId: tagEditObjectId, reloadModel: true });
}

async function onTagEditorAdd() {
  const select = el("tagEditorAdd");
  if (!(select instanceof HTMLSelectElement) || !select.value) return;
  await updateObjectTag("add", select.value);
}

async function onTagEditorReset() {
  if (!sessionId || !tagEditObjectId) return;
  const res = await fetch(`/api/object-tags/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: tagEditObjectId, action: "reset" }),
  });
  let data = {};
  try {
    data = await res.json();
  } catch {
    alert(`タグのリセットに失敗しました (HTTP ${res.status})`);
    return;
  }
  if (!res.ok) {
    alert(data.error || `タグのリセットに失敗しました (HTTP ${res.status})`);
    return;
  }
  await applyObjectTagsResponse(data, { preserveEditId: tagEditObjectId, reloadModel: true });
}

function objectGroupKey(obj) {
  const name = String(obj?.name || "").trim();
  if (!name) return String(obj?.id || "object");
  const base = name.replace(/\d+$/, "").trim();
  return base || name;
}

function groupObjectsByName(list) {
  const groups = new Map();
  for (const obj of list) {
    const key = objectGroupKey(obj);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(obj);
  }
  return [...groups.entries()].map(([name, items]) => ({ name, items }));
}

function createObjectTagList(obj, canEditTags) {
  const wrap = document.createElement("div");
  wrap.className = "object-tag-list";
  const tags = getObjectTags(obj).filter((id) => id !== "other");
  for (const tagId of tags) {
    const chip = document.createElement("span");
    chip.className = "object-tag-chip";
    chip.textContent = tagLabel(tagId);
    wrap.appendChild(chip);
  }

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "object-tag-add";
  addBtn.textContent = "+";
  addBtn.title = "Edit RoomPlan tags";
  addBtn.setAttribute("aria-label", "Edit tags");
  addBtn.disabled = !canEditTags;
  addBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    openTagEditor(obj);
  });
  wrap.appendChild(addBtn);

  return wrap;
}

function createObjectItemElement(obj, canEditTags) {
  const row = document.createElement("div");
  row.className = "object-item" + (selected && selected.id === obj.id ? " selected" : "");

  const body = document.createElement("div");
  body.className = "object-item-body";
  const nameEl = document.createElement("div");
  nameEl.className = "object-item-name";
  nameEl.textContent = obj.name || obj.id || "";
  body.appendChild(nameEl);
  body.appendChild(createObjectTagList(obj, canEditTags));
  body.addEventListener("click", () => selectObject(obj));
  row.appendChild(body);

  return row;
}

function renderObjectList() {
  const box = el("objects");
  if (!box) return;
  box.innerHTML = "";
  box.className = "object-list";
  const visibleObjects = getAssetsObjects(objects);
  if (!visibleObjects.length) {
    box.innerHTML = '<div class="object-empty">No objects</div>';
    refreshReplacementUI();
    refreshTagBulkUI();
    return;
  }
  const canEditTags = Boolean(sessionId);
  const groups = groupObjectsByName(visibleObjects);

  for (const group of groups) {
    const details = document.createElement("details");
    details.className = "object-group";
    const hasSelected = group.items.some((obj) => selected?.id === obj.id);
    if (hasSelected) details.open = true;

    const summary = document.createElement("summary");
    summary.className = "object-group-summary";
    summary.innerHTML = `
      <span class="object-group-chevron" aria-hidden="true"></span>
      <span class="object-group-name">${escapeHtml(group.name)}</span>
      <span class="object-group-count">${group.items.length}</span>
    `;

    const list = document.createElement("div");
    list.className = "object-group-list";
    for (const obj of group.items) {
      list.appendChild(createObjectItemElement(obj, canEditTags));
    }

    details.appendChild(summary);
    details.appendChild(list);
    box.appendChild(details);
  }
  refreshReplacementUI();
  refreshTagBulkUI();
}

function selectObject(o) {
  if (!o || isObjectLayerLocked(o)) return;
  selected = o;
  renderObjectList();
  fillEditor(o);
  updateSelectionHighlight3D();
  syncObjectDisplayTargets();
  syncFloorPlanEditCursor();
  refreshReplacementUI();
  drawFloorPlanView();
  if (isSurfaceTextureOnlyObject(o)) {
    transformControls?.detach();
    return;
  }
  if (editMode === "translate") {
    transformControls?.detach();
    return;
  }
  attachTransform(o.id);
}

function attachTransform(id) {
  if (!transformControls || editMode === "translate") return;
  const mesh = objectMeshes.get(id);
  if (mesh) {
    transformControls.attach(mesh);
    enforcePlanarTransformAxes();
    restyleTransformControlsGizmo();
    syncPlanarTransformAccent();
  } else {
    transformControls.detach();
    syncPlanarTransformAccent();
  }
}

function fillEditor(o) {
  if (!o) return;
  el("editor")?.classList.remove("disabled");
  const pos = normalizePosition(o.position);
  el("x").value = pos[0].toFixed(4);
  el("y").value = pos[1].toFixed(4);
  el("z").value = pos[2].toFixed(4);
  const pivot = objectMeshes.get(o.id);
  const yawDeg = pivot ? quaternionYawDeg(pivot.quaternion) : safeNumber(o.yaw_deg, 0);
  el("yaw").value = safeNumber(yawDeg, 0).toFixed(2);
}

function normalizeCategory(cat) {
  const c = String(cat || "").toLowerCase();
  if (c === "wall" || c === "walls") return "wall";
  if (c === "floor" || c === "floors") return "floor";
  if (c === "door" || c === "doors") return "door";
  if (c === "window" || c === "windows") return "window";
  if (c === "opening" || c === "openings") return "opening";
  if (c === "chair") return "chair";
  if (c === "table") return "table";
  if (c === "storage" || c === "closet" || c === "wardrobe") return "storage";
  return "object";
}

function isInteriorCategory(cat) {
  return cat === "chair" || cat === "table" || cat === "storage";
}

function shouldRenderObjectInScene(obj) {
  return getObjectTags(obj).some((layerId) => isLayerVisible(layerId));
}
function normalize2d(v) {
  const len = Math.hypot(v[0], v[1]);
  if (len < 1e-8) return [1, 0];
  return [v[0] / len, v[1] / len];
}

function clamp(value, minValue, maxValue) {
  return Math.min(Math.max(value, minValue), maxValue);
}

function normalizeQuaternion(quat) {
  if (!Array.isArray(quat) || quat.length < 4) return null;
  const q = [safeNumber(quat[0], 0), safeNumber(quat[1], 0), safeNumber(quat[2], 0), safeNumber(quat[3], 1)];
  const len = Math.hypot(q[0], q[1], q[2], q[3]);
  if (len < 1e-8) return [0, 0, 0, 1];
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

function quaternionYawDeg(quat) {
  return THREE.MathUtils.radToDeg(new THREE.Euler().setFromQuaternion(quat, "YXZ").y);
}

function applyYawInputToPivot(pivot, targetYawDeg) {
  const currentYawDeg = quaternionYawDeg(pivot.quaternion);
  const deltaDeg = targetYawDeg - currentYawDeg;
  if (!Number.isFinite(deltaDeg) || Math.abs(deltaDeg) < 1e-6) return;
  const qDelta = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(deltaDeg));
  pivot.quaternion.premultiply(qDelta).normalize();
}

function normalizeDimensions(dim) {
  if (!Array.isArray(dim) || dim.length < 3) return [0.5, 0.5, 0.5];
  return [safeNumber(dim[0], 0.5), safeNumber(dim[1], 0.5), safeNumber(dim[2], 0.5)];
}

function normalizePosition(pos) {
  if (!Array.isArray(pos) || pos.length < 3) return [0, 0, 0];
  return [safeNumber(pos[0], 0), safeNumber(pos[1], 0), safeNumber(pos[2], 0)];
}

function normalizeLocalBBoxCenter(center) {
  if (!Array.isArray(center) || center.length < 3) return [0, 0, 0];
  return [safeNumber(center[0], 0), safeNumber(center[1], 0), safeNumber(center[2], 0)];
}

function safeNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function colorForCategory(cat) {
  const c = String(cat || "").toLowerCase();
  if (c === "wall") return 0x7f8c8d;
  if (c === "floor") return 0xb0b7c3;
  if (c === "chair") return 0x2e86de;
  if (c === "table") return 0x8e44ad;
  if (c === "storage") return 0x27ae60;
  if (c === "door") return 0xd35400;
  if (c === "window") return 0x16a085;
  return 0xf39c12;
}

function isFlatCategory(cat) {
  const c = String(cat || "").toLowerCase();
  return c === "wall" || c === "floor" || c === "window" || c === "door" || c === "opening";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function disposeObject(obj) {
  obj.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
      else child.material.dispose();
    }
  });
}

publishWalkAPI();
