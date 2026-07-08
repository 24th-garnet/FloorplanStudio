import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { USDZLoader } from 'three/addons/loaders/USDZLoader.js';
import {
  getSource,
  readStoredSource,
  storeSource,
  validateFileForSource,
} from '../input-sources.js';

let sessionId = null;
let objects = [];
let selected = null;

let scene = null;
let camera = null;
let renderer = null;
let controls = null;
let transformControls = null;
let modelRoot = null;
let realModelRoot = null;
let loadingModel = false;

let uiBound = false;
let uploading = false;
let transformTimer = null;

const objectMeshes = new Map();
const objectPickMeshes = new Map();
const usdzLoader = new USDZLoader();

let idPrefix = '';
let initialized = false;

const el = (id) => document.getElementById(idPrefix + id);

export function initSimpleEditor(prefix = '') {
  if (initialized) return;
  initialized = true;
  idPrefix = prefix;
  console.log('[simple/main.js] init', { prefix });
  initViewer();
  bindUI();
}

function getQuickInputSource() {
  const checked = document.querySelector('input[name="quickInputSource"]:checked');
  const value = checked?.value;
  if (value && getSource(value)) return value;
  const stored = readStoredSource('roomplan_usdz');
  return getSource(stored)?.kind === 'usdz' ? stored : 'roomplan_usdz';
}

function initQuickInputSourcePicker() {
  const picker = document.getElementById('quickInputSourcePicker');
  if (!picker) return;

  const stored = readStoredSource('roomplan_usdz');
  const initial = getSource(stored)?.kind === 'usdz' ? stored : 'roomplan_usdz';
  const radio = picker.querySelector(`input[value="${initial}"]`);
  if (radio instanceof HTMLInputElement) radio.checked = true;

  picker.querySelectorAll('input[name="quickInputSource"]').forEach((input) => {
    input.addEventListener('change', () => {
      if (!(input instanceof HTMLInputElement) || !input.checked) return;
      storeSource(input.value);
      updateQuickImportPanel(input.value);
    });
  });

  updateQuickImportPanel(getQuickInputSource());
}

function updateQuickImportPanel(sourceId) {
  const source = getSource(sourceId);
  const hint = document.getElementById('quickInputSourceHint');
  const fileInput = el('file');
  if (hint && source) hint.textContent = `${source.description} — transform のみ編集する軽量モードです。`;
  if (fileInput && source) fileInput.accept = source.extensions.join(',');
}

document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('app-simple')) return;
  initSimpleEditor('');
});

function bindUI() {
  if (uiBound) return;
  uiBound = true;

  initQuickInputSourcePicker();

  const uploadBtn = el('upload');
  const fileInput = el('file');
  const applyBtn = el('apply');
  const showRealModel = el('showRealModel');
  const showProxy = el('showProxy');

  if (!uploadBtn) {
    console.error('[bindUI] upload button not found: id="upload"');
    return;
  }

  if (!fileInput) {
    console.error('[bindUI] file input not found: id="file"');
    return;
  }

  uploadBtn.addEventListener('click', async (event) => {
    event.preventDefault();
    event.stopPropagation();

    if (uploading) {
      console.warn('[upload] already uploading');
      return;
    }

    uploading = true;
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Importing...';

    try {
      const file = fileInput.files[0];
      const sourceId = getQuickInputSource();
      const validationError = validateFileForSource(file, sourceId);
      if (validationError) {
        alert(validationError);
        return;
      }

      console.log('[upload] file:', file.name, file.size);

      const fd = new FormData();
      fd.append('file', file);
      fd.append('source', sourceId);

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: fd,
      });

      const data = await res.json();

      console.log('[upload] response:', data);

      if (!res.ok) {
        alert(data.error || 'upload failed');
        return;
      }

      sessionId = data.session_id;
      objects = Array.isArray(data.objects) ? data.objects : [];
      selected = null;

      console.log('[upload] sessionId:', sessionId);
      console.log('[upload] object count:', objects.length);

      renderObjectList();

      const download = el('download');
      if (download) {
        download.classList.remove('disabled');
        download.href = `/api/download/${sessionId}`;
      }

      if (data.usdz_url) {
        await loadRealModel(`${data.usdz_url}?t=${Date.now()}`, { fitCamera: true });
      } else {
        buildProxyScene(objects, { fitCamera: true });
      }
    } catch (err) {
      console.error('[upload] import failed:', err);
      alert('import failed. Consoleを確認してください。');
    } finally {
      uploading = false;
      uploadBtn.disabled = false;
      uploadBtn.textContent = 'インポート';
    }
  });

  if (applyBtn) {
    applyBtn.addEventListener('click', async (event) => {
      event.preventDefault();
      event.stopPropagation();

      if (!selected || !sessionId) {
        console.warn('[apply] no selected object or sessionId');
        return;
      }

      const pivot = objectMeshes.get(selected.id);
      if (!pivot) {
        console.warn('[apply] selected pivot not found');
        return;
      }

      const yawInput = el('yaw');
      if (yawInput) {
        const targetYaw = Number.parseFloat(yawInput.value);
        if (Number.isFinite(targetYaw)) {
          applyYawInputToPivot(pivot, targetYaw);
        }
      }

      await sendUpdate(
        selected.id,
        [
          parseFloat(el('x').value),
          parseFloat(el('y').value),
          parseFloat(el('z').value),
        ],
        parseFloat(el('yaw').value),
        [pivot.quaternion.x, pivot.quaternion.y, pivot.quaternion.z, pivot.quaternion.w],
        { reloadModel: true }
      );
    });
  }

  const modeTranslate = el('modeTranslate');
  if (modeTranslate) {
    modeTranslate.addEventListener('click', (event) => {
      event.preventDefault();
      transformControls?.setMode('translate');
    });
  }

  const modeRotate = el('modeRotate');
  if (modeRotate) {
    modeRotate.addEventListener('click', (event) => {
      event.preventDefault();
      transformControls?.setMode('rotate');
    });
  }

  if (showRealModel) {
    showRealModel.addEventListener('change', () => {
      if (realModelRoot) {
        realModelRoot.visible = showRealModel.checked;
      }
    });
  }

  if (showProxy) {
    showProxy.addEventListener('change', () => {
      if (modelRoot) {
        modelRoot.visible = showProxy.checked;
      }
    });
  }
}

function initViewer() {
  const viewer = el('viewer');

  if (!viewer) {
    console.error('[initViewer] viewer element not found: id="viewer"');
    return;
  }

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeeeeee);

  camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
  camera.position.set(5, 4, 7);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  viewer.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.update();

  transformControls = new TransformControls(camera, renderer.domElement);
  transformControls.setMode('translate');

  transformControls.addEventListener('dragging-changed', (e) => {
    controls.enabled = !e.value;
  });

  transformControls.addEventListener('objectChange', onTransformChange);

  scene.add(transformControls);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.0));

  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(5, 8, 5);
  scene.add(dir);

  scene.add(new THREE.GridHelper(10, 10));

  window.addEventListener('resize', resize);
  renderer.domElement.addEventListener('pointerdown', onPointerDown);

  resize();
  animate();

  console.log('[initViewer] viewer initialized');
}

async function loadRealModel(usdzUrl, options = {}) {
  if (!scene || !usdzUrl || loadingModel) return;
  loadingModel = true;

  try {
    if (realModelRoot) {
      scene.remove(realModelRoot);
      disposeObject(realModelRoot);
      realModelRoot = null;
    }

    const loaded = await usdzLoader.loadAsync(usdzUrl);
    realModelRoot = loaded;
    realModelRoot.name = 'RoomPlanUsdModel';
    realModelRoot.visible = el('showRealModel')?.checked ?? true;
    scene.add(realModelRoot);
    syncUsdOverlayTransforms(realModelRoot, objects);
    console.log('[loadRealModel] loaded:', usdzUrl);
    if (objects.length) {
      buildProxyScene(objects, { ...options, usdRoot: realModelRoot });
      if (selected?.id) {
        fillEditor(selected);
        attachTransform(selected.id);
      }
    }
  } catch (err) {
    console.error('[loadRealModel] failed:', err);
  } finally {
    loadingModel = false;
  }
}

function resize() {
  const viewer = el('viewer');
  if (!viewer || !renderer || !camera) return;

  const rect = viewer.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    console.warn('[resize] viewer size invalid:', rect.width, rect.height);
    return;
  }

  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
  renderer.setSize(rect.width, rect.height);
}

function animate() {
  requestAnimationFrame(animate);

  if (!renderer || !scene || !camera) return;

  renderer.render(scene, camera);
}

function expectedUsdReferenceNodeName(obj) {
  const name = String(obj?.name || '').trim();
  const cat = String(obj?.category || '').toLowerCase();
  const idxMatch = name.match(/(\d+)$/);
  const idx = idxMatch ? idxMatch[1] : '';
  if (cat === 'floor' || name === 'Floor0') return 'Floor_grp';
  if (cat === 'table' || name === 'Table0') return 'Table_grp';
  if (cat === 'wall' || name === 'Wall0') return 'Wall_0_grp';
  if (cat === 'chair' && idx !== '') return `Chair_grp__r${idx}`;
  if (cat === 'storage' && idx !== '') return `Storage_grp__r${idx}`;
  return name;
}

function findUsdNodeForObject(usdRoot, obj) {
  if (!usdRoot?.traverse || !obj) return null;
  const candidates = [expectedUsdReferenceNodeName(obj), String(obj?.name || '').trim()].filter(Boolean);
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

function syncUsdOverlayTransforms(usdRoot, items) {
  if (!usdRoot?.traverse || !Array.isArray(items)) return;
  for (const obj of items) {
    const rel = String(obj?.path || obj?.id || '').replace(/\\/g, '/');
    if (!rel.startsWith('assets/')) continue;
    const node = findUsdNodeForObject(usdRoot, obj);
    if (node) applyManifestTransformToNode(node, obj);
  }
}

function createManifestProxyPivot(o) {
  const dim = normalizeDimensions(o.dimensions);
  const pos = normalizePosition(o.matrix_position ?? o.position);
  const geometry = new THREE.BoxGeometry(
    Math.max(Math.abs(dim[0]), 0.10),
    Math.max(Math.abs(dim[1]), 0.10),
    Math.max(Math.abs(dim[2]), 0.10)
  );
  const material = new THREE.MeshStandardMaterial({
    color: colorForCategory(o.category),
    transparent: true,
    opacity: isFlatCategory(o.category) ? 0.50 : 0.80,
    roughness: 0.7,
    metalness: 0.0,
  });
  const pivot = new THREE.Group();
  pivot.name = `Pivot:${o.name || o.id}`;
  pivot.userData.objectId = o.id;
  pivot.position.set(pos[0], pos[1], pos[2]);
  const quat = normalizeQuaternion(o.quaternion_xyzw);
  if (quat) {
    pivot.quaternion.set(quat[0], quat[1], quat[2], quat[3]);
  } else {
    pivot.rotation.y = THREE.MathUtils.degToRad(safeNumber(o.yaw_deg, 0));
  }
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = o.name || o.id;
  mesh.userData.objectId = o.id;
  const bboxCenter = normalizeLocalBBoxCenter(o.local_bbox_center);
  mesh.position.set(bboxCenter[0], bboxCenter[1], bboxCenter[2]);
  pivot.add(mesh);
  return { pivot, mesh };
}

function buildProxyScene(items, options = {}) {
  console.log('[buildProxyScene] items:', items);

  if (!scene) {
    console.error('[buildProxyScene] scene is not initialized');
    return;
  }

  if (!Array.isArray(items)) {
    console.error('[buildProxyScene] items is not array:', items);
    return;
  }

  if (modelRoot) {
    scene.remove(modelRoot);
    disposeObject(modelRoot);
  }

  if (transformControls) {
    transformControls.detach();
  }

  objectMeshes.clear();
  objectPickMeshes.clear();

  modelRoot = new THREE.Group();
  modelRoot.name = 'RoomPlanProxyRoot';

  if (items.length === 0) {
    console.warn('[buildProxyScene] No objects returned from server.');

    const box = el('objects');
    if (box) {
      box.innerHTML = `
        <div class="object">
          <b>No objects found</b><br>
          <span class="meta">
            サーバは応答しましたが、編集対象objectが0件です。<br>
            app.py側のUSDA解析条件を確認してください。
          </span>
        </div>
      `;
    }

    scene.add(modelRoot);
    return;
  }

  for (const o of items) {
    console.log('[buildProxyScene] creating proxy:', o);

    const built = createManifestProxyPivot(o);
    if (!built) continue;
    const { pivot, mesh } = built;

    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({ color: 0x000000 })
    );

    mesh.add(edges);
    modelRoot.add(pivot);
    objectMeshes.set(o.id, pivot);
    objectPickMeshes.set(o.id, mesh);
  }

  scene.add(modelRoot);
  modelRoot.visible = el('showProxy')?.checked ?? true;

  console.log('[buildProxyScene] proxy mesh count:', objectMeshes.size);

  if (options.fitCamera !== false) {
    fitCamera(modelRoot);
  }
}

function isFlatCategory(cat) {
  const c = String(cat || '').toLowerCase();
  return c === 'wall' || c === 'floor' || c === 'window' || c === 'door' || c === 'opening';
}

function normalizeDimensions(dim) {
  if (!Array.isArray(dim) || dim.length < 3) {
    return [0.5, 0.5, 0.5];
  }

  return [
    safeNumber(dim[0], 0.5),
    safeNumber(dim[1], 0.5),
    safeNumber(dim[2], 0.5),
  ];
}

function normalizePosition(pos) {
  if (!Array.isArray(pos) || pos.length < 3) {
    return [0, 0, 0];
  }

  return [
    safeNumber(pos[0], 0),
    safeNumber(pos[1], 0),
    safeNumber(pos[2], 0),
  ];
}

function normalizeLocalBBoxCenter(center) {
  if (!Array.isArray(center) || center.length < 3) {
    return [0, 0, 0];
  }

  return [
    safeNumber(center[0], 0),
    safeNumber(center[1], 0),
    safeNumber(center[2], 0),
  ];
}

function safeNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function colorForCategory(cat) {
  switch (String(cat || '').toLowerCase()) {
    case 'wall':
      return 0x7f8c8d;
    case 'floor':
      return 0xb0b7c3;
    case 'chair':
      return 0x2e86de;
    case 'table':
      return 0x8e44ad;
    case 'storage':
      return 0x27ae60;
    case 'door':
      return 0xd35400;
    case 'window':
      return 0x16a085;
    case 'opening':
      return 0xc0392b;
    default:
      return 0xf39c12;
  }
}

function fitCamera(root) {
  if (!root || !camera || !controls) return;

  const box = new THREE.Box3().setFromObject(root);

  if (box.isEmpty()) {
    console.warn('[fitCamera] bounding box is empty');
    return;
  }

  const center = box.getCenter(new THREE.Vector3());
  const sizeVec = box.getSize(new THREE.Vector3());
  const size = Math.max(sizeVec.length(), 1.0);

  controls.target.copy(center);

  camera.position.copy(center).add(
    new THREE.Vector3(size * 0.8, size * 0.6, size * 0.9)
  );

  camera.near = Math.max(size / 1000, 0.001);
  camera.far = size * 100;
  camera.updateProjectionMatrix();

  controls.update();

  console.log('[fitCamera] center:', center, 'size:', size);
}

function onPointerDown(event) {
  if (!renderer || !camera || !modelRoot) return;

  const rect = renderer.domElement.getBoundingClientRect();

  const mouse = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1
  );

  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(mouse, camera);

  const hits = raycaster.intersectObjects([...objectPickMeshes.values()], false);

  if (hits.length > 0) {
    const id = hits[0].object.userData.objectId;
    const obj = objects.find((o) => o.id === id);

    if (obj) {
      selectObject(obj);
    }
  }
}

function onTransformChange() {
  if (!selected) return;

  const mesh = objectMeshes.get(selected.id);
  if (!mesh) return;

  const xInput = el('x');
  const yInput = el('y');
  const zInput = el('z');
  const yawInput = el('yaw');

  if (xInput) xInput.value = mesh.position.x.toFixed(4);
  if (yInput) yInput.value = mesh.position.y.toFixed(4);
  if (zInput) zInput.value = mesh.position.z.toFixed(4);
  if (yawInput) yawInput.value = THREE.MathUtils.radToDeg(mesh.rotation.y).toFixed(2);

  clearTimeout(transformTimer);

  transformTimer = setTimeout(() => {
    sendUpdate(
      selected.id,
      [mesh.position.x, mesh.position.y, mesh.position.z],
      THREE.MathUtils.radToDeg(mesh.rotation.y),
      [mesh.quaternion.x, mesh.quaternion.y, mesh.quaternion.z, mesh.quaternion.w],
      { reloadModel: false }
    );
  }, 300);
}

async function sendUpdate(id, position, yawDeg, quaternionXyzw, options = {}) {
  if (!sessionId) return;
  const reloadModel = Boolean(options.reloadModel);

  console.log('[sendUpdate]', id, position, yawDeg);

  const res = await fetch(`/api/object/${sessionId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      id,
      position,
      yaw_deg: yawDeg,
      quaternion_xyzw: quaternionXyzw,
    }),
  });

  const data = await res.json();

  console.log('[sendUpdate] response:', data);

  if (!res.ok) {
    alert(data.error || 'update failed');
    return;
  }

  objects = Array.isArray(data.objects) ? data.objects : [];
  selected = objects.find((o) => o.id === id) || null;

  renderObjectList();
  fillEditor(selected);

  if (reloadModel && data.usdz_url) {
    await loadRealModel(`${data.usdz_url}?t=${Date.now()}`);
  } else {
    buildProxyScene(objects);
    if (selected) attachTransform(selected.id);
  }
}

function renderObjectList() {
  const box = el('objects');

  if (!box) {
    console.error('[renderObjectList] objects element not found: id="objects"');
    return;
  }

  console.log('[renderObjectList] objects:', objects);

  box.innerHTML = '';

  if (!Array.isArray(objects) || objects.length === 0) {
    box.innerHTML = `
      <div class="object">
        <b>No objects</b><br>
        <span class="meta">
          objects配列が空です。Consoleの "[upload] response:" の中身を確認してください。
        </span>
      </div>
    `;
    return;
  }

  objects.forEach((o) => {
    const div = document.createElement('div');
    div.className = 'object' + (selected && selected.id === o.id ? ' selected' : '');

    const name = escapeHtml(o.name || o.id || '');
    const category = escapeHtml(o.category || '');
    const path = escapeHtml(o.path || '');
    const uuid = escapeHtml(o.uuid || '');

    div.innerHTML = `
      <b>${name}</b>
      <span class="meta">${category}</span>
      <br>
      <div class="meta">${path}<br>${uuid}</div>
    `;

    div.addEventListener('click', () => {
      selectObject(o);
    });

    box.appendChild(div);
  });
}

function selectObject(o) {
  selected = o;

  console.log('[selectObject]', o);

  renderObjectList();
  fillEditor(o);
  attachTransform(o.id);
}

function attachTransform(id) {
  if (!transformControls) return;

  const mesh = objectMeshes.get(id);

  if (mesh) {
    transformControls.attach(mesh);
  } else {
    console.warn('[attachTransform] mesh not found:', id);
  }
}

function fillEditor(o) {
  if (!o) return;

  const editor = el('editor');
  if (editor) {
    editor.classList.remove('disabled');
  }

  const pos = normalizePosition(o.position);

  const xInput = el('x');
  const yInput = el('y');
  const zInput = el('z');
  const yawInput = el('yaw');

  if (xInput) xInput.value = pos[0].toFixed(4);
  if (yInput) yInput.value = pos[1].toFixed(4);
  if (zInput) zInput.value = pos[2].toFixed(4);
  if (yawInput) {
    const pivot = objectMeshes.get(o.id);
    const yawDeg = pivot ? quaternionYawDeg(pivot.quaternion) : safeNumber(o.yaw_deg, 0);
    yawInput.value = safeNumber(yawDeg, 0).toFixed(2);
  }
}

function disposeObject(obj) {
  obj.traverse((child) => {
    if (child.geometry) {
      child.geometry.dispose();
    }

    if (child.material) {
      if (Array.isArray(child.material)) {
        child.material.forEach((m) => m.dispose());
      } else {
        child.material.dispose();
      }
    }
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function normalizeQuaternion(quat) {
  if (!Array.isArray(quat) || quat.length < 4) {
    return null;
  }
  const q = [
    safeNumber(quat[0], 0),
    safeNumber(quat[1], 0),
    safeNumber(quat[2], 0),
    safeNumber(quat[3], 1),
  ];
  const len = Math.hypot(q[0], q[1], q[2], q[3]);
  if (len < 1e-8) {
    return [0, 0, 0, 1];
  }
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

function quaternionYawDeg(quat) {
  const euler = new THREE.Euler().setFromQuaternion(quat, 'YXZ');
  return THREE.MathUtils.radToDeg(euler.y);
}

function applyYawInputToPivot(pivot, targetYawDeg) {
  const currentYawDeg = quaternionYawDeg(pivot.quaternion);
  const deltaDeg = targetYawDeg - currentYawDeg;
  if (!Number.isFinite(deltaDeg) || Math.abs(deltaDeg) < 1e-6) return;

  const qDelta = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(0, 1, 0),
    THREE.MathUtils.degToRad(deltaDeg)
  );
  pivot.quaternion.premultiply(qDelta).normalize();
}