/**
 * Interactive 3D Tabletop Raycasting & Drag Controller (`stageInteractionController.ts`)
 *
 * Implements high-performance Three.js pointer picking, throttled hover
 * detection, and tabletop plane dragging (`THREE.Plane(0, 1, 0)`) so users can
 * directly grab and reposition the reconstructed target object or obstacles
 * inside the 3D simulation stage while automatically handing off `OrbitControls`.
 *
 * Key engineering guarantees:
 * - Split-viewport aware NDC calculation: maps pointer coordinates accurately
 *   in both `Single` mode (full canvas) and `Dual` mode (left half primary
 *   camera viewport).
 * - Throttled `pointermove` hover raycasting (~25 Hz max) so hovering never
 *   causes frame drops on software WebGL.
 * - Pre-allocated `Raycaster`, `Vector2`, `Vector3`, and `Plane` instances
 *   for zero per-event heap garbage.
 *
 * Use cases:
 * - Dragging the ceramic mug or an obstacle into a tighter corridor on the
 *   tabletop to watch the robot arm dynamically re-plan or collide.
 * - Highlighting interactive tabletop assets on hover with cursor feedback.
 */

import * as THREE from 'three';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { StageViewModeId } from '../config/appConfig';
import { logFunctionCall } from '../utils/logger';

export interface DraggableObjectEntry {
  bodyId: string;
  meshGroup: THREE.Group;
}

export interface StageInteractionHandlers {
  getDraggables: () => DraggableObjectEntry[];
  getLayoutMode: () => StageViewModeId;
  isInteractiveEnabled: () => boolean;
  onDragStart: (bodyId: string) => void;
  onDragMove: (bodyId: string, worldX: number, worldZ: number) => void;
  onDragEnd: (bodyId: string) => void;
  onHoverChange: (bodyId: string | null) => void;
}

/**
 * Attaches raycast hover and ray-plane tabletop drag listeners to the Three.js
 * WebGL canvas and returns a clean disposer function.
 */
export function attachStageInteractionController(params: {
  domElement: HTMLCanvasElement;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  handlers: StageInteractionHandlers;
}): () => void {
  logFunctionCall('attachStageInteractionController');
  const { domElement, camera, controls, handlers } = params;

  const raycaster = new THREE.Raycaster();
  const pointerNdc = new THREE.Vector2();
  const tabletopPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  const planeHit = new THREE.Vector3();
  const dragOffset = new THREE.Vector3();

  let activeBodyId: string | null = null;
  let hoveredBodyId: string | null = null;
  let lastHoverCheckMs = 0;

  /**
   * Computes normalized device coordinates for the primary orbitable viewport
   * (full canvas in Single mode, left half in Dual mode). Returns false if the
   * pointer lies outside the primary viewport (e.g., over the right wrist cam).
   */
  const updatePointerNdc = (clientX: number, clientY: number): boolean => {
    const rect = domElement.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;

    const layout = handlers.getLayoutMode();
    const activeWidth = layout === 'single' ? rect.width : rect.width * 0.5;
    const localX = clientX - rect.left;
    const localY = clientY - rect.top;

    if (localX < 0 || localX > activeWidth || localY < 0 || localY > rect.height) {
      return false;
    }

    pointerNdc.x = (localX / activeWidth) * 2 - 1;
    pointerNdc.y = -(localY / rect.height) * 2 + 1;
    return true;
  };

  /**
   * Finds the topmost draggable tabletop body under the current pointer ray.
   */
  const pickDraggableBody = (): DraggableObjectEntry | null => {
    const entries = handlers.getDraggables();
    if (entries.length === 0) return null;

    raycaster.setFromCamera(pointerNdc, camera);
    const groups = entries.map((e) => e.meshGroup);
    const hits = raycaster.intersectObjects(groups, true);
    if (hits.length === 0) return null;

    let cursor: THREE.Object3D | null = hits[0].object;
    while (cursor) {
      const match = entries.find((e) => e.meshGroup === cursor);
      if (match) return match;
      cursor = cursor.parent;
    }
    return null;
  };

  const onPointerDown = (evt: PointerEvent): void => {
    if (evt.button !== 0 || !handlers.isInteractiveEnabled()) return;
    if (!updatePointerNdc(evt.clientX, evt.clientY)) return;

    const picked = pickDraggableBody();
    if (!picked) return;

    raycaster.setFromCamera(pointerNdc, camera);
    if (!raycaster.ray.intersectPlane(tabletopPlane, planeHit)) return;

    activeBodyId = picked.bodyId;
    dragOffset.set(
      picked.meshGroup.position.x - planeHit.x,
      0,
      picked.meshGroup.position.z - planeHit.z,
    );
    controls.enabled = false;
    domElement.style.cursor = 'grabbing';
    domElement.setPointerCapture?.(evt.pointerId);
    handlers.onDragStart(activeBodyId);
  };

  const onPointerMove = (evt: PointerEvent): void => {
    if (!handlers.isInteractiveEnabled()) {
      if (hoveredBodyId !== null) {
        hoveredBodyId = null;
        domElement.style.cursor = 'crosshair';
        handlers.onHoverChange(null);
      }
      return;
    }

    if (activeBodyId) {
      updatePointerNdc(evt.clientX, evt.clientY);
      raycaster.setFromCamera(pointerNdc, camera);
      if (raycaster.ray.intersectPlane(tabletopPlane, planeHit)) {
        handlers.onDragMove(activeBodyId, planeHit.x + dragOffset.x, planeHit.z + dragOffset.z);
      }
      return;
    }

    // Throttle hover raycasting to <= 25 Hz for silky performance
    const now = performance.now();
    if (now - lastHoverCheckMs < 40) return;
    lastHoverCheckMs = now;

    if (!updatePointerNdc(evt.clientX, evt.clientY)) {
      if (hoveredBodyId !== null) {
        hoveredBodyId = null;
        domElement.style.cursor = 'crosshair';
        handlers.onHoverChange(null);
      }
      return;
    }

    const hit = pickDraggableBody();
    const nextHoverId = hit?.bodyId ?? null;
    if (nextHoverId !== hoveredBodyId) {
      hoveredBodyId = nextHoverId;
      domElement.style.cursor = nextHoverId ? 'grab' : 'crosshair';
      handlers.onHoverChange(nextHoverId);
    }
  };

  const endActiveDrag = (evt?: PointerEvent): void => {
    if (!activeBodyId) return;
    const finishedId = activeBodyId;
    activeBodyId = null;
    controls.enabled = true;
    domElement.style.cursor = hoveredBodyId ? 'grab' : 'crosshair';
    if (evt && domElement.hasPointerCapture?.(evt.pointerId)) {
      domElement.releasePointerCapture(evt.pointerId);
    }
    handlers.onDragEnd(finishedId);
  };

  domElement.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', endActiveDrag);
  window.addEventListener('pointercancel', endActiveDrag);

  return () => {
    domElement.removeEventListener('pointerdown', onPointerDown);
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', endActiveDrag);
    window.removeEventListener('pointercancel', endActiveDrag);
    controls.enabled = true;
  };
}
