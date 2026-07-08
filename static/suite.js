const SIDEBAR_COLLAPSED_KEY = "floorplan-studio.sidebar.collapsed";
const SIDEBAR_TRANSITION_MS = 420;

function notifyViewerResize() {
  window.dispatchEvent(new Event("resize"));
}

function scheduleViewerResizeAfterSidebarTransition() {
  window.setTimeout(notifyViewerResize, SIDEBAR_TRANSITION_MS + 20);
}

function setSidebarCollapsed(app, toggle, collapsed) {
  app.classList.toggle("is-sidebar-collapsed", collapsed);
  toggle.setAttribute("aria-expanded", String(!collapsed));
  const label = collapsed ? "サイドバーを表示" : "サイドバーを隠す";
  toggle.setAttribute("aria-label", label);
  toggle.title = label;
  try {
    if (!collapsed) localStorage.removeItem(SIDEBAR_COLLAPSED_KEY);
    else localStorage.setItem(SIDEBAR_COLLAPSED_KEY, "1");
  } catch (_) {
    /* ignore storage errors */
  }
  scheduleViewerResizeAfterSidebarTransition();
}

function initSidebarToggle() {
  const app = document.getElementById("app");
  const toggle = document.getElementById("sidebarEdgeToggle");
  if (!app || !toggle) return;

  // Always start expanded; collapsed state is session-only (avoids stuck hidden sidebar).
  setSidebarCollapsed(app, toggle, false);

  toggle.addEventListener("click", () => {
    setSidebarCollapsed(app, toggle, !app.classList.contains("is-sidebar-collapsed"));
  });
}

function initSidebarPanels() {
  const panels = document.querySelectorAll(".sidebar-nav > .sidebar-panel");
  for (const panel of panels) {
    if (!(panel instanceof HTMLDetailsElement)) continue;
    panel.open = panel.dataset.sidebar === "import";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initSidebarPanels();
  initSidebarToggle();
});
