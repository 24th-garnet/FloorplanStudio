(function () {
  function setStatus(message) {
    const status = document.getElementById("walkModeStatus");
    if (status) status.textContent = message;
  }

  window.__walkClick = function walkClick(buttonId, event) {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    const api = window.__topViewerWalk;
    if (!api) {
      setStatus(
        "Walkthrough JS が未読込です。Ctrl+Shift+R で再読込するか、ページを開き直してください。"
      );
      return false;
    }
    try {
      switch (buttonId) {
        case "walkPickSpawn":
          api.toggleWalkSpawnPick();
          break;
        case "walkModeToggle":
          setStatus(api.canUseWalkMode?.() ? "Walkthrough を切り替え中…" : "USDZオーバーレイ読込後に利用できます。");
          api.toggleWalkMode();
          break;
        default:
          break;
      }
    } catch (err) {
      console.error("walk click failed:", err);
      setStatus(`操作エラー: ${err?.message || err}`);
    }
    return false;
  };
})();
