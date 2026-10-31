"use strict";

(() => {
  const card = document.querySelector(".memory-card");
  if (!card) return;

  const percent = card.querySelector("[data-memory-percent]");
  const bar = card.querySelector("[data-memory-bar]");
  const detail = card.querySelector("[data-memory-detail]");
  const button = card.querySelector("[data-clear-models]");
  const status = card.querySelector("[data-memory-status]");
  let timer;
  let paused = false;
  let unloadId = null;
  let posting = false;

  const showStatus = (message, error = false) => {
    status.hidden = false;
    status.textContent = message;
    status.dataset.error = String(error);
  };

  const fetchJSON = async (url, options = {}) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch(url, {cache: "no-store", ...options, signal: controller.signal});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Request failed. Please try again.");
      return data;
    } finally {
      clearTimeout(timeout);
    }
  };

  const refreshMemory = async () => {
    try {
      const data = await fetchJSON(card.dataset.memoryUrl);
      percent.textContent = `${data.percent.toFixed(1)}%`;
      bar.value = data.percent;
      const gib = (bytes) => (bytes / 1024 ** 3).toFixed(1);
      detail.textContent = `${gib(data.used_bytes)} / ${gib(data.total_bytes)} GiB used`;
      bar.setAttribute("aria-valuetext", `${percent.textContent}, ${detail.textContent}`);
    } catch {
      percent.textContent = "—";
      bar.removeAttribute("value");
      bar.removeAttribute("aria-valuetext");
      detail.textContent = "Memory unavailable · retrying…";
    }
  };

  const refreshUnload = async () => {
    if (!unloadId) return;
    try {
      const data = await fetchJSON(`${card.dataset.unloadUrl}/${encodeURIComponent(unloadId)}`);
      if (data.status === "pending") {
        showStatus("Waiting for the worker; active generation will finish first.");
        return;
      }
      unloadId = null;
      button.disabled = false;
      button.textContent = "Clear all models";
      showStatus(data.status === "completed" ? "All models cleared from memory." : data.error, data.status !== "completed");
      await refreshMemory();
    } catch {
      showStatus("Cleanup status unavailable. Retrying…", true);
    }
  };

  const poll = async () => {
    clearTimeout(timer);
    if (paused || document.hidden) return;
    await Promise.allSettled([refreshMemory(), refreshUnload()]);
    if (!paused && !document.hidden) timer = setTimeout(poll, unloadId ? 1000 : 3000);
  };

  button.addEventListener("click", async () => {
    if (posting || unloadId) return;
    posting = true;
    button.disabled = true;
    button.textContent = "Clearing…";
    showStatus("Requesting model cleanup…");
    try {
      const data = await fetchJSON(card.dataset.unloadUrl, {
        method: "POST", headers: {"X-CSRF-Token": card.dataset.csrfToken},
      });
      unloadId = data.id;
      showStatus("Waiting for the worker; active generation will finish first.");
    } catch (error) {
      button.disabled = false;
      button.textContent = "Clear all models";
      showStatus(error.message, true);
    } finally {
      posting = false;
      poll();
    }
  });

  document.addEventListener("visibilitychange", () => {
    clearTimeout(timer);
    if (!document.hidden) poll();
  });
  window.addEventListener("pagehide", () => { paused = true; clearTimeout(timer); });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) { paused = false; poll(); }
  });
  poll();
})();
