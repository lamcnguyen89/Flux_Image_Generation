"use strict";

const jobView = document.getElementById("job-view");
if (jobView) {
  let paused = false;
  let timer;
  let controller;
  let version = 0;
  let outputIds = jobView.dataset.outputIds;
  let eventSignature;
  const statusText = document.getElementById("connection-status");

  const link = (label, url) => {
    const element = document.createElement("a");
    element.textContent = label;
    element.href = url;
    return element;
  };

  const renderOutputs = (outputs, terminal) => {
    const empty = document.getElementById("outputs-empty");
    empty.hidden = outputs.length > 0;
    empty.textContent = terminal ? "No outputs were saved." : "Outputs will appear here when saved.";
    const ids = outputs.map((output) => output.id).join(",");
    if (ids === outputIds) return;
    outputIds = ids;
    const figures = outputs.map((output) => {
      const figure = document.createElement("figure");
      figure.className = "job-output";
      figure.dataset.outputId = output.id;
      if (output.kind === "image") {
        const image = document.createElement("img");
        image.src = output.url;
        image.alt = `Generated image, seed ${output.seed}`;
        image.loading = "lazy";
        if (output.width > 0 && output.height > 0) {
          image.width = output.width;
          image.height = output.height;
        }
        const full = link("", output.url);
        full.append(image);
        figure.append(full);
      } else {
        const video = document.createElement("video");
        video.src = output.url;
        video.controls = true;
        video.preload = "metadata";
        figure.append(video);
      }
      const caption = document.createElement("figcaption");
      const seed = document.createElement("p");
      seed.textContent = `Seed ${output.seed}`;
      const links = document.createElement("div");
      links.className = "output-links";
      links.append(link("View full output", output.url), link("Download", output.download_url), link("Metadata", output.metadata_url));
      caption.append(seed, links);
      figure.append(caption);
      return figure;
    });
    document.getElementById("job-outputs").replaceChildren(...figures);
  };

  const render = (job) => {
    jobView.dataset.terminal = String(job.terminal);
    document.getElementById("job-state").textContent = job.status[0].toUpperCase() + job.status.slice(1);
    document.getElementById("progress-stage").textContent = job.progress.label;
    document.getElementById("progress-count").textContent = job.progress.text;
    const progress = document.getElementById("job-progress");
    if (job.progress.total > 0 && job.progress.current !== null) {
      progress.max = job.progress.total;
      progress.value = job.progress.current;
    } else {
      progress.max = 1;
      if (job.terminal) progress.value = 0;
      else progress.removeAttribute("value");
    }
    const notice = document.getElementById("worker-notice");
    notice.textContent = job.worker_notice;
    notice.hidden = !job.worker_notice;
    document.getElementById("job-error").hidden = !job.error;
    document.getElementById("job-error-message").textContent = job.error || "";
    const signature = JSON.stringify(job.events);
    if (signature !== eventSignature) {
      eventSignature = signature;
      document.getElementById("execution-events").replaceChildren(...job.events.map((event) => {
        const item = document.createElement("li");
        item.textContent = event.label + (event.text ? ` · ${event.text}` : "");
        return item;
      }));
    }
    renderOutputs(job.outputs, job.terminal);
    statusText.textContent = job.terminal ? "Job finished." : "Live updates every second.";
  };

  const poll = async () => {
    if (paused || jobView.dataset.terminal === "true") return;
    const currentVersion = ++version;
    const requestController = new AbortController();
    controller = requestController;
    const timeout = setTimeout(() => requestController.abort(), 10000);
    try {
      const response = await fetch(jobView.dataset.statusUrl, {cache: "no-store", signal: requestController.signal});
      if (response.status === 404 && currentVersion === version) {
        paused = true;
        statusText.textContent = "This job is no longer available. Return to the jobs list.";
        return;
      }
      if (!response.ok) throw new Error("Status unavailable");
      const job = await response.json();
      if (!paused && currentVersion === version) render(job);
    } catch (error) {
      if (!paused && currentVersion === version) statusText.textContent = "Live status is unavailable. Retrying; you can also refresh this page.";
    } finally {
      clearTimeout(timeout);
      if (!paused && currentVersion === version && jobView.dataset.terminal !== "true") timer = setTimeout(poll, 1000);
    }
  };

  window.addEventListener("pagehide", () => { paused = true; version++; clearTimeout(timer); controller?.abort(); });
  window.addEventListener("pageshow", (event) => { if (event.persisted) { paused = false; poll(); } });
  poll();
}
