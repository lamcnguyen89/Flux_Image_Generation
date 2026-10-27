"use strict";

// Optional hints only. Python validates the submitted values independently.
const field = (name) => document.getElementById(`field-${name.replaceAll(".", "-")}`);
const frames = field("num_frames");
const fps = field("fps");
const duration = document.getElementById("video-duration");
if (frames && fps && duration) {
  const updateDuration = () => {
    const count = Number(frames.value);
    const rate = Number(fps.value);
    duration.textContent = count > 0 && rate > 0 && Number.isFinite(count / rate)
      ? `Estimated playback: ${(count / rate).toFixed(2)} seconds (${count} frames at ${rate} FPS).`
      : "Enter a positive frame count and FPS to estimate duration.";
  };
  frames.addEventListener("input", updateDuration);
  fps.addEventListener("input", updateDuration);
  updateDuration();
}

const cfg = field("true_cfg_scale");
const negative = field("negative_prompt");
const cfgStatus = document.getElementById("cfg-status");
if (cfg && negative && cfgStatus) {
  const updateCFG = () => {
    cfgStatus.textContent = Number(cfg.value) > 1 && negative.value.trim()
      ? "Negative prompting is active with these settings."
      : "Negative prompting is inactive. Set true CFG above 1 and enter a negative prompt to enable it.";
  };
  cfg.addEventListener("input", updateCFG);
  negative.addEventListener("input", updateCFG);
  updateCFG();
}

const source = field("input_image");
const image = document.getElementById("source-preview");
if (source && image) {
  let imageURL;
  source.addEventListener("change", () => {
    if (imageURL) URL.revokeObjectURL(imageURL);
    const file = source.files[0];
    image.hidden = true;
    image.removeAttribute("src");
    if (file && file.type.startsWith("image/")) {
      imageURL = URL.createObjectURL(file);
      image.src = imageURL;
      image.hidden = false;
    }
  });
  image.addEventListener("error", () => { image.hidden = true; });
  window.addEventListener("pagehide", () => { if (imageURL) URL.revokeObjectURL(imageURL); });
}
