"use strict";

const headerMenu = document.querySelector(".header-menu");

if (headerMenu) {
  document.addEventListener("click", (event) => {
    if (!headerMenu.contains(event.target)) headerMenu.open = false;
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && headerMenu.open) {
      headerMenu.open = false;
      headerMenu.querySelector("summary").focus();
    }
  });
}
