document.addEventListener("DOMContentLoaded", () => {
  const toggles = document.querySelectorAll("[data-toggle='wy-nav-top']");
  if (!toggles.length) {
    return;
  }
  toggles.forEach((toggle) => {
    toggle.addEventListener("click", () => {
      document
        .querySelectorAll("[data-toggle='wy-nav-shift']")
        .forEach((el) => el.classList.toggle("shift"));
      document
        .querySelectorAll("[data-toggle='rst-versions']")
        .forEach((el) => el.classList.toggle("shift"));
    });
  });
});
