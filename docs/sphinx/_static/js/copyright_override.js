(() => {
  const applyOverride = () => {
    const blocks = document.querySelectorAll('div[role="contentinfo"]');
    if (blocks.length) {
      const html =
        'Lucidy © Copyright 2026 -- <a href="https://www.lucidy.site">www.lucidy.site</a>';
      blocks.forEach((block) => {
        block.innerHTML = html;
      });
    }

  };

  const applyLogoOverride = () => {
    const brand = document.querySelector(".wy-side-nav-search .brand");
    if (!brand) {
      return false;
    }
    brand.innerHTML =
      '<img src="_static/images/OpenInferIcon.png" class="logo" alt="OpenInfer">';
    return true;
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      applyOverride();
      let attempts = 0;
      const maxAttempts = 20;
      const timer = setInterval(() => {
        attempts += 1;
        if (applyLogoOverride() || attempts >= maxAttempts) {
          clearInterval(timer);
        }
      }, 250);
    });
  } else {
    applyOverride();
    let attempts = 0;
    const maxAttempts = 20;
    const timer = setInterval(() => {
      attempts += 1;
      if (applyLogoOverride() || attempts >= maxAttempts) {
        clearInterval(timer);
      }
    }, 250);
  }

})();
