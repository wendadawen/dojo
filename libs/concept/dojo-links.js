(function setDojoLinkTargets() {
  const desktopLinks = matchMedia("(hover: hover) and (pointer: fine)");

  function updateLinkTargets() {
    document.querySelectorAll("a[href]").forEach((link) => {
      const href = link.getAttribute("href");
      if (href && href[0] !== "#" && (/^[a-z][a-z0-9+.\-]*:/i.test(href) || /\.html?(?:[?#]|$)/i.test(href))) {
        link.target = desktopLinks.matches ? "_blank" : "_self";
        link.rel = "noopener";
      }
    });
  }

  desktopLinks.addEventListener("change", updateLinkTargets);
  updateLinkTargets();
}());
