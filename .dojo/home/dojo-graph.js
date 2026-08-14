(function attachDojoGraph(root) {
  let globalCy = null;

  function cssVar(name, fallback) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  }

  function graphStyle() {
    const colors = {
      ink: cssVar("--ink", "#171815"),
      muted: cssVar("--muted", "#686a64"),
      surface: cssVar("--surface", "#fffefa"),
      accent: cssVar("--accent", "#284f44"),
      paper: cssVar("--paper-node", "#9b5c35"),
      note: cssVar("--note-node", "#7a6230"),
      line: cssVar("--graph-line", "#7d807a"),
    };

    return [
      {
        selector: "node",
        style: {
          label: "",
          width: 15,
          height: 15,
          "background-color": colors.accent,
          "border-width": 2,
          "border-color": colors.surface,
          "overlay-opacity": 0,
        },
      },
      { selector: 'node[type = "paper"]', style: { "background-color": colors.paper } },
      { selector: 'node[type = "note"]', style: { "background-color": colors.note } },
      {
        selector: "node.show-label",
        style: {
          label: "data(shortLabel)",
          color: colors.ink,
          "font-size": 9,
          "font-family": "system-ui, sans-serif",
          "font-weight": 600,
          "text-background-color": colors.surface,
          "text-background-opacity": .94,
          "text-background-padding": 3,
          "text-background-shape": "roundrectangle",
          "text-margin-y": -12,
          "text-wrap": "none",
        },
      },
      {
        selector: "node[degree >= 8]",
        style: {
          label: "data(shortLabel)",
          color: colors.muted,
          "font-size": 8,
          "font-family": "system-ui, sans-serif",
          "font-weight": 600,
          "text-background-color": colors.surface,
          "text-background-opacity": .9,
          "text-background-padding": 2,
          "text-background-shape": "roundrectangle",
          "text-margin-y": -11,
          "text-wrap": "none",
        },
      },
      {
        selector: "edge",
        style: {
          width: 1.45,
          opacity: .58,
          "line-color": colors.line,
          "target-arrow-color": colors.line,
          "target-arrow-shape": "triangle",
          "arrow-scale": .7,
          "curve-style": "bezier",
          "overlay-opacity": 0,
        },
      },
      {
        selector: ".is-dimmed",
        style: { opacity: .09 },
      },
      {
        selector: "edge.is-focused",
        style: {
          opacity: .95,
          width: 2.6,
          "line-color": colors.accent,
          "target-arrow-color": colors.accent,
        },
      },
      {
        selector: "node.is-focused",
        style: {
          opacity: 1,
          width: 25,
          height: 25,
          "border-width": 4,
          "border-color": colors.surface,
          "z-index": 20,
        },
      },
    ];
  }

  function requireCytoscape() {
    if (!root.cytoscape) throw new Error("Cytoscape 未加载");
  }

  function destroyGlobal() {
    if (globalCy) globalCy.destroy();
    globalCy = null;
  }

  function clearFocus(cy) {
    cy.elements().removeClass("is-focused is-dimmed show-label");
  }

  function focusNode(cy, node) {
    const neighborhood = node.closedNeighborhood();
    clearFocus(cy);
    cy.elements().not(neighborhood).addClass("is-dimmed");
    neighborhood.addClass("is-focused");
    node.addClass("show-label");
  }

  function renderGlobal(container, catalog, visibleIds, onSelect, onClear) {
    requireCytoscape();
    destroyGlobal();
    const elements = root.DojoHomeModel.makeGlobalElements(catalog, visibleIds);
    globalCy = root.cytoscape({
      container,
      elements: [...elements.nodes, ...elements.edges],
      style: graphStyle(),
      minZoom: .28,
      maxZoom: 2.4,
      wheelSensitivity: .22,
      layout: {
        name: "cose",
        animate: false,
        fit: true,
        padding: 54,
        randomize: true,
        componentSpacing: 90,
        nodeRepulsion: 14000,
        nodeOverlap: 30,
        idealEdgeLength: 112,
        edgeElasticity: 85,
        nestingFactor: 1.15,
        gravity: .45,
        numIter: 1800,
      },
    });

    globalCy.on("tap", "node", (event) => {
      const node = event.target;
      focusNode(globalCy, node);
      onSelect(node.id());
    });

    globalCy.on("mouseover", "node", (event) => {
      const node = event.target;
      if (!globalCy.$("node.is-focused").length) node.addClass("show-label");
      container.style.cursor = "pointer";
    });

    globalCy.on("mouseout", "node", (event) => {
      const node = event.target;
      if (!node.hasClass("is-focused")) node.removeClass("show-label");
      container.style.cursor = "";
    });

    globalCy.on("tap", (event) => {
      if (event.target !== globalCy) return;
      clearFocus(globalCy);
      if (onClear) onClear();
    });
  }

  function focusGlobal(id) {
    if (!globalCy) return;
    const node = globalCy.$id(id);
    if (!node.length) return;
    focusNode(globalCy, node);
  }

  root.DojoGraph = { renderGlobal, destroyGlobal, focusGlobal };
}(window));
