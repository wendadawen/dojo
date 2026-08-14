(function attachDojoGraph(root) {
  let globalCy = null;
  let localCy = null;

  if (root.cytoscape && root.cytoscapeDagre) {
    root.cytoscape.use(root.cytoscapeDagre);
  }

  function cssVar(name, fallback) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  }

  function graphStyle() {
    const colors = {
      text: cssVar("--muted", "#59636e"),
      muted: cssVar("--muted", "#8c959f"),
      accent: cssVar("--accent", "#0969da"),
      green: cssVar("--green", "#1a7f37"),
      orange: cssVar("--orange", "#bc4c00"),
      purple: cssVar("--purple", "#8250df"),
      focus: "#d29922",
    };
    return [
      {
        selector: "node",
        style: {
          label: "data(label)",
          "font-size": 10,
          "font-family": "system-ui, sans-serif",
          color: colors.text,
          "text-wrap": "wrap",
          "text-max-width": 110,
          "background-color": colors.muted,
          width: 18,
          height: 18,
        },
      },
      { selector: 'node[type = "concept"]', style: { "background-color": colors.accent } },
      { selector: 'node[type = "paper"]', style: { "background-color": colors.green } },
      { selector: 'node[type = "note"]', style: { "background-color": colors.orange } },
      {
        selector: "edge",
        style: {
          width: 1,
          opacity: .16,
          "line-color": colors.muted,
          "target-arrow-color": colors.muted,
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
        },
      },
      { selector: ".is-dimmed", style: { opacity: .12 } },
      {
        selector: ".is-focused",
        style: {
          opacity: 1,
          width: 3,
          "line-color": colors.purple,
          "target-arrow-color": colors.purple,
        },
      },
      {
        selector: "node.is-focused",
        style: { width: 26, height: 26, "border-width": 3, "border-color": colors.focus },
      },
      {
        selector: 'node[role = "center"]',
        style: { width: 30, height: 30, "border-width": 3, "border-color": colors.focus },
      },
      { selector: 'node[role = "incoming"]', style: { "background-color": colors.accent } },
      { selector: 'node[role = "outgoing"]', style: { "background-color": colors.green } },
      { selector: 'node[role = "both"]', style: { "background-color": colors.purple } },
    ];
  }

  function requireCytoscape() {
    if (!root.cytoscape) throw new Error("Cytoscape 未加载");
  }

  function destroyGlobal() {
    if (globalCy) globalCy.destroy();
    globalCy = null;
  }

  function destroyLocal() {
    if (localCy) localCy.destroy();
    localCy = null;
  }

  function renderGlobal(container, catalog, visibleIds, onSelect) {
    requireCytoscape();
    destroyGlobal();
    const elements = root.DojoHomeModel.makeGlobalElements(catalog, visibleIds);
    globalCy = root.cytoscape({
      container,
      elements: [...elements.nodes, ...elements.edges],
      style: graphStyle(),
      layout: {
        name: "cose",
        animate: false,
        fit: true,
        padding: 35,
        nodeRepulsion: 9000,
        idealEdgeLength: 95,
      },
    });
    globalCy.on("tap", "node", (event) => {
      const node = event.target;
      const neighborhood = node.closedNeighborhood();
      globalCy.elements().removeClass("is-focused is-dimmed");
      globalCy.elements().not(neighborhood).addClass("is-dimmed");
      neighborhood.addClass("is-focused");
      onSelect(node.id());
    });
    globalCy.on("tap", (event) => {
      if (event.target === globalCy) {
        globalCy.elements().removeClass("is-focused is-dimmed");
      }
    });
  }

  function renderLocal(container, catalog, centerId, onSelect) {
    requireCytoscape();
    destroyLocal();
    const elements = root.DojoHomeModel.makeLocalGraph(catalog, centerId);
    localCy = root.cytoscape({
      container,
      elements: [...elements.nodes, ...elements.edges],
      style: graphStyle(),
      layout: { name: "dagre", rankDir: "LR", nodeSep: 38, rankSep: 78, padding: 24 },
    });
    localCy.on("tap", "node", (event) => {
      if (event.target.id() !== centerId) onSelect(event.target.id());
    });
  }

  root.DojoGraph = { renderGlobal, renderLocal, destroyGlobal, destroyLocal };
}(window));
