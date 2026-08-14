(function attachDojoGraph(root) {
  let globalCy = null;
  let global3d = null;
  let graph3dContainer = null;
  let graph3dResize = null;
  let graph3dNeighbors = new Map();
  let graph3dNodes = new Map();
  let spriteTextClass = null;
  let forceGraph3dPromise = null;
  let selectedId = null;
  let hoveredId = null;
  let activeMode = "2d";
  let renderGeneration = 0;
  const MAX_VISIBLE_LABELS = 8;

  function cssVar(name, fallback) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  }

  function colors() {
    return {
      ink: cssVar("--ink", "#171815"),
      muted: cssVar("--muted", "#686a64"),
      surface: cssVar("--surface", "#fffefa"),
      accent: cssVar("--accent", "#284f44"),
      paper: cssVar("--paper-node", "#9b5c35"),
      note: cssVar("--note-node", "#7a6230"),
      line: cssVar("--graph-line", "#7d807a"),
      incoming: cssVar("--incoming", "#2f6f9f"),
      outgoing: cssVar("--outgoing", "#b45f3d"),
    };
  }

  function graphStyle() {
    const color = colors();
    return [
      {
        selector: "node",
        style: {
          label: "",
          width: "mapData(degree, 0, 17, 15, 28)",
          height: "mapData(degree, 0, 17, 15, 28)",
          "background-color": color.accent,
          "border-width": 2,
          "border-color": color.surface,
          "overlay-opacity": 0,
        },
      },
      {
        selector: "node.show-label",
        style: {
          label: "data(shortLabel)",
          color: color.ink,
          "font-size": 10,
          "font-family": "system-ui, sans-serif",
          "font-weight": 600,
          "text-wrap": "wrap",
          "text-max-width": 96,
          "text-valign": "bottom",
          "text-margin-y": 8,
          "text-background-color": color.surface,
          "text-background-opacity": .86,
          "text-background-padding": 2,
          "text-background-shape": "roundrectangle",
          "min-zoomed-font-size": 7,
        },
      },
      { selector: 'node[type = "paper"]', style: { "background-color": color.paper } },
      { selector: 'node[type = "note"]', style: { "background-color": color.note } },
      {
        selector: "edge",
        style: {
          width: 1.1,
          opacity: .32,
          "line-color": color.line,
          "target-arrow-color": color.line,
          "target-arrow-shape": "triangle",
          "arrow-scale": .65,
          "curve-style": "bezier",
          "overlay-opacity": 0,
        },
      },
      { selector: ".is-dimmed", style: { opacity: .055 } },
      {
        selector: "edge.is-focused",
        style: {
          opacity: .96,
          width: 2.35,
          "line-color": color.accent,
          "target-arrow-color": color.accent,
          "arrow-scale": .8,
        },
      },
      {
        selector: "edge.is-incoming",
        style: {
          opacity: .98,
          width: 2.6,
          "line-color": color.incoming,
          "target-arrow-color": color.incoming,
          "arrow-scale": .85,
        },
      },
      {
        selector: "edge.is-outgoing",
        style: {
          opacity: .98,
          width: 2.6,
          "line-color": color.outgoing,
          "target-arrow-color": color.outgoing,
          "arrow-scale": .85,
        },
      },
      {
        selector: "node.is-focused",
        style: {
          opacity: 1,
          width: 29,
          height: 29,
          "border-width": 4,
          "border-color": color.surface,
          "font-size": 12,
          "text-max-width": 120,
          "text-background-opacity": 1,
          "text-background-padding": 3,
          "min-zoomed-font-size": 6,
          "z-index": 20,
        },
      },
    ];
  }

  function requireCytoscape() {
    if (!root.cytoscape) throw new Error("Cytoscape 未加载");
  }

  function clearFocus(cy) {
    cy.elements().removeClass("is-focused is-dimmed is-incoming is-outgoing show-label");
  }

  function topNodes(nodes, limit = MAX_VISIBLE_LABELS) {
    return nodes
      .slice()
      .sort((a, b) => (b.data("degree") || 0) - (a.data("degree") || 0))
      .slice(0, limit);
  }

  function showCoreLabels(cy) {
    cy.nodes().removeClass("show-label");
    topNodes(cy.nodes().toArray()).forEach((node) => node.addClass("show-label"));
  }

  function focusNode(cy, node) {
    const neighborhood = node.closedNeighborhood();
    clearFocus(cy);
    cy.elements().not(neighborhood).addClass("is-dimmed");
    neighborhood.addClass("is-focused");
    node.connectedEdges().forEach((edge) => {
      edge.addClass(edge.target().id() === node.id() ? "is-incoming" : "is-outgoing");
    });
    const neighbors = topNodes(node.neighborhood("node").toArray(), MAX_VISIBLE_LABELS - 1);
    node.addClass("show-label");
    neighbors.forEach((neighbor) => neighbor.addClass("show-label"));
  }

  function destroyGlobal() {
    if (globalCy) globalCy.destroy();
    globalCy = null;
  }

  function destroy3d() {
    renderGeneration += 1;
    if (graph3dResize) graph3dResize.disconnect();
    graph3dResize = null;
    if (global3d) {
      global3d.pauseAnimation();
      if (typeof global3d._destructor === "function") global3d._destructor();
    }
    if (graph3dContainer) graph3dContainer.replaceChildren();
    global3d = null;
    graph3dContainer = null;
    hoveredId = null;
  }

  function renderGlobal(container, catalog, visibleIds, onSelect, onClear) {
    requireCytoscape();
    destroyGlobal();
    selectedId = null;
    hoveredId = null;
    activeMode = "2d";
    const elements = root.DojoHomeModel.makeGlobalElements(catalog, visibleIds);
    globalCy = root.cytoscape({
      container,
      elements: [...elements.nodes, ...elements.edges],
      style: graphStyle(),
      minZoom: .25,
      maxZoom: 2.6,
      wheelSensitivity: .2,
      layout: {
        name: "cose",
        animate: false,
        fit: true,
        padding: 68,
        randomize: true,
        componentSpacing: 130,
        nodeRepulsion: 22000,
        nodeOverlap: 70,
        idealEdgeLength: 145,
        edgeElasticity: 80,
        nestingFactor: 1.15,
        gravity: .32,
        numIter: 2400,
        nodeDimensionsIncludeLabels: false,
      },
    });
    showCoreLabels(globalCy);

    globalCy.on("tap", "node", (event) => {
      selectedId = event.target.id();
      focusNode(globalCy, event.target);
      onSelect(selectedId);
    });

    globalCy.on("mouseover", "node", (event) => {
      container.style.cursor = "pointer";
      if (!selectedId) focusNode(globalCy, event.target);
    });

    globalCy.on("mouseout", "node", () => {
      container.style.cursor = "";
      if (!selectedId) {
        clearFocus(globalCy);
        showCoreLabels(globalCy);
      }
    });

    globalCy.on("tap", (event) => {
      if (event.target !== globalCy) return;
      selectedId = null;
      clearFocus(globalCy);
      showCoreLabels(globalCy);
      if (onClear) onClear();
    });
  }

  function graph3dData(catalog, visibleIds) {
    const elements = root.DojoHomeModel.makeGlobalElements(catalog, visibleIds);
    return {
      nodes: elements.nodes.map((item) => ({ ...item.data })),
      links: elements.edges.map((item) => ({ ...item.data })),
    };
  }

  function buildNeighborMap(links) {
    const neighbors = new Map();
    const add = (source, target) => {
      if (!neighbors.has(source)) neighbors.set(source, new Set());
      neighbors.get(source).add(target);
    };
    links.forEach((link) => {
      add(link.source, link.target);
      add(link.target, link.source);
    });
    return neighbors;
  }

  function ranked3dIds(ids, limit) {
    return [...ids]
      .map((id) => graph3dNodes.get(id))
      .filter(Boolean)
      .sort((a, b) => (b.degree || 0) - (a.degree || 0))
      .slice(0, limit)
      .map((node) => node.id);
  }

  function visible3dLabelIds(focusId) {
    if (!focusId) {
      return new Set(ranked3dIds(graph3dNodes.keys(), MAX_VISIBLE_LABELS));
    }
    const neighbors = graph3dNeighbors.get(focusId) || new Set();
    return new Set([
      focusId,
      ...ranked3dIds(neighbors, MAX_VISIBLE_LABELS - 1),
    ]);
  }

  function nodeId(value) {
    return typeof value === "object" ? value.id : value;
  }

  function isNeighbor(id, centerId) {
    return id === centerId || (graph3dNeighbors.get(centerId) || new Set()).has(id);
  }

  function colorWithAlpha(hex, alpha) {
    const value = hex.replace("#", "");
    const full = value.length === 3
      ? value.split("").map((part) => part + part).join("")
      : value;
    const number = Number.parseInt(full, 16);
    return `rgba(${number >> 16}, ${(number >> 8) & 255}, ${number & 255}, ${alpha})`;
  }

  function loadScript(url) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${url}"]`);
      if (existing) {
        if (root.ForceGraph3D) resolve();
        else existing.addEventListener("load", resolve, { once: true });
        return;
      }
      const script = document.createElement("script");
      script.src = url;
      script.onload = resolve;
      script.onerror = () => reject(new Error("3D 资源加载错误"));
      document.head.appendChild(script);
    });
  }

  async function load3dDependencies() {
    if (!root.ForceGraph3D) {
      if (!forceGraph3dPromise) {
        const url = new URL("libs/3d-force-graph.min.js", document.baseURI).href;
        forceGraph3dPromise = loadScript(url);
      }
      await forceGraph3dPromise;
    }
    if (!spriteTextClass) {
      const moduleUrl = new URL("libs/three-spritetext.mjs", document.baseURI).href;
      const module = await import(moduleUrl);
      spriteTextClass = module.default;
    }
  }

  function refresh3d() {
    if (!global3d || !spriteTextClass) return;
    const color = colors();
    const focusId = selectedId || hoveredId;
    const labelIds = visible3dLabelIds(focusId);
    const nodeBaseColor = (node) => (
      node.type === "paper" ? color.paper : node.type === "note" ? color.note : color.accent
    );
    const focusedLink = (link) => focusId
      && (nodeId(link.source) === focusId || nodeId(link.target) === focusId);
    const linkDirection = (link) => {
      if (!focusId) return "";
      if (nodeId(link.target) === focusId) return "incoming";
      if (nodeId(link.source) === focusId) return "outgoing";
      return "";
    };
    const focusedLinkColor = (link) => (
      linkDirection(link) === "incoming" ? color.incoming : color.outgoing
    );

    global3d
      .nodeColor((node) => {
        const base = nodeBaseColor(node);
        return !focusId || isNeighbor(node.id, focusId) ? base : colorWithAlpha(base, .1);
      })
      .linkColor((link) => (
        focusedLink(link)
          ? focusedLinkColor(link)
          : colorWithAlpha(color.line, focusId ? .045 : .4)
      ))
      .linkWidth((link) => focusedLink(link) ? .5 : .18)
      .linkDirectionalArrowLength((link) => focusedLink(link) ? 3.4 : 1.8)
      .linkDirectionalArrowColor((link) => (
        focusedLink(link) ? focusedLinkColor(link) : colorWithAlpha(color.line, .32)
      ))
      .nodeThreeObject((node) => {
        if (!labelIds.has(node.id)) return null;
        const label = new spriteTextClass(node.shortLabel);
        label.material.depthWrite = false;
        label.color = color.ink;
        label.backgroundColor = node.id === focusId
          ? "rgba(255,254,250,.96)"
          : "rgba(255,254,250,.72)";
        label.padding = 1.2;
        label.borderRadius = .8;
        label.textHeight = node.id === focusId ? 6.2 : 5;
        label.center.y = -.9;
        return label;
      });
  }

  function focus3dCamera(id, duration = 650) {
    if (!global3d) return;
    const node = global3d.graphData().nodes.find((item) => item.id === id);
    if (!node || !Number.isFinite(node.x)) return;
    const camera = global3d.cameraPosition();
    let dx = camera.x - node.x;
    let dy = camera.y - node.y;
    let dz = camera.z - node.z;
    let length = Math.hypot(dx, dy, dz);
    if (length < 1) {
      dx = 0;
      dy = 0;
      dz = 1;
      length = 1;
    }
    const distance = 260;
    global3d.cameraPosition(
      {
        x: node.x + (dx / length) * distance,
        y: node.y + (dy / length) * distance,
        z: node.z + (dz / length) * distance,
      },
      node,
      duration,
    );
  }

  async function render3d(container, catalog, visibleIds, onSelect, onClear) {
    destroy3d();
    selectedId = null;
    hoveredId = null;
    activeMode = "3d";
    const generation = renderGeneration;
    await load3dDependencies();
    if (generation !== renderGeneration || activeMode !== "3d") return;

    graph3dContainer = container;
    const data = graph3dData(catalog, visibleIds);
    graph3dNeighbors = buildNeighborMap(data.links);
    graph3dNodes = new Map(data.nodes.map((node) => [node.id, node]));
    let fitted = false;

    global3d = root.ForceGraph3D()(container)
      .backgroundColor(cssVar("--surface", "#fffefa"))
      .graphData(data)
      .showNavInfo(false)
      .nodeLabel(() => "")
      .nodeVal((node) => 2 + Math.sqrt(node.degree || 0) * .72)
      .nodeResolution(18)
      .nodeOpacity(.96)
      .nodeThreeObjectExtend(true)
      .linkOpacity(1)
      .linkDirectionalArrowRelPos(1)
      .warmupTicks(120)
      .cooldownTicks(260)
      .onNodeHover((node) => {
        hoveredId = node ? node.id : null;
        container.style.cursor = node ? "pointer" : "";
        if (!selectedId) refresh3d();
      })
      .onNodeClick((node) => {
        selectedId = node.id;
        hoveredId = null;
        refresh3d();
        focus3dCamera(node.id);
        onSelect(node.id);
      })
      .onBackgroundClick(() => {
        selectedId = null;
        hoveredId = null;
        refresh3d();
        if (onClear) onClear();
      })
      .onEngineStop(() => {
        if (fitted || selectedId) return;
        fitted = true;
        global3d.zoomToFit(650, 55);
      });

    global3d.d3Force("charge").strength(-155);
    global3d.d3Force("link").distance(52);
    graph3dResize = new ResizeObserver(() => {
      if (!global3d) return;
      global3d.width(container.clientWidth).height(container.clientHeight);
    });
    graph3dResize.observe(container);
    refresh3d();
  }

  function focusGlobal(id, moveCamera = false) {
    selectedId = id;
    if (globalCy) {
      const node = globalCy.$id(id);
      if (node.length) focusNode(globalCy, node);
    }
    if (global3d) {
      refresh3d();
      if (moveCamera) focus3dCamera(id);
    }
  }

  function reset(mode = activeMode) {
    selectedId = null;
    hoveredId = null;
    if (mode === "3d" && global3d) {
      refresh3d();
      global3d.zoomToFit(650, 55);
      return;
    }
    if (globalCy) {
      clearFocus(globalCy);
      globalCy.fit(undefined, 68);
    }
  }

  function setMode(mode) {
    activeMode = mode;
  }

  function resize() {
    if (globalCy) globalCy.resize();
    if (global3d && graph3dContainer) {
      global3d.width(graph3dContainer.clientWidth).height(graph3dContainer.clientHeight);
    }
  }

  root.DojoGraph = {
    renderGlobal,
    render3d,
    destroyGlobal,
    destroy3d,
    focusGlobal,
    reset,
    setMode,
    resize,
  };
}(window));
