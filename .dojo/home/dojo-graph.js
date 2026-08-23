(function attachDojoGraph(root) {
  let global3d = null;
  let graph3dContainer = null;
  let graph3dResize = null;
  let graph3dNeighbors = new Map();
  let spriteTextClass = null;
  let forceGraph3dPromise = null;
  let selectedId = null;
  let hoveredId = null;
  let activeMode = "3d";
  let renderGeneration = 0;
  let cameraListenerAbort = null;
  let graph3dSprites = new Map();
  let labelFocusId = null;

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








  function destroy3d() {
    renderGeneration += 1;
    if (cameraListenerAbort) cameraListenerAbort.abort();
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
    graph3dSprites.clear();
    labelFocusId = null;
  }

  function pause3d() {
    if (global3d) global3d.pauseAnimation();
  }

  function has3d() {
    return Boolean(global3d);
  }

  function resume3d() {
    if (!global3d) return false;
    if (graph3dContainer) {
      global3d.width(graph3dContainer.clientWidth).height(graph3dContainer.clientHeight);
    }
    global3d.resumeAnimation();
    return true;
  }


  function graph3dData(catalog, visibleIds, matchIds) {
    const elements = root.DojoHomeModel.makeGlobalElements(catalog, visibleIds, matchIds);
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

  // 标签精灵按节点缓存：创建一次反复复用，hover 时只改颜色/字高，不重建对象
  function labelSpriteFor(node) {
    let sprite = graph3dSprites.get(node.id);
    if (!sprite) {
      sprite = new spriteTextClass(node.shortLabel);
      sprite.material.depthWrite = false;
      sprite.center.y = -.9;
      sprite.textHeight = 5;
      sprite.color = colors().ink;
      graph3dSprites.set(node.id, sprite);
    }
    return sprite;
  }

  function setLabelEmphasis(id) {
    if (labelFocusId === id) return;
    const palette = colors();
    const previous = graph3dSprites.get(labelFocusId);
    if (previous) {
      previous.color = palette.ink;
      previous.textHeight = 5;
    }
    const current = graph3dSprites.get(id);
    if (current) {
      current.color = palette.accent;
      current.textHeight = 6.2;
    }
    labelFocusId = id;
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

  // hover/点击时的强调更新：只改材质颜色，不重建标签与连线几何体
  function refresh3d() {
    if (!global3d) return;
    const color = colors();
    const focusId = selectedId || hoveredId;
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

    setLabelEmphasis(focusId);
    global3d
      .nodeColor((node) => {
        const base = nodeBaseColor(node);
        if (focusId) return isNeighbor(node.id, focusId) ? base : colorWithAlpha(base, .1);
        return node.match === false ? colorWithAlpha(base, .3) : base;
      })
      .linkColor((link) => (
        focusedLink(link)
          ? focusedLinkColor(link)
          : colorWithAlpha(color.line, focusId ? .045 : (link.match === false ? .12 : .4))
      ))
      .linkDirectionalArrowColor((link) => (
        focusedLink(link) ? focusedLinkColor(link) : colorWithAlpha(color.line, .32)
      ));
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

  async function render3d(container, catalog, visibleIds, onSelect, onClear, matchIds) {
    destroy3d();
    selectedId = null;
    hoveredId = null;
    activeMode = "3d";
    const generation = renderGeneration;
    await load3dDependencies();
    if (generation !== renderGeneration || activeMode !== "3d") return;

    graph3dContainer = container;
    const data = graph3dData(catalog, visibleIds, matchIds);
    graph3dNeighbors = buildNeighborMap(data.links);
    let fitted = false;
    // 用户一旦操作过视角，引擎停止时不再自动缩放到全图
    let userAdjustedCamera = false;
    if (cameraListenerAbort) cameraListenerAbort.abort();
    cameraListenerAbort = new AbortController();
    const signal = cameraListenerAbort.signal;
    container.addEventListener("wheel", () => { userAdjustedCamera = true; }, { passive: true, signal });
    container.addEventListener("pointerdown", () => { userAdjustedCamera = true; }, { signal });

    global3d = root.ForceGraph3D()(container)
      .backgroundColor(cssVar("--paper", "#f7f5ef"))
      .graphData(data)
      .showNavInfo(false)
      .nodeLabel(() => "")
      .nodeVal((node) => 2 + Math.sqrt(node.degree || 0) * .72)
      .nodeResolution(18)
      .nodeOpacity(.96)
      .nodeThreeObjectExtend(true)
      .nodeThreeObject((node) => labelSpriteFor(node))
      .linkOpacity(1)
      .linkWidth(() => .18)
      .linkDirectionalArrowLength(() => 1.8)
      .linkDirectionalArrowRelPos(1)
      .warmupTicks(50)
      .cooldownTicks(180)
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
        if (fitted || selectedId || userAdjustedCamera) return;
        fitted = true;
        global3d.zoomToFit(650, 55);
      });

    const controls = global3d.controls();
    if (controls) {
      controls.zoomSpeed = .55;
      controls.enablePan = true;
    }
    // Retina 屏封顶 1.5 倍像素比：2 倍绘制像素近乎翻倍，视觉差异小、滚轮流畅度差异大
    if (typeof global3d.renderer === "function") {
      const renderer = global3d.renderer();
      if (renderer && renderer.setPixelRatio) {
        renderer.setPixelRatio(Math.min(root.devicePixelRatio || 1, 1.5));
        renderer.getSize && renderer.setSize(graph3dContainer.clientWidth, graph3dContainer.clientHeight);
      }
    }
    global3d.d3VelocityDecay(.45);
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
    if (global3d) {
      refresh3d();
      if (moveCamera) focus3dCamera(id);
    }
  }

  function resize() {
    if (global3d && graph3dContainer) {
      global3d.width(graph3dContainer.clientWidth).height(graph3dContainer.clientHeight);
    }
  }

  root.DojoGraph = {
    render3d,
    destroy3d,
    pause3d,
    has3d,
    resume3d,
    preload3d: load3dDependencies,
    focusGlobal,
    resize,
  };
}(window));
