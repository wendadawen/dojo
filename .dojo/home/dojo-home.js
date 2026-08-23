(function startDojoHome() {
  const model = window.DojoHomeModel;
  const graph = window.DojoGraph;
  const elements = {
    status: document.getElementById("app-status"),
    search: document.getElementById("search-input"),
    type: document.getElementById("type-filter"),
    topic: document.getElementById("topic-filter"),
    sort: document.getElementById("sort-order"),
    tabs: [...document.querySelectorAll("[data-view]")],
    libraryView: document.getElementById("library-view"),
    mapView: document.getElementById("map-view"),
    grid: document.getElementById("library-grid"),
    count: document.getElementById("result-count"),
    empty: document.getElementById("empty-state"),
    clear: document.getElementById("clear-filters"),
    globalGraph: document.getElementById("global-graph"),
    globalGraph3d: document.getElementById("global-graph-3d"),
    mapCount: document.getElementById("map-count"),
    mapWorkspace: document.getElementById("map-workspace"),
    mapDetails: document.getElementById("map-details"),
    resetMap: document.getElementById("reset-map"),
    graphModes: [...document.querySelectorAll("[data-graph-mode]")],
  };
  const initialGraphMode = window.matchMedia("(max-width: 760px)").matches ? "2d" : "3d";
  const SORT_STORAGE_KEY = "dojo-home-sort";
  const SORT_OPTIONS = new Set(["newest", "oldest", "title"]);

  function loadStoredSort() {
    try {
      const stored = window.localStorage.getItem(SORT_STORAGE_KEY);
      return SORT_OPTIONS.has(stored) ? stored : "newest";
    } catch (error) {
      return "newest";
    }
  }

  function storeSort(value) {
    try {
      window.localStorage.setItem(SORT_STORAGE_KEY, value);
    } catch (error) {
      /* file:// or restricted contexts: ignore */
    }
  }

  const state = {
    catalog: null,
    query: "",
    type: "",
    topic: "",
    sort: loadStoredSort(),
    view: "library",
    graphMode: initialGraphMode,
  };

  function setStatus(message, isError = false) {
    elements.status.textContent = message;
    elements.status.hidden = !message;
    elements.status.classList.toggle("is-error", isError);
  }

  function optionMarkup(value, label) {
    return `<option value="${model.escapeHtml(value)}">${model.escapeHtml(label)}</option>`;
  }

  function populateFilters() {
    const options = model.getFilterOptions(state.catalog.pages);
    elements.type.innerHTML = optionMarkup("", "全部类型")
      + options.types.map((value) => optionMarkup(value, model.typeLabel(value))).join("");
    elements.topic.innerHTML = optionMarkup("", "全部主题")
      + options.topics.map((value) => optionMarkup(value, value)).join("");
  }

  function filteredPages() {
    return model.filterPages(state.catalog.pages, state);
  }

  function renderMath(container) {
    if (!window.renderMathInElement || !container) return;
    window.renderMathInElement(container, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  }

  function renderLibrary() {
    if (!state.catalog) return;
    const pages = filteredPages();
    elements.grid.innerHTML = pages.map(model.renderCard).join("");
    renderMath(elements.grid);
    elements.count.textContent = pages.length === state.catalog.pages.length
      ? `共 ${pages.length} 篇`
      : `${pages.length} / ${state.catalog.pages.length} 篇`;
    elements.empty.hidden = pages.length !== 0;
  }

  function relationList(ids) {
    const pages = new Map(state.catalog.pages.map((page) => [page.id, page]));
    if (!ids.length) return '<p class="relation-empty">暂无关联文档</p>';
    return `<ul class="relation-list">${ids.map((id) => {
      const page = pages.get(id);
      if (!page) return "";
      return `<li><button type="button" data-map-id="${model.escapeHtml(id)}">${model.escapeHtml(page.title)}</button></li>`;
    }).join("")}</ul>`;
  }

  function relationGroup(title, note, ids, direction) {
    return `
      <section class="relation-group is-${model.escapeHtml(direction)}">
        <div class="relation-group-header">
          <h4>${model.escapeHtml(title)}</h4>
          <span>${ids.length} ${model.escapeHtml(note)}</span>
        </div>
        ${relationList(ids)}
      </section>`;
  }

  function resetMapDetails() {
    elements.mapDetails.replaceChildren();
    elements.mapDetails.hidden = true;
    elements.mapWorkspace.classList.remove("has-details");
    requestAnimationFrame(graph.resize);
  }

  function renderMapDetails(id) {
    const page = state.catalog.pages.find((item) => item.id === id);
    if (!page) return;
    elements.mapDetails.hidden = false;
    elements.mapWorkspace.classList.add("has-details");
    elements.mapDetails.innerHTML = `
      <p class="detail-type type-${model.escapeHtml(page.type)}">${model.escapeHtml(model.typeLabel(page.type))}</p>
      <h3>${model.escapeHtml(page.title)}</h3>
      <p class="detail-description">${model.escapeHtml(page.summary || page.description || "暂无摘要")}</p>
      <div class="detail-actions">
        <a class="detail-open" href="${model.escapeHtml(page.path)}" target="_blank" rel="noopener">阅读文档&nbsp; ↗</a>
      </div>
      ${relationGroup("引用本文", "篇", page.incoming || [], "incoming")}
      ${relationGroup("本文引用", "篇", page.outgoing || [], "outgoing")}`;
    renderMath(elements.mapDetails);
    requestAnimationFrame(graph.resize);
  }

  async function renderMap() {
    if (!state.catalog) return;
    resetMapDetails();
    const matched = filteredPages();
    const isFiltered = Boolean(state.query || state.type || state.topic);
    const matchIds = isFiltered ? new Set(matched.map((page) => page.id)) : null;
    // 搜索时把每个命中节点的引用和被引用（一跳邻居）一并纳入地图
    const visibleIds = new Set(matchIds || matched.map((page) => page.id));
    if (matchIds) {
      for (const page of matched) {
        (page.incoming || []).forEach((id) => visibleIds.add(id));
        (page.outgoing || []).forEach((id) => visibleIds.add(id));
      }
    }
    const edgeCount = state.catalog.edges.filter(
      (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
    ).length;
    elements.mapCount.textContent = matchIds
      ? `${matchIds.size} 篇匹配 · ${visibleIds.size} 个节点 · ${edgeCount} 条引用`
      : `${visibleIds.size} 个节点 · ${edgeCount} 条引用`;
    elements.globalGraph.hidden = state.graphMode !== "2d";
    elements.globalGraph3d.hidden = state.graphMode !== "3d";
    graph.setMode(state.graphMode);
    try {
      if (state.graphMode === "3d") {
        graph.destroyGlobal();
        await graph.render3d(
          elements.globalGraph3d,
          state.catalog,
          visibleIds,
          renderMapDetails,
          resetMapDetails,
          matchIds,
        );
      } else {
        graph.destroy3d();
        graph.renderGlobal(
          elements.globalGraph,
          state.catalog,
          visibleIds,
          renderMapDetails,
          resetMapDetails,
          matchIds,
        );
      }
      setStatus("");
    } catch (error) {
      if (state.graphMode === "3d") {
        state.graphMode = "2d";
        elements.graphModes.forEach((button) => {
          const active = button.dataset.graphMode === "2d";
          button.classList.toggle("is-active", active);
          button.setAttribute("aria-pressed", String(active));
        });
        elements.globalGraph.hidden = false;
        elements.globalGraph3d.hidden = true;
        graph.destroy3d();
        graph.renderGlobal(
          elements.globalGraph,
          state.catalog,
          visibleIds,
          renderMapDetails,
          resetMapDetails,
          matchIds,
        );
        setStatus("3D 不可用，当前显示 2D。", true);
        return;
      }
      elements.mapDetails.innerHTML = `<div class="map-empty"><h3>知识地图错误</h3><p>${model.escapeHtml(error.message)}</p></div>`;
      setStatus("知识地图错误。", true);
    }
  }

  function setGraphMode(mode) {
    state.graphMode = mode;
    elements.graphModes.forEach((button) => {
      const active = button.dataset.graphMode === mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    if (state.view === "map") renderMap();
  }

  function setView(view) {
    state.view = view;
    elements.libraryView.hidden = view !== "library";
    elements.mapView.hidden = view !== "map";
    elements.tabs.forEach((tab) => {
      const active = tab.dataset.view === view;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    if (view === "map" && state.catalog) {
      renderMap();
    } else {
      graph.destroyGlobal();
      graph.destroy3d();
    }
  }

  function applyFilters() {
    if (!state.catalog) return;
    renderLibrary();
    if (state.view === "map") renderMap();
  }

  function clearFilters() {
    state.query = "";
    state.type = "";
    state.topic = "";
    elements.search.value = "";
    elements.type.value = "";
    elements.topic.value = "";
    applyFilters();
  }

  function bindEvents() {
    elements.search.addEventListener("input", () => {
      state.query = elements.search.value;
      applyFilters();
    });
    elements.type.addEventListener("change", () => {
      state.type = elements.type.value;
      applyFilters();
    });
    elements.topic.addEventListener("change", () => {
      state.topic = elements.topic.value;
      applyFilters();
    });
    if (elements.sort) {
      elements.sort.value = state.sort;
      elements.sort.addEventListener("change", () => {
        state.sort = elements.sort.value;
        storeSort(state.sort);
        applyFilters();
      });
    }
    elements.tabs.forEach((tab) => tab.addEventListener("click", () => setView(tab.dataset.view)));
    elements.graphModes.forEach((button) => {
      button.addEventListener("click", () => setGraphMode(button.dataset.graphMode));
    });
    elements.clear.addEventListener("click", clearFilters);
    elements.resetMap.addEventListener("click", () => {
      graph.reset(state.graphMode);
      resetMapDetails();
    });
    document.addEventListener("click", (event) => {
      const open = event.target.closest("[data-open-path]");
      if (open) {
        window.open(open.dataset.openPath, "_blank", "noopener");
        return;
      }
      const mapTarget = event.target.closest("[data-map-id]");
      if (mapTarget) {
        graph.focusGlobal(mapTarget.dataset.mapId, state.graphMode === "3d");
        renderMapDetails(mapTarget.dataset.mapId);
      }
    });
  }

  function showLoadError(error) {
    setStatus(`目录错误：${error.message}`, true);
    elements.grid.innerHTML = `
      <article class="document-card">
        <div class="card-main">
          <h2>目录错误</h2>
          <p>catalog.json 未能加载。</p>
        </div>
      </article>`;
  }

  async function load() {
    try {
      const response = await fetch("catalog.json");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.catalog = await response.json();
      populateFilters();
      renderLibrary();
      if (state.catalog.warnings.length) {
        setStatus(`目录中有 ${state.catalog.warnings.length} 条链接警告。`, true);
      } else {
        setStatus("");
      }
    } catch (error) {
      showLoadError(error);
    }
  }

  bindEvents();
  elements.graphModes.forEach((button) => {
    const active = button.dataset.graphMode === state.graphMode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  resetMapDetails();
  load();
}());
