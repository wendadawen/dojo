(function startDojoHome() {
  const model = window.DojoHomeModel;
  const graph = window.DojoGraph;
  const elements = {
    status: document.getElementById("app-status"),
    search: document.getElementById("search-input"),
    type: document.getElementById("type-filter"),
    topic: document.getElementById("topic-filter"),
    tabs: [...document.querySelectorAll("[data-view]")],
    libraryView: document.getElementById("library-view"),
    mapView: document.getElementById("map-view"),
    grid: document.getElementById("library-grid"),
    count: document.getElementById("result-count"),
    empty: document.getElementById("empty-state"),
    clear: document.getElementById("clear-filters"),
    globalGraph: document.getElementById("global-graph"),
    mapCount: document.getElementById("map-count"),
    mapDetails: document.getElementById("map-details"),
    resetMap: document.getElementById("reset-map"),
  };
  const state = { catalog: null, query: "", type: "", topic: "", view: "library" };

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

  function renderLibrary() {
    if (!state.catalog) return;
    const pages = filteredPages();
    elements.grid.innerHTML = pages.map(model.renderCard).join("");
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

  function relationGroup(title, note, ids) {
    return `
      <section class="relation-group">
        <div class="relation-group-header">
          <h4>${model.escapeHtml(title)}</h4>
          <span>${ids.length} ${model.escapeHtml(note)}</span>
        </div>
        ${relationList(ids)}
      </section>`;
  }

  function resetMapDetails() {
    elements.mapDetails.innerHTML = `
      <div class="map-empty">
        <span class="map-step">↗</span>
        <h3>从一个节点开始</h3>
        <p>点击节点查看它附近的一跳关系；悬停节点可查看简称，滚轮缩放，拖动画布移动。</p>
      </div>`;
  }

  function renderMapDetails(id) {
    const page = state.catalog.pages.find((item) => item.id === id);
    if (!page) return;
    elements.mapDetails.innerHTML = `
      <p class="detail-type type-${model.escapeHtml(page.type)}">${model.escapeHtml(model.typeLabel(page.type))}</p>
      <h3>${model.escapeHtml(page.title)}</h3>
      <p class="detail-description">${model.escapeHtml(page.description || "暂无摘要")}</p>
      <div class="detail-actions">
        <a class="detail-open" href="${model.escapeHtml(page.path)}">阅读文档&nbsp; ↗</a>
        <span class="detail-hint">图中已突出相邻节点</span>
      </div>
      ${relationGroup("引用本文", "篇", page.incoming || [])}
      ${relationGroup("本文引用", "篇", page.outgoing || [])}`;
  }

  function renderMap() {
    if (!state.catalog) return;
    resetMapDetails();
    const visibleIds = new Set(filteredPages().map((page) => page.id));
    const edgeCount = state.catalog.edges.filter(
      (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
    ).length;
    elements.mapCount.textContent = `${visibleIds.size} 个节点 · ${edgeCount} 条引用`;
    try {
      graph.renderGlobal(
        elements.globalGraph,
        state.catalog,
        visibleIds,
        renderMapDetails,
        resetMapDetails,
      );
    } catch (error) {
      elements.mapDetails.innerHTML = `<div class="map-empty"><h3>知识地图不可用</h3><p>${model.escapeHtml(error.message)}</p></div>`;
      setStatus("知识地图加载失败，文档索引仍可使用。", true);
    }
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
    if (view === "map" && state.catalog) renderMap();
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
    elements.tabs.forEach((tab) => tab.addEventListener("click", () => setView(tab.dataset.view)));
    elements.clear.addEventListener("click", clearFilters);
    elements.resetMap.addEventListener("click", renderMap);
    document.addEventListener("click", (event) => {
      const open = event.target.closest("[data-open-path]");
      if (open) window.location.href = open.dataset.openPath;
      const mapTarget = event.target.closest("[data-map-id]");
      if (mapTarget) {
        graph.focusGlobal(mapTarget.dataset.mapId);
        renderMapDetails(mapTarget.dataset.mapId);
      }
    });
  }

  function showLoadError(error) {
    setStatus(`目录加载失败：${error.message}`, true);
    elements.grid.innerHTML = `
      <article class="document-card">
        <div class="card-main">
          <h2>无法加载目录</h2>
          <p>请检查最近一次 GitHub Pages 构建是否成功。</p>
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
  resetMapDetails();
  load();
}());
