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
    mapDetails: document.getElementById("map-details"),
    panel: document.getElementById("relation-panel"),
    backdrop: document.getElementById("relation-backdrop"),
    close: document.getElementById("close-relation"),
    relationTitle: document.getElementById("relation-title"),
    relationOpen: document.getElementById("relation-open"),
    localGraph: document.getElementById("local-graph"),
    incoming: document.getElementById("incoming-list"),
    outgoing: document.getElementById("outgoing-list"),
  };
  const state = { catalog: null, query: "", type: "", topic: "", view: "library" };

  function setStatus(message, isError = false) {
    elements.status.textContent = message;
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
    elements.count.textContent = `${pages.length} / ${state.catalog.pages.length} 篇`;
    elements.empty.hidden = pages.length !== 0;
  }

  function renderRelationList(container, ids) {
    const pages = new Map(state.catalog.pages.map((page) => [page.id, page]));
    container.innerHTML = ids.length
      ? ids.map((id) => {
        const page = pages.get(id);
        if (!page) return "";
        return `<div class="relation-list-item">
          <button class="relation-link" type="button" data-relation-id="${model.escapeHtml(id)}">${model.escapeHtml(page.title)}</button>
          <a href="${model.escapeHtml(page.path)}">打开</a>
        </div>`;
      }).join("")
      : '<p class="subtitle">暂无站内链接</p>';
  }

  function openRelationPanel(id) {
    const page = state.catalog.pages.find((item) => item.id === id);
    if (!page) return;
    elements.relationTitle.textContent = page.title;
    elements.relationOpen.href = page.path;
    renderRelationList(elements.incoming, page.incoming || []);
    renderRelationList(elements.outgoing, page.outgoing || []);
    elements.panel.hidden = false;
    elements.backdrop.hidden = false;
    elements.localGraph.textContent = "";
    try {
      graph.renderLocal(elements.localGraph, state.catalog, id, openRelationPanel);
    } catch (error) {
      elements.localGraph.textContent = `关系图加载失败：${error.message}`;
    }
    elements.close.focus();
  }

  function closeRelationPanel() {
    elements.panel.hidden = true;
    elements.backdrop.hidden = true;
    graph.destroyLocal();
  }

  function resetMapDetails() {
    elements.mapDetails.innerHTML = `
      <h2>知识地图</h2>
      <p>选择节点后查看它的入链和出链。</p>
      <button type="button" id="reset-map" class="secondary-button">重置地图</button>`;
    document.getElementById("reset-map").addEventListener("click", renderMap);
  }

  function renderMapDetails(id) {
    const page = state.catalog.pages.find((item) => item.id === id);
    if (!page) return;
    elements.mapDetails.innerHTML = `
      <p class="eyebrow">${model.escapeHtml(model.typeLabel(page.type))}</p>
      <h2>${model.escapeHtml(page.title)}</h2>
      <p>${model.escapeHtml(page.description || "暂无摘要")}</p>
      <p>入链 ${page.incoming_count || 0} · 出链 ${page.outgoing_count || 0}</p>
      <a class="action-link" href="${model.escapeHtml(page.path)}">打开文章</a>
      <button class="relation-link" type="button" data-relation-id="${model.escapeHtml(page.id)}">查看局部关系</button>
      <button type="button" id="reset-map" class="secondary-button">重置地图</button>`;
    document.getElementById("reset-map").addEventListener("click", renderMap);
  }

  function renderMap() {
    if (!state.catalog) return;
    resetMapDetails();
    const visibleIds = new Set(filteredPages().map((page) => page.id));
    try {
      graph.renderGlobal(elements.globalGraph, state.catalog, visibleIds, renderMapDetails);
    } catch (error) {
      elements.mapDetails.innerHTML = `<h2>知识地图不可用</h2><p>${model.escapeHtml(error.message)}</p>`;
      setStatus("知识地图加载失败，全部内容仍可使用。", true);
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
    document.getElementById("reset-map").addEventListener("click", renderMap);
    elements.close.addEventListener("click", closeRelationPanel);
    elements.backdrop.addEventListener("click", closeRelationPanel);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !elements.panel.hidden) closeRelationPanel();
    });
    document.addEventListener("click", (event) => {
      const open = event.target.closest("[data-open-path]");
      if (open) window.location.href = open.dataset.openPath;
      const relation = event.target.closest("[data-relation-id]");
      if (relation) openRelationPanel(relation.dataset.relationId);
    });
  }

  function showLoadError(error) {
    setStatus(`目录加载失败：${error.message}`, true);
    elements.grid.innerHTML = `
      <article class="document-card">
        <div class="card-main">
          <h2>无法加载 catalog.json</h2>
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
      const warningText = state.catalog.warnings.length
        ? `，${state.catalog.warnings.length} 条链接警告`
        : "";
      setStatus(`已加载 ${state.catalog.pages.length} 篇内容${warningText}`);
    } catch (error) {
      showLoadError(error);
    }
  }

  bindEvents();
  load();
}());
