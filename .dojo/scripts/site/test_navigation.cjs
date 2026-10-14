const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const root = path.resolve(__dirname, "../../..");
const read = (file) => readFileSync(path.join(root, file), "utf8");
const query = "(hover: hover) and (pointer: fine)";

function mediaQuery(matches) {
  const listeners = [];
  return {
    matches,
    addEventListener(event, listener) {
      assert.equal(event, "change");
      listeners.push(listener);
    },
    change(matches) {
      this.matches = matches;
      listeners.forEach((listener) => listener());
    },
  };
}

function element(dataset = {}) {
  return {
    dataset,
    events: {},
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener(event, listener) { this.events[event] = listener; },
    setAttribute() {},
    replaceChildren() {},
    querySelectorAll() { return []; },
  };
}

for (const desktop of [false, true]) {
  const device = desktop ? "desktop" : "touch";
  const expectedTarget = desktop ? "_blank" : "_self";

  test(`${device}: concept links and input-device changes`, () => {
    const media = mediaQuery(desktop);
    const hrefs = ["../moe/index.html", "overview.html", "index.html#s1", "https://example.com", "#s1", ""];
    const links = hrefs.map((href) => ({ getAttribute: () => href, target: "", rel: "" }));
    vm.runInNewContext(read("libs/concept/dojo-links.js"), {
      matchMedia(value) { assert.equal(value, query); return media; },
      document: { querySelectorAll() { return links; } },
    });
    links.slice(0, 4).forEach((link) => {
      assert.equal(link.target, expectedTarget);
      assert.equal(link.rel, "noopener");
    });
    links.slice(4).forEach((link) => assert.equal(link.target, ""));
    media.change(!desktop);
    links.slice(0, 4).forEach((link) => assert.equal(link.target, desktop ? "_self" : "_blank"));
    links.slice(4).forEach((link) => assert.equal(link.target, ""));
  });

  test(`${device}: library cards and knowledge-map links`, async () => {
    const media = mediaQuery(desktop);
    const elements = new Map();
    const mapTab = element({ view: "map" });
    const clicks = [];
    const page = { id: "wiki/moe/index.html", path: "wiki/moe/index.html", title: "MoE" };
    const document = {
      events: {},
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
      },
      querySelectorAll() { return [mapTab]; },
      addEventListener(event, listener) { this.events[event] = listener; },
    };
    const context = vm.createContext({
      document,
      requestAnimationFrame() {},
      fetch: async () => ({ ok: true, json: async () => ({ pages: [page], warnings: [] }) }),
      window: {
        matchMedia(value) { assert.equal(value, query); return media; },
        open(...args) { clicks.push(["open", ...args]); },
        location: { assign(url) { clicks.push(["assign", url]); } },
        requestIdleCallback() {},
        DojoGraph: {
          resize() {}, pause3d() {}, has3d() { return false; },
          async render3d(container, catalog, ids, onSelect) { onSelect(page.id); },
        },
      },
    });
    vm.runInContext(read(".dojo/site/dojo-home-model.js"), context);
    vm.runInContext(read(".dojo/site/dojo-home.js"), context);
    await new Promise(setImmediate);
    document.events.click({ target: { closest: (selector) => selector === "[data-open-path]" ? { dataset: { openPath: page.path } } : null } });
    assert.deepEqual(clicks, desktop ? [["open", page.path, "_blank", "noopener"]] : [["assign", page.path]]);
    mapTab.events.click();
    assert.ok(elements.get("map-details").innerHTML.includes(`target="${expectedTarget}"`));
    media.change(!desktop);
    const link = { target: expectedTarget };
    document.events.click({ target: { closest: (selector) => selector === ".detail-open" ? link : null } });
    assert.equal(link.target, desktop ? "_self" : "_blank");
  });
}

test("both concept templates load the shared link policy", () => {
  for (const template of ["index", "overview"]) {
    const source = read(`.dojo/templates/concept/${template}.html`);
    assert.ok(source.includes('<script defer src="../../libs/concept/dojo-links.js"></script>'));
    assert.ok(!source.includes("a.target = '_blank'"));
  }
});
