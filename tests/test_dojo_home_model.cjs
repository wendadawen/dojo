const test = require('node:test');
const assert = require('node:assert/strict');
const model = require('../assets/dojo-home-model.js');

const catalog = {
  pages: [
    {
      id: 'wiki/attention/index.html',
      path: 'wiki/attention/index.html',
      title: '标准注意力',
      description: 'Transformer attention basics',
      type: 'concept',
      topics: ['注意力机制'],
      tag: '基础',
      incoming: ['wiki/kda/index.html'],
      outgoing: [],
      incoming_count: 1,
      outgoing_count: 0,
    },
    {
      id: 'wiki/kda/index.html',
      path: 'wiki/kda/index.html',
      title: 'KDA',
      description: 'Delta rule linear attention',
      type: 'paper',
      topics: ['注意力机制'],
      tag: '模型架构',
      incoming: [],
      outgoing: ['wiki/attention/index.html'],
      incoming_count: 0,
      outgoing_count: 1,
    },
  ],
  edges: [
    {
      id: 'wiki/kda/index.html::wiki/attention/index.html',
      source: 'wiki/kda/index.html',
      target: 'wiki/attention/index.html',
      count: 2,
    },
  ],
};

test('search and filters combine', () => {
  assert.deepEqual(
    model.filterPages(catalog.pages, {
      query: 'delta',
      type: 'paper',
      topic: '注意力机制',
    }).map((page) => page.id),
    ['wiki/kda/index.html'],
  );
});

test('local graph separates incoming nodes', () => {
  const graph = model.makeLocalGraph(catalog, 'wiki/attention/index.html');
  assert.deepEqual(graph.nodes.map((node) => node.data.role), ['center', 'incoming']);
  assert.equal(graph.edges[0].data.count, 2);
});

test('card markup escapes html and exposes actions', () => {
  const markup = model.renderCard({ ...catalog.pages[0], title: '<Attention>' });
  assert.match(markup, /&lt;Attention&gt;/);
  assert.match(markup, /data-open-path=/);
  assert.match(markup, /data-relation-id=/);
});
