const terminal = document.getElementById('terminal');
const input = document.getElementById('prompt-input');
const chipRow = document.getElementById('chip-row');
const viewMode = document.getElementById('view-mode');
const mapEl = document.getElementById('system-map');
const mapNodesEl = document.getElementById('map-nodes');
const mapLinesEl = document.getElementById('map-lines');
const mapPlaceholder = document.getElementById('map-placeholder');
const mapToolbar = document.getElementById('map-toolbar');
const miniMap = document.getElementById('mini-map');

const divider = document.getElementById('divider');
const workspace = document.querySelector('.workspace');

const nodeDefs = {
  frontend: { title: 'Frontend', sub: 'React + Design System', x: 70, y: 70 },
  api: { title: 'API Gateway', sub: 'Routing + Validation', x: 320, y: 170 },
  task: { title: 'Task Service', sub: 'Domain Rules', x: 240, y: 300 },
  db: { title: 'Database', sub: 'Postgres', x: 560, y: 300 },
  auth: { title: 'Auth', sub: 'JWT + Roles', x: 560, y: 70 },
  queue: { title: 'Queue', sub: 'Async Jobs', x: 610, y: 190 },
  search: { title: 'Search', sub: 'Filters + Index', x: 600, y: 400 }
};

const edgeDefs = {
  'front-api': { from: 'frontend', to: 'api' },
  'api-task': { from: 'api', to: 'task' },
  'task-db': { from: 'task', to: 'db' },
  'api-auth': { from: 'api', to: 'auth' },
  'api-queue': { from: 'api', to: 'queue' },
  'task-search': { from: 'task', to: 'search' }
};

const functionNodeDefs = {
  taskCore: { title: 'Task Service', sub: 'Core domain', x: 310, y: 160 },
  createTask: { title: 'createTask()', sub: 'Validate + persist', x: 120, y: 70 },
  transition: { title: 'transitionTaskStatus()', sub: 'State machine', x: 120, y: 270 },
  assign: { title: 'assignTask()', sub: 'Ownership rules', x: 120, y: 370 },
  audit: { title: 'writeAuditLog()', sub: 'Compliance', x: 520, y: 120 },
  index: { title: 'updateSearchIndex()', sub: 'Filters', x: 520, y: 320 }
};

const functionEdgeDefs = {
  'task-create': { from: 'taskCore', to: 'createTask' },
  'task-transition': { from: 'taskCore', to: 'transition' },
  'task-assign': { from: 'taskCore', to: 'assign' },
  'task-audit': { from: 'taskCore', to: 'audit' },
  'task-index': { from: 'taskCore', to: 'index' }
};

const componentMaps = {
  frontend: {
    label: 'Frontend',
    nodes: {
      appShell: { title: 'App Shell', sub: 'Routing + Layout', x: 300, y: 70 },
      taskList: { title: 'TaskList', sub: 'Filters + Sort', x: 120, y: 200 },
      taskDetail: { title: 'TaskDetail', sub: 'Drawer', x: 120, y: 320 },
      editor: { title: 'TaskEditor', sub: 'Create/Edit', x: 480, y: 200 },
      design: { title: 'Design Tokens', sub: 'Color + Type', x: 480, y: 320 }
    },
    edges: {
      'shell-list': { from: 'appShell', to: 'taskList' },
      'shell-detail': { from: 'appShell', to: 'taskDetail' },
      'shell-editor': { from: 'appShell', to: 'editor' },
      'editor-design': { from: 'editor', to: 'design' }
    }
  },
  api: {
    label: 'API Gateway',
    nodes: {
      gateway: { title: 'Gateway', sub: 'Routing', x: 310, y: 90 },
      tasks: { title: 'Tasks Router', sub: 'CRUD', x: 120, y: 210 },
      authz: { title: 'AuthZ', sub: 'Policy checks', x: 480, y: 210 },
      events: { title: 'Event Bus', sub: 'TaskUpdated', x: 310, y: 320 }
    },
    edges: {
      'gate-tasks': { from: 'gateway', to: 'tasks' },
      'gate-authz': { from: 'gateway', to: 'authz' },
      'tasks-events': { from: 'tasks', to: 'events' }
    }
  },
  task: {
    label: 'Task Service',
    nodes: functionNodeDefs,
    edges: functionEdgeDefs
  },
  db: {
    label: 'Database',
    nodes: {
      tasks: { title: 'tasks', sub: 'Core table', x: 120, y: 120 },
      events: { title: 'task_events', sub: 'Audit log', x: 120, y: 280 },
      assignments: { title: 'task_assignments', sub: 'Ownership', x: 420, y: 120 },
      indexes: { title: 'Indexes', sub: 'Status + Tag', x: 420, y: 280 }
    },
    edges: {
      'tasks-events': { from: 'tasks', to: 'events' },
      'tasks-assign': { from: 'tasks', to: 'assignments' },
      'tasks-index': { from: 'tasks', to: 'indexes' }
    }
  },
  auth: {
    label: 'Auth',
    nodes: {
      authCore: { title: 'Auth Core', sub: 'JWT + Sessions', x: 300, y: 90 },
      roles: { title: 'Roles', sub: 'Admin, Team', x: 120, y: 220 },
      policy: { title: 'Policy Engine', sub: 'Permissions', x: 480, y: 220 },
      audit: { title: 'Access Log', sub: 'Security', x: 300, y: 320 }
    },
    edges: {
      'core-roles': { from: 'authCore', to: 'roles' },
      'core-policy': { from: 'authCore', to: 'policy' },
      'policy-audit': { from: 'policy', to: 'audit' }
    }
  },
  queue: {
    label: 'Queue',
    nodes: {
      broker: { title: 'Broker', sub: 'Topics', x: 300, y: 90 },
      email: { title: 'Email Worker', sub: 'Notifications', x: 120, y: 240 },
      webhook: { title: 'Webhook Worker', sub: 'Outbound', x: 480, y: 240 },
      retries: { title: 'Retry Policy', sub: 'DLQ', x: 300, y: 340 }
    },
    edges: {
      'broker-email': { from: 'broker', to: 'email' },
      'broker-webhook': { from: 'broker', to: 'webhook' },
      'broker-retry': { from: 'broker', to: 'retries' }
    }
  },
  search: {
    label: 'Search',
    nodes: {
      indexer: { title: 'Indexer', sub: 'Ingest tasks', x: 120, y: 120 },
      store: { title: 'Index Store', sub: 'Filters', x: 420, y: 120 },
      query: { title: 'Query Engine', sub: 'Search API', x: 120, y: 280 },
      rank: { title: 'Ranker', sub: 'Relevance', x: 420, y: 280 }
    },
    edges: {
      'index-store': { from: 'indexer', to: 'store' },
      'store-query': { from: 'store', to: 'query' },
      'query-rank': { from: 'query', to: 'rank' }
    }
  }
};

const systemState = {
  nodes: new Set(),
  edges: new Set()
};

let activeNodes = new Set();
let activeEdges = new Set();
let isTyping = false;
let mapMode = 'system';
let mapFocusComponent = null;
let currentComponentDefs = null;

const script = {
  projectOptions: [
    { label: 'Task Manager', value: 'task' },
    { label: 'Tiny CRM', value: 'crm' },
    { label: 'Social App', value: 'social' }
  ],
  taskManager: {
    questions: [
      {
        id: 'screens',
        prompt: 'Which screens should the frontend include? (List, Detail, Create/Edit?)',
        focusNode: 'frontend',
        onAnswer: () => {
          addSystemNode('frontend');
        },
        detail: {
          sub: 'Frontend screens',
          body: 'Task list, task detail, create/edit drawer, and a focused status kanban view.'
        }
      },
      {
        id: 'states',
        prompt: 'What task states should we support? (Backlog → In Progress → Done?)',
        focusNode: 'task',
        onAnswer: () => {
          addSystemNode('api');
          addSystemNode('task');
          addSystemNode('db');
          addSystemEdge('front-api');
          addSystemEdge('api-task');
          addSystemEdge('task-db');
        },
        detail: {
          sub: 'State model',
          body: 'Backlog → In Progress → Blocked → Done. Task Service enforces transitions.'
        }
      },
      {
        id: 'auth',
        prompt: 'Who can create, assign, and delete tasks? (Team vs admin?)',
        focusNode: 'auth',
        onAnswer: () => {
          addSystemNode('auth');
          addSystemEdge('api-auth');
        },
        detail: {
          sub: 'Permissions',
          body: 'Team members can create/assign. Admins can delete and override statuses.'
        }
      },
      {
        id: 'notifications',
        prompt: 'Any async jobs or notifications? (Email, webhook, reminders?)',
        focusNode: 'queue',
        onAnswer: () => {
          addSystemNode('queue');
          addSystemEdge('api-queue');
        },
        detail: {
          sub: 'Async pipeline',
          body: 'Task updates emit events → queue → email + webhook workers.'
        }
      },
      {
        id: 'search',
        prompt: 'Need filtering or search? (Tags, assignee, status?)',
        focusNode: 'search',
        onAnswer: () => {
          addSystemNode('search');
          addSystemEdge('task-search');
        },
        detail: {
          sub: 'Search index',
          body: 'Filters for tags, assignee, due date, and status with a lightweight index.'
        }
      }
    ],
    plan: {
      text:
        'Plan:\n1) Define data model + API endpoints\n2) Implement Task Service state machine + permissions\n3) Build UI screens using design system tokens\n4) Wire search + async notifications\n\nApprove plan? (approve / revise)'
    },
    buildSteps: [
      {
        label: 'Scaffold UI',
        focusNode: 'frontend',
        mode: 'system',
        detail: {
          sub: 'UI scaffold',
          body: 'Create task list, detail drawer, and create/edit modal using design tokens.'
        },
        actions: ['[scaffold] ui/layout', '[write] components/TaskList', '[write] components/TaskDetail']
      },
      {
        label: 'Define models',
        focusNode: 'db',
        mode: 'system',
        detail: {
          sub: 'Data model',
          body: 'tasks, task_assignments, task_events tables with audit fields.'
        },
        actions: ['[write] schema/tasks.sql', '[write] schema/task_events.sql'],
        edits: [
          {
            file: 'schema/tasks.sql',
            summary: 'Edited schema/tasks.sql (+12 -0)',
            lines: [
              '1  CREATE TABLE tasks (',
              '2    id uuid PRIMARY KEY,',
              '3    title text NOT NULL,',
              '4    status text NOT NULL,',
              '5    assignee_id uuid,',
              '6    due_at timestamp,',
              '7    created_at timestamp DEFAULT now()',
              '8  );'
            ]
          }
        ]
      },
      {
        label: 'Implement transitions',
        focusNode: 'task',
        mode: 'function',
        detail: {
          sub: 'transitionTaskStatus()',
          body: 'Guards, persistence, audit log, and event emission.'
        },
        actions: ['[write] task_service/transition.ts', '[update] task_service/index.ts'],
        edits: [
          {
            file: 'task_service/transition.ts',
            summary: 'Edited task_service/transition.ts (+9 -1)',
            lines: [
              '12 export function transitionTaskStatus(task, nextStatus, user) {',
              '13   assertCanTransition(task.status, nextStatus);',
              '14   assertPermissions(user, task);',
              '15   task.status = nextStatus;',
              '16   writeAuditLog(task.id, user.id, nextStatus);',
              '17   emitTaskUpdated(task);',
              '18   updateSearchIndex(task);',
              '19   return task;',
              '20 }'
            ]
          }
        ]
      },
      {
        label: 'Auth rules',
        focusNode: 'auth',
        mode: 'system',
        detail: {
          sub: 'Access control',
          body: 'Role matrix enforced at gateway and service boundary.'
        },
        actions: ['[update] api/middleware/auth.ts', '[write] permissions/roles.ts'],
        edits: [
          {
            file: 'permissions/roles.ts',
            summary: 'Edited permissions/roles.ts (+6 -0)',
            lines: [
              '1 export const roles = {',
              '2   admin: [\"delete\", \"assign\", \"transition\"],',
              '3   team: [\"create\", \"assign\", \"transition\"],',
              '4   viewer: [\"read\"]',
              '5 };'
            ]
          }
        ]
      },
      {
        label: 'Search index',
        focusNode: 'search',
        mode: 'system',
        detail: {
          sub: 'Search layer',
          body: 'Sync filters after task updates for fast list views.'
        },
        actions: ['[write] search/indexer.ts', '[update] task_service/events.ts'],
        edits: [
          {
            file: 'search/indexer.ts',
            summary: 'Edited search/indexer.ts (+8 -0)',
            lines: [
              '4 export function indexTask(task) {',
              '5   const doc = {',
              '6     id: task.id,',
              '7     status: task.status,',
              '8     tags: task.tags || [],',
              '9     assignee: task.assignee_id',
              '10  };',
              '11  return searchStore.upsert(doc);',
              '12 }'
            ]
          }
        ]
      },
      {
        label: 'Notifications',
        focusNode: 'queue',
        mode: 'system',
        detail: {
          sub: 'Queue workers',
          body: 'Email + webhook workers subscribed to TaskUpdated events.'
        },
        actions: ['[write] workers/email.ts', '[write] workers/webhook.ts'],
        edits: [
          {
            file: 'workers/email.ts',
            summary: 'Edited workers/email.ts (+7 -0)',
            lines: [
              '3 export async function sendTaskEmail(event) {',
              '4   const subject = `Task updated: ${event.task.title}`;',
              '5   const to = event.task.assigneeEmail;',
              '6   return mailer.send(to, subject, event.payload);',
              '7 }'
            ]
          }
        ]
      }
    ]
  }
};

const state = {
  phase: 'pick_project',
  project: null,
  questionIndex: 0
};

function addLine(role, text) {
  const line = document.createElement('div');
  line.className = `terminal-line ${role}`;
  line.textContent = text;
  terminal.appendChild(line);
  requestAnimationFrame(() => {
    terminal.scrollTop = terminal.scrollHeight;
  });
  return line;
}

function typeText(lineEl, text, callback) {
  let i = 0;
  const speed = 12;
  const interval = setInterval(() => {
    lineEl.textContent = text.slice(0, i);
    i += 1;
    requestAnimationFrame(() => {
      terminal.scrollTop = terminal.scrollHeight;
    });
    if (i > text.length) {
      clearInterval(interval);
      callback();
    }
  }, speed);
}

function runAssistant(text, callback) {
  isTyping = true;
  const line = addLine('assistant', 'co> …');
  setTimeout(() => {
    typeText(line, `co> ${text}`, () => {
      isTyping = false;
      if (callback) callback();
    });
  }, 200);
}

function setChips(items) {
  chipRow.innerHTML = '';
  items.forEach((item) => {
    const btn = document.createElement('button');
    btn.className = 'chip';
    btn.dataset.seed = item.label;
    btn.textContent = item.label;
    btn.addEventListener('click', () => handleSubmit(item.label));
    chipRow.appendChild(btn);
  });
}

function updateViewMode(label) {
  viewMode.textContent = label;
}

function updateMapPlaceholder() {
  const hasNodes = systemState.nodes.size > 0;
  mapPlaceholder.style.display = hasNodes ? 'none' : 'grid';
}

function renderMap(nodeSet, edgeSet, defs) {
  mapNodesEl.innerHTML = '';
  mapLinesEl.innerHTML = '';

  nodeSet.forEach((id) => {
    const def = defs.nodes[id];
    if (!def) return;
    const node = document.createElement('div');
    node.className = 'node';
    if (activeNodes.has(id)) node.classList.add('active');
    node.dataset.node = id;
    node.style.left = `${def.x}px`;
    node.style.top = `${def.y}px`;
    node.innerHTML = `<div class="node-title">${def.title}</div><div class="node-sub">${def.sub}</div>`;
    node.addEventListener('click', () => {
      if (mapMode === 'system') {
        enterComponentMap(id);
      } else {
        focusNodeById(id);
      }
    });
    mapNodesEl.appendChild(node);
  });

  const mapRect = mapEl.getBoundingClientRect();
  mapLinesEl.setAttribute('viewBox', `0 0 ${mapRect.width} ${mapRect.height}`);

  edgeSet.forEach((edgeId) => {
    const edge = defs.edges[edgeId];
    if (!edge) return;
    const fromEl = mapNodesEl.querySelector(`[data-node="${edge.from}"]`);
    const toEl = mapNodesEl.querySelector(`[data-node="${edge.to}"]`);
    if (!fromEl || !toEl) return;
    const startX = fromEl.offsetLeft + fromEl.offsetWidth / 2;
    const startY = fromEl.offsetTop + fromEl.offsetHeight / 2;
    const endX = toEl.offsetLeft + toEl.offsetWidth / 2;
    const endY = toEl.offsetTop + toEl.offsetHeight / 2;
    const midX = (startX + endX) / 2;
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('id', edgeId);
    const d = `M${startX} ${startY} C${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}`;
    path.setAttribute('d', d);
    if (activeEdges.has(edgeId)) path.classList.add('active');
    mapLinesEl.appendChild(path);
  });
}

function renderSystemMap() {
  mapMode = 'system';
  mapEl.classList.remove('function');
  renderMiniMap(null);
  renderMap(systemState.nodes, systemState.edges, { nodes: nodeDefs, edges: edgeDefs });
  updateMapPlaceholder();
}

function renderFunctionMap() {
  mapMode = 'function';
  mapEl.classList.add('function');
  const nodeSet = new Set(Object.keys(functionNodeDefs));
  const edgeSet = new Set(Object.keys(functionEdgeDefs));
  renderMap(nodeSet, edgeSet, { nodes: functionNodeDefs, edges: functionEdgeDefs });
}

function renderComponentMap(componentId) {
  const defs = componentMaps[componentId];
  if (!defs) return;
  mapMode = 'component';
  mapEl.classList.add('function');
  currentComponentDefs = defs;
  renderMiniMap(componentId);
  const nodeSet = new Set(Object.keys(defs.nodes));
  const edgeSet = new Set(Object.keys(defs.edges));
  renderMap(nodeSet, edgeSet, { nodes: defs.nodes, edges: defs.edges });
}

function setActiveFocus(nodeIds, edgeIds) {
  activeNodes = new Set(nodeIds);
  activeEdges = new Set(edgeIds);
}

function focusNodeById(id) {
  if (mapMode === 'function') {
    setActiveFocus([id], []);
    renderFunctionMap();
    addLine('system', `co> Focused on ${functionNodeDefs[id]?.title || id}.`);
    return;
  }

  if (mapMode === 'component' && currentComponentDefs) {
    const relatedEdges = Object.entries(currentComponentDefs.edges)
      .filter(([, edge]) => edge.from === id || edge.to === id)
      .map(([edgeId]) => edgeId);
    setActiveFocus([id], relatedEdges);
    renderComponentMap(mapFocusComponent);
    addLine('system', `co> Focused on ${currentComponentDefs.nodes[id]?.title || id}.`);
    return;
  }

  const relatedEdges = Object.entries(edgeDefs)
    .filter(([, edge]) => edge.from === id || edge.to === id)
    .map(([edgeId]) => edgeId);
  setActiveFocus([id], relatedEdges);
  renderSystemMap();
  addLine('system', `co> Focused on ${nodeDefs[id]?.title || id}.`);
}

function enterComponentMap(id) {
  if (!componentMaps[id]) {
    focusNodeById(id);
    addLine('system', 'co> No internal function map for this component yet.');
    return;
  }
  mapFocusComponent = id;
  setActiveFocus([Object.keys(componentMaps[id].nodes)[0]], []);
  renderComponentMap(id);
  addLine('system', `co> ${componentMaps[id].label} functions view.`);
}

function addSystemNode(id) {
  systemState.nodes.add(id);
  renderSystemMap();
}

function addSystemEdge(id) {
  systemState.edges.add(id);
  renderSystemMap();
}

function handleSubmit(value) {
  if (!value || isTyping) return;
  addLine('user', `you> ${value}`);
  input.value = '';

  if (state.phase === 'pick_project') {
    handleProjectPick(value);
    return;
  }

  if (state.phase === 'requirements') {
    handleAnswer(value);
    return;
  }

  if (state.phase === 'plan') {
    handlePlanResponse(value);
    return;
  }

  if (state.phase === 'build') {
    addLine('system', 'co> Build in progress. Sit tight.');
  }
}

function handleProjectPick(value) {
  const picked = value.toLowerCase();
  if (picked.includes('task')) {
    state.project = 'taskManager';
    state.phase = 'requirements';
    updateViewMode('Discovery');
    runAssistant('Starting Task Manager discovery. I will ask a few questions.', () => {
      askQuestion();
    });
    setChips([
      { label: 'List + Detail + Create/Edit' },
      { label: 'Kanban + List' },
      { label: 'Just List' }
    ]);
    return;
  }

  runAssistant('That demo is not scripted yet. Choose Task Manager to proceed.');
}

function askQuestion() {
  const project = script.taskManager;
  const question = project.questions[state.questionIndex];
  if (!question) {
    presentPlan();
    return;
  }

  setActiveFocus([question.focusNode], []);
  renderSystemMap();
  runAssistant(question.prompt);
}

function handleAnswer() {
  const project = script.taskManager;
  const question = project.questions[state.questionIndex];
  if (!question) return;

  question.onAnswer();
  setActiveFocus([question.focusNode], []);
  renderSystemMap();
  updateChipsForQuestion(state.questionIndex + 1);

  state.questionIndex += 1;

  setTimeout(() => {
    askQuestion();
  }, 250);

  
}

function updateChipsForQuestion(index) {
  const chipSets = [
    [
      { label: 'List + Detail + Create/Edit' },
      { label: 'Kanban + List' },
      { label: 'Just List' }
    ],
    [
      { label: 'Backlog → In Progress → Done' },
      { label: 'Add Blocked state' },
      { label: 'Custom workflow' }
    ],
    [
      { label: 'Team + Admin' },
      { label: 'Owner-only edits' },
      { label: 'Admin-only deletes' }
    ],
    [
      { label: 'Email + Webhook' },
      { label: 'Just email' },
      { label: 'Slack notifications' }
    ],
    [
      { label: 'Tags + Assignee + Status' },
      { label: 'Status only' },
      { label: 'Advanced filters' }
    ]
  ];

  const set = chipSets[index] || [
    { label: 'approve' },
    { label: 'revise' }
  ];
  setChips(set);
}

function presentPlan() {
  state.phase = 'plan';
  updateViewMode('Plan Review');
  setChips([
    { label: 'approve' },
    { label: 'revise' }
  ]);
  runAssistant(script.taskManager.plan.text);
}

function handlePlanResponse(value) {
  const lower = value.toLowerCase();
  if (lower.includes('approve')) {
    state.phase = 'build';
    updateViewMode('Build');
    runAssistant('Plan approved. Building now…', () => {
      runBuildStep(0);
    });
    return;
  }

  runAssistant('Revise noted. For this demo, type approve to continue.');
}

function runBuildStep(index) {
  const steps = script.taskManager.buildSteps;
  if (index >= steps.length) {
    updateViewMode('Build Complete');
    setChips([
      { label: 'Restart' },
      { label: 'Explore functions' }
    ]);
    appendBuildSummary();
    return;
  }

  const step = steps[index];
  setActiveFocus([step.focusNode], []);
  if (step.mode === 'function') {
    mapFocusComponent = 'task';
    renderComponentMap('task');
  } else {
    renderSystemMap();
  }
  

  const introLine = addLine('assistant', `co> ${step.label}…`);
  step.actions.forEach((action, idx) => {
    setTimeout(() => {
      addLine('action', action);
    }, 300 + idx * 350);
  });

  const actionsDelay = 300 + step.actions.length * 350;
  if (step.edits) {
    setTimeout(() => {
      appendEditBlocks(step.edits);
    }, actionsDelay + 150);
  }

  setTimeout(() => {
    if (step.mode === 'function') {
      renderSystemMap();
    }
    runBuildStep(index + 1);
  }, actionsDelay + (step.edits ? 900 : 500));
}

function appendBuildSummary() {
  addLine('assistant', 'co> Summary');
  addLine('action', 'Changes');
  addLine('action', '  - Added UI scaffold: TaskList, TaskDetail, create/edit drawer');
  addLine('action', '  - Defined schema: tasks, task_events');
  addLine('action', '  - Implemented Task Service transitions + audit log');
  addLine('action', '  - Wired auth roles + permissions matrix');
  addLine('action', '  - Added search indexing and notifications workers');
  addLine('assistant', 'co> Ready for iteration. Ask to refine or extend.');
}

function appendEditBlocks(edits) {
  edits.forEach((edit) => {
    addLine('action', edit.summary);
    edit.lines.forEach((line) => {
      addLine('action', `  ${line}`);
    });
  });
}

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    handleSubmit(input.value.trim());
  }
});

setChips(script.projectOptions);
renderSystemMap();
updateMapPlaceholder();

miniMap.addEventListener('click', () => {
  mapFocusComponent = null;
  currentComponentDefs = null;
  renderSystemMap();
});


window.addEventListener('resize', () => {
  if (mapMode === 'component' && mapFocusComponent) {
    renderComponentMap(mapFocusComponent);
    return;
  }
  if (mapMode === 'function') {
    renderFunctionMap();
    return;
  }
  renderSystemMap();
});

function renderMiniMap(activeId) {
  miniMap.innerHTML = '';
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.classList.add('mini-lines');
  svg.setAttribute('viewBox', '0 0 120 86');
  miniMap.appendChild(svg);

  const nodes = Object.keys(nodeDefs);
  const baseWidth = 800;
  const baseHeight = 520;
  const positions = {};

  nodes.forEach((id) => {
    if (systemState.nodes.size && !systemState.nodes.has(id)) return;
    const def = nodeDefs[id];
    if (!def) return;
    const leftPct = (def.x + 70) / baseWidth;
    const topPct = (def.y + 24) / baseHeight;
    positions[id] = {
      x: leftPct * 120,
      y: topPct * 86
    };
  });

  Object.entries(edgeDefs).forEach(([edgeId, edge]) => {
    if (!positions[edge.from] || !positions[edge.to]) return;
    const start = positions[edge.from];
    const end = positions[edge.to];
    const midX = (start.x + end.x) / 2;
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', `M${start.x} ${start.y} C${midX} ${start.y}, ${midX} ${end.y}, ${end.x} ${end.y}`);
    if (edge.from === activeId || edge.to === activeId) {
      path.classList.add('active');
    }
    svg.appendChild(path);
  });

  nodes.forEach((id) => {
    if (systemState.nodes.size && !systemState.nodes.has(id)) return;
    const def = nodeDefs[id];
    if (!def) return;
    const node = document.createElement('div');
    node.className = 'mini-node';
    if (id === activeId) node.classList.add('active');
    const pos = positions[id];
    node.style.left = `${(pos.x / 120) * 100}%`;
    node.style.top = `${(pos.y / 86) * 100}%`;
    miniMap.appendChild(node);
  });
}

let isDragging = false;

function onDrag(event) {
  if (!isDragging) return;
  const bounds = workspace.getBoundingClientRect();
  const minLeft = 280;
  const maxLeft = bounds.width - 320;
  const raw = event.clientX - bounds.left;
  const left = Math.min(Math.max(raw, minLeft), maxLeft);
  const right = bounds.width - left;
  workspace.style.gridTemplateColumns = `${left}px 6px ${right}px`;
}

function stopDrag() {
  isDragging = false;
  document.body.style.userSelect = '';
}

divider.addEventListener('pointerdown', () => {
  isDragging = true;
  document.body.style.userSelect = 'none';
});

window.addEventListener('pointermove', onDrag);
window.addEventListener('pointerup', stopDrag);
