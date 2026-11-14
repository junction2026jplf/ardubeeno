'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(window.location.search);
  const initialSource = params.get('source');
  const initialDevice = params.get('device');
  const initialRange = params.get('range');

  const RANGES = {
    '1h': { ms: 3600000, bucket: '5m', pollMs: 5000, title: '5-minute ', unit: '5 min', csv: '1h' },
    '24h': { ms: 86400000, bucket: 'hour', pollMs: 15000, title: 'Hourly ', unit: 'hour', csv: '24h' },
    '7d': { ms: 7 * 86400000, bucket: 'day', pollMs: 15000, title: 'Daily ', unit: 'day', csv: '7d' },
    '30d': { ms: 30 * 86400000, bucket: 'day', pollMs: 20000, title: 'Daily ', unit: 'day', csv: '30d' },
  };
  const SOURCE_LABELS = {
    field: 'Field observations',
    screen_demo: 'Screen demonstration',
    replay: 'Recorded replay',
  };
  const statusNames = {
    never_seen: 'Not yet received',
    offline: 'Device not reachable',
    recording: 'Recording',
    idle: 'Idle',
    camera_error: 'Camera fault',
    storage_error: 'Storage fault',
  };

  if (['field', 'screen_demo', 'replay'].includes(initialSource)) $('source').value = initialSource;
  const startRange = initialRange === 'live' ? '1h' : (RANGES[initialRange] ? initialRange : '7d');

  const number = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
  const decimal = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
  const state = {
    devices: [],
    device: initialDevice || '',
    range: startRange,
    history: null,
    status: null,
    selected: -1,
    solo: 'both',
    timer: null,
    liveDebounce: null,
    pulseTimer: null,
    stream: null,
    streamConnected: false,
    failures: 0,
    controller: null,
    generation: 0,
    query: null,
  };

  // Sync range button pressed states on startup
  document.querySelectorAll('[data-range]').forEach((btn) => {
    btn.setAttribute('aria-pressed', String(btn.dataset.range === state.range));
  });

  // Shared theme toggle (matches applab/ardubeeno/assets/index.html)
  const storage = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
  };
  function applyTheme(theme) {
    if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
  }
  applyTheme(storage.get('ardubeeno.theme'));
  $('theme-btn').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === 'dark'
      : window.matchMedia('(prefers-color-scheme: dark)').matches;
    const next = dark ? 'light' : 'dark';
    applyTheme(next);
    storage.set('ardubeeno.theme', next);
  });

  const count = (value) => value === null || value === undefined ? '—' : number.format(value);
  const durationText = (seconds) => {
    const s = seconds || 0;
    if (state.range === '1h') return decimal.format(s / 60) + ' min';
    return decimal.format(s / 3600) + ' h';
  };
  const zone = () => $('timezone').value || 'UTC';

  function timestamp(value, withSeconds = false) {
    if (!value) return '—';
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return '—';
    return new Intl.DateTimeFormat(undefined, {
      timeZone: zone(),
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      ...(withSeconds ? { second: '2-digit' } : {}),
    }).format(date);
  }

  function notice(message, actionLabel = '', onAction = null) {
    const node = $('notice');
    node.replaceChildren();
    node.hidden = !message;
    if (!message) return;
    const span = document.createElement('span');
    span.textContent = message;
    node.append(span);
    if (actionLabel && onAction) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = actionLabel;
      button.addEventListener('click', onAction);
      node.append(button);
    }
  }

  async function response(url, signal) {
    const result = await fetch(url, { signal, cache: 'no-store', credentials: 'same-origin' });
    if (!result.ok) {
      let message = 'Request failed (' + result.status + ')';
      try {
        const body = await result.json();
        if (typeof body.detail === 'string') message += ': ' + body.detail;
      } catch { /* HTTP status remains useful if proxy does not return JSON */ }
      throw new Error(message);
    }
    return result;
  }

  function configureTimezone(device) {
    const current = device.timezone || 'UTC';
    const browser = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const zones = [...new Set([current, browser, 'UTC'].filter(Boolean))];
    $('timezone').replaceChildren();
    zones.forEach((value) => {
      try { new Intl.DateTimeFormat(undefined, { timeZone: value }); }
      catch { return; }
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      $('timezone').append(option);
    });
    $('timezone').value = zones.includes(current) ? current : 'UTC';
  }

  function flashPulse() {
    const badge = $('connection');
    badge.classList.add('pulse');
    clearTimeout(state.pulseTimer);
    state.pulseTimer = setTimeout(() => badge.classList.remove('pulse'), 900);
  }

  function connectStream() {
    if (state.stream || document.hidden || typeof EventSource === 'undefined') return;
    try {
      const es = new EventSource('/api/v1/stream');
      state.stream = es;
      es.addEventListener('connected', () => {
        state.streamConnected = true;
        renderStatus();
      });
      es.addEventListener('ingest', (ev) => {
        state.streamConnected = true;
        let payload = null;
        try { payload = JSON.parse(ev.data); } catch { /* ignore malformed frame */ }
        if (payload && state.device && payload.device_id !== state.device) return;
        flashPulse();
        clearTimeout(state.liveDebounce);
        state.liveDebounce = setTimeout(() => {
          if (state.device) refresh(true);
          else loadDevices();
        }, 180);
      });
      es.onerror = () => {
        state.streamConnected = false;
        renderStatus();
      };
    } catch {
      state.streamConnected = false;
    }
  }

  function disconnectStream() {
    if (state.stream) {
      state.stream.close();
      state.stream = null;
    }
    state.streamConnected = false;
  }

  async function loadDevices() {
    clearTimeout(state.timer);
    notice('');
    $('connection-text').textContent = 'Connecting';
    try {
      const body = await (await response('/api/v1/devices')).json();
      state.devices = body.devices || [];
      $('device').replaceChildren();
      state.devices.forEach((device) => {
        const option = document.createElement('option');
        option.value = device.id;
        option.textContent = device.name || device.id;
        $('device').append(option);
      });
      if (!state.devices.length) {
        state.device = '';
        $('device').append(new Option('No configured devices', ''));
        $('device').disabled = true;
        $('connection-text').textContent = 'No devices';
        notice('No devices are configured on this server. Configure a device and its ingestion credential, then refresh.', 'Try again', loadDevices);
        renderEmpty('No configured devices', 'A device must be configured on the server before its history can be received.');
        $('history-panel').setAttribute('aria-busy', 'false');
        return;
      }
      $('device').disabled = false;
      state.device = state.devices.some((device) => device.id === state.device) ? state.device : state.devices[0].id;
      $('device').value = state.device;
      configureTimezone(state.devices.find((device) => device.id === state.device));
      connectStream();
      await refresh();
    } catch (error) {
      $('connection-text').textContent = 'Server unavailable';
      $('connection').className = 'badge stale';
      $('device').replaceChildren(new Option('Could not load devices', ''));
      notice(error.message + '. Unable to load configured devices.', 'Try again', loadDevices);
      renderEmpty('Cannot reach the server', 'Check the dashboard connection and try again. No sample data is substituted.');
      $('history-panel').setAttribute('aria-busy', 'false');
      state.timer = setTimeout(loadDevices, 15000);
    }
  }

  function buildQuery() {
    const cfg = RANGES[state.range] || RANGES['7d'];
    const end = new Date();
    const start = new Date(end.getTime() - cfg.ms);
    return new URLSearchParams({
      from: start.toISOString(),
      to: end.toISOString(),
      tz: zone(),
      bucket: cfg.bucket,
      source: $('source').value,
    });
  }

  function schedule() {
    clearTimeout(state.timer);
    if (!document.hidden && state.device) {
      const base = (RANGES[state.range] || RANGES['7d']).pollMs;
      state.timer = setTimeout(() => refresh(true), Math.min(base * 2 ** state.failures, 300000));
    }
  }

  function otherSourceHint(status, historyEmpty) {
    if (!historyEmpty || !status || !status.sources) return null;
    const current = $('source').value;
    if (status.sources[current]) return null;
    const preferred = status.active_source && status.sources[status.active_source]
      ? status.active_source
      : Object.keys(status.sources)[0];
    return preferred && preferred !== current ? preferred : null;
  }

  async function refresh(silent = false) {
    if (!state.device || document.hidden) return;
    clearTimeout(state.timer);
    state.controller?.abort();
    state.controller = new AbortController();
    const generation = ++state.generation;
    const query = buildQuery();
    const path = '/api/v1/devices/' + encodeURIComponent(state.device);
    if (!silent) $('history-panel').setAttribute('aria-busy', 'true');
    if (!state.history) $('period').textContent = 'Loading history…';
    try {
      const [history, status] = await Promise.all([
        response(path + '/history?' + query, state.controller.signal).then((r) => r.json()),
        response(path + '/status', state.controller.signal).then((r) => r.json()),
      ]);
      if (generation !== state.generation) return;
      if (!history.summary || !Array.isArray(history.buckets)) throw new Error('The server returned an invalid history response');
      const wasAtLatest = state.selected < 0 || (state.history && state.selected >= state.history.buckets.length - 1);
      state.history = history;
      state.status = status;
      state.query = query;
      state.failures = 0;
      if (wasAtLatest || state.selected >= history.buckets.length) state.selected = history.buckets.length - 1;
      render();

      const emptyPeriod = !history.buckets.length || !history.buckets.some((b) => b.bees_observed !== null);
      const altSource = otherSourceHint(status, emptyPeriod);
      const hasUncoveredEvents = history.summary.events_outside_coverage > 0 || history.buckets.some(
        (b) => b.events_outside_coverage > 0 || b.availability === 'events_only' || (!b.recorded_seconds && b.bees_observed !== null && b.bees_observed > 0)
      );

      if (altSource) {
        const label = SOURCE_LABELS[altSource] || altSource;
        notice(
          `No ${SOURCE_LABELS[$('source').value].toLowerCase()} in this window, but this device has recorded ${label.toLowerCase()}.`,
          `Switch to ${label}`,
          () => { $('source').value = altSource; changeFilter(); }
        );
      } else if (hasUncoveredEvents) {
        notice('Some bee passages arrived without recording coverage. Their counts are retained, but coverage and passages/hour may be incomplete.');
      } else if (status.stale || status.status === 'offline') {
        notice('The device has not sent a recent heartbeat. Showing saved history; buffered events will appear automatically when it reconnects.');
      } else if (['camera_error', 'storage_error'].includes(status.status)) {
        notice(statusNames[status.status] + '. Saved history remains available; current recording may be interrupted.');
      } else {
        notice('');
      }
    } catch (error) {
      if (error.name === 'AbortError' || generation !== state.generation) return;
      state.failures += 1;
      $('connection-text').textContent = state.history ? 'Saved view · stale' : 'Server unavailable';
      $('connection').className = 'badge stale';
      notice(error.message + (state.history ? '. Showing the last successful result; it is now stale.' : '. No history could be loaded.'), 'Try again', () => refresh());
      if (!state.history) renderEmpty('Unable to load history', 'Check the connection and try again.');
    } finally {
      if (generation === state.generation) {
        $('history-panel').setAttribute('aria-busy', 'false');
        schedule();
      }
    }
  }

  function renderEmpty(title, message) {
    $('empty').hidden = false;
    $('empty').querySelector('h3').textContent = title;
    $('empty').querySelector('p').textContent = message;
    $('chart-scroll').hidden = true;
    $('inspection').textContent = 'No period selected.';
    $('export').disabled = true;
    $('history-table').replaceChildren();
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 6;
    cell.textContent = 'No recording history';
    row.append(cell);
    $('history-table').append(row);
  }

  function renderStatus() {
    const status = state.status || {};
    const stale = status.stale || status.status === 'offline' || status.status === 'never_seen';
    const fault = ['camera_error', 'storage_error'].includes(status.status);
    const livePrefix = state.streamConnected ? 'Live · ' : '';
    const badge = $('connection');

    if (fault) {
      badge.className = 'badge fault';
      $('connection-text').textContent = statusNames[status.status];
    } else if (stale) {
      badge.className = 'badge stale';
      $('connection-text').textContent = status.status === 'never_seen'
        ? (state.streamConnected ? 'Live · Waiting for device' : 'Waiting for device')
        : 'Device offline · saved history';
    } else {
      badge.className = 'badge live';
      $('connection-text').textContent = livePrefix + (statusNames[status.status] || 'Connected');
    }

    $('device-status').textContent = statusNames[status.status] || status.status || 'Unknown';
    $('stream-status').textContent = state.streamConnected ? 'Connected (instant push)' : 'Polling fallback';
    $('active-source').textContent = status.active_source ? (SOURCE_LABELS[status.active_source] || status.active_source) : '—';
    $('last-received').textContent = timestamp(status.last_received_at, state.range === '1h');
    $('last-event').textContent = timestamp(status.last_occurred_at || status.last_seen, state.range === '1h');
    $('pending').textContent = status.pending_events === null || status.pending_events === undefined
      ? 'Unknown'
      : count(status.pending_events) + ' events';
    $('last-updated').textContent = timestamp(state.history?.generated_at, true);

    const liveLink = $('live-link');
    if (status.live_url) {
      liveLink.href = status.live_url;
      liveLink.hidden = false;
    } else {
      liveLink.hidden = true;
    }
  }

  function notDetectedCount(item) {
    if (item.bees_observed === null || item.bees_observed === undefined) return null;
    if (item.bees_not_detected !== null && item.bees_not_detected !== undefined) return item.bees_not_detected;
    return Math.max(0, item.bees_observed - (item.bees_with_detected_mites || 0) - (item.bees_assessment_uncertain || 0));
  }

  function render() {
    const { summary, buckets } = state.history;
    const cfg = RANGES[state.range] || RANGES['7d'];
    const clean = notDetectedCount(summary);

    $('total-bees').textContent = count(summary.bees_observed);
    $('bees-detail').textContent = summary.bees_observed !== null
      ? `${count(clean)} no mite · ${count(summary.bees_assessment_uncertain)} uncertain`
      : 'All completed bee passages';
    $('total-mites').textContent = count(summary.bees_with_detected_mites);
    $('mite-stat').classList.toggle('has-mites', (summary.bees_with_detected_mites || 0) > 0);

    const uncovered = summary.rate_available === false || buckets.some(
      (b) => b.events_outside_coverage > 0 || b.availability === 'events_only' || (!b.recorded_seconds && (b.bees_observed || 0) > 0)
    );
    $('rate').textContent = summary.recorded_seconds > 0 && summary.bees_observed !== null && !uncovered
      ? decimal.format(summary.bees_observed / summary.recorded_seconds * 3600)
      : '—';
    $('coverage').textContent = summary.requested_seconds > 0
      ? decimal.format(summary.recorded_seconds / summary.requested_seconds * 100) + '%'
      : '—';
    $('coverage-detail').textContent = durationText(summary.recorded_seconds) + ' of ' + durationText(summary.requested_seconds);
    $('period').textContent = timestamp(state.query.get('from')) + ' – ' + timestamp(state.query.get('to')) + ' · ' + zone();
    $('chart-title').textContent = cfg.title + (state.solo === 'mites' ? 'bees with mites' : 'bees observed');
    $('chart-unit').textContent = 'Bees / ' + cfg.unit;
    $('clean-legend').hidden = state.solo === 'mites';
    $('uncertain-legend').hidden = state.solo !== 'both';
    $('mite-legend').hidden = state.solo === 'bees';
    $('clean-label').textContent = state.solo === 'bees' ? 'All bees observed' : 'No mite detected (2+ checks)';
    $('export').disabled = false;
    renderStatus();

    if (!buckets.length || !buckets.some((b) => b.bees_observed !== null)) {
      renderEmpty('No recording history in this period', 'Choose another time range or source, or leave this page open—live uploads appear automatically.');
      $('export').disabled = false;
      return;
    }
    $('empty').hidden = true;
    $('chart-scroll').hidden = false;
    renderChart();
    renderTable();
    inspect();
  }

  function barValue(bucket) {
    return state.solo === 'mites' ? bucket.bees_with_detected_mites : bucket.bees_observed;
  }

  function bucketLabel(bucket, short = false) {
    const date = new Date(bucket.start);
    if (!Number.isFinite(date.getTime())) return bucket.label || '';
    if (state.range === '1h' || state.range === '24h') {
      return new Intl.DateTimeFormat(undefined, {
        timeZone: zone(),
        hour: '2-digit',
        minute: '2-digit',
        ...(short ? {} : { day: 'numeric', month: 'short' }),
      }).format(date);
    }
    return new Intl.DateTimeFormat(undefined, { timeZone: zone(), day: 'numeric', month: 'short' }).format(date);
  }

  function niceMaximum(value) {
    if (value <= 4) return 4;
    const scale = 10 ** Math.floor(Math.log10(value));
    return Math.ceil(value / scale / 0.5) * scale * 0.5;
  }

  function describe(bucket) {
    const clean = notDetectedCount(bucket);
    return bucketLabel(bucket) + ': ' + (bucket.bees_observed === null
      ? 'no observations or recording coverage'
      : `${count(bucket.bees_observed)} bees observed (${count(clean)} no mite, ${count(bucket.bees_assessment_uncertain)} uncertain, ${count(bucket.bees_with_detected_mites)} with mites)`) +
      `. Recorded ${durationText(bucket.recorded_seconds)} of ${durationText(bucket.requested_seconds)}.`;
  }

  function renderChart() {
    const buckets = state.history.buckets;
    const maximum = niceMaximum(Math.max(0, ...buckets.map((b) => barValue(b) || 0)));
    $('grid').replaceChildren();
    [maximum, maximum / 2, 0].forEach((value) => {
      const line = document.createElement('div');
      line.className = 'gridline';
      const label = document.createElement('span');
      label.textContent = number.format(value);
      line.append(label);
      $('grid').append(line);
    });
    $('plot').classList.toggle('dense', buckets.length > 10);
    $('bars').replaceChildren();
    buckets.forEach((bucket, index) => {
      const slot = document.createElement('div');
      slot.className = 'bar-slot';
      const bar = document.createElement('button');
      bar.type = 'button';
      const value = barValue(bucket);
      const missing = value === null;
      bar.className = 'bar' + (missing ? ' gap' : value === 0 ? ' zero' : '');
      bar.style.height = (value || 0) / maximum * 100 + '%';
      bar.setAttribute('aria-label', describe(bucket));
      bar.setAttribute('aria-pressed', String(index === state.selected));
      if (missing) bar.textContent = buckets.length > 10 ? '—' : 'No recording';
      else {
        const fill = state.solo === 'mites' ? 'var(--red)' : 'var(--green)';
        bar.style.backgroundColor = fill;
        const layers = [];
        if (bucket.recorded_seconds < bucket.requested_seconds) {
          layers.push('repeating-linear-gradient(135deg,transparent,transparent 5px,var(--card) 5px,var(--card) 7px)');
        }
        if (state.solo === 'both' && bucket.bees_observed > 0) {
          const redPct = Math.max(0, Math.min(100, (bucket.bees_with_detected_mites || 0) / bucket.bees_observed * 100));
          const ambEnd = Math.max(redPct, Math.min(100, ((bucket.bees_with_detected_mites || 0) + (bucket.bees_assessment_uncertain || 0)) / bucket.bees_observed * 100));
          layers.push(
            `linear-gradient(to bottom,var(--red) 0%,var(--red) ${redPct}%,var(--amber) ${redPct}%,var(--amber) ${ambEnd}%,var(--green) ${ambEnd}%,var(--green) 100%)`
          );
        }
        bar.style.backgroundImage = layers.length ? layers.join(',') : 'none';
      }
      bar.addEventListener('click', () => selectBucket(index));
      bar.addEventListener('keydown', (event) => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? buckets.length - 1 : Math.max(0, Math.min(buckets.length - 1, index + (event.key === 'ArrowRight' ? 1 : -1)));
        selectBucket(next);
        $('bars').querySelectorAll('button')[next].focus();
      });
      slot.append(bar);
      const tickEvery = Math.max(1, Math.ceil(buckets.length / 7));
      if (index % tickEvery === 0 || index === buckets.length - 1) {
        const tick = document.createElement('span');
        tick.className = 'tick';
        tick.textContent = bucketLabel(bucket, true);
        slot.append(tick);
      }
      $('bars').append(slot);
    });
  }

  function selectBucket(index) {
    state.selected = index;
    $('bars').querySelectorAll('button').forEach((bar, i) => bar.setAttribute('aria-pressed', String(i === index)));
    inspect();
  }

  function inspect() {
    const bucket = state.history?.buckets[state.selected];
    if (!bucket) return;
    const clean = notDetectedCount(bucket);
    $('inspection').replaceChildren();
    const primary = document.createElement('strong');
    primary.textContent = bucketLabel(bucket) + ' · ' + (bucket.bees_observed === null
      ? 'No recording'
      : `${count(bucket.bees_observed)} observed · ${count(clean)} no mite · ${count(bucket.bees_assessment_uncertain)} uncertain · ${count(bucket.bees_with_detected_mites)} with mites`);
    const secondary = document.createElement('span');
    secondary.className = 'muted';
    secondary.textContent = durationText(bucket.recorded_seconds) + ' / ' + durationText(bucket.requested_seconds) + ' recorded' +
      (bucket.events_outside_coverage > 0 || bucket.availability === 'events_only' || (!bucket.recorded_seconds && bucket.bees_observed > 0)
        ? ' · coverage unavailable'
        : bucket.recorded_seconds < bucket.requested_seconds ? ' · partial coverage' : ' · complete');
    $('inspection').append(primary, secondary);
  }

  function renderTable() {
    $('history-table').replaceChildren();
    [...state.history.buckets].reverse().forEach((bucket) => {
      const row = document.createElement('tr');
      const clean = notDetectedCount(bucket);
      const cells = [
        bucketLabel(bucket),
        count(bucket.bees_observed),
        count(clean),
        count(bucket.bees_assessment_uncertain),
        count(bucket.bees_with_detected_mites),
        durationText(bucket.recorded_seconds) + ' / ' + durationText(bucket.requested_seconds),
      ];
      cells.forEach((text, idx) => {
        const cell = document.createElement('td');
        cell.textContent = text;
        if (idx === 4 && (bucket.bees_with_detected_mites || 0) > 0) cell.className = 'mite-col';
        row.append(cell);
      });
      $('history-table').append(row);
    });
  }

  function changeFilter() {
    state.history = null;
    state.query = null;
    state.selected = -1;
    ['total-bees', 'total-mites', 'rate', 'coverage'].forEach((id) => { $(id).textContent = '—'; });
    $('bees-detail').textContent = 'Loading selected period';
    $('coverage-detail').textContent = 'Loading selected period';
    renderEmpty('Loading history…', 'Retrieving observations for the selected filters.');
    refresh();
  }

  $('device').addEventListener('change', () => {
    state.device = $('device').value;
    configureTimezone(state.devices.find((device) => device.id === state.device));
    changeFilter();
  });
  $('source').addEventListener('change', changeFilter);
  $('timezone').addEventListener('change', changeFilter);
  document.querySelectorAll('[data-range]').forEach((button) => button.addEventListener('click', () => {
    state.range = button.dataset.range;
    document.querySelectorAll('[data-range]').forEach((other) => other.setAttribute('aria-pressed', String(other === button)));
    changeFilter();
  }));
  $('solo').addEventListener('change', () => {
    state.solo = $('solo').value;
    if (state.history) render();
  });
  $('export').addEventListener('click', async () => {
    if (!state.query || !state.device) return;
    $('export').disabled = true;
    const cfg = RANGES[state.range] || RANGES['7d'];
    try {
      const result = await response('/api/v1/devices/' + encodeURIComponent(state.device) + '/export.csv?' + state.query);
      const objectUrl = URL.createObjectURL(await result.blob());
      const link = document.createElement('a');
      link.href = objectUrl;
      link.download = 'ardubeeno-' + state.device.replace(/[^a-zA-Z0-9_-]/g, '_') + '-' + cfg.csv + '.csv';
      link.click();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch (error) { notice('CSV export failed: ' + error.message, 'Try again', () => refresh()); }
    finally { $('export').disabled = !state.history; }
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      clearTimeout(state.timer);
      clearTimeout(state.liveDebounce);
      state.controller?.abort();
      disconnectStream();
    } else if (state.device) {
      connectStream();
      refresh(true);
    } else {
      loadDevices();
    }
  });
  loadDevices();
})();
