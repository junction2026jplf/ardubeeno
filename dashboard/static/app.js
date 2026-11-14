'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const initialSource = new URLSearchParams(window.location.search).get('source');
  if (['field', 'screen_demo', 'replay'].includes(initialSource)) $('source').value = initialSource;
  const number = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
  const decimal = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
  const state = { devices: [], device: '', days: 7, history: null, status: null, selected: -1,
    solo: 'both', timer: null, failures: 0, controller: null, generation: 0, query: null };
  const count = (value) => value === null || value === undefined ? '—' : number.format(value);
  const hours = (seconds) => decimal.format((seconds || 0) / 3600) + ' h';
  const zone = () => $('timezone').value || 'UTC';
  const statusNames = { never_seen: 'Not yet received', offline: 'Device not reachable',
    recording: 'Recording', idle: 'Idle', camera_error: 'Camera fault', storage_error: 'Storage fault' };

  function timestamp(value) {
    if (!value) return '—';
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return '—';
    return new Intl.DateTimeFormat(undefined, { timeZone: zone(), month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit' }).format(date);
  }

  function notice(message, retry = false) {
    const node = $('notice');
    node.replaceChildren();
    node.hidden = !message;
    if (!message) return;
    node.append(document.createTextNode(message));
    if (retry) {
      const button = document.createElement('button');
      button.textContent = 'Try again';
      button.addEventListener('click', () => state.device ? refresh() : loadDevices());
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
      } catch { /* HTTP status remains useful if the proxy does not return JSON. */ }
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

  async function loadDevices() {
    clearTimeout(state.timer);
    notice('');
    $('connection').textContent = 'Connecting';
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
        $('connection').textContent = 'No devices';
        notice('No devices are configured on this server. Configure a device and its ingestion credential, then refresh this page.', true);
        renderEmpty('No configured devices', 'A device must be configured on the server before its history can be received.');
        $('history-panel').setAttribute('aria-busy', 'false');
        return;
      }
      $('device').disabled = false;
      state.device = state.devices.some((device) => device.id === state.device) ? state.device : state.devices[0].id;
      $('device').value = state.device;
      configureTimezone(state.devices.find((device) => device.id === state.device));
      await refresh();
    } catch (error) {
      $('connection').textContent = 'Server unavailable';
      $('connection').classList.add('stale');
      $('device').replaceChildren(new Option('Could not load devices', ''));
      notice(error.message + '. Unable to load configured devices.', true);
      renderEmpty('Cannot reach the server', 'Check the dashboard connection and try again. No sample data is substituted.');
      $('history-panel').setAttribute('aria-busy', 'false');
      state.timer = setTimeout(loadDevices, 30000);
    }
  }

  function buildQuery() {
    const end = new Date();
    const start = new Date(end.getTime() - state.days * 86400000);
    return new URLSearchParams({ from: start.toISOString(), to: end.toISOString(), tz: zone(),
      bucket: state.days === 1 ? 'hour' : 'day', source: $('source').value });
  }

  function schedule() {
    clearTimeout(state.timer);
    if (!document.hidden && state.device) {
      state.timer = setTimeout(refresh, Math.min(30000 * 2 ** state.failures, 300000));
    }
  }

  async function refresh() {
    if (!state.device || document.hidden) return;
    clearTimeout(state.timer);
    state.controller?.abort();
    state.controller = new AbortController();
    const generation = ++state.generation;
    const query = buildQuery();
    const path = '/api/v1/devices/' + encodeURIComponent(state.device);
    $('history-panel').setAttribute('aria-busy', 'true');
    if (!state.history) $('period').textContent = 'Loading history…';
    try {
      const [history, status] = await Promise.all([
        response(path + '/history?' + query, state.controller.signal).then((r) => r.json()),
        response(path + '/status', state.controller.signal).then((r) => r.json())
      ]);
      if (generation !== state.generation) return;
      if (!history.summary || !Array.isArray(history.buckets)) throw new Error('The server returned an invalid history response');
      state.history = history;
      state.status = status;
      state.query = query;
      state.failures = 0;
      if (state.selected < 0 || state.selected >= history.buckets.length) state.selected = history.buckets.length - 1;
      render();
      const hasUncoveredEvents = history.summary.events_outside_coverage > 0 || history.buckets.some((b) => b.events_outside_coverage > 0 || b.availability === 'events_only' || (!b.recorded_seconds && b.bees_observed !== null && b.bees_observed > 0));
      if (hasUncoveredEvents) notice('Some bee passages arrived without recording coverage. Their counts are retained, but coverage and the hourly rate may be incomplete.');
      else if (status.stale || status.status === 'offline') notice('The device has not sent a recent heartbeat. Showing saved history; a recent upload may contain older observations.');
      else if (['camera_error', 'storage_error'].includes(status.status)) notice(statusNames[status.status] + '. Saved history remains available; current recording may be interrupted.');
      else notice('');
    } catch (error) {
      if (error.name === 'AbortError' || generation !== state.generation) return;
      state.failures += 1;
      $('connection').textContent = state.history ? 'Saved view · stale' : 'Server unavailable';
      $('connection').classList.add('stale');
      notice(error.message + (state.history ? '. Showing the last successful result; it is now stale.' : '. No history could be loaded.'), true);
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
    cell.colSpan = 5;
    cell.textContent = 'No recording history';
    row.append(cell);
    $('history-table').append(row);
  }

  function renderStatus() {
    const status = state.status || {};
    const stale = status.stale || status.status === 'offline';
    $('connection').textContent = stale ? 'Device offline · saved history' : statusNames[status.status] || 'History received';
    $('connection').classList.toggle('stale', Boolean(stale || ['camera_error', 'storage_error'].includes(status.status)));
    $('device-status').textContent = statusNames[status.status] || status.status || 'Unknown';
    $('last-received').textContent = timestamp(status.last_received_at);
    $('last-event').textContent = timestamp(status.last_occurred_at || status.last_seen);
    $('pending').textContent = status.pending_events === null || status.pending_events === undefined ? 'Unknown' : count(status.pending_events) + ' events';
    $('last-updated').textContent = timestamp(state.history?.generated_at);
  }

  function render() {
    const { summary, buckets } = state.history;
    $('total-bees').textContent = count(summary.bees_observed);
    $('total-mites').textContent = count(summary.bees_with_detected_mites);
    const uncovered = summary.rate_available === false || buckets.some((b) => b.events_outside_coverage > 0 || b.availability === 'events_only' || (!b.recorded_seconds && (b.bees_observed || 0) > 0));
    $('rate').textContent = summary.recorded_seconds > 0 && summary.bees_observed !== null && !uncovered ? decimal.format(summary.bees_observed / summary.recorded_seconds * 3600) : '—';
    $('coverage').textContent = summary.requested_seconds > 0 ? decimal.format(summary.recorded_seconds / summary.requested_seconds * 100) + '%' : '—';
    $('coverage-detail').textContent = hours(summary.recorded_seconds) + ' of ' + hours(summary.requested_seconds);
    $('period').textContent = timestamp(state.query.get('from')) + ' – ' + timestamp(state.query.get('to')) + ' · ' + zone();
    $('chart-title').textContent = (state.days === 1 ? 'Hourly ' : 'Daily ') + (state.solo === 'mites' ? 'bees with detected mites' : 'bees observed');
    $('chart-unit').textContent = 'Bees / ' + (state.days === 1 ? 'hour' : 'day');
    $('other-legend').hidden = state.solo === 'mites';
    $('mite-legend').hidden = state.solo === 'bees';
    $('other-label').textContent = state.solo === 'bees' ? 'All bees observed' : 'Other observed bees';
    $('export').disabled = false;
    renderStatus();
    if (!buckets.length || !buckets.some((b) => b.bees_observed !== null)) {
      renderEmpty('No recording history in this period', 'Choose another time range or source. The device may still have observations waiting to upload.');
      // An empty period can still be exported with its explicit coverage gaps.
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
    return new Intl.DateTimeFormat(undefined, state.days === 1 ? { timeZone: zone(), hour: '2-digit', minute: '2-digit', ...(short ? {} : { day: 'numeric', month: 'short' }) } : { timeZone: zone(), day: 'numeric', month: 'short' }).format(date);
  }

  function niceMaximum(value) {
    if (value <= 4) return 4;
    const scale = 10 ** Math.floor(Math.log10(value));
    return Math.ceil(value / scale / 0.5) * scale * 0.5;
  }

  function describe(bucket) {
    return bucketLabel(bucket) + ': ' + (bucket.bees_observed === null ? 'no observations or recording coverage' : count(bucket.bees_observed) + ' observed bees, ' + count(bucket.bees_with_detected_mites) + ' with detected mites, ' + count(bucket.bees_assessment_uncertain) + ' uncertain assessments') + '. Recorded ' + hours(bucket.recorded_seconds) + ' of ' + hours(bucket.requested_seconds) + '.';
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
        if (bucket.recorded_seconds < bucket.requested_seconds) layers.push('repeating-linear-gradient(135deg,transparent,transparent 5px,var(--card) 5px,var(--card) 7px)');
        if (state.solo === 'both' && bucket.bees_observed > 0) {
          const red = Math.max(0, Math.min(100, bucket.bees_with_detected_mites / bucket.bees_observed * 100));
          layers.push('linear-gradient(to bottom,var(--red) 0%,var(--red) ' + red + '%,var(--green) ' + red + '%,var(--green) 100%)');
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
    $('inspection').replaceChildren();
    const primary = document.createElement('strong');
    primary.textContent = bucketLabel(bucket) + ' · ' + (bucket.bees_observed === null ? 'No recording' : count(bucket.bees_observed) + ' bees · ' + count(bucket.bees_with_detected_mites) + ' with detected mites');
    const secondary = document.createElement('span');
    secondary.className = 'muted';
    secondary.textContent = hours(bucket.recorded_seconds) + ' / ' + hours(bucket.requested_seconds) + ' recorded' + (bucket.events_outside_coverage > 0 || bucket.availability === 'events_only' || (!bucket.recorded_seconds && bucket.bees_observed > 0) ? ' · coverage unavailable' : bucket.recorded_seconds < bucket.requested_seconds ? ' · partial coverage' : ' · complete') + (bucket.bees_assessment_uncertain ? ' · ' + count(bucket.bees_assessment_uncertain) + ' uncertain' : '');
    $('inspection').append(primary, secondary);
  }

  function renderTable() {
    $('history-table').replaceChildren();
    [...state.history.buckets].reverse().forEach((bucket) => {
      const row = document.createElement('tr');
      [bucketLabel(bucket), count(bucket.bees_observed), count(bucket.bees_with_detected_mites), count(bucket.bees_assessment_uncertain), hours(bucket.recorded_seconds) + ' / ' + hours(bucket.requested_seconds)].forEach((text) => {
        const cell = document.createElement('td');
        cell.textContent = text;
        row.append(cell);
      });
      $('history-table').append(row);
    });
  }

  function changeFilter() {
    state.history = null;
    state.status = null;
    state.query = null;
    state.selected = -1;
    ['total-bees', 'total-mites', 'rate', 'coverage'].forEach((id) => $(id).textContent = '—');
    $('coverage-detail').textContent = 'Loading selected period';
    ['device-status', 'last-received', 'last-event', 'pending', 'last-updated'].forEach((id) => $(id).textContent = '—');
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
  document.querySelectorAll('[data-days]').forEach((button) => button.addEventListener('click', () => {
    state.days = Number(button.dataset.days);
    document.querySelectorAll('[data-days]').forEach((other) => other.setAttribute('aria-pressed', String(other === button)));
    changeFilter();
  }));
  $('solo').addEventListener('change', () => {
    state.solo = $('solo').value;
    if (state.history) render();
  });
  $('export').addEventListener('click', async () => {
    if (!state.query || !state.device) return;
    $('export').disabled = true;
    try {
      const result = await response('/api/v1/devices/' + encodeURIComponent(state.device) + '/export.csv?' + state.query);
      const objectUrl = URL.createObjectURL(await result.blob());
      const link = document.createElement('a');
      link.href = objectUrl;
      link.download = 'ardubeeno-' + state.device.replace(/[^a-zA-Z0-9_-]/g, '_') + '-' + state.days + 'd.csv';
      link.click();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch (error) { notice('CSV export failed: ' + error.message, true); }
    finally { $('export').disabled = !state.history; }
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      clearTimeout(state.timer);
      state.controller?.abort();
    } else if (state.device) refresh();
    else loadDevices();
  });
  loadDevices();
})();
