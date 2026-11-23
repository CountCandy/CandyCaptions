// Candy Factory (Run): the caption queue, with live progress.

(function () {
    const panel = $('#panel-run');
    const FLAVOR_MESSAGES = [
        'Melting the pixels into a smooth syrup…',
        'Sorting the sprinkles by colour…',
        'Pulling the taffy on a long description…',
        'Tasting for sour notes…',
        'Pressing the caption into its candy mould…',
        'Putting a cherry on top…',
    ];
    const STAGES = { look: 'looking at the image', compose: 'writing the caption', check: 'Taste Check (field checks)' };
    let flavorTimer = null, flavorIdx = 0, crustWidth = 0;
    let misshapes = [];
    const cards = {};   // image id -> { el, list, stage }

    const status = () => App.run || { state: 'idle', total: 0, done: 0, failed: 0, skipped: 0, in_flight: [], recent: [], message: '' };
    const busy = () => ['running', 'pausing', 'paused', 'stopping'].includes(status().state);

    async function control(action, button, body) {
        await act(button, async () => {
            if (action === 'start') await flushDatasetSettings();
            App.run = await api(`/api/run/${action}`, 'POST', body || {});
            update();
        });
    }

    function setFlavor(message) {
        const el = $('#runFlavor');
        if (!el || el.textContent === message) return;
        el.classList.add('swap');
        setTimeout(() => { el.textContent = message; el.classList.remove('swap'); }, 300);
    }

    function rotateFlavor() {
        const s = status();
        if (s.state !== 'running') return;
        flavorIdx = (flavorIdx + 1) % (FLAVOR_MESSAGES.length + 1);
        const finished = s.done + s.failed + s.skipped;
        setFlavor(flavorIdx === 0 ? `Unwrapping image ${Math.min(finished + 1, s.total)} of ${s.total}…` : FLAVOR_MESSAGES[flavorIdx - 1]);
    }

    function buildCrust() {
        const track = $('#runTrack'), reveal = $('#runReveal');
        if (!track || !reveal) return;
        const width = Math.max(1, Math.round(track.clientWidth));
        if (width === crustWidth && reveal.firstChild) return;
        crustWidth = width;
        reveal.innerHTML = buildCrustSVG(width, 42);
    }

    function fieldList(text) {
        const names = {};
        const prefill = App.prefills.find((p) => p.id === App.dataset.settings.prefill_id);
        (prefill ? prefill.fields : []).forEach((f) => { names[f.key] = f.name; });
        const fields = partialFields(text);
        return Object.entries(fields).map(([key, value]) => [h('dt', null, names[key] || key), h('dd', { title: value }, value || '…')]);
    }

    function flightCard(item) {
        let card = cards[item.id];
        if (!card) {
            const list = h('dl');
            const stage = h('span', { class: 'hint' });
            const el = h('div', { class: 'flight-card' }, h('img', { src: thumbUrl(item.id) }), h('div', { class: 'body' }, h('b', { title: item.rel }, item.rel), stage, list));
            card = cards[item.id] = { el, list, stage };
        }
        card.stage.textContent = STAGES[item.stage] || '';
        if (item.text) card.list.replaceChildren(...fieldList(item.text).flat());
        return card.el;
    }

    async function loadMisshapes() {
        if (!App.dataset.open) return;
        try { misshapes = (await api('/api/review?q=' + encodeURIComponent('status:failed'))).rows; } catch (err) { misshapes = []; }
        const slot = $('#runMisshapes');
        if (slot) slot.replaceChildren(...misshapeNodes());
    }

    function misshapeNodes() {
        if (!misshapes.length) return [];
        return [h('div', { class: 'card' },
            h('div', { class: 'spread' },
                h('h2', null, candy('Misshapes', `retry list · ${misshapes.length}`)),
                h('button', { class: 'quiet small', disabled: busy(), onclick: (e) => control('start', e.target, { mode: 'retry' }) }, 'Retry them all')),
            h('table', { style: 'margin-top:8px' }, misshapes.map((row) => h('tr', null,
                h('td', { style: 'width:56px' }, h('img', { src: thumbUrl(row.id), style: 'width:44px; height:44px; object-fit:cover; border-radius:6px' })),
                h('td', null, h('b', null, row.rel), h('div', { class: 'hint' }, row.error))))))];
    }

    // Refresh the live parts without touching the settings inputs above them.
    function update() {
        const s = status();
        const live = $('#runLive');
        if (!live) return;
        const finished = s.done + s.failed + s.skipped;
        const percent = s.total ? (finished / s.total) * 100 : 0;
        $('#runReveal').style.width = `${Math.min(100, percent)}%`;
        $('#runPercent').textContent = s.total ? `${finished} of ${s.total}` : '';
        const idleLine = s.message || (App.dataset.images.length ? `${App.dataset.images.length} images in the jar, ready when you are.` : 'This folder has no images in it.');
        if (s.state !== 'running') $('#runFlavor').textContent = idleLine;
        if (s.state === 'running' && !flavorTimer) { flavorIdx = -1; rotateFlavor(); flavorTimer = setInterval(rotateFlavor, 2600); }
        if (s.state !== 'running' && flavorTimer) { clearInterval(flavorTimer); flavorTimer = null; }

        const running = s.state === 'running';
        $('#runButtons').replaceChildren(
            !busy() ? h('button', { onclick: (e) => control('start', e.target, { mode: 'caption' }) }, s.state === 'done' || s.state === 'stopped' ? 'Start another batch' : 'Start the batch') : null,
            running ? h('button', { class: 'quiet', onclick: (e) => control('pause', e.target) }, 'Pause') : null,
            ['paused', 'pausing'].includes(s.state) ? h('button', { disabled: s.state === 'pausing', onclick: (e) => control('resume', e.target) }, 'Resume') : null,
            busy() ? h('button', { class: 'danger', disabled: s.state === 'stopping', onclick: (e) => control('stop', e.target) }, 'Stop') : null,
            !busy() ? h('button', {
                class: 'quiet', title: 'Rewrite every caption from its stored fields with the current recipe, without looking at the images again',
                onclick: (e) => { if (confirm('Rewrite every caption from its stored fields, using the current recipe? Old captions stay in history.')) control('start', e.target, { mode: 'compose' }); },
            }, 'Recompose only') : null);

        const interrupted = s.interrupted && !busy();
        $('#runNotice').replaceChildren(...(interrupted ? [h('div', { class: 'notice', style: 'margin-bottom:16px' },
            'A batch was interrupted before it finished. ',
            h('button', { class: 'small', onclick: (e) => control('start', e.target, { mode: 'caption', resume: true }) }, 'Resume it'), ' ',
            h('button', { class: 'quiet small', onclick: (e) => control('forget', e.target) }, 'Forget it'))] : []));

        const stat = (value, label) => h('div', { class: 'stat' }, h('b', null, value), h('span', null, label));
        $('#runStats').replaceChildren(
            stat(s.done, 'wrapped (done)'),
            stat(s.skipped, 'skipped, already captioned'),
            stat(s.failed, 'misshapes (failed)'),
            stat(s.seconds_per_image != null ? `${s.seconds_per_image}s` : '–', 'average per image'),
            stat(s.tokens_per_second != null ? s.tokens_per_second : '–', 'tokens per second'),
            stat(running || s.state === 'pausing' ? (s.eta_seconds != null ? `~${formatDuration(s.eta_seconds)}` : 'tasting…') : '–', 'time remaining'),
            s.check_seconds ? stat(`${formatDuration(s.check_seconds)}`, `Taste Check time (${s.check_seconds_per_image}s per image)`) : null);

        const ids = new Set(s.in_flight.map((i) => String(i.id)));
        Object.keys(cards).forEach((id) => { if (!ids.has(id)) delete cards[id]; });
        live.replaceChildren(
            s.in_flight.length ? h('div', { class: 'card' },
                h('h2', { style: 'margin-bottom:10px' }, candy('On the line', 'images in flight')),
                h('div', { class: 'flight' }, s.in_flight.map(flightCard))) : null,
            s.recent.length ? h('div', { class: 'card' },
                h('h2', { style: 'margin-bottom:6px' }, candy('Fresh off the line', 'recent captions')),
                s.recent.map((item) => h('div', { class: 'recent-item', title: 'Open in the Sorting Table (review)', onclick: () => App.tabs.review.open(item.id) },
                    h('img', { src: thumbUrl(item.id) }),
                    h('div', { style: 'min-width:0; flex:1' }, h('div', { class: 'spread' }, h('b', { style: 'font-size:12.5px' }, item.rel), flagIcons(item.flags)), h('p', null, item.caption))))) : null);
    }

    function render() {
        if (needsDataset(panel)) return;
        const settings = App.dataset.settings;
        const recipe = h('div');
        renderRecipe(recipe);
        const verifier = App.settings.verifier_model && App.settings.verifier_model !== App.settings.model ? App.settings.verifier_model : null;
        const parallelOut = h('output', null, settings.parallel);

        panel.replaceChildren(
            h('details', { class: 'card', open: !busy() },
                h('summary', null, 'This dataset\'s recipe (run settings, shared with the Taste Test tab)'), recipe),
            h('div', { class: 'card' },
                h('div', { class: 'grid3' },
                    h('div', null, h('label', null, candy('Batch Size', 'parallel requests, 1 to 8')),
                        h('div', { class: 'row' }, h('input', { type: 'range', min: 1, max: 8, step: 1, value: settings.parallel, style: 'flex:1', oninput: (e) => { parallelOut.textContent = e.target.value; patchDataset({ parallel: Number(e.target.value) }); } }), parallelOut),
                        h('p', { class: 'hint' }, 'Each image is still its own request. In LM Studio, also raise Max Concurrent Predictions for the model.')),
                    h('div', null, h('label', null, candy('Taste Check', 'field checks, one short request per field')),
                        h('select', { onchange: (e) => patchDataset({ field_checks: e.target.value }) },
                            [['off', 'Off'], ['flagged', 'Flagged images only'], ['all', 'All images']].map(([value, label]) => h('option', { value, selected: settings.field_checks === value }, label))),
                        h('p', { class: 'hint' }, verifier ? `Answered by the Taste Tester (verifier): ${verifier}.` : 'Answered by the captioning model. Set a Taste Tester (verifier model) in the Candy Shop tab for a second opinion.')),
                    h('div', null, h('label', null, 'Existing descriptions'),
                        h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: settings.overwrite, onchange: (e) => patchDataset({ overwrite: e.target.checked }) }),
                            'Overwrite', h('span', { class: 'plain' }, 'the old caption stays in history')),
                        h('p', { class: 'hint' }, 'When off, images that already have a _nl.txt are skipped, which is also how a stopped batch picks up again.')))),
            h('div', { id: 'runNotice' }),
            h('div', { class: 'gen-progress', role: 'status', 'aria-live': 'polite' },
                h('div', { class: 'gen-progress-top' }, h('span', { class: 'gen-flavor', id: 'runFlavor' }), h('span', { class: 'gen-percent', id: 'runPercent' })),
                h('div', { class: 'gen-track', id: 'runTrack' }, h('div', { class: 'gen-reveal', id: 'runReveal' })),
                h('div', { class: 'spread', style: 'margin-top:12px' }, h('div', { class: 'row', id: 'runButtons' })),
                h('div', { class: 'stats', id: 'runStats' })),
            h('div', { id: 'runLive' }),
            h('div', { id: 'runMisshapes' }, misshapeNodes()));
        crustWidth = 0;
        buildCrust();
        update();
    }

    on('state', (event) => {
        const before = status();
        App.run = event;
        if (App.tab === 'run') update();
        if (event.failed !== before.failed || (event.state !== before.state && ['done', 'stopped'].includes(event.state))) loadMisshapes();
        if (event.state !== before.state && ['done', 'stopped'].includes(event.state) && App.tabs.review.refreshCount) App.tabs.review.refreshCount();
    });
    on('delta', (event) => {
        const item = status().in_flight.find((i) => i.id === event.id);
        if (item) item.text = event.text;
        if (App.tab === 'run' && cards[event.id]) cards[event.id].list.replaceChildren(...fieldList(event.text).flat());
    });
    on('stage', (event) => {
        const item = status().in_flight.find((i) => i.id === event.id);
        if (item) item.stage = event.stage;
        if (App.tab === 'run' && cards[event.id]) cards[event.id].stage.textContent = STAGES[event.stage] || '';
    });
    window.addEventListener('resize', debounce(() => { if (App.tab === 'run') buildCrust(); }, 150));

    App.tabs.run = {
        show: () => act(null, async () => {
            if (App.dataset.open) { App.run = await api('/api/run'); loadMisshapes(); }
            render();
        }),
    };
})();
