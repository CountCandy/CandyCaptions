// Taste Test (Test bench): prove a recipe on a few pinned images before a full run.

(function () {
    const panel = $('#panel-bench');
    let bench = { pins: [], runs: [] };
    let selA = null, selB = null;
    let tasting = null;   // progress of the test run in flight, or the last one finished
    const details = {};   // run id -> full run

    const VERDICTS = { captured: ['good', '✓'], wrong: ['bad', '✗'], missing: ['neutral', '–'] };

    async function runDetail(id) {
        if (!details[id]) details[id] = await api(`/api/test/run/${id}`);
        return details[id];
    }

    function showRequest(pin) {
        act(null, async () => {
            await flushDatasetSettings();
            const data = await api('/api/test/request', 'POST', { id: pin.id });
            overlay(h('div', { class: 'modal' },
                h('h2', null, 'Exact request for ', pin.rel),
                h('p', { class: 'hint', style: 'margin:6px 0 12px' }, 'Each request stands alone: the server keeps no memory between requests and nothing from another image is ever resent.'),
                h('div', { class: 'row', style: 'align-items:flex-start' },
                    h('img', { src: thumbUrl(pin.id), style: 'width:120px; border-radius:8px' }),
                    h('div', { class: 'grow' },
                        h('h3', null, '1. Look: ', h('code', null, 'POST ' + data.look.endpoint)),
                        h('pre', { class: 'payload' }, JSON.stringify(data.look.body, null, 2)))),
                h('h3', { style: 'margin-top:14px' }, '2. Compose (text only): ', h('code', null, 'POST ' + data.compose.endpoint)),
                data.compose.note ? h('p', { class: 'hint' }, data.compose.note) : null,
                h('pre', { class: 'payload' }, JSON.stringify(data.compose.body, null, 2))), { center: true });
        });
    }

    function pinPicker() {
        const pinned = new Set(bench.pins.map((p) => p.id));
        const grid = h('div', { class: 'thumbs' }, App.dataset.images.filter((i) => !i.health.some((b) => b.check === 'unreadable')).map((image) => {
            const button = h('button', { class: 'pin' + (pinned.has(image.id) ? ' on' : ''), title: 'Pin to the Sampler Box' }, '★');
            const card = h('div', { class: 'thumb' }, h('img', { src: thumbUrl(image.id), loading: 'lazy' }), button, h('div', { class: 'name' }, image.rel));
            card.onclick = () => act(null, async () => {
                const now = !pinned.has(image.id);
                bench = await api('/api/test/pin', 'POST', { id: image.id, pinned: now });
                if (now) pinned.add(image.id); else pinned.delete(image.id);
                image.pinned = now;
                button.classList.toggle('on', now);
            });
            return card;
        }));
        overlay(h('div', { class: 'modal' },
            h('h2', null, candy('Fill the Sampler Box', 'pin 5 to 10 test images')),
            h('p', { class: 'hint', style: 'margin:6px 0 12px' }, 'Pick hard ones: crowds, visible text, odd angles, busy backgrounds.'), grid), { center: true, onclose: render });
    }

    function verdictChips(result) {
        if (result.error) return h('span', { class: 'chip bad' }, '✗ failed: ', result.error);
        return h('div', { class: 'chips' },
            (result.facts || []).map((fact, index) => {
                const [cls, icon] = VERDICTS[result.verdicts[index]] || VERDICTS.missing;
                return h('span', { class: `chip ${cls} verdict`, title: result.verdicts[index] }, icon, ' ', fact);
            }),
            [...new Set((result.reasons || []).map((r) => r.signal))].map((signal) => {
                const reason = result.reasons.find((r) => r.signal === signal);
                const chip = flagChip(signal, false);
                chip.title = reason.message;
                return chip;
            }));
    }

    async function comparison() {
        if (!selA || !bench.runs.some((r) => r.id === selA)) return null;
        if (selB && !bench.runs.some((r) => r.id === selB)) selB = null;
        const a = await runDetail(selA);
        const b = selB && selB !== selA ? await runDetail(selB) : null;
        const head = (run) => h('div', null, h('b', null, run.label),
            h('div', { class: 'hint' }, `${run.summary.model || 'model not recorded'} · ${run.summary.prefill} · ${run.summary.preset} · facts ${run.summary.captured}/${run.summary.facts} captured, ${run.summary.wrong} wrong`));
        return h('div', { class: 'card' },
            h('h2', null, b ? 'Side by side' : 'Results'),
            h('div', { class: 'compare' + (b ? '' : ' single'), style: 'padding-top:6px' }, h('span'), head(a), b ? head(b) : null),
            a.results.map((ra) => {
                const rb = b ? b.results.find((r) => r.image_id === ra.image_id) : null;
                return h('div', { class: 'compare' + (b ? '' : ' single') },
                    h('div', null, h('img', { src: thumbUrl(ra.image_id), loading: 'lazy', style: 'cursor:zoom-in', title: 'Click to see the image at full size', onclick: () => viewImage(ra.image_id, ra.rel) }), h('div', { class: 'hint', style: 'word-break:break-all' }, ra.rel)),
                    h('div', null, h('p', null, ra.caption), verdictChips(ra)),
                    b ? h('div', null, rb ? [h('p', null, wordDiff(ra.caption, rb.caption)), verdictChips(rb)] : h('span', { class: 'hint' }, 'not in this run')) : null);
            }),
            b ? h('p', { class: 'hint', style: 'margin-top:10px' }, 'In the right column, ', h('ins', null, 'added'), ' and ', h('del', null, 'removed'), ' words are measured against the left one.') : null);
    }

    async function render() {
        if (needsDataset(panel)) return;
        const recipe = h('div');
        renderRecipe(recipe);
        const label = h('input', { type: 'text', placeholder: 'Name this version, such as "shorter rules"', style: 'flex:1' });
        const twice = h('input', { type: 'checkbox' });

        const pins = h('div', { class: 'card' },
            h('div', { class: 'spread' },
                h('h2', null, candy('Sampler Box', 'test set')),
                h('button', { class: 'quiet small', onclick: pinPicker }, bench.pins.length ? 'Change the pins' : 'Pin some images')),
            h('p', { class: 'hint', style: 'margin:6px 0 4px' }, bench.pins.length
                ? 'Two boxes per image, and they do opposite jobs. The Hint is told to the model as fact. Tasting Notes are the answer key: the model never sees them, and each test batch is scored on how many it got right.'
                : 'Pin 5 to 10 hard images. Then write a few facts for each once, and every recipe version is scored on them.'),
            h('div', { class: 'pins-grid' }, bench.pins.map((pin) => h('div', { class: 'pinned' },
                h('img', { src: thumbUrl(pin.id), loading: 'lazy', style: 'cursor:zoom-in', title: 'Click to see the image at full size', onclick: () => viewImage(pin.id, pin.rel) }),
                h('div', null,
                    h('div', { class: 'spread' }, h('b', { style: 'font-size:13.5px' }, pin.rel),
                        h('span', null, h('button', { class: 'quiet small', onclick: () => showRequest(pin) }, 'Show request'))),
                    hintBox(pin.id, pin.note, (saved) => { pin.note = saved; }, { style: 'margin-top:6px' }),
                    pin.facts.length && !pin.note ? h('button', {
                        class: 'quiet small', style: 'margin-top:6px', title: 'Tasting Notes are never shown to the model. This moves the text into the Hint, which is.',
                        onclick: (e) => act(e.target, async () => {
                            await api(`/api/image/${pin.id}/note`, 'PUT', { text: pin.facts.join(' ') });
                            bench = await api('/api/test/facts', 'PUT', { id: pin.id, facts: [] });
                            render();
                        }),
                    }, '↑ Move these Tasting Notes into the Hint') : null,
                    h('label', { style: 'margin-top:8px' }, candy('Tasting Notes', 'answer key, one fact per line. Used only to score captions; the model never sees it')),
                    h('textarea', {
                        rows: 3, placeholder: 'two people\nred scarf\nnight', value: pin.facts.join('\n'),
                        onchange: (e) => act(null, async () => { bench = await api('/api/test/facts', 'PUT', { id: pin.id, facts: e.target.value.split('\n') }); }),
                    }))))));

        const runCard = h('div', { class: 'card' },
            h('h2', null, candy('Take a Bite', 'run the test')),
            h('p', { class: 'hint', style: 'margin:4px 0 10px' }, 'One image at a time with the fixed seed. Nothing is written to the dataset folder.'),
            h('div', { class: 'row' }, label,
                h('button', {
                    disabled: !bench.pins.length, onclick: (e) => act(e.target, async () => {
                        e.target.textContent = 'Tasting…';
                        await flushDatasetSettings();
                        const rounds = twice.checked ? 2 : 1;
                        tasting = { done: 0, total: bench.pins.length * rounds, rel: '', finished: false, round: 0, rounds, label: '' };
                        showProgress();
                        let data;
                        try {
                            data = await api('/api/test/run', 'POST', { label: label.value, twice: twice.checked });
                        } catch (err) {
                            tasting = null;
                            showProgress();
                            throw err;
                        }
                        tasting = { ...tasting, done: tasting.total, finished: true };
                        playGumdrops();
                        bench = data;
                        if (data.run_ids.length === 2) [selA, selB] = data.run_ids;
                        else { selB = selA; selA = data.run_ids[0]; if (!selB) selB = null; }
                        render();
                    }),
                }, 'Run on the Sampler Box')),
            h('label', { class: 'check', style: 'margin-top:10px' }, twice, 'Run it twice to see normal variation',
                h('span', { class: 'plain' }, 'some servers are not exactly reproducible even with a fixed seed')),
            h('div', { class: 'gen-progress', role: 'status', 'aria-live': 'polite', style: 'margin:12px 0 0' },
                h('div', { class: 'gen-progress-top' }, h('span', { class: 'gen-flavor', id: 'benchFlavor' }), h('span', { class: 'gen-percent', id: 'benchPercent' })),
                h('div', { class: 'gen-track', id: 'benchTrack' }, h('div', { class: 'gen-reveal', id: 'benchReveal' }))));

        const runs = h('div', { class: 'card' },
            h('div', { class: 'spread' },
                h('h2', null, candy('Test Batches', 'versions tried; the newest 8 are kept')),
                bench.runs.length ? h('button', {
                    class: 'quiet small', onclick: (e) => act(e.target, async () => {
                        if (!confirm('Clear every version tried for this dataset? Nothing in the dataset folder changes.')) return;
                        bench = await api('/api/test/runs', 'DELETE');
                        App.tabs.bench.reset();
                        render();
                    }),
                }, 'Clear all') : null),
            bench.runs.length ? h('table', { style: 'margin-top:8px' },
                h('tr', null, h('th', { title: 'Left column' }, 'A'), h('th', { title: 'Right column' }, 'B'), h('th', null, 'Version'), h('th', null, 'Model'), h('th', null, 'Recipe'), h('th', null, 'Facts captured'), h('th', null, 'Wrong'), h('th', null, 'Flagged'), h('th')),
                bench.runs.map((run) => h('tr', null,
                    h('td', null, h('input', { type: 'radio', name: 'benchA', checked: run.id === selA, onchange: () => { selA = run.id; render(); } })),
                    h('td', null, h('input', { type: 'radio', name: 'benchB', checked: run.id === selB, onchange: () => { selB = run.id; render(); } })),
                    h('td', null, run.label, h('div', { class: 'hint' }, new Date(run.created * 1000).toLocaleString())),
                    h('td', null, run.model || '–', run.verifier_model && run.verifier_model !== run.model ? h('div', { class: 'hint' }, `Taste Tester: ${run.verifier_model}`) : null),
                    h('td', null, run.prefill, h('div', { class: 'hint' }, run.preset)),
                    h('td', null, run.facts ? `${run.captured} of ${run.facts}` : h('span', { class: 'hint' }, 'no facts written')),
                    h('td', null, run.wrong || h('span', { class: 'muted' }, '0')),
                    h('td', null, `${run.flagged} of ${run.images}`, run.failed ? h('div', { class: 'hint' }, `${run.failed} failed`) : null),
                    h('td', null, h('button', {
                        class: 'quiet small', title: 'Make this tested version the dataset\'s run settings', onclick: (e) => act(e.target, async () => {
                            App.dataset.settings = (await api('/api/test/promote', 'POST', { run_id: run.id })).settings;
                            await App.loadPrefills();
                            toast(`"${run.label}" is now this dataset's recipe.`);
                            render();
                        }),
                    }, 'Promote'), ' ',
                    h('button', {
                        class: 'quiet icon-btn', title: 'Remove this version', onclick: (e) => act(e.target, async () => {
                            bench = await api(`/api/test/run/${run.id}`, 'DELETE');
                            delete details[run.id];
                            if (selA === run.id) selA = bench.runs.length ? bench.runs[0].id : null;
                            if (selB === run.id) selB = null;
                            render();
                        }),
                    }, '✕')))))
                : h('p', { class: 'hint', style: 'margin-top:8px' }, 'No test batches yet. Each run is kept here so you can put two side by side and promote the better one to be this dataset\'s recipe.'));

        const compareSlot = h('div');
        panel.replaceChildren(
            h('div', { class: 'card' }, h('h2', { style: 'margin-bottom:12px' }, candy('This dataset\'s recipe', 'run settings, shared with the Candy Factory tab')), recipe),
            pins, h('div', { class: 'grid2 wide-right' }, runCard, runs), compareSlot);
        showProgress();
        try {
            const view = await comparison();
            if (view) compareSlot.replaceChildren(view);
        } catch (err) { toast(err.message, true); }
    }

    // The rock-candy bar for a test run. `tasting` is null until a test has been started in this session.
    function showProgress() {
        const track = $('#benchTrack'), reveal = $('#benchReveal');
        if (!track) return;
        const width = Math.max(1, Math.round(track.clientWidth));
        if (reveal.dataset.width !== String(width)) { reveal.innerHTML = buildCrustSVG(width, 42); reveal.dataset.width = String(width); }
        const flavor = $('#benchFlavor'), percent = $('#benchPercent');
        if (!tasting) {
            reveal.style.width = '0%';
            flavor.textContent = bench.pins.length ? `${bench.pins.length} pieces in the Sampler Box, ready to taste.` : 'The Sampler Box is empty. Pin a few images first.';
            percent.textContent = '';
            return;
        }
        reveal.style.width = `${tasting.total ? Math.min(100, (tasting.done / tasting.total) * 100) : 0}%`;
        percent.textContent = `${tasting.done} of ${tasting.total}`;
        const model = App.settings.model ? ` with ${App.settings.model}` : '';
        flavor.textContent = tasting.finished ? `Taste test done${model}.`
            : tasting.rel ? `Tasting ${tasting.rel}${model}…` : `Warming up the taste buds${model}…`;
    }

    on('test_progress', (event) => {
        if (!tasting || tasting.finished) return;
        // With "run it twice" the server reports each round from zero, so rounds are added up here.
        if (event.label !== tasting.label) { tasting.label = event.label; tasting.round += 1; }
        const perRound = tasting.total / tasting.rounds;
        tasting.done = Math.min(tasting.total, (tasting.round - 1) * perRound + event.done);
        tasting.rel = event.rel || tasting.rel;
        showProgress();
    });
    window.addEventListener('resize', debounce(() => { if (App.tab === 'bench') showProgress(); }, 150));

    App.tabs.bench = {
        show: () => act(null, async () => {
            if (App.dataset.open) {
                bench = await api('/api/test');
                if (!selA && bench.runs.length) selA = bench.runs[0].id;
            }
            await render();
        }),
        reset: () => { selA = selB = null; for (const key of Object.keys(details)) delete details[key]; },
    };
})();
