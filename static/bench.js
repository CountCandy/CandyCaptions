// Taste Test (Test bench): prove a recipe on a few pinned images before a full run.

(function () {
    const panel = $('#panel-bench');
    let bench = { pins: [], runs: [] };
    let selA = null, selB = null;
    let progress = '';
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
        if (!selA) return null;
        const a = await runDetail(selA);
        const b = selB && selB !== selA ? await runDetail(selB) : null;
        const head = (run) => h('div', null, h('b', null, run.label),
            h('div', { class: 'hint' }, `${run.summary.prefill} · ${run.summary.preset} · facts ${run.summary.captured}/${run.summary.facts} captured, ${run.summary.wrong} wrong`));
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
            bench.pins.length ? null : h('p', { class: 'hint', style: 'margin-top:8px' }, 'Pin 5 to 10 hard images. Then write a few facts for each once, and every recipe version is scored on them.'),
            bench.pins.map((pin) => h('div', { class: 'pinned' },
                h('img', { src: thumbUrl(pin.id), loading: 'lazy', style: 'cursor:zoom-in', title: 'Click to see the image at full size', onclick: () => viewImage(pin.id, pin.rel) }),
                h('div', null,
                    h('div', { class: 'spread' }, h('b', { style: 'font-size:12.5px' }, pin.rel),
                        h('span', null, h('button', { class: 'quiet small', onclick: () => showRequest(pin) }, 'Show request'))),
                    h('label', { style: 'margin-top:6px' }, candy('Tasting Notes', 'reference facts, one per line')),
                    h('textarea', {
                        rows: 3, placeholder: 'two people\nred scarf\nnight', value: pin.facts.join('\n'),
                        onchange: (e) => act(null, async () => { bench = await api('/api/test/facts', 'PUT', { id: pin.id, facts: e.target.value.split('\n') }); }),
                    })))));

        const runCard = h('div', { class: 'card' },
            h('h2', null, 'Run the test'),
            h('p', { class: 'hint', style: 'margin:4px 0 10px' }, 'One image at a time with the fixed seed. Nothing is written to the dataset folder.'),
            h('div', { class: 'row' }, label,
                h('button', {
                    disabled: !bench.pins.length, onclick: (e) => act(e.target, async () => {
                        e.target.textContent = 'Tasting…';
                        await flushDatasetSettings();
                        const data = await api('/api/test/run', 'POST', { label: label.value, twice: twice.checked });
                        bench = data;
                        if (data.run_ids.length === 2) [selA, selB] = data.run_ids;
                        else { selB = selA; selA = data.run_ids[0]; if (!selB) selB = null; }
                        progress = '';
                        render();
                    }),
                }, 'Run on the Sampler Box')),
            h('label', { class: 'check', style: 'margin-top:10px' }, twice, 'Run it twice to see normal variation',
                h('span', { class: 'plain' }, 'some servers are not exactly reproducible even with a fixed seed')),
            h('p', { class: 'hint', id: 'benchProgress' }, progress));

        const runs = h('div', { class: 'card' },
            h('h2', null, 'Versions tried'),
            bench.runs.length ? h('table', { style: 'margin-top:8px' },
                h('tr', null, h('th', { title: 'Left column' }, 'A'), h('th', { title: 'Right column' }, 'B'), h('th', null, 'Version'), h('th', null, 'Recipe'), h('th', null, 'Facts captured'), h('th', null, 'Wrong'), h('th', null, 'Flagged'), h('th')),
                bench.runs.map((run) => h('tr', null,
                    h('td', null, h('input', { type: 'radio', name: 'benchA', checked: run.id === selA, onchange: () => { selA = run.id; render(); } })),
                    h('td', null, h('input', { type: 'radio', name: 'benchB', checked: run.id === selB, onchange: () => { selB = run.id; render(); } })),
                    h('td', null, run.label, h('div', { class: 'hint' }, new Date(run.created * 1000).toLocaleString())),
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
                    }, 'Promote')))))
                : h('p', { class: 'hint', style: 'margin-top:8px' }, 'No test runs yet.'));

        const compareSlot = h('div');
        panel.replaceChildren(
            h('div', { class: 'card' }, h('h2', { style: 'margin-bottom:12px' }, candy('This dataset\'s recipe', 'run settings, shared with the Candy Factory tab')), recipe),
            h('div', { class: 'grid2' }, pins, h('div', null, runCard, runs)), compareSlot);
        try {
            const view = await comparison();
            if (view) compareSlot.replaceChildren(view);
        } catch (err) { toast(err.message, true); }
    }

    on('test_progress', (event) => {
        progress = event.done < event.total ? `${event.label}: ${event.done + 1} of ${event.total}, ${event.rel}` : '';
        const el = $('#benchProgress');
        if (el) el.textContent = progress;
    });

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
