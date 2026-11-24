// Flavor Profile (Style): which images do not fit the rest of the set.
// Macro pass: source quality from the pixels, no model. Style lens: each image checked against the set's own style.

(function () {
    const panel = $('#panel-style');
    let data = null;
    let ringerResult = null;

    const percent = (p) => `${Math.round(p * 100)}%`;
    const busy = () => data && data.job.state === 'running';

    function thumb(row, size = 64) {
        return h('img', {
            src: thumbUrl(row.id), loading: 'lazy', title: 'Click to see the image at full size',
            style: `width:${size}px; height:${size}px; object-fit:cover; border-radius:8px; cursor:zoom-in; flex:none`,
            onclick: () => viewImage(row.id, row.rel),
        });
    }

    function macroCard() {
        return h('div', { class: 'card' },
            h('h2', null, candy('Source quality', 'macro pass: measured from the pixels, no model needed')),
            h('p', { class: 'hint', style: 'margin:4px 0 10px' },
                'Each image is compared with the rest of this folder, not with a fixed limit. Compression or softness that the whole set shares is not flagged, because then it is not the odd one out.'),
            data.macro.length
                ? data.macro.map((row) => h('div', { class: 'row', style: 'padding:8px 0; border-bottom:1px solid var(--input-border)' },
                    thumb(row), h('div', { class: 'grow', style: 'min-width:0' }, h('b', { style: 'font-size:13px' }, row.rel),
                        h('div', { class: 'chips', style: 'margin-top:4px' }, row.macro.map((flag) => h('span', { class: 'chip health', title: flag.detail }, flag.name))),
                        h('div', { class: 'hint', style: 'margin-top:2px' }, row.macro.map((flag) => flag.detail).join('; ')))))
                : h('p', null, h('span', { class: 'chip good' }, '✓'), ' No image stands apart from the rest in sharpness, resolution, compression or grain.'));
    }

    function oddRow(row) {
        const whySlot = h('div', { class: 'hint', style: 'margin-top:4px' }, row.why ? ['How it differs: ', row.why] : null);
        const byModel = row.match != null && row.match < 0.5;
        return h('div', { class: 'row', style: 'align-items:flex-start; padding:10px 0; border-bottom:1px solid var(--input-border)' },
            thumb(row, 96),
            h('div', { class: 'grow', style: 'min-width:0' },
                h('b', { style: 'font-size:13px' }, row.rel),
                h('div', { class: 'chips', style: 'margin-top:4px' },
                    row.match != null ? h('span', { class: 'chip ' + (byModel ? 'bad' : 'neutral'), title: row.measured ? 'Probability the model says this image is in the set\'s style' : 'A plain yes or no; this server gives no probabilities' },
                        `Model: ${percent(row.match)} that it fits`) : null,
                    row.pixels.reasons.length ? h('span', { class: 'chip health', title: 'Palette, light and texture against its nearest neighbors in the set' }, `Pixels: ${row.pixels.reasons.join(', ')}`) : null),
                row.note ? h('div', { class: 'hint', style: 'margin-top:4px' }, 'Its style note: ', row.note) : null,
                whySlot),
            h('button', {
                class: 'quiet small', title: 'Shows this image to the model next to three typical ones and asks how it differs', disabled: busy(),
                onclick: (e) => act(e.target, async () => {
                    e.target.textContent = 'Comparing…';
                    row.why = (await api('/api/style/explain', 'POST', { id: row.id })).why;
                    whySlot.replaceChildren('How it differs: ', row.why);
                }),
            }, row.why ? 'Ask again' : 'How does it differ?'));
    }

    function lensCard() {
        const redo = h('input', { type: 'checkbox' });
        const made = data.made && data.made.model;
        const taster = made && data.made.verifier_model && data.made.verifier_model !== data.made.model ? ` · Taste Tester: ${data.made.verifier_model}` : '';
        const ranked = data.rows.slice().sort((a, b) => (a.match ?? 2) - (b.match ?? 2) || b.pixels.z - a.pixels.z);
        return h('div', { class: 'card' },
            h('h2', null, candy('Odd One Out', 'style lens: each image checked against the set\'s own style')),
            h('p', { class: 'hint', style: 'margin:4px 0 12px' },
                'The model writes a short style note for each image, sums the notes up into one style for the set, then looks at each image again and answers one question: is this made in that style? One image per request. The result is a ranking for your eye, not a verdict.'),
            data.card ? h('div', { class: 'notice', style: 'margin-bottom:12px' }, h('b', null, 'This set\'s style: '), data.card,
                h('div', { class: 'hint', style: 'margin-top:4px' }, `From ${data.made.count} images on ${new Date(data.made.at * 1000).toLocaleString()}. Captioner: ${data.made.model}${taster}`)) : null,
            h('div', { class: 'row' },
                h('button', {
                    disabled: busy(), onclick: (e) => act(e.target, async () => {
                        await flushDatasetSettings();
                        data.job = await api('/api/style/run', 'POST', { redo: redo.checked });
                        render();
                    }),
                }, busy() ? 'Tasting…' : data.checked ? 'Taste the batch again' : 'Taste the whole batch'),
                data.checked ? h('label', { class: 'check', style: 'margin:0' }, redo, 'Write the style notes again too') : null,
                h('span', { class: 'hint' }, App.settings.model ? `Will use ${App.settings.model}` : 'No model chosen yet')),
            h('div', { class: 'gen-progress', role: 'status', 'aria-live': 'polite', style: 'margin:12px 0' },
                h('div', { class: 'gen-progress-top' }, h('span', { class: 'gen-flavor', id: 'styleFlavor' }), h('span', { class: 'gen-percent', id: 'stylePercent' })),
                h('div', { class: 'gen-track', id: 'styleTrack' }, h('div', { class: 'gen-reveal', id: 'styleReveal' }))),
            data.checked ? (data.odd.length
                ? [h('h3', null, `${data.odd.length} image${data.odd.length === 1 ? '' : 's'} to look at, least like the set first`), data.odd.map(oddRow)]
                : h('p', null, h('span', { class: 'chip good' }, '✓'), ' Nothing stands out. Every image fits the set\'s style, by the model and by the pixels.'))
                : (data.odd.length ? [h('h3', null, 'By the pixels alone, before any model check'), data.odd.map(oddRow)] : null),
            data.checked ? h('details', { style: 'margin-top:12px' },
                h('summary', null, `Every image, least like the set first (${data.rows.length})`),
                h('table', null,
                    h('tr', null, h('th'), h('th', null, 'Image'), h('th', null, 'Model: fits'), h('th', null, 'Pixels'), h('th', null, 'Style note')),
                    ranked.map((row) => h('tr', null,
                        h('td', { style: 'width:52px' }, thumb(row, 40)), h('td', null, row.rel),
                        h('td', null, row.match != null ? percent(row.match) : '–'),
                        h('td', null, row.pixels.reasons.join(', ') || h('span', { class: 'muted' }, 'like the rest')),
                        h('td', { class: 'hint' }, row.note || ''))))) : null);
    }

    function ringerCard() {
        const input = h('input', { type: 'text', class: 'grow', placeholder: 'Full path of an image that does not belong, such as C:\\other\\drawing.png' });
        const slot = h('div', { style: 'margin-top:10px' });
        const show = () => {
            const r = ringerResult;
            if (!r) return;
            slot.replaceChildren(h('div', { class: 'notice' },
                r.model_rank != null
                    ? h('p', null, h('b', null, `By the model: least like the set, place ${r.model_rank} of ${r.count + 1}`), ` (${percent(r.match)} that it fits). `,
                        r.match < 0.5 ? 'The check caught it.' : 'The check did not catch it, so do not rely on the model\'s ranking for this kind of difference.')
                    : h('p', { class: 'hint' }, 'Taste the whole batch first to test the model\'s part.'),
                h('p', null, h('b', null, `By the pixels: place ${r.pixel_rank} of ${r.count + 1}`), r.pixels.reasons.length ? ` (${r.pixels.reasons.join(', ')}).` : ' (palette, light and texture like the rest).'),
                r.macro.length ? h('p', null, h('b', null, 'Source quality: '), r.macro.map((f) => f.detail).join('; '), '.') : null));
        };
        show();
        return h('div', { class: 'card' },
            h('h2', null, candy('Plant a Ringer', 'test the check with an image you know does not belong')),
            h('p', { class: 'hint', style: 'margin:4px 0 10px' }, 'The image is measured and ranked against this set but never added to it. If your ringer does not come out near the top, the ranking cannot be trusted for that kind of difference.'),
            h('div', { class: 'row' }, input,
                h('button', { class: 'quiet', disabled: busy(), onclick: (e) => act(e.target, async () => { e.target.textContent = 'Tasting…'; ringerResult = await api('/api/style/ringer', 'POST', { path: input.value }); show(); }) }, 'Test this image')),
            slot);
    }

    function showProgress() {
        const track = $('#styleTrack'), reveal = $('#styleReveal');
        if (!track || !data) return;
        const width = Math.max(1, Math.round(track.clientWidth));
        if (reveal.dataset.width !== String(width)) { reveal.innerHTML = buildCrustSVG(width, 42); reveal.dataset.width = String(width); }
        const job = data.job;
        const idle = data.checked ? `${data.checked} of ${data.count} images checked against the set's style.` : `${data.count} images, not tasted yet.`;
        reveal.style.width = job.state === 'running' ? `${job.total ? Math.min(100, (job.done / job.total) * 100) : 0}%` : (job.state === 'done' ? '100%' : '0%');
        $('#stylePercent').textContent = job.state === 'running' && job.total ? `${job.done} of ${job.total}` : '';
        $('#styleFlavor').textContent = job.state === 'running' ? `${job.step || 'Warming up'}…` : (job.message || idle);
    }

    function render() {
        if (needsDataset(panel)) return;
        if (!data) { panel.replaceChildren(h('div', { class: 'empty' }, 'Measuring the batch…')); return; }
        if (!data.enough) {
            panel.replaceChildren(h('div', { class: 'card empty' },
                `This folder has ${data.count} readable image${data.count === 1 ? '' : 's'}. A set needs at least ${data.min_set} before anything can be called the odd one out.`));
            return;
        }
        panel.replaceChildren(macroCard(), lensCard(), ringerCard());
        showProgress();
    }

    on('style_progress', (event) => {
        if (!data) return;
        const before = data.job.state;
        data.job = event;
        if (event.state !== 'running' && before === 'running') {
            if (event.state === 'done') playGumdrops();
            if (event.state === 'failed') toast(event.message, true);
            act(null, async () => { data = await api('/api/style'); if (App.tab === 'style') render(); });
        } else if (App.tab === 'style') showProgress();
    });
    window.addEventListener('resize', debounce(() => { if (App.tab === 'style') showProgress(); }, 150));

    App.tabs.style = {
        show: () => act(null, async () => {
            if (App.dataset.open) data = await api('/api/style');
            render();
        }),
        reset: () => { data = null; ringerResult = null; },
    };
})();
