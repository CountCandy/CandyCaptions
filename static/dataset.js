// Pantry (Dataset): open a folder, run the Freshness Check (health checks), find and replace.

(function () {
    const panel = $('#panel-dataset');
    let filter = null;     // health check key, or null for everything
    let shown = 240;
    let order = localStorage.getItem('pantryOrder') || 'az';
    let changes = null;    // find-and-replace preview

    async function openFolder(path, button) {
        await act(button, async () => {
            if (button) button.textContent = 'Checking every image…';
            App.dataset = await api('/api/dataset/open', 'POST', { path });
            filter = null; changes = null; shown = 240;
            if (App.tabs.bench.reset) App.tabs.bench.reset();
            if (App.tabs.review.reset) App.tabs.review.reset();
            refreshHeader();
            render();
            const problems = Object.values(App.dataset.health_summary).reduce((a, b) => a + b, 0);
            toast(`${App.dataset.images.length} images in the jar` + (problems ? `, ${problems} freshness notes to look at.` : ', all fresh.'));
        });
    }

    async function reload() {
        App.dataset = await api('/api/dataset');
        refreshHeader();
        render();
    }

    function browse(input) {
        const body = h('div');
        // With no path the picker opens at your start folder; `roots` asks for the drive list instead.
        const go = (path, roots = false) => act(null, async () => {
            const data = await api('/api/browse?path=' + encodeURIComponent(path) + (roots ? '&roots=true' : ''));
            const isStart = data.path && data.path === data.start_folder;
            body.replaceChildren(
                h('div', { class: 'row', style: 'margin-bottom:10px' },
                    h('code', { class: 'grow' }, data.path || 'Pick a starting place'),
                    data.parent !== null ? h('button', { class: 'quiet small', onclick: () => go(data.parent, !data.parent) }, '↑ Up') : null,
                    data.path ? h('button', { class: 'small', onclick: () => { input.value = data.path; close(); openFolder(data.path); } }, 'Open this folder') : null),
                h('div', { class: 'row', style: 'margin-bottom:10px' },
                    data.path ? h('button', {
                        class: 'quiet small', disabled: isStart, title: 'The folder picker will open here from now on',
                        onclick: (e) => act(e.target, async () => {
                            const saved = await api('/api/settings', 'PUT', { start_folder: data.path });
                            App.settings = saved.settings;
                            toast('The folder picker will start here from now on.');
                            go(data.path);
                        }),
                    }, isStart ? '★ This is your start folder' : '☆ Start here every time') : null,
                    data.start_folder && !isStart ? h('button', { class: 'quiet small', title: data.start_folder, onclick: () => go(data.start_folder) }, '★ Go to start folder') : null,
                    data.path ? h('button', { class: 'quiet small', onclick: () => go('', true) }, 'Drives') : null),
                h('div', { class: 'stack', style: 'max-height:55vh; overflow:auto' },
                    data.folders.length ? data.folders.map((f) => h('button', { class: 'quiet', style: 'width:100%; text-align:left', onclick: () => go(f.path) }, '📁 ', f.name))
                        : h('p', { class: 'hint' }, 'No folders in here.')));
        });
        const close = overlay(h('div', { class: 'modal', style: 'width:min(640px,100%)' }, h('h2', { style: 'margin-bottom:10px' }, 'Choose a dataset folder'), body), { center: true });
        go(App.dataset.open ? '' : input.value.trim());
    }

    function openBar() {
        const d = App.dataset;
        const input = h('input', { type: 'text', class: 'grow', placeholder: 'C:\\datasets\\my_lora\\images', value: d.open ? d.folder : '', onkeydown: (e) => { if (e.key === 'Enter') openFolder(input.value); } });
        return h('div', { class: 'card' },
            h('h2', null, candy('Dataset folder', 'images, with their tag files next to them')),
            h('div', { class: 'row', style: 'margin-top:10px' }, input,
                h('button', { class: 'quiet', onclick: () => browse(input) }, 'Browse…'),
                h('button', { onclick: (e) => openFolder(input.value, e.target) }, d.open ? 'Open' : 'Open folder')),
            (d.recent || []).length ? h('div', { class: 'chips', style: 'margin-top:10px' }, h('span', { class: 'hint' }, 'Recent:'),
                d.recent.map((path) => h('span', { class: 'chip neutral clickable', title: path, onclick: () => openFolder(path) }, path.split(/[\\/]/).slice(-2).join('/')))) : null,
            d.open ? null : h('div', { class: 'empty' }, 'The jar is empty. Point me at a folder of images.'));
    }

    function healthCard() {
        const d = App.dataset, s = d.settings;
        const total = Object.values(d.health_summary).reduce((a, b) => a + b, 0);
        const fixButton = (kind, label) => d.health_summary[kind] ? h('button', {
            class: 'quiet small', onclick: (e) => act(e.target, async () => {
                const what = kind === 'sideways' ? 'Save upright copies' : `Fill transparency with ${s.background} and save upright copies`;
                if (!confirm(`${what} of ${d.health_summary[kind]} image(s), in place? The originals are kept in CandyCaption's backup folder.`)) return;
                const result = await api('/api/dataset/fix', 'POST', { kind });
                toast(`${result.fixed} fixed. Originals are in ${result.backup}` + (result.errors.length ? `. ${result.errors.length} could not be fixed.` : '.'), result.errors.length > 0);
                await reload();
            }),
        }, label) : null;

        return h('div', { class: 'card' },
            h('div', { class: 'spread' },
                h('h2', null, candy('Freshness Check', 'dataset health checks')),
                h('button', { class: 'quiet small', onclick: (e) => act(e.target, async () => { App.dataset = await api('/api/dataset/rescan', 'POST'); refreshHeader(); render(); }) }, 'Check again')),
            h('p', { class: 'hint', style: 'margin:4px 0 12px' }, `${d.images.length} images, ${d.captioned} with a description. `,
                total ? 'Click a badge to see only those images.' : 'Everything looks fresh.'),
            h('div', { class: 'chips' },
                Object.entries(d.health_summary).filter(([, count]) => count).map(([check, count]) => {
                    const chip = healthChip(check, d.health_checks[check].why, true);
                    chip.append(` · ${count}`);
                    chip.classList.add('clickable');
                    if (filter === check) chip.classList.add('selected');
                    chip.onclick = () => { filter = filter === check ? null : check; shown = 240; render(); };
                    return chip;
                })),
            filter ? h('p', { class: 'notice', style: 'margin-top:12px' }, h('b', null, d.health_checks[filter].name, ': '), d.health_checks[filter].why) : null,
            h('div', { class: 'row', style: 'margin-top:12px' },
                fixButton('sideways', 'Save all sideways images upright'),
                fixButton('transparency', 'Fill all transparency')),
            h('div', { class: 'row', style: 'margin-top:14px' },
                h('label', { class: 'check', style: 'margin:0' },
                    h('input', { type: 'checkbox', checked: s.recursive, onchange: (e) => act(null, async () => { await api('/api/dataset/settings', 'PUT', { recursive: e.target.checked }); await reload(); }) }), 'Include subfolders'),
                h('label', { style: 'margin:0 0 0 14px' }, 'Size floor (megapixels)'),
                h('input', { type: 'number', step: 0.05, min: 0, max: 4, value: s.min_area_mp, style: 'width:90px', onchange: (e) => act(null, async () => { await api('/api/dataset/settings', 'PUT', { min_area_mp: Number(e.target.value) }); await reload(); }) }),
                h('label', { style: 'margin:0 0 0 14px', title: 'Used to fill see-through areas, both for the captioner and for the one-click fix' }, 'Background color'),
                h('input', { type: 'color', value: s.background, onchange: (e) => act(null, async () => { await api('/api/dataset/settings', 'PUT', { background: e.target.value }); await reload(); }) })),
            h('p', { class: 'hint', style: 'margin-top:8px' }, 'Token counts use: ', d.tokenizer.source, d.tokenizer.exact ? '.' : '. Set a tokenizer in the Candy Shop tab for exact counts.'));
    }

    function grid() {
        const d = App.dataset;
        const images = (filter ? d.images.filter((i) => i.health.some((b) => b.check === filter)) : d.images).slice();
        const sorters = {
            az: (a, b) => byName(a.rel, b.rel),
            za: (a, b) => byName(b.rel, a.rel),
            todo: (a, b) => a.has_caption - b.has_caption || byName(a.rel, b.rel),
            problems: (a, b) => b.health.length - a.health.length || byName(a.rel, b.rel),
        };
        images.sort(sorters[order] || sorters.az);
        return h('div', { class: 'card' },
            h('div', { class: 'spread', style: 'margin-bottom:10px' },
                h('h2', null, filter ? `${images.length} images: ${d.health_checks[filter].name.toLowerCase()}` : 'All images'),
                h('span', { class: 'row' },
                    h('span', { class: 'hint' }, 'Click an image to see it large. ★ pins it to the Sampler Box (test set).'),
                    h('label', { style: 'margin:0' }, 'Order'),
                    h('select', { style: 'width:auto', onchange: (e) => { order = e.target.value; localStorage.setItem('pantryOrder', order); render(); } },
                        [['az', 'File name, A to Z'], ['za', 'File name, Z to A'], ['todo', 'Not captioned first'], ['problems', 'Most freshness notes first']]
                            .map(([value, label]) => h('option', { value, selected: order === value }, label))))),
            h('div', { class: 'thumbs' }, images.slice(0, shown).map((image) => {
                const unreadable = image.health.some((b) => b.check === 'unreadable');
                const pin = h('button', {
                    class: 'pin' + (image.pinned ? ' on' : ''), title: 'Pin to the Sampler Box (test set)', onclick: (e) => {
                        e.stopPropagation();
                        act(null, async () => { await api('/api/test/pin', 'POST', { id: image.id, pinned: !image.pinned }); image.pinned = !image.pinned; pin.classList.toggle('on', image.pinned); });
                    },
                }, '★');
                return h('div', { class: 'thumb', title: `${image.rel} · ${image.width}×${image.height}`, onclick: () => { if (!unreadable) viewImage(image.id, image.rel, image.has_caption ? [['Open in the Sorting Table', () => App.tabs.review.open(image.id)]] : []); } },
                    unreadable ? h('div', { style: 'aspect-ratio:1; display:grid; place-items:center', class: 'muted' }, 'cannot be read') : h('img', { src: thumbUrl(image.id), loading: 'lazy' }),
                    unreadable ? null : pin,
                    image.has_caption ? h('span', { class: 'chip good state', title: 'Has a description' }, '✓') : null,
                    h('div', { class: 'name' }, image.rel),
                    h('div', { class: 'chips' }, image.health.map((b) => {
                        const chip = healthChip(b.check, b.detail);
                        if (['sideways', 'transparency'].includes(b.check)) {
                            chip.classList.add('clickable');
                            chip.title += '. Click to fix this image.';
                            chip.onclick = (e) => {
                                e.stopPropagation();
                                act(null, async () => {
                                    if (!confirm(`Fix ${image.rel} in place? The original is kept in CandyCaption's backup folder.`)) return;
                                    await api('/api/dataset/fix', 'POST', { kind: b.check, id: image.id });
                                    await reload();
                                });
                            };
                        }
                        return chip;
                    })));
            })),
            images.length > shown ? h('div', { style: 'text-align:center; margin-top:12px' }, h('button', { class: 'quiet', onclick: () => { shown += 480; render(); } }, `Show more (${images.length - shown} left)`)) : null);
    }

    function replaceCard() {
        const find = h('input', { type: 'text', placeholder: 'Find', class: 'grow' });
        const replace = h('input', { type: 'text', placeholder: 'Replace with (empty removes it)', class: 'grow' });
        const options = { case: h('input', { type: 'checkbox' }), whole: h('input', { type: 'checkbox' }), regex: h('input', { type: 'checkbox' }) };
        const query = () => ({ find: find.value, replace: replace.value, case: options.case.checked, whole: options.whole.checked, regex: options.regex.checked });
        const results = h('div', { style: 'margin-top:12px' });
        let last = null;

        const showChanges = () => {
            if (!changes) { results.replaceChildren(); return; }
            if (!changes.length) { results.replaceChildren(h('p', { class: 'hint' }, 'No description contains that.')); return; }
            const boxes = changes.map(() => h('input', { type: 'checkbox', checked: true }));
            results.replaceChildren(
                h('div', { class: 'spread', style: 'margin-bottom:8px' }, h('b', null, `${changes.length} descriptions would change`),
                    h('button', {
                        onclick: (e) => act(e.target, async () => {
                            const ids = changes.filter((c, i) => boxes[i].checked).map((c) => c.id);
                            const result = await api('/api/replace/apply', 'POST', { ...last, ids });
                            toast(`${result.changed} descriptions changed. The old versions are in each image's history.`);
                            changes = null; showChanges();
                        }),
                    }, 'Apply to the ticked ones')),
                h('div', { style: 'max-height:420px; overflow:auto' }, changes.map((change, i) => h('div', { class: 'row', style: 'align-items:flex-start; padding:8px 0; border-bottom:1px solid var(--input-border)' },
                    boxes[i], h('div', { class: 'grow' }, h('b', { style: 'font-size:12px' }, change.rel), h('div', { class: 'diff', style: 'margin-top:4px' }, wordDiff(change.before, change.after)))))));
        };

        return h('div', { class: 'card' },
            h('h2', null, candy('Swap the Sprinkles', 'find and replace across every description')),
            h('div', { class: 'row', style: 'margin-top:10px' }, find, replace,
                h('button', { class: 'quiet', onclick: (e) => act(e.target, async () => { last = query(); changes = (await api('/api/replace/preview', 'POST', last)).changes; showChanges(); }) }, 'Preview changes')),
            h('div', { class: 'row', style: 'margin-top:8px' },
                h('label', { class: 'check', style: 'margin:0' }, options.case, 'Match case'),
                h('label', { class: 'check', style: 'margin:0' }, options.whole, 'Whole words only'),
                h('label', { class: 'check', style: 'margin:0' }, options.regex, 'Regular expression')),
            results);
    }

    function render() {
        if (!App.dataset.open) { panel.replaceChildren(openBar()); return; }
        panel.replaceChildren(openBar(), healthCard(), grid(), replaceCard());
    }

    App.tabs.dataset = { show: () => act(null, async () => { if (App.dataset.open) await reload(); else render(); }), openFolder };
})();
