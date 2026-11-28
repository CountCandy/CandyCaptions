// Pantry (Dataset): open a folder, run the Freshness Check (health checks), find and replace.

(function () {
    const panel = $('#panel-dataset');
    let filter = null;     // health check key, or null for everything
    let shown = 240;
    let order = localStorage.getItem('pantryOrder') || 'az';
    let view = localStorage.getItem('pantryView') === 'grid' ? 'grid' : 'list';

    async function openFolder(path, button) {
        await act(button, async () => {
            if (button) button.textContent = 'Checking every image…';
            App.dataset = await api('/api/dataset/open', 'POST', { path });
            filter = null; shown = 240;
            if (App.tabs.bench.reset) App.tabs.bench.reset();
            if (App.tabs.review.reset) App.tabs.review.reset();
            if (App.tabs.style && App.tabs.style.reset) App.tabs.style.reset();
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
            h('h2', null, candy('The Jar', 'dataset folder: images, with their tag files next to them')),
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

    // Label Maker: tag files from the image, and the model's check of them.
    let tagScope = 'missing', tagCheck = true;
    function labelCard() {
        const d = App.dataset, job = d.tag_job || {}, running = job.state === 'running';
        const untagged = d.images.length - d.tagged;
        const line = running
            ? `${job.steps > 1 && job.step_number ? `Step ${job.step_number} of ${job.steps}: ` : ''}${job.step || 'Getting ready'}… ${job.step_total ? `${job.step_done} of ${job.step_total} images` : ''}`
            : (job.message || '');
        const scopes = [
            ['missing', `Tag the ${untagged} image${untagged === 1 ? '' : 's'} with no tag file`],
            ['all', 'Tag every image again (old tags are kept in each image\'s history)'],
            ['none', 'Do not tag; only check the tags that are there'],
        ];
        return h('div', { class: 'card' },
            h('h2', null, candy('Label Maker', 'auto-tagger: writes a tag file for each image')),
            h('p', { class: 'hint', style: 'margin:4px 0 10px' },
                `${d.tagged} of ${d.images.length} images have a tag file. `,
                'The tagger reads each image and writes its tags next to it. Then, if you tick the box, your Captioner model looks at the image too and strikes or corrects tags that are clearly wrong. It keeps a tag whenever it is unsure, never adds tags, and never touches the trigger word. Each change is listed with the image and one click undoes it.'),
            h('div', { class: 'row', style: 'align-items:center' },
                h('select', { style: 'width:auto; flex:1; min-width:260px', disabled: running, onchange: (e) => { tagScope = e.target.value; } },
                    scopes.map(([value, label]) => h('option', { value, selected: tagScope === value }, label))),
                h('label', { class: 'check', style: 'margin:0' },
                    h('input', { type: 'checkbox', checked: tagCheck, disabled: running, onchange: (e) => { tagCheck = e.target.checked; } }),
                    'Let the model check the tags', h('span', { class: 'plain' }, 'one request per image')),
                running
                    ? h('button', { class: 'quiet', onclick: (e) => act(e.target, async () => { await api('/api/tagger/stop', 'POST'); }) }, 'Stop')
                    : h('button', { onclick: (e) => act(e.target, async () => {
                        if (tagScope === 'all' && d.tagged && !confirm(`Replace the tag files of all ${d.images.length} images? The old tags stay in each image's history.`)) return;
                        d.tag_job = await api('/api/tagger/run', 'POST', { scope: tagScope, check: tagCheck });
                        render();
                    }) }, 'Make the labels')),
            line ? h('p', { id: 'tagLine', class: job.state === 'failed' ? 'notice' : 'hint', style: 'margin-top:10px' }, line) : h('p', { id: 'tagLine', class: 'hint', style: 'margin-top:10px' }),
            d.tagger.ready ? null : h('p', { class: 'hint', style: 'margin-top:6px' }, d.tagger.local
                ? `The tagger folder ${d.tagger.folder} does not hold model.onnx and selected_tags.csv yet.`
                : `The tagger (${d.tagger.source}, about 1.3 GB) is downloaded the first time you tag. That needs the internet once.`),
            d.tags_changed ? h('p', { class: 'hint', style: 'margin-top:6px' }, `The model changed tags on ${d.tags_changed} image${d.tags_changed === 1 ? '' : 's'}. Each one is marked in the list below.`) : null);
    }

    on('tag_progress', (event) => {
        if (!App.dataset.open) return;
        const was = (App.dataset.tag_job || {}).state;
        App.dataset.tag_job = event;
        if (App.tab !== 'dataset') return;
        if (event.state !== 'running' && was === 'running') { reload(); return; }      // finished: tag files and marks changed
        if (was !== 'running') { render(); return; }
        const line = $('#tagLine');
        if (line) line.textContent = `${event.steps > 1 && event.step_number ? `Step ${event.step_number} of ${event.steps}: ` : ''}${event.step || 'Getting ready'}… ${event.step_total ? `${event.step_done} of ${event.step_total} images` : ''}`;
    });

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
                    h('label', { style: 'margin:0' }, 'View'),
                    h('select', { style: 'width:auto', onchange: (e) => { view = e.target.value; localStorage.setItem('pantryView', view); render(); } },
                        [['list', 'List with hint lines'], ['grid', 'Grid']].map(([value, label]) => h('option', { value, selected: view === value }, label))),
                    h('label', { style: 'margin:0' }, 'Order'),
                    h('select', { style: 'width:auto', onchange: (e) => { order = e.target.value; localStorage.setItem('pantryOrder', order); render(); } },
                        [['az', 'File name, A to Z'], ['za', 'File name, Z to A'], ['todo', 'Not captioned first'], ['problems', 'Most freshness notes first']]
                            .map(([value, label]) => h('option', { value, selected: order === value }, label))))),
            h('p', { class: 'hint', style: 'margin-bottom:10px' }, view === 'list'
                ? 'A hint is given to the model as fact when it looks at that image, in the Taste Test and in the Candy Factory. Leave it empty for images that need no help. Click a picture to see it large; ★ pins it to the Sampler Box (test set).'
                : 'Click an image to see it large and give it a hint. ★ pins it to the Sampler Box (test set).'),
            h('div', { class: view === 'grid' ? 'thumbs' : 'pantry-list' }, images.slice(0, shown).map((image) => {
                const unreadable = image.health.some((b) => b.check === 'unreadable');
                const pin = h('button', {
                    class: 'pin' + (image.pinned ? ' on' : ''), title: 'Pin to the Sampler Box (test set)', onclick: (e) => {
                        e.stopPropagation();
                        act(null, async () => { await api('/api/test/pin', 'POST', { id: image.id, pinned: !image.pinned }); image.pinned = !image.pinned; pin.classList.toggle('on', image.pinned); });
                    },
                }, '★');
                const open = () => { if (!unreadable) viewImage(image.id, image.rel, image.has_caption ? [['Open in the Sorting Table', () => App.tabs.review.open(image.id)]] : [], { value: image.note, onsave: (saved) => { image.note = saved; render(); } }); };
                const picture = unreadable ? h('div', { style: 'aspect-ratio:1; display:grid; place-items:center', class: 'muted' }, 'cannot be read') : h('img', { src: thumbUrl(image.id), loading: 'lazy' });
                // One badge per kind of problem: an image that is a near duplicate of four others shows it once.
                const kinds = [...new Map(image.health.map((b) => [b.check, { check: b.check, detail: image.health.filter((x) => x.check === b.check).map((x) => x.detail).filter(Boolean).join('; ') }])).values()];
                const badges = kinds.map((b) => {
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
                });
                const captioned = image.has_caption ? h('span', { class: 'chip good', title: 'Has a description' }, '✓ captioned') : null;
                if (view === 'list') {
                    // One row per image with its hint line: fill in only the ones the model needs help with.
                    const saved = h('span', { class: 'hint', style: 'min-width:44px' });
                    const hint = h('input', {
                        type: 'text', value: image.note || '', disabled: unreadable, placeholder: 'Hint (optional): one line telling the model what is in this image',
                        onchange: () => act(null, async () => { image.note = (await api(`/api/image/${image.id}/note`, 'PUT', { text: hint.value })).note; hint.value = image.note; saved.textContent = image.note ? 'saved' : 'cleared'; }),
                    });
                    return h('div', { class: 'pantry-row' },
                        h('div', { class: 'thumb', title: 'Click to see the image at full size', onclick: open }, picture, unreadable ? null : pin),
                        h('div', { style: 'min-width:0' }, h('b', { title: `${image.width}×${image.height}` }, image.rel), h('div', { class: 'chips', style: 'margin-top:4px' }, captioned, badges),
                            tagChanges(image.id, image.tagging, (reply) => { image.tagging = reply.tagging && (reply.tagging.removed.length || reply.tagging.replaced.length || reply.tagging.refused) ? reply.tagging : null; render(); })),
                        hint, saved);
                }
                return h('div', { class: 'thumb', title: `${image.rel} · ${image.width}×${image.height}`, onclick: open },
                    picture, unreadable ? null : pin,
                    h('span', { class: 'state chips' },
                        image.has_caption ? h('span', { class: 'chip good', title: 'Has a description' }, '✓') : null,
                        image.note ? h('span', { class: 'chip neutral', style: 'background:rgba(0,0,0,0.6)', title: `Hint: ${image.note}` }, '✎ hint') : null),
                    h('div', { class: 'name' }, image.rel),
                    h('div', { class: 'chips' }, badges));
            })),
            images.length > shown ? h('div', { style: 'text-align:center; margin-top:12px' }, h('button', { class: 'quiet', onclick: () => { shown += 480; render(); } }, `Show more (${images.length - shown} left)`)) : null);
    }

    function render() {
        if (!App.dataset.open) { panel.replaceChildren(openBar()); return; }
        panel.replaceChildren(openBar(), healthCard(), labelCard(), grid());
    }

    App.tabs.dataset = { show: () => act(null, async () => { if (App.dataset.open) await reload(); else render(); }), openFolder };
})();
