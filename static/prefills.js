// Recipe Box (Prefills): everything that differs between kinds of LoRA, all of it editable.

(function () {
    const panel = $('#panel-prefills');
    let selected = null;   // prefill id
    let draft = null;      // working copy of the selected prefill
    let dirty = false;

    async function load() {
        const data = await api('/api/prefills');
        Object.assign(App, { prefills: data.prefills, presets: data.presets, wordlists: data.wordlists });
    }

    function pick(id) {
        const found = App.prefills.find((p) => p.id === id) || App.prefills[0];
        selected = found.id;
        draft = JSON.parse(JSON.stringify(found));
        dirty = false;
    }

    function touch() {
        dirty = true;
        const button = $('#prefillSave');
        if (button) { button.disabled = false; button.textContent = 'Save recipe'; }
    }

    const bind = (object, key, extra = {}) => ({
        value: object[key] ?? '', oninput: (e) => { object[key] = e.target.type === 'number' ? Number(e.target.value) : e.target.value; touch(); }, ...extra,
    });

    function fieldsTable() {
        const move = (index, by) => {
            const to = index + by;
            if (to < 0 || to >= draft.fields.length) return;
            [draft.fields[index], draft.fields[to]] = [draft.fields[to], draft.fields[index]];
            touch(); render();
        };
        return h('table', null,
            h('tr', null, h('th', null, 'Order'), h('th', null, 'Field'), h('th', null, 'Holds'), h('th', { title: 'Whether this field reaches the caption' }, 'In caption'), h('th', null, 'Caption note'), h('th')),
            draft.fields.map((field, index) => h('tr', null,
                h('td', { style: 'white-space:nowrap' },
                    h('button', { class: 'quiet icon-btn', title: 'Move up', onclick: () => move(index, -1) }, '↑'), ' ',
                    h('button', { class: 'quiet icon-btn', title: 'Move down', onclick: () => move(index, 1) }, '↓')),
                h('td', { style: 'width:20%' }, h('input', { type: 'text', ...bind(field, 'name') }),
                    h('input', { type: 'text', class: 'mono', style: 'margin-top:4px', title: 'Key used in the JSON reply', ...bind(field, 'key') })),
                h('td', { style: 'width:34%' }, h('textarea', { rows: 2, ...bind(field, 'holds') })),
                h('td', null, h('input', { type: 'checkbox', checked: field.in_caption, onchange: (e) => { field.in_caption = e.target.checked; touch(); } })),
                h('td', { style: 'width:26%' }, h('textarea', { rows: 2, placeholder: 'optional, such as "defects only"', ...bind(field, 'caption_note') })),
                h('td', null, h('button', { class: 'quiet icon-btn', title: 'Remove this field', onclick: () => { draft.fields.splice(index, 1); touch(); render(); } }, '✕')))));
    }

    function togglesTable() {
        return h('table', null,
            h('tr', null, h('th', null, 'On'), h('th', null, 'Checkbox label'), h('th', null, 'Line added to the instructions'), h('th', null, 'Used when'), h('th')),
            draft.toggles.map((toggle, index) => h('tr', null,
                h('td', null, h('input', { type: 'checkbox', checked: toggle.on, onchange: (e) => { toggle.on = e.target.checked; touch(); } })),
                h('td', { style: 'width:30%' }, h('input', { type: 'text', ...bind(toggle, 'label') })),
                h('td', null, h('input', { type: 'text', ...bind(toggle, 'line') })),
                h('td', null, h('select', { onchange: (e) => { toggle.stage = e.target.value; touch(); } },
                    [['compose', 'writing the caption'], ['look', 'looking at the image'], ['both', 'both']].map(([value, label]) => h('option', { value, selected: toggle.stage === value }, label)))),
                h('td', null, h('button', { class: 'quiet icon-btn', title: 'Remove', onclick: () => { draft.toggles.splice(index, 1); touch(); render(); } }, '✕')))));
    }

    function wordlistEditor() {
        const area = h('textarea', { class: 'mono', rows: 16, spellcheck: 'false' }, JSON.stringify(App.wordlists, null, 2));
        return h('details', { class: 'card' },
            h('summary', null, 'Word lists and tag map (used by Mixed Flavors and Flavor Clash)'),
            h('p', { class: 'hint', style: 'margin-bottom:8px' },
                'In "categories", words in the same inner list are wordings of one fact (red, crimson); words in different lists are different facts (red, blue). Only these categories are compared when the model wavers. "tag_map" says which booru tags are checked against which field.'),
            area,
            h('div', { class: 'row', style: 'margin-top:10px' },
                h('button', {
                    onclick: (e) => act(e.target, async () => {
                        let parsed;
                        try { parsed = JSON.parse(area.value); } catch (err) { throw new Error(`That is not valid JSON: ${err.message}`); }
                        App.wordlists = (await api('/api/wordlists', 'PUT', { wordlists: parsed })).wordlists;
                        toast('Word lists saved.');
                    }),
                }, 'Save word lists'),
                h('button', {
                    class: 'quiet', onclick: (e) => act(e.target, async () => {
                        if (!confirm('Reset the word lists and tag map to their defaults?')) return;
                        App.wordlists = (await api('/api/wordlists', 'PUT', { reset: true })).wordlists;
                        render();
                    }),
                }, 'Reset to default')));
    }

    function render() {
        if (!draft) pick(selected || 'style');
        const list = h('div', { class: 'card' },
            h('h2', null, candy('Recipes', 'prefills')),
            h('div', { class: 'stack', style: 'margin:10px 0 14px' }, App.prefills.map((p) => h('button', {
                class: p.id === selected ? '' : 'quiet', style: 'width:100%; text-align:left',
                onclick: () => { if (dirty && !confirm('Discard the unsaved changes to this recipe?')) return; pick(p.id); render(); },
            }, p.name, p.builtin ? '' : h('span', { class: 'plain' }, 'variant')))),
            h('div', { class: 'row' },
                h('button', {
                    class: 'quiet small', onclick: (e) => act(e.target, async () => {
                        const name = prompt('Name for the copy, such as "Character, chibi":', `${draft.name} copy`);
                        if (!name) return;
                        const data = await api(`/api/prefills/${selected}/duplicate`, 'POST', { name });
                        App.prefills = data.prefills; pick(data.id); render();
                    }),
                }, 'Duplicate'),
                draft.builtin ? h('button', {
                    class: 'quiet small', onclick: (e) => act(e.target, async () => {
                        if (!confirm(`Reset ${draft.name} to its default text? Your word range and preset are kept; everything else goes back to the default.`)) return;
                        App.prefills = (await api(`/api/prefills/${selected}/reset`, 'POST')).prefills; pick(selected); render();
                    }),
                }, 'Reset to default') : h('button', {
                    class: 'danger small', onclick: (e) => act(e.target, async () => {
                        if (!confirm(`Delete ${draft.name}?`)) return;
                        App.prefills = (await api(`/api/prefills/${selected}`, 'DELETE')).prefills; pick('style'); render();
                    }),
                }, 'Delete')),
            h('h3', { style: 'margin-top:18px' }, 'Share'),
            h('div', { class: 'row' },
                h('a', { href: '/api/share/export', download: 'candycaption-recipes.json' }, h('button', { class: 'quiet small' }, 'Export JSON')),
                h('button', {
                    class: 'quiet small', onclick: () => {
                        const input = h('input', { type: 'file', accept: '.json,application/json' });
                        input.onchange = () => act(null, async () => {
                            const data = await api('/api/share/import', 'POST', JSON.parse(await input.files[0].text()));
                            Object.assign(App, { prefills: data.prefills, presets: data.presets, wordlists: data.wordlists });
                            pick(selected); render(); toast('Recipes imported.');
                        });
                        input.click();
                    },
                }, 'Import JSON')),
            h('p', { class: 'hint', style: 'margin-top:8px' }, 'Exports prefills, your sampling presets and the word lists.'));

        const editor = h('div', null,
            h('div', { class: 'card' },
                h('div', { class: 'spread' },
                    h('h2', null, draft.name, h('span', { class: 'plain' }, draft.builtin ? 'built-in prefill' : 'your variant')),
                    h('button', {
                        id: 'prefillSave', disabled: !dirty, onclick: (e) => act(e.target, async () => {
                            draft.banned_words = draft.banned_words.map((w) => w.trim()).filter(Boolean);
                            draft.rules = draft.rules.map((r) => r.trim()).filter(Boolean);
                            draft.caption_rules = (draft.caption_rules || []).map((r) => r.trim()).filter(Boolean);
                            App.prefills = (await api(`/api/prefills/${selected}`, 'PUT', draft)).prefills;
                            dirty = false; toast(`${draft.name} saved.`); render();
                        }),
                    }, dirty ? 'Save recipe' : 'Saved')),
                h('div', { class: 'grid2', style: 'margin-top:10px' },
                    h('div', { class: 'field' }, h('label', null, 'Name'), h('input', { type: 'text', ...bind(draft, 'name') })),
                    h('div', { class: 'grid3' },
                        h('div', { class: 'field' }, h('label', null, 'Min words'), h('input', { type: 'number', min: 5, max: 400, ...bind(draft, 'min_words') })),
                        h('div', { class: 'field' }, h('label', null, 'Max words'), h('input', { type: 'number', min: 5, max: 400, ...bind(draft, 'max_words') })),
                        h('div', { class: 'field' }, h('label', null, 'Default preset'),
                            h('select', { onchange: (e) => { draft.preset = e.target.value; touch(); } }, App.presets.map((p) => h('option', { value: p.name, selected: p.name === draft.preset }, p.name)))))),
                h('div', { class: 'field' }, h('label', null, 'What this recipe is for'), h('input', { type: 'text', ...bind(draft, 'summary') })),
                h('p', { class: 'hint' }, 'Placeholders work in any text below: ',
                    ['{tags}', '{trigger}', '{filename}', '{min_words}', '{max_words}'].map((p) => [h('code', null, p), ' ']),
                    '. A line that uses {trigger} is left out when the dataset has no trigger word.')),
            h('div', { class: 'card' },
                h('h3', null, 'System prompts'),
                h('div', { class: 'field' }, h('label', null, 'Looking at the image', h('span', { class: 'plain' }, 'fills in the fields')), h('textarea', { rows: 3, ...bind(draft, 'look_prompt') })),
                h('div', { class: 'field' }, h('label', null, 'Writing the caption', h('span', { class: 'plain' }, 'text only, from the fields')), h('textarea', { rows: 3, ...bind(draft, 'compose_prompt') })),
                h('div', { class: 'field' }, h('label', null, 'Rules', h('span', { class: 'plain' }, 'one per line, sent with both requests')),
                    h('textarea', { rows: 8, value: draft.rules.join('\n'), oninput: (e) => { draft.rules = e.target.value.split('\n'); touch(); } })),
                h('div', { class: 'field' }, h('label', null, 'Caption rules', h('span', { class: 'plain' }, 'one per line, only when writing the caption; this is where a recipe says what to leave out')),
                    h('textarea', { rows: 3, value: (draft.caption_rules || []).join('\n'), oninput: (e) => { draft.caption_rules = e.target.value.split('\n'); touch(); } }))),
            h('div', { class: 'card' },
                h('div', { class: 'spread' }, h('h3', null, 'Caption fields'),
                    h('button', { class: 'quiet small', onclick: () => { draft.fields.push({ key: `field_${draft.fields.length + 1}`, name: 'New field', holds: '', in_caption: true, caption_note: '' }); touch(); render(); } }, 'Add a field')),
                h('p', { class: 'hint', style: 'margin-bottom:8px' }, 'The model fills in every field for every image. "In caption" decides which reach the caption, and the caption follows this order.'),
                fieldsTable()),
            h('div', { class: 'card' },
                h('div', { class: 'spread' }, h('h3', null, 'Option toggles'),
                    h('button', { class: 'quiet small', onclick: () => { draft.toggles.push({ id: `toggle_${Date.now()}`, label: 'New option', line: '', stage: 'compose', on: true }); touch(); render(); } }, 'Add an option')),
                togglesTable()),
            h('div', { class: 'card' },
                h('h3', null, candy('Off-Menu Words', 'banned words')),
                h('p', { class: 'hint', style: 'margin-bottom:8px' }, 'One word or phrase per line. If any slip into a caption, it is rewritten once and the cleaner version is kept.'),
                h('textarea', { rows: 6, value: draft.banned_words.join('\n'), oninput: (e) => { draft.banned_words = e.target.value.split('\n'); touch(); } })),
            wordlistEditor());

        panel.replaceChildren(h('div', { style: 'display:grid; grid-template-columns: 250px 1fr; gap:16px; align-items:start' }, list, editor));
    }

    App.tabs.prefills = { show: () => act(null, async () => { if (!dirty) { await load(); draft = null; } render(); }) };
    App.loadPrefills = load;
})();
