// This dataset's recipe: prefill, trigger word, extra instructions and sampling controls.
// Shown on both the Taste Test and Candy Factory tabs; both edit the same saved settings.

const SLIDERS = [
    // key, label, min, max, step, backend setting it depends on, note
    ['temperature', 'Temperature', 0, 1.5, 0.05, 'temperature', 'Higher values vary wording but risk invented details'],
    ['top_p', 'Top P', 0, 1, 0.01, 'top_p', ''],
    ['top_k', 'Top K', 0, 100, 1, 'top_k', '0 turns it off'],
    ['min_p', 'Min P', 0, 0.3, 0.01, 'min_p', ''],
    ['repeat_penalty', 'Repeat penalty', 1, 1.3, 0.01, 'repeat_penalty', ''],
    ['max_tokens_fields', 'Max tokens, fields', 200, 1500, 50, 'max_tokens', 'Twelve fields of JSON can pass 300 tokens'],
    ['max_tokens_compose', 'Max tokens, caption', 100, 600, 10, 'max_tokens', 'A 70-word caption is about 100 tokens'],
    ['long_side', 'Image long side (px)', 768, 2048, 64, null, 'Smaller is faster; larger catches small details'],
];

let saveTimer = null;

// Send any pending edits now. Called before a test or a run starts.
async function flushDatasetSettings() {
    clearTimeout(saveTimer);
    saveTimer = null;
    if (App.dataset.open) await api('/api/dataset/settings', 'PUT', App.dataset.settings);
}

function patchDataset(patch) {
    const settings = App.dataset.settings;
    for (const [key, value] of Object.entries(patch)) {
        if (key === 'sampling') Object.assign(settings.sampling, value); else settings[key] = value;
    }
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => flushDatasetSettings().catch((err) => toast(err.message, true)), 350);
}

function renderRecipe(container) {
    const settings = App.dataset.settings;
    const profile = App.backends[App.settings.backend] || { sampling: [] };
    const prefill = App.prefills.find((p) => p.id === settings.prefill_id) || App.prefills[0];
    const rerender = () => renderRecipe(container);
    const honored = (name) => !name || profile.sampling.includes(name);

    const slider = ([key, label, min, max, step, needs, note]) => {
        const off = !honored(needs);
        const number = h('input', { type: 'number', min, max, step, value: settings.sampling[key], disabled: off });
        const range = h('input', { type: 'range', min, max, step, value: settings.sampling[key], disabled: off });
        const set = (value) => {
            range.value = number.value = value;
            patchDataset({ sampling: { [key]: Number(value) }, preset: ['long_side', 'max_tokens_fields', 'max_tokens_compose'].includes(key) ? settings.preset : 'Custom' });
            const select = $('.preset-select', container);
            if (select) select.value = settings.preset;
        };
        range.oninput = () => set(range.value);
        number.onchange = () => set(number.value);
        return h('div', { class: 'slider-row' + (off ? ' grayed' : ''), title: off ? `${profile.name} ignores this setting, so it is not sent` : note },
            h('span', null, label), range, number);
    };

    const onToggles = prefill.toggles.filter((t) => t.on).map((t) => t.label);
    const custom = settings.preset === 'Custom';
    const presetNames = App.presets.map((p) => p.name);

    container.replaceChildren(
        h('div', { class: 'grid2' },
            h('div', null,
                h('div', { class: 'grid2' },
                    h('div', { class: 'field' }, h('label', null, candy('Recipe', 'prefill')),
                        h('select', {
                            onchange: (e) => {
                                const chosen = App.prefills.find((p) => p.id === e.target.value);
                                const preset = App.presets.find((p) => p.name === chosen.preset);
                                patchDataset({ prefill_id: chosen.id, ...(preset ? { preset: preset.name, sampling: preset.sampling } : {}) });
                                rerender();
                            },
                        }, App.prefills.map((p) => h('option', { value: p.id, selected: p.id === prefill.id }, p.name)))),
                    h('div', { class: 'field' }, h('label', null, 'Trigger word', h('span', { class: 'plain' }, 'fills {trigger}')),
                        h('input', { type: 'text', value: settings.trigger, placeholder: prefill.id === 'style' ? 'optional' : 'names the character or concept', oninput: (e) => patchDataset({ trigger: e.target.value }) }))),
                h('p', { class: 'hint', style: 'margin:-4px 0 10px' }, prefill.summary),
                h('div', { class: 'field' }, h('label', null, 'Extra instructions for this dataset', h('span', { class: 'plain' }, 'one per line')),
                    h('textarea', { rows: 4, value: settings.extra_instructions, placeholder: 'The character\'s trigger is kyarine.', oninput: (e) => patchDataset({ extra_instructions: e.target.value }) })),
                h('p', { class: 'hint' }, 'Options switched on: ', onToggles.length ? onToggles.join(' · ') : 'none', '. ',
                    h('a', { href: '#', onclick: (e) => { e.preventDefault(); showTab('prefills'); } }, 'Edit in the Recipe Box'))),
            h('div', null,
                h('div', { class: 'row', style: 'margin-bottom:10px' },
                    h('label', { style: 'margin:0' }, 'Sampling preset'),
                    h('select', {
                        class: 'preset-select', style: 'width:auto; flex:1', onchange: (e) => {
                            const preset = App.presets.find((p) => p.name === e.target.value);
                            if (preset) patchDataset({ preset: preset.name, sampling: preset.sampling });
                            rerender();
                        },
                    }, [...presetNames, ...(custom ? ['Custom'] : [])].map((name) => h('option', { value: name, selected: name === settings.preset }, name))),
                    h('button', {
                        class: 'quiet small', title: 'Save these sampling values as your own preset', onclick: (e) => act(e.target, async () => {
                            const name = prompt('Name for this preset:');
                            if (!name) return;
                            const s = settings.sampling;
                            const sampling = { temperature: s.temperature, top_p: s.top_p, top_k: s.top_k, min_p: s.min_p, repeat_penalty: s.repeat_penalty };
                            const presets = [...App.presets.filter((p) => !p.builtin && p.name !== name), { name, sampling }];
                            App.presets = (await api('/api/presets', 'PUT', { presets })).presets;
                            patchDataset({ preset: name });
                            rerender();
                        }),
                    }, 'Save as preset')),
                SLIDERS.map(slider),
                h('div', { class: 'slider-row' }, h('span', null, 'Seed'),
                    h('span', { class: 'hint' }, 'fixed, so test runs can be compared'),
                    h('input', { type: 'number', value: settings.sampling.seed, disabled: !honored('seed'), onchange: (e) => patchDataset({ sampling: { seed: Number(e.target.value) || 0 } }) })),
                h('label', { class: 'check' + (profile.thinking ? '' : ' grayed'), title: profile.thinking ? '' : `${profile.name} does not take this setting per request. Turn thinking off in the model's own settings there.` },
                    h('input', { type: 'checkbox', checked: settings.sampling.thinking, disabled: !profile.thinking, onchange: (e) => patchDataset({ sampling: { thinking: e.target.checked } }) }),
                    'Thinking', h('span', { class: 'plain' }, 'slower, rarely helps captions')))));
}
