// Candy Shop (Connect): pick the model server and models, and test what the server supports.

(function () {
    const panel = $('#panel-connect');
    let models = [];
    let findings = null;

    async function save(patch) {
        const data = await api('/api/settings', 'PUT', patch);
        Object.assign(App, { settings: data.settings, caps: data.caps });
        refreshHeader();
    }

    async function loadModels(button) {
        await act(button, async () => {
            models = (await api('/api/connect/models', 'POST')).models;
            if (!models.length) toast('The server is up but lists no models. Load one there first.', true);
            render();
        });
    }

    function modelSelect(key, emptyLabel) {
        const current = App.settings[key] || '';
        const options = [...new Set([...(current ? [current] : []), ...models])];
        return h('select', { onchange: (e) => act(null, () => save({ [key]: e.target.value }).then(render)) },
            h('option', { value: '' }, emptyLabel),
            options.map((m) => h('option', { value: m, selected: m === current }, m)));
    }

    function render() {
        const s = App.settings, profile = App.backends[s.backend] || {}, caps = App.caps;
        const SAMPLING = ['temperature', 'top_p', 'top_k', 'min_p', 'repeat_penalty', 'seed', 'max_tokens', 'thinking'];
        const probs = caps.chat_logprobs ? 'on chat completions' : caps.responses_logprobs ? 'on /v1/responses' : null;

        const server = h('div', { class: 'card' },
            h('h2', null, candy('Model server', 'where your models live')),
            h('p', { class: 'hint', style: 'margin-bottom:14px' }, 'Images only ever go to the server you connect here, normally on this PC.'),
            h('div', { class: 'grid2' },
                h('div', { class: 'field' }, h('label', null, 'Backend'),
                    h('select', {
                        onchange: (e) => act(null, async () => {
                            models = []; findings = null;
                            await save({ backend: e.target.value, base_url: App.backends[e.target.value].default_url, model: '', verifier_model: '' });
                            render();
                        }),
                    }, Object.entries(App.backends).map(([id, b]) => h('option', { value: id, selected: id === s.backend }, b.name)))),
                h('div', { class: 'field' }, h('label', null, 'Base URL'),
                    h('input', { type: 'text', value: s.base_url, onchange: (e) => act(null, () => { findings = null; return save({ base_url: e.target.value.trim() }).then(render); }) }))),
            h('div', { class: 'field' + (profile.api_key === 'none' ? ' grayed' : '') },
                h('label', null, 'API key', h('span', { class: 'plain' }, profile.api_key === 'required' ? 'required by this backend' : profile.api_key === 'none' ? 'not used by this backend' : 'optional')),
                h('input', { type: 'password', value: s.api_key, placeholder: profile.api_key === 'required' ? 'sk-unsloth-…' : '', onchange: (e) => act(null, () => save({ api_key: e.target.value.trim() })) })),
            h('div', { class: 'grid2' },
                h('div', { class: 'field' }, h('label', null, candy('Captioner', 'the model that describes each image')), modelSelect('model', 'choose a model…')),
                h('div', { class: 'field' }, h('label', null, candy('Taste Tester', 'verifier model for field checks')), modelSelect('verifier_model', 'same as the captioner'))),
            h('div', { class: 'row' },
                h('button', { class: 'quiet', onclick: (e) => loadModels(e.target) }, 'Load model list'),
                h('button', {
                    onclick: (e) => act(e.target, async () => {
                        e.target.textContent = 'Sending one tiny image…';
                        const data = await api('/api/connect/test', 'POST');
                        Object.assign(App, { settings: data.settings, caps: data.caps });
                        findings = data.lines;
                        if (data.models.length) models = data.models;
                        refreshHeader();
                        render();
                    }),
                }, 'Test connection')),
            findings ? h('ul', { class: 'findings', style: 'margin-top:14px' },
                findings.map((line) => h('li', { class: line.ok === true ? 'yes' : line.ok === false ? 'no' : '' }, line.text))) : null,
            profile.notes ? h('p', { class: 'hint', style: 'margin-top:12px' }, profile.notes) : null);

        const honors = h('div', { class: 'card' },
            h('h2', null, 'What this server honors'),
            h('p', { class: 'hint', style: 'margin-bottom:10px' }, 'A setting the backend ignores is grayed out here and in the sampling controls, and is never sent.'),
            h('div', { class: 'chips' }, SAMPLING.map((name) => {
                const yes = (profile.sampling || []).includes(name) && (name !== 'thinking' || profile.thinking);
                return h('span', { class: 'chip ' + (yes ? 'good' : 'neutral grayed'), title: yes ? 'sent' : 'not sent to this backend' }, yes ? '✓' : '–', ' ', name);
            })),
            h('p', { style: 'margin-top:12px' }, 'Token probabilities: ',
                probs ? h('span', { class: 'chip good' }, '✓ ', probs) : h('span', { class: 'chip neutral' }, '– ', caps.tested ? 'not available' : 'not tested yet')),
            h('p', { class: 'hint', style: 'margin-top:6px' }, probs
                ? 'Mixed Flavors (uncertain words) and measured Taste Check (field check) scores are on.'
                : 'Without them, Mixed Flavors (uncertain words) is off and Taste Check (field checks) falls back to plain yes or no.'));

        const tokenizerStatus = h('span', { class: 'hint' }, 'checking…');
        api('/api/tokenizer').then((t) => {
            tokenizerStatus.replaceChildren(h('span', { class: 'chip ' + (t.exact ? 'good' : 'health') }, t.exact ? '✓ exact' : '≈ approximate'), ' ', t.source);
        }).catch(() => {});
        const tokenizer = h('div', { class: 'card' },
            h('h2', null, candy('Token counter', 'checks captions against the trainer\'s limit')),
            h('div', { class: 'field' }, h('label', null, 'Local tokenizer folder', h('span', { class: 'plain' }, 'optional, works offline')),
                h('input', { type: 'text', value: s.tokenizer_path, placeholder: 'for example the tokenizer folder BluFlow bundles', onchange: (e) => act(null, () => save({ tokenizer_path: e.target.value.trim() }).then(render)) })),
            h('div', { class: 'grid2' },
                h('div', { class: 'field' }, h('label', null, 'Download from', h('span', { class: 'plain' }, 'Hugging Face repo, first run only')),
                    h('input', { type: 'text', value: s.tokenizer_repo, onchange: (e) => act(null, () => save({ tokenizer_repo: e.target.value.trim() }).then(render)) })),
                h('div', { class: 'field' }, h('label', null, 'Token limit', h('span', { class: 'plain' }, 'BluFlow default 512')),
                    h('input', { type: 'number', min: 32, max: 4096, value: s.token_limit, onchange: (e) => act(null, () => save({ token_limit: Number(e.target.value) || 512 })) }))),
            h('p', null, 'Now using: ', tokenizerStatus));

        const w = s.score_weights;
        const weight = (key, label) => h('div', { class: 'field' }, h('label', null, candy(FLAGS[key].candy, label)),
            h('input', { type: 'number', step: 0.05, min: 0, max: 1, value: w[key], onchange: (e) => act(null, () => save({ score_weights: { ...App.settings.score_weights, [key]: Number(e.target.value) } })) }));
        const advanced = h('details', { class: 'card' },
            h('summary', null, 'Review score weights and thresholds'),
            h('p', { class: 'hint', style: 'margin-bottom:10px' }, 'The review queue sorts by a weighted sum of the four signals. These are starting points to tune on real datasets.'),
            h('div', { class: 'grid4' }, weight('uncertain', 'uncertain words'), weight('checks', 'field checks'), weight('tags', 'tag contradictions'), weight('template', 'template repetition')),
            h('div', { class: 'grid2' },
                h('div', { class: 'field' }, h('label', null, 'Flag a word when an alternative reaches', h('span', { class: 'plain' }, 'probability')),
                    h('input', { type: 'number', step: 0.05, min: 0.01, max: 0.5, value: s.uncertain_min_alt, onchange: (e) => act(null, () => save({ uncertain_min_alt: Number(e.target.value) })) })),
                h('div', { class: 'field' }, h('label', null, 'Flag a field when "yes" falls below', h('span', { class: 'plain' }, 'probability')),
                    h('input', { type: 'number', step: 0.05, min: 0.05, max: 0.95, value: s.check_threshold, onchange: (e) => act(null, () => save({ check_threshold: Number(e.target.value) })) }))));

        const level = h('output', null, `${Math.round(soundVolume() * 100)}%`);
        const sounds = h('div', { class: 'card' },
            h('h2', null, candy('Finish sound', 'gumdrops when a taste test or a batch finishes')),
            h('label', { class: 'check', style: 'margin-top:10px' },
                h('input', { type: 'checkbox', checked: soundOn(), onchange: (e) => setSound(e.target.checked) }), 'Play the sound'),
            h('div', { class: 'slider-row' + (soundOn() ? '' : ' grayed'), style: 'grid-template-columns: 70px 1fr 48px auto' },
                h('span', null, 'Volume'),
                h('input', { type: 'range', min: 0, max: 1, step: 0.05, value: soundVolume(), disabled: !soundOn(), oninput: (e) => { localStorage.setItem('volume', e.target.value); level.textContent = `${Math.round(e.target.value * 100)}%`; } }),
                level,
                h('button', { class: 'quiet small', title: 'Plays even when the sound is switched off', onclick: () => playGumdrops(true) }, 'Test sound')),
            h('p', { class: 'hint' }, 'Kept in this browser. The bell in the header also switches it on and off.'));

        panel.replaceChildren(h('div', { class: 'grid2' }, h('div', null, server, sounds), h('div', null, honors, tokenizer, advanced)));
    }

    App.tabs.connect = { show: render };
})();
