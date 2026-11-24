// Shared helpers: DOM building, the API, tabs, themes, live events and the rock-candy bar.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
        if (value === false || value == null) continue;
        if (key === 'class') el.className = value;
        else if (key === 'html') el.innerHTML = value;
        else if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
        else if (key === 'value') el.value = value;
        else if (value === true) el.setAttribute(key, '');
        else el.setAttribute(key, value);
    }
    const add = (child) => {
        if (child == null || child === false) return;
        if (Array.isArray(child)) child.forEach(add);
        else el.append(child.nodeType ? child : document.createTextNode(String(child)));
    };
    children.forEach(add);
    return el;
}

// replaceChildren that skips null and false and flattens arrays, like h() does,
// so a render can write `condition ? node : null` without printing "null".
const nativeReplaceChildren = Element.prototype.replaceChildren;
Element.prototype.replaceChildren = function (...children) {
    return nativeReplaceChildren.apply(this, children.flat(Infinity).filter((child) => child != null && child !== false));
};

// A candy label with its plain name underneath: puns never hide meaning.
const candy = (name, plain) => [name, h('span', { class: 'plain' }, plain)];

const App = {
    settings: null, backends: {}, caps: {}, defaultSampling: {},
    dataset: { open: false, recent: [] },
    prefills: [], presets: [], wordlists: null,
    run: null, tab: null, tabs: {},
};

async function api(path, method = 'GET', body) {
    const options = { method, headers: {} };
    if (body !== undefined) {
        options.headers['Content-Type'] = 'application/json';
        options.body = JSON.stringify(body);
    }
    let response;
    try {
        response = await fetch(path, options);
    } catch (err) {
        throw new Error('CandyCaption is not answering. Is its window still open?');
    }
    if (!response.ok) {
        let detail = `Something went wrong (${response.status}).`;
        try { detail = (await response.json()).detail || detail; } catch (err) { /* not JSON */ }
        throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return response.json();
}

function toast(message, isError = false) {
    const el = h('div', { class: 'toast' + (isError ? ' error' : '') }, message);
    $('#toasts').append(el);
    setTimeout(() => el.remove(), isError ? 9000 : 4000);
}

// Run an action from a button: disable it meanwhile and show any error in plain words.
async function act(button, fn) {
    const label = button ? button.textContent : '';
    if (button) button.disabled = true;
    try {
        return await fn();
    } catch (err) {
        toast(err.message, true);
        return undefined;
    } finally {
        if (button && button.isConnected) { button.disabled = false; button.textContent = label; }
    }
}

function debounce(fn, ms = 500) {
    let timer = null;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

// --- Flags and health badges: each has an icon as well as a color ---
const FLAGS = {
    uncertain: { candy: 'Mixed Flavors', plain: 'uncertain words', icon: '◐' },
    checks: { candy: 'Taste Check', plain: 'field checks', icon: '?' },
    tags: { candy: 'Flavor Clash', plain: 'tag contradictions', icon: '≠' },
    template: { candy: 'Same Old Flavor', plain: 'template repetition', icon: '↻' },
};
const HEALTH_ICONS = {
    too_small: '▫', unreadable: '✗', sideways: '⟳', transparency: '◌', extreme_shape: '▭',
    stale_caption: '⌛', exact_duplicate: '⧉', near_duplicate: '≈', missing_files: '∅', too_many_tokens: '⇥', name_clash: '⚑',
};

function flagChip(signal, withPlain = true) {
    const flag = FLAGS[signal];
    return h('span', { class: `chip ${signal}`, title: `${flag.plain}: ${FLAG_MEANING[signal]}` }, flag.icon, ' ', flag.candy,
        withPlain ? h('span', { class: 'plain' }, flag.plain) : null);
}

function flagIcons(signals) {
    return h('span', { class: 'icons' }, signals.map((s) =>
        h('i', { style: `color: var(--flag-${s})`, title: `${FLAGS[s].candy} (${FLAGS[s].plain}): ${FLAG_MEANING[s]}` }, FLAGS[s].icon)));
}

function healthChip(check, detail, withPlain = false) {
    const info = (App.dataset.health_checks || {})[check] || { name: check, candy: check, why: '' };
    return h('span', { class: 'chip ' + (check === 'unreadable' ? 'bad' : 'health'), title: `${info.name}: ${detail || info.why}` },
        HEALTH_ICONS[check] || '!', ' ', info.candy, withPlain ? h('span', { class: 'plain' }, info.name.toLowerCase()) : null);
}

function formatDuration(seconds) {
    if (seconds == null || !isFinite(seconds) || seconds < 0) return '–';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.round(seconds / 60);
    if (mins < 60) return `${mins} min`;
    const hours = Math.floor(mins / 60), rem = mins % 60;
    return rem ? `${hours}h ${rem}m` : `${hours}h`;
}

// A browser reuses a picture it has already loaded for the same address, whatever the server says.
// Image numbers start again in every dataset, so each address also carries the dataset and the
// file's version; a different dataset or a replaced file is then a different address.
const imageVersions = new WeakMap();
function thumbUrl(id, size = 'thumb') {
    const dataset = App.dataset || {};
    let version = '';
    if (dataset.open) {
        if (!imageVersions.has(dataset)) imageVersions.set(dataset, new Map(dataset.images.map((image) => [image.id, image.v])));
        version = `&v=${dataset.id}-${imageVersions.get(dataset).get(id) || ''}`;
    }
    return `/api/image/${id}?size=${size}${version}`;
}

// Word-level diff of two captions, as a fragment with <ins> and <del>.
function wordDiff(before, after) {
    const a = before.split(/\s+/).filter(Boolean), b = after.split(/\s+/).filter(Boolean);
    const m = a.length, n = b.length;
    const table = Array.from({ length: m + 1 }, () => new Uint16Array(n + 1));
    for (let i = m - 1; i >= 0; i--) {
        for (let j = n - 1; j >= 0; j--) {
            table[i][j] = a[i] === b[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1]);
        }
    }
    const out = document.createDocumentFragment();
    let i = 0, j = 0;
    const push = (tag, word) => { out.append(tag ? h(tag, null, word) : word, ' '); };
    while (i < m && j < n) {
        if (a[i] === b[j]) { push(null, a[i]); i++; j++; }
        else if (table[i + 1][j] >= table[i][j + 1]) push('del', a[i++]);
        else push('ins', b[j++]);
    }
    while (i < m) push('del', a[i++]);
    while (j < n) push('ins', b[j++]);
    return out;
}

// Text with the wavered words wrapped in <mark>. Each item names a word and which occurrence
// of it (nth, counted from 0) the model wavered on, so only that one is marked.
function markWavered(text, items) {
    const spots = [];
    for (const item of items) {
        const re = new RegExp(`(?<![\\w-])${item.word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\w-])`, 'gi');
        let match, seen = 0;
        while ((match = re.exec(text)) !== null) {
            if (item.nth == null || seen === item.nth) spots.push([match.index, match.index + match[0].length, item]);
            seen++;
        }
    }
    spots.sort((a, b) => a[0] - b[0]);
    const out = [];
    let at = 0;
    for (const [start, end, item] of spots) {
        if (start < at) continue;
        out.push(text.slice(at, start), h('mark', { title: `Nearly wrote: ${item.alts.map((a) => `${a.word} (${Math.round(a.p * 100)}%)`).join(', ')}` }, text.slice(start, end)));
        at = end;
    }
    out.push(text.slice(at));
    return out;
}

// What the four flags mean, in plain words. Shown wherever the flag icons appear on their own.
const FLAG_MEANING = {
    uncertain: 'While writing a word, the model nearly wrote a different fact instead, such as orange where it wrote red. Check that one word against the image.',
    checks: 'Asked "is this field accurate?" about the image, the model or the Taste Tester would not say yes.',
    tags: 'A field says one thing and the tag file says another, such as blonde hair against red_hair.',
    template: 'The caption shares long stretches of wording with many other captions in this run.',
};

function flagLegend() {
    return h('details', { class: 'legend' },
        h('summary', null, 'What the flag symbols mean'),
        h('table', null, Object.keys(FLAGS).map((signal) => h('tr', null,
            h('td', { style: 'white-space:nowrap' }, flagChip(signal)),
            h('td', null, FLAG_MEANING[signal])))),
        h('p', { class: 'hint', style: 'margin-top:6px' }, 'A flag is a reason to look, not a verdict. Captions with no symbol raised none of these. Click a caption to open it in the Sorting Table (review).'));
}

function overlay(content, { center = false, onclose } = {}) {
    const close = () => { wrap.remove(); document.removeEventListener('keydown', onKey, true); if (onclose) onclose(); };
    const onKey = (event) => { if (event.key === 'Escape') { event.stopPropagation(); close(); } };
    const wrap = h('div', { class: 'overlay' + (center ? ' center' : ''), onmousedown: (event) => { if (event.target === wrap) close(); } }, content);
    document.addEventListener('keydown', onKey, true);
    document.body.append(wrap);
    return close;
}

// Names compare the way people read them: "ta (2)" before "ta (16)".
const byName = (a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });

// Show one image large. `actions` are extra buttons for the place it was opened from.
// With `hint`, the viewer also has the image's Hint box: { value, onsave(text) }.
function viewImage(id, rel, actions = [], hint = null) {
    let close = () => {};
    const buttons = actions.map(([label, fn]) => h('button', { class: 'quiet small', onclick: () => { close(); fn(); } }, label));
    close = overlay(h('div', { class: 'modal', style: 'width:min(1500px,100%)' },
        h('div', { class: 'spread', style: 'margin-bottom:10px' }, h('b', null, rel),
            h('span', { class: 'row' }, buttons, h('button', { class: 'quiet small', onclick: () => close() }, 'Close'))),
        hint ? hintBox(id, hint.value, hint.onsave, { style: 'margin-bottom:10px' }) : null,
        h('img', { class: 'zoom', src: thumbUrl(id, 'full'), alt: rel })), { center: true });
}

// The Hint box: one line telling the model what is in this image. It is sent to the model as fact
// whenever the image is captioned, and saved the moment you click away.
function hintBox(id, value, onsave, attrs = {}) {
    const status = h('span', { class: 'plain' }, 'one line telling the model what is in this image');
    const area = h('textarea', {
        rows: 2, value: value || '', placeholder: 'An elf girl sits on a tree stump and plays an ocarina.',
        onchange: () => act(null, async () => {
            const saved = (await api(`/api/image/${id}/note`, 'PUT', { text: area.value })).note;
            status.textContent = saved ? 'saved; used the next time this image is captioned' : 'cleared';
            if (onsave) onsave(saved);
        }),
    });
    return h('div', attrs, h('label', null, 'Hint', status), area);
}

// --- Tabs ---
function showTab(name) {
    App.tab = name;
    $$('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === name));
    $$('.panel').forEach((panel) => panel.classList.toggle('active', panel.id === `panel-${name}`));
    localStorage.setItem('tab', name);
    if (App.tabs[name] && App.tabs[name].show) App.tabs[name].show();
}

// The friendly "nothing here yet" panel for tabs that need a dataset.
function needsDataset(panel) {
    if (App.dataset.open) return false;
    panel.replaceChildren(h('div', { class: 'empty' },
        'The jar is empty. Point me at a folder of images.',
        h('div', null, h('button', { onclick: () => showTab('dataset') }, 'Open the Pantry'))));
    return true;
}

function refreshHeader() {
    const s = App.settings || {};
    const taster = s.verifier_model && s.verifier_model !== s.model ? ` · Taste Tester: ${s.verifier_model}` : '';
    $('#modelPillText').textContent = s.model ? `Captioner: ${s.model}${taster}` : 'no model chosen';
    $('#modelDot').classList.toggle('on', Boolean(s.model && App.caps.tested));
    $('#modelPill').title = s.model ? (App.caps.tested ? 'Connection tested' : 'Not tested yet. Open the Candy Shop tab') : 'Open the Candy Shop tab';
    const d = App.dataset;
    $('#datasetPill').textContent = d.open ? `📁 ${d.folder.split(/[\\/]/).pop()} · ${d.images.length} images` : '📁 no dataset open';
    if (d.open) $('#datasetPill').title = d.folder;
}

// --- Themes (cycles candy → dark → light) ---
const THEMES = ['candy', 'dark', 'light'];
const THEME_ICONS = { dark: '🌙', light: '☀️', candy: '🍬' };
function setTheme(theme) {
    if (!THEMES.includes(theme)) theme = 'candy';
    document.documentElement.setAttribute('data-theme', theme);
    $('#themeToggle').textContent = THEME_ICONS[theme];
    $('#themeToggle').title = `Theme: ${theme} — click to switch`;
    localStorage.setItem('theme', theme);
}

// --- The finish sound: a handful of gumdrops dropping into a jar ---
// Made with the browser's own tone generator, so there is no sound file to ship.
let audio = null;
const soundOn = () => localStorage.getItem('sound') !== 'off';
const soundVolume = () => { const v = parseFloat(localStorage.getItem('volume')); return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0.6; };
function wakeAudio() {
    try {
        audio = audio || new (window.AudioContext || window.webkitAudioContext)();
        if (audio.state === 'suspended') audio.resume();
    } catch (err) { audio = null; }
}
// Browsers only allow sound after a click, so the generator is woken on the first one.
document.addEventListener('pointerdown', wakeAudio, { once: true });

function playGumdrops(force = false) {
    if (!soundOn() && !force) return;
    wakeAudio();
    if (!audio) return;
    const start = audio.currentTime + 0.03;
    const loudness = soundVolume() / 0.6;   // 0.6 is the level the drops were tuned at
    if (loudness <= 0) return;
    // Each drop is a short tone that falls in pitch (the "plop"), then a quieter bounce.
    const drops = [[0.00, 880], [0.14, 700], [0.25, 990], [0.41, 620], [0.52, 830], [0.72, 560]];
    for (const [at, pitch] of drops) {
        for (const [delay, level, scale] of [[0, 0.22, 1], [0.085, 0.07, 0.8]]) {
            const tone = audio.createOscillator(), volume = audio.createGain();
            const t = start + at + delay;
            tone.type = 'sine';
            tone.frequency.setValueAtTime(pitch * scale * 1.3, t);
            tone.frequency.exponentialRampToValueAtTime(pitch * scale * 0.55, t + 0.1);
            volume.gain.setValueAtTime(0.0001, t);
            volume.gain.exponentialRampToValueAtTime(Math.max(0.0002, level * loudness), t + 0.008);
            volume.gain.exponentialRampToValueAtTime(0.0001, t + 0.16);
            tone.connect(volume).connect(audio.destination);
            tone.start(t);
            tone.stop(t + 0.18);
        }
    }
}

function setSound(on) {
    localStorage.setItem('sound', on ? 'on' : 'off');
    const button = $('#soundToggle');
    if (App.tab === 'connect' && App.tabs.connect) App.tabs.connect.show();
    button.textContent = on ? '🔔' : '🔕';
    button.title = on ? 'Finish sound is on (gumdrops when a test or batch finishes). Click to turn it off.' : 'Finish sound is off. Click to turn it on.';
}

// --- Live events ---
const eventHandlers = {};
function on(type, fn) { (eventHandlers[type] = eventHandlers[type] || []).push(fn); }
function connectEvents() {
    const source = new EventSource('/api/events');
    source.onmessage = (message) => {
        const event = JSON.parse(message.data);
        (eventHandlers[event.type] || []).forEach((fn) => fn(event));
    };
}

// Fields as they stream in: read whatever key/value pairs are complete enough to show.
function partialFields(text) {
    const out = {};
    const re = /"([a-z0-9_]+)"\s*:\s*"((?:[^"\\]|\\.)*)("?)/gi;
    let match;
    while ((match = re.exec(text || '')) !== null) out[match[1]] = match[2].replace(/\\"/g, '"').replace(/\\n/g, ' ');
    return out;
}

// --- Rock-candy crust, from Candy Echo: a deterministic crystal field drawn once as inline SVG,
// then revealed left to right by growing the clip window. ---
function mulberry32(a) {
    return function () {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
const _cClamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const _cLerp = (a, b, t) => a + (b - a) * t;
// Color flows along the stick: bubblegum pink -> orchid -> grape.
const _CRUST_STOPS = [[0, [255, 99, 176]], [0.5, [201, 92, 238]], [1, [167, 89, 245]]];
function _colorAt(t) {
    t = _cClamp(t, 0, 1);
    for (let i = 1; i < _CRUST_STOPS.length; i++) {
        if (t <= _CRUST_STOPS[i][0]) {
            const [t0, c0] = _CRUST_STOPS[i - 1], [t1, c1] = _CRUST_STOPS[i];
            const k = (t - t0) / (t1 - t0);
            return [_cLerp(c0[0], c1[0], k), _cLerp(c0[1], c1[1], k), _cLerp(c0[2], c1[2], k)];
        }
    }
    return _CRUST_STOPS[_CRUST_STOPS.length - 1][1];
}
const _cRgba = (c, a) => 'rgba(' + (c[0] | 0) + ',' + (c[1] | 0) + ',' + (c[2] | 0) + ',' + a + ')';
const _cShade = (c, f) => [_cClamp(c[0] * f, 0, 255), _cClamp(c[1] * f, 0, 255), _cClamp(c[2] * f, 0, 255)];
const _cMix = (c, d, t) => [_cLerp(c[0], d[0], t), _cLerp(c[1], d[1], t), _cLerp(c[2], d[2], t)];

// One faceted quartz-like crystal: body + lit facet + shadow facet + tip glint.
function _crystal(cx, cy, size, rot, base, rng) {
    const w = size * (0.46 + 0.34 * rng()), h = size * (1.05 + 0.75 * rng());
    const jg = (rng() - 0.5) * 0.18;
    const bottom = [w * 0.10 * jg, h / 2];
    const P = [[0, -h / 2], [w / 2, -h * 0.20], [w * 0.40, h * 0.30], bottom, [-w * 0.40, h * 0.28], [-w / 2, -h * 0.18]];
    const fmt = (p) => p[0].toFixed(1) + ',' + p[1].toFixed(1);
    const body = P.map(fmt).join(' ');
    const litFace = [P[0], P[5], P[4], bottom].map(fmt).join(' ');
    const shFace = [P[0], P[1], P[2], bottom].map(fmt).join(' ');
    const lit = _cShade(base, 1.28), dk = _cShade(base, 0.72);
    let g = '<g transform="translate(' + cx.toFixed(1) + ',' + cy.toFixed(1) + ') rotate(' + rot.toFixed(1) + ')">';
    g += '<polygon points="' + body + '" fill="' + _cRgba(base, 0.60) + '" stroke="' + _cRgba(_cShade(base, 1.5), 0.7) + '" stroke-width="0.5"/>';
    g += '<polygon points="' + litFace + '" fill="' + _cRgba(lit, 0.55) + '"/>';
    g += '<polygon points="' + shFace + '" fill="' + _cRgba(dk, 0.5) + '"/>';
    if (rng() > 0.45) {
        const gl = (w * 0.16).toFixed(1);
        g += '<polygon points="0,' + (-h / 2).toFixed(1) + ' ' + gl + ',' + (-h * 0.24).toFixed(1) + ' ' + (-gl) + ',' + (-h * 0.24).toFixed(1) + '" fill="rgba(255,255,255,0.75)"/>';
    }
    return g + '</g>';
}

function buildCrustSVG(W, H) {
    const rng = mulberry32(0x1F5A2C);       // fixed seed -> stable crust for a given width
    const cy = H / 2, stickH = 5;
    let s = '<svg class="gen-crust" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + W + ' ' + H
        + '" preserveAspectRatio="none" style="width:' + W + 'px;height:100%">'
        + '<defs>'
        + '<linearGradient id="genCore" x1="0" x2="1"><stop offset="0" stop-color="#ff63b0"/>'
        + '<stop offset="0.5" stop-color="#c95cee"/><stop offset="1" stop-color="#a759f5"/></linearGradient>'
        + '<filter id="genSoft" x="-20%" y="-60%" width="140%" height="220%"><feGaussianBlur stdDeviation="2.6"/></filter>'
        + '</defs>';
    // Coated spine (a soft glow copy + the solid core) spanning the full stick;
    // the outer clip window decides how much of it shows.
    s += '<rect x="0" y="' + (cy - 8) + '" width="' + W + '" height="16" rx="8" fill="url(#genCore)" opacity="0.4" filter="url(#genSoft)"/>';
    s += '<rect x="0" y="' + (cy - stickH / 2) + '" width="' + W + '" height="' + stickH + '" rx="' + (stickH / 2) + '" fill="url(#genCore)"/>';

    // Dense crystal field across the whole stick; small grains first, big last.
    const crystals = [];
    const STEP = 6.2;
    for (let x = 2; x <= W; x += STEP) {
        const per = 2 + Math.floor(rng() * 2);
        for (let k = 0; k < per; k++) {
            const cxj = x + (rng() - 0.5) * STEP * 1.7;
            const size = 3 + Math.pow(rng(), 1.9) * 12;
            const spread = 3 + (size / 15) * 7 + rng() * 5;
            const cyj = cy + (rng() < 0.5 ? -1 : 1) * rng() * spread;
            const rot = (rng() - 0.5) * 80;
            let base = _colorAt(cxj / W);
            base = _cShade(base, 0.82 + rng() * 0.4);
            if (rng() < 0.10) base = _cMix(base, [235, 255, 248], 0.55);  // occasional white sparkle grain
            crystals.push({ x: cxj, y: cyj, size, rot, base });
        }
    }
    crystals.sort((a, b) => a.size - b.size);
    for (const c of crystals) s += _crystal(c.x, c.y, c.size, c.rot, c.base, rng);

    // Larger turquoise milestone gem at every 5% mark, on top with a soft glow.
    const TURQ = [64, 224, 208];
    for (let i = 1; i <= 20; i++) {
        const px = W * (i * 5) / 100;
        const mr = mulberry32(0x7EA1 + i * 997);
        const size = 18 + mr() * 4;          // a little bigger than the crust crystals
        const rot = (mr() - 0.5) * 34;
        const y = cy - 2 - mr() * 3;
        s += '<ellipse cx="' + px.toFixed(1) + '" cy="' + cy + '" rx="' + (size * 0.72).toFixed(1) + '" ry="' + (size * 0.85).toFixed(1) + '" fill="rgba(64,224,208,0.30)" filter="url(#genSoft)"/>';
        s += _crystal(px, y, size, rot, TURQ, mr);
    }
    return s + '</svg>';
}
