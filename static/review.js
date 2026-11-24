// Sorting Table (Review): the captions most likely to be wrong, worst first, and Sweet Talk (review chat).

(function () {
    const panel = $('#panel-review');
    let rows = [], totals = { done: 0, reviewed: 0, flagged: 0, failed: 0 };
    let query = '';
    let sortBy = localStorage.getItem('reviewSort') === 'name' ? 'name' : 'score';
    let currentId = null, detail = null;
    let chatOpen = false;

    const QUICK = [
        ['Wrong color', 'The  is , not .'],
        ['Wrong count', 'There are , not .'],
        ['Missed something', 'You missed: '],
        ['Too long', 'Too long. Keep only the essentials in each field.'],
        ['Mentions the style', 'It mentions the art style. Take style words out of every field except Style and medium.'],
    ];

    function setCount() {
        const badge = $('#reviewCount');
        badge.hidden = !totals.flagged;
        badge.textContent = totals.flagged;
        badge.title = `${totals.flagged} flagged captions not reviewed yet`;
    }

    async function loadList() {
        const data = await api('/api/review?q=' + encodeURIComponent(query) + '&sort=' + sortBy);
        rows = data.rows; totals = data.totals;
        setCount();
    }

    async function select(id, { focusList = true } = {}) {
        currentId = id;
        detail = id == null ? null : await api(`/api/review/${id}`);
        renderList();
        renderDetail();
        if (focusList) {
            const active = $('.queue-item.active', panel);
            if (active) active.scrollIntoView({ block: 'nearest' });
        }
    }

    const indexOfCurrent = () => rows.findIndex((r) => r.id === currentId);
    function step(by) {
        if (!rows.length) return;
        const next = Math.max(0, Math.min(rows.length - 1, indexOfCurrent() + by));
        if (rows[next].id !== currentId) act(null, () => select(rows[next].id));
    }
    function nextUnreviewed() {
        const start = indexOfCurrent();
        for (let offset = 1; offset <= rows.length; offset++) {
            const row = rows[(start + offset) % rows.length];
            if (!row.reviewed && row.status !== 'failed') { act(null, () => select(row.id)); return; }
        }
        toast('Nothing left unreviewed in this list.');
    }

    async function approve() {
        if (!detail) return;
        const reviewed = !detail.reviewed;
        await api(`/api/review/${detail.id}/approve`, 'POST', { reviewed });
        const index = indexOfCurrent();
        const keepsReviewed = /reviewed:/i.test(query);
        totals.reviewed += reviewed ? 1 : -1;
        if (detail.reasons.length) totals.flagged += reviewed ? -1 : 1;
        setCount();
        if (index >= 0) {
            rows[index].reviewed = reviewed;
            if (reviewed && !keepsReviewed) rows.splice(index, 1);
        }
        if (reviewed && index >= 0) {
            const next = keepsReviewed ? (rows[index + 1] || rows[index]) : rows[Math.min(index, rows.length - 1)];
            await select(next ? next.id : null);
        } else {
            await select(detail.id);
        }
    }

    async function saveText(kind, text) {
        if (!detail || text.trim() === (kind === 'caption' ? detail.caption : detail.tags)) return;
        detail = await api(`/api/review/${detail.id}/${kind}`, 'PUT', { text });
        const row = rows.find((r) => r.id === detail.id);
        if (row) Object.assign(row, { caption: detail.caption, score: detail.score, reasons: detail.reasons, words: detail.words, tokens: detail.tokens });
        renderList();
        renderDetail();
    }

    function refreshFrom(fresh) {
        detail = fresh;
        const row = rows.find((r) => r.id === fresh.id);
        if (row) Object.assign(row, { caption: fresh.caption, score: fresh.score, reasons: fresh.reasons, words: fresh.words, tokens: fresh.tokens, status: fresh.status });
        if (sortBy === 'score') rows.sort((a, b) => (a.status === 'failed' ? 0 : 1) - (b.status === 'failed' ? 0 : 1) || b.score - a.score);
        renderList();
        renderDetail();
    }

    // --- Sweet Talk (review chat) ---
    function openChat() {
        if (!detail || chatOpen) return;
        if (!detail.fields) { toast('This image has no stored fields yet. Regenerate it first.', true); return; }
        chatOpen = true;
        const image = detail;
        const messages = [];      // lives only while the drawer is open
        let proposal = null;
        const log = h('div', { class: 'chat-log' });
        const input = h('textarea', { rows: 2, placeholder: 'She\'s holding a phone, not a cup, and the scarf is orange.' });
        const ruleSlot = h('div');
        const names = Object.fromEntries(image.field_defs.map((f) => [f.key, f.name]));

        const send = (button) => act(button, async () => {
            const text = input.value.trim();
            if (!text) return;
            messages.push({ role: 'user', content: text });
            log.append(h('div', { class: 'bubble user' }, text));
            input.value = '';
            const waiting = h('div', { class: 'bubble model hint' }, 'Rewriting…');
            log.append(waiting);
            waiting.scrollIntoView({ block: 'nearest' });
            try {
                proposal = await api(`/api/review/${image.id}/chat`, 'POST', { messages });
            } catch (err) {
                messages.pop();
                waiting.remove();
                throw err;
            }
            messages.push({ role: 'assistant', content: proposal.assistant });
            const changed = Object.keys(proposal.fields).filter((k) => proposal.fields[k] !== (image.fields[k] || ''));
            const mine = proposal;
            const buttons = h('div', { class: 'row', style: 'margin-top:10px' },
                h('button', {
                    class: 'small', onclick: (e) => act(e.target, async () => {
                        e.target.textContent = 'Saving and re-checking…';
                        const fresh = await api(`/api/review/${image.id}/chat/accept`, 'POST', { fields: mine.fields, caption: mine.caption });
                        refreshFrom(fresh);
                        buttons.replaceChildren(h('span', { class: 'chip good' }, '✓ saved'));
                        showRule(text);
                    }),
                }, 'Accept'),
                h('button', { class: 'quiet small', onclick: () => { buttons.replaceChildren(h('span', { class: 'chip neutral' }, 'rejected')); proposal = null; } }, 'Reject'),
                h('span', { class: 'hint' }, 'or keep talking'));
            waiting.replaceWith(h('div', { class: 'bubble model' },
                changed.length ? h('div', { class: 'hint', style: 'margin-bottom:6px' }, 'Changed: ', changed.map((k) => names[k] || k).join(', ')) : h('div', { class: 'hint', style: 'margin-bottom:6px' }, 'No field changed.'),
                h('div', { class: 'diff' }, wordDiff(image.caption, proposal.caption)), buttons));
            log.lastChild.scrollIntoView({ block: 'nearest' });
        });

        // Add to the Recipe: the only way information moves between images, and only on your say-so.
        function showRule(text) {
            const rule = h('input', { type: 'text', value: text, placeholder: 'The creature is a fox spirit, never a dog.' });
            const old = h('input', { type: 'text', placeholder: 'old word to look for, such as dog' });
            const result = h('div', { style: 'margin-top:8px' });
            ruleSlot.replaceChildren(h('div', { class: 'card', style: 'margin:0' },
                h('h3', null, candy('Add to the Recipe', 'promote this correction to a rule for the whole dataset')),
                h('div', { class: 'stack' }, rule, old,
                    h('button', {
                        class: 'small', onclick: (e) => act(e.target, async () => {
                            App.dataset.settings = (await api('/api/rules/promote', 'POST', { line: rule.value })).settings;
                            toast('Added to this dataset\'s extra instructions.');
                            const matches = old.value.trim() ? (await api('/api/rules/matches', 'POST', { word: old.value })).matches : [];
                            result.replaceChildren(matches.length
                                ? h('div', { class: 'row' }, h('span', null, `${matches.length} image${matches.length === 1 ? '' : 's'} mention "${old.value.trim()}" in their fields.`),
                                    h('button', {
                                        class: 'small', onclick: (ev) => act(ev.target, async () => {
                                            App.run = await api('/api/run/start', 'POST', { image_ids: matches.map((m) => m.id) });
                                            close();
                                            showTab('run');
                                        }),
                                    }, 'Re-run them'))
                                : h('span', { class: 'hint' }, old.value.trim() ? 'No other image mentions that word.' : 'Rule added. Give an old word to find images worth re-running.'));
                        }),
                    }, 'Add rule')), result));
        }

        input.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(null); } });
        const close = overlay(h('div', { class: 'drawer' },
            h('div', { class: 'spread' }, h('h2', null, candy('Sweet Talk', 'review chat')), h('button', { class: 'quiet small', onclick: () => close() }, 'Close')),
            h('div', { class: 'row', style: 'align-items:flex-start' },
                h('img', { src: thumbUrl(image.id), style: 'width:96px; height:96px; object-fit:cover; border-radius:8px' }),
                h('p', { class: 'grow', style: 'font-size:13px' }, image.caption)),
            h('p', { class: 'hint' }, 'Say what is wrong and the model does the rewriting. This conversation holds only this image, and closing it discards it.'),
            log,
            h('div', { class: 'chips' }, QUICK.filter(([label]) => label !== 'Mentions the style' || (image.snapshot && /style/i.test(image.snapshot.prefill)))
                .map(([label, template]) => h('span', { class: 'chip neutral clickable', onclick: () => { input.value = template; input.focus(); const at = template.indexOf('  '); if (at >= 0) input.setSelectionRange(at + 1, at + 1); } }, label))),
            h('div', { class: 'row', style: 'align-items:flex-end' }, h('div', { class: 'grow' }, input), h('button', { onclick: (e) => send(e.target) }, 'Send')),
            ruleSlot), { onclose: () => { chatOpen = false; } });
        setTimeout(() => input.focus(), 50);
    }

    // --- Rendering ---
    function renderList() {
        const list = $('#reviewQueue');
        if (!list) return;
        $('#reviewTotals').textContent = `${rows.length} shown · ${totals.reviewed} of ${totals.done} approved · ${totals.flagged} flagged to go`;
        if (!rows.length) {
            list.replaceChildren(h('div', { class: 'empty', style: 'padding:30px 14px; font-size:13.5px' },
                totals.done ? (query ? 'No caption matches this filter.' : 'The table is clear. Every caption here is approved.') : 'No captions yet. Run a batch in the Candy Factory tab.'));
            return;
        }
        list.replaceChildren(...rows.map((row) => {
            const signals = [...new Set(row.reasons.map((r) => r.signal))];
            const first = row.status === 'failed' ? `failed: ${row.error}` : (row.reasons[0] ? row.reasons[0].message : 'no flags');
            return h('div', { class: 'queue-item' + (row.id === currentId ? ' active' : '') + (row.reviewed ? ' reviewed' : ''), onclick: () => act(null, () => select(row.id, { focusList: false })) },
                h('img', { src: thumbUrl(row.id), loading: 'lazy' }),
                h('div', { class: 'info' }, h('b', null, row.reviewed ? '✓ ' : '', row.rel), h('span', { title: first }, first)),
                row.status === 'failed' ? h('span', { class: 'chip bad' }, '✗') : flagIcons(signals));
        }));
    }

    function renderDetail() {
        const slot = $('#reviewDetail');
        if (!slot) return;
        if (!detail) {
            slot.replaceChildren(h('div', { class: 'card empty' }, rows.length ? 'Pick a caption on the left.' : 'Nothing to sort right now.'));
            return;
        }
        const d = detail;
        const checks = (d.signals && d.signals.checks) || { values: {} };
        const uncertain = ((d.signals && d.signals.uncertain) || { items: [] }).items || [];
        const flaggedFields = new Set(d.reasons.map((r) => r.field).filter(Boolean));
        const over = d.tokens > d.token_limit;

        const caption = h('textarea', { id: 'reviewCaption', rows: 5, value: d.caption, onchange: (e) => act(null, () => saveText('caption', e.target.value)) });
        const counter = h('span', { class: 'counter' + (over ? ' over' : ''), title: `Tags plus description, counted with: ${d.tokenizer.source}` },
            `${d.words} words · ${d.tokens} of ${d.token_limit} tokens${d.tokenizer.exact ? '' : ' (approximate)'}${over ? ' · over the limit' : ''}`);
        caption.addEventListener('input', () => { counter.textContent = `${caption.value.trim().split(/\s+/).filter(Boolean).length} words · saves when you click away`; counter.className = 'counter'; });
        const extra = h('input', { type: 'text', placeholder: 'optional extra instructions for this one image', class: 'grow' });

        const fields = (d.field_defs || []).map((def) => {
            const value = (d.fields || {})[def.key] || '';
            const p = (checks.values || {})[def.key];
            const wavered = uncertain.filter((u) => u.field === def.key);
            return [
                h('div', { class: 'field-row' + (flaggedFields.has(def.key) ? ' flagged' : '') + (def.in_caption ? '' : ' off') },
                    h('label', { title: def.holds }, def.name, def.in_caption ? null : h('span', { class: 'plain' }, 'not in caption')),
                    h('input', {
                        type: 'text', value, disabled: !d.fields, onchange: (e) => act(null, async () => {
                            e.target.disabled = true;
                            refreshFrom(await api(`/api/review/${d.id}/fields`, 'PUT', { fields: { [def.key]: e.target.value } }));
                        }),
                    }),
                    h('span', { class: 'p' + (p != null && p < 0.6 ? ' low' : ''), title: p != null ? `Taste Check (field check): probability the ${checks.measured ? '' : 'plain yes/no '}answer is yes` : '' }, p != null ? `${Math.round(p * 100)}% yes` : '')),
                wavered.length ? h('div', { class: 'field-row', style: 'margin-top:-4px' }, h('span'),
                    h('span', { class: 'hint' }, markWavered(value, wavered), h('br'),
                        wavered.map((u) => `Wrote "${u.word}" (${Math.round(u.p * 100)}%) but nearly wrote ${u.alts.map((a) => `"${a.word}" (${Math.round(a.p * 100)}%)`).join(' or ')}`).join('. '), '. Check the marked word.'), h('span')) : null,
            ];
        });

        const history = d.history.length ? h('details', { style: 'margin-top:12px' },
            h('summary', null, `History (${d.history.length} earlier version${d.history.length === 1 ? '' : 's'})`),
            d.history.map((entry) => h('div', { style: 'padding:8px 0; border-bottom:1px solid var(--input-border)' },
                h('div', { class: 'spread' }, h('span', { class: 'hint' }, `${entry.kind} · ${entry.reason} · ${new Date(entry.created * 1000).toLocaleString()}`),
                    entry.kind === 'caption' || entry.kind === 'tags' ? h('button', { class: 'quiet small', onclick: () => act(null, () => saveText(entry.kind, entry.old)) }, 'Restore') : null),
                h('div', { style: 'font-size:12.5px; margin-top:4px; word-break:break-word' }, entry.old)))) : null;

        slot.replaceChildren(h('div', { class: 'card' },
            h('div', { class: 'spread', style: 'margin-bottom:12px' },
                h('h2', null, d.rel, d.reviewed ? h('span', { class: 'chip good', style: 'margin-left:10px' }, '✓ approved') : null),
                h('span', { class: 'hint' }, [['J', 'next'], ['K', 'previous'], ['A', 'approve'], ['E', 'edit'], ['C', 'Sweet Talk'], ['N', 'next unreviewed'], ['R', 'regenerate']]
                    .map(([key, label]) => [h('span', { class: 'kbd' }, key), label, '  ']))),
            h('div', { class: 'detail-top' },
                h('img', { class: 'picture', src: thumbUrl(d.id, 'prepared'), title: 'Shown as the model saw it. Click to enlarge.', onclick: () => overlay(h('div', { class: 'modal' }, h('img', { class: 'zoom', src: thumbUrl(d.id, 'prepared') })), { center: true }) }),
                h('div', null,
                    d.status === 'failed' ? h('div', { class: 'notice error', style: 'margin-bottom:10px' }, 'This image failed: ', d.error) : null,
                    d.reasons.length ? h('ul', { class: 'reasons' }, d.reasons.map((r) => h('li', null, flagChip(r.signal), h('span', null, r.message))))
                        : h('p', { class: 'hint', style: 'margin-bottom:12px' }, d.fields ? 'No flags on this caption.'
                            : d.status === 'failed' ? 'Regenerate to try this image again, or write its description by hand.'
                                : 'This caption was not written by CandyCaption, so there are no fields or flags for it. Regenerate to get them.'),
                    d.health.length ? h('div', { class: 'chips', style: 'margin-bottom:12px' }, d.health.map((b) => healthChip(b.check, b.detail, true))) : null,
                    h('div', { class: 'spread' }, h('label', null, 'Description', h('span', { class: 'plain' }, d.rel.replace(/\.[^.]+$/, '') + '_nl.txt')), counter),
                    caption,
                    h('div', { class: 'row', style: 'margin:10px 0 14px' },
                        h('button', { onclick: (e) => act(e.target, approve) }, d.reviewed ? 'Put back in the queue' : 'Approve'),
                        h('button', { class: 'quiet', onclick: openChat }, candy('Sweet Talk', 'review chat'))),
                    h('label', null, 'Tags', h('span', { class: 'plain' }, d.rel.replace(/\.[^.]+$/, '') + '.txt')),
                    h('textarea', { rows: 3, value: d.tags, onchange: (e) => act(null, () => saveText('tags', e.target.value)) }),
                    h('div', { class: 'row', style: 'margin-top:10px' }, extra,
                        h('button', {
                            id: 'reviewRegenerate', class: 'quiet', onclick: (e) => act(e.target, async () => {
                                e.target.textContent = 'Looking again…';
                                refreshFrom(await api(`/api/review/${d.id}/regenerate`, 'POST', { extra: extra.value }));
                            }),
                        }, 'Regenerate')))),
            d.fields ? [h('h3', { style: 'margin-top:18px' }, 'Fields', h('span', { class: 'plain' }, 'editing one rewrites the description')), fields] : null,
            d.snapshot ? h('p', { class: 'hint', style: 'margin-top:10px' }, `Made with the ${d.snapshot.prefill} recipe, ${d.snapshot.model}, ${d.snapshot.preset} preset.`) : null,
            history));
    }

    function applyFilter(text) {
        return act(null, async () => {
            const before = query;
            query = text;
            try {
                await loadList();
            } catch (err) {
                query = before;   // a filter with a mistake in it leaves the queue as it was
                throw err;
            }
            render();
            await select(rows.some((r) => r.id === currentId) ? currentId : (rows.length ? rows[0].id : null));
        });
    }

    function render() {
        if (needsDataset(panel)) return;
        const help = h('div', { class: 'notice hint', hidden: true, style: 'margin-bottom:10px' },
            'Combine with AND, OR, NOT and brackets. ',
            h('code', null, 'signal:uncertain'), ' ', h('code', null, 'signal:checks'), ' ', h('code', null, 'signal:tags'), ' ', h('code', null, 'signal:template'), ' ',
            h('code', null, 'field:clothing'), ' ', h('code', null, 'caption:"red scarf"'), ' ', h('code', null, 'file:cat'), ' ', h('code', null, 'tags:1girl'), ' ',
            h('code', null, 'words>70'), ' ', h('code', null, 'tokens>512'), ' ', h('code', null, 'reviewed:yes'), ' ', h('code', null, 'rating:explicit'), ' ', h('code', null, 'health:transparency'),
            '. A bare word searches the caption.');
        const search = h('input', {
            type: 'search', value: query, placeholder: 'Filter, such as signal:tags AND NOT field:clothing',
            onkeydown: (e) => { if (e.key === 'Enter') applyFilter(e.target.value); },
        });
        panel.replaceChildren(h('div', { class: 'review' },
            h('div', null,
                h('div', { class: 'row', style: 'margin-bottom:8px' }, h('div', { class: 'grow' }, search),
                    h('button', { class: 'quiet small', title: 'Reload the queue from the dataset folder', onclick: (e) => act(e.target, async () => { await loadList(); await select(rows.some((r) => r.id === currentId) ? currentId : (rows.length ? rows[0].id : null)); }) }, '↻ Refresh'),
                    h('button', { class: 'quiet small', title: 'How filters work', onclick: () => { help.hidden = !help.hidden; } }, '?')),
                help,
                h('div', { class: 'chips', style: 'margin-bottom:8px' },
                    (() => {
                        const all = h('span', { class: 'chip neutral clickable' + (query.trim() ? '' : ' selected'), title: 'Clear the filter and show the whole queue' }, 'All');
                        all.onclick = () => applyFilter('');
                        return all;
                    })(),
                    Object.keys(FLAGS).map((signal) => {
                        const chip = flagChip(signal, false);
                        chip.classList.add('clickable');
                        if (query.trim() === `signal:${signal}`) chip.classList.add('selected');
                        // Clicking the chip that is already on turns it off again.
                        chip.onclick = () => applyFilter(query.trim() === `signal:${signal}` ? '' : `signal:${signal}`);
                        return chip;
                    })),
                h('div', { class: 'row', style: 'margin-bottom:8px' },
                    h('label', { style: 'margin:0' }, 'Order'),
                    h('select', {
                        style: 'width:auto; flex:1', onchange: (e) => {
                            sortBy = e.target.value;
                            localStorage.setItem('reviewSort', sortBy);
                            act(null, async () => { await loadList(); renderList(); const active = $('.queue-item.active', panel); if (active) active.scrollIntoView({ block: 'nearest' }); });
                        },
                    }, [['score', 'Most doubtful first'], ['name', 'By file name']].map(([value, label]) => h('option', { value, selected: sortBy === value }, label)))),
                flagLegend(),
                h('p', { class: 'hint', id: 'reviewTotals', style: 'margin:8px 0' }),
                h('div', { class: 'queue', id: 'reviewQueue' })),
            h('div', { id: 'reviewDetail' })));
        renderList();
        renderDetail();
    }

    document.addEventListener('keydown', (event) => {
        if (App.tab !== 'review' || event.ctrlKey || event.metaKey || event.altKey || $('.overlay')) return;
        const target = event.target;
        if (target.matches('input, textarea, select')) {
            if (event.key === 'Escape') target.blur();
            return;
        }
        const key = event.key.toLowerCase();
        const actions = {
            j: () => step(1), arrowdown: () => step(1), k: () => step(-1), arrowup: () => step(-1),
            a: () => act(null, approve), n: nextUnreviewed, c: openChat,
            e: () => { const el = $('#reviewCaption'); if (el) { el.focus(); el.setSelectionRange(el.value.length, el.value.length); } },
            r: () => { const el = $('#reviewRegenerate'); if (el && !el.disabled) el.click(); },
        };
        if (actions[key]) { event.preventDefault(); actions[key](); }
    });

    const reloadSoon = debounce(() => { if (App.tab === 'review' && !$('.overlay') && !document.activeElement.matches('input, textarea')) act(null, async () => { await loadList(); renderList(); }); }, 1500);
    on('image_done', reloadSoon);
    on('image_failed', reloadSoon);

    App.tabs.review = {
        show: () => act(null, async () => {
            if (!App.dataset.open) { render(); return; }
            await loadList();
            render();
            const keep = currentId != null && (rows.some((r) => r.id === currentId) || detail);
            await select(keep ? currentId : (rows.length ? rows[0].id : null));
        }),
        open: (id) => { currentId = id; detail = { id }; showTab('review'); },
        reset: () => { rows = []; currentId = null; detail = null; query = ''; totals = { done: 0, reviewed: 0, flagged: 0, failed: 0 }; setCount(); },
        // Called when a run ends: the queue changes then, because template repetition is scored across the whole run.
        refreshCount: () => {
            if (!App.dataset.open) return;
            if (App.tab === 'review') act(null, async () => { await loadList(); renderList(); });
            else api('/api/review').then((data) => { totals = data.totals; setCount(); }).catch(() => {});
        },
    };
})();
