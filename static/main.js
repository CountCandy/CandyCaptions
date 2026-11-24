// Start-up: load what every tab shares, then open the last tab used.

(async function () {
    setTheme(localStorage.getItem('theme') || 'candy');
    $('#themeToggle').addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        setTheme(THEMES[(THEMES.indexOf(current) + 1) % THEMES.length]);
    });
    setSound(soundOn());
    $('#soundToggle').addEventListener('click', () => { setSound(!soundOn()); if (soundOn()) playGumdrops(); });
    $$('.tab[data-tab]').forEach((tab) => tab.addEventListener('click', () => showTab(tab.dataset.tab)));
    $('#modelPill').addEventListener('click', () => showTab('connect'));
    $('#datasetPill').addEventListener('click', () => showTab('dataset'));

    try {
        const [settings, , dataset, run] = await Promise.all([
            api('/api/settings'), App.loadPrefills(), api('/api/dataset'), api('/api/run'),
        ]);
        Object.assign(App, { settings: settings.settings, backends: settings.backends, caps: settings.caps, defaultSampling: settings.default_sampling, dataset, run });
    } catch (err) {
        toast(err.message, true);
        return;
    }
    refreshHeader();
    connectEvents();
    if (App.dataset.open) App.tabs.review.refreshCount();
    const saved = localStorage.getItem('tab');
    showTab(App.tabs[saved] ? saved : 'connect');
})();
