

        /* == Dock widget toggles (open/close real OS windows) == */
        const dockState = {};
        function dockToggle(type, iconEl) {
            dockState[type] = !dockState[type];
            iconEl.classList.toggle('on', dockState[type]);
            if (window.aegis) window.aegis.toggleWidget(type);
        }

        // Sync dock icon when a widget window is closed by its own X button
        if (window.aegis) {
            const { ipcRenderer } = require && typeof require === 'function' ? {} : {};
        }

        /* ── Clock ──────────────────────────────────────────────────── */
        const clk = () => {
            const n = new Date();
            document.getElementById('clk').textContent = n.toISOString().slice(11, 19) + ' UTC';
        };
        clk(); setInterval(clk, 1000);

        /* ── Opacity ────────────────────────────────────────────────── */
        let currOp = localStorage.getItem('aegis-opacity') || '0.52';
        function setOp(val, init) {
            currOp = val;
            document.documentElement.style.setProperty('--g-op', val);
            const ov = document.getElementById('opa-v');
            if (ov) ov.textContent = Math.round(val * 100) + '%';
            const oInp = document.getElementById('opa');
            if (oInp) oInp.value = val;
            
            if (!init) {
                localStorage.setItem('aegis-opacity', val);
                if (window.aegis) window.aegis.forwardToWidgets({ type: 'opacity', value: val });
            }
        }
        setOp(currOp, true);

        /* ── Theme ──────────────────────────────────────────────────── */
        function applyTheme(name) {
            document.documentElement.setAttribute('data-theme', name);
            document.querySelectorAll('.theme-dot').forEach(d => d.classList.remove('on'));
            const dot = document.getElementById('td-' + name);
            if (dot) dot.classList.add('on');
            localStorage.setItem('aegis-theme', name);
        }
        applyTheme(localStorage.getItem('aegis-theme') || 'mocha');

        /* ── Chart tab switch ────────────────────────────────────────── */
        function swt(el, sym) {
            document.querySelectorAll('.tab-pill').forEach(t => t.classList.remove('on'));
            el.classList.add('on');
            initTV('BINANCE:' + sym + '.P');
        }

        /* ── Settings modal ─────────────────────────────────────────── */
        function openSettings() {
            document.getElementById('settingsModal').classList.add('active');
            document.getElementById('apiKeyInp').value = localStorage.getItem('binance_api_key') || '';
            document.getElementById('apiSecretInp').value = localStorage.getItem('binance_api_secret') || '';
            document.getElementById('testnetCheck').checked = localStorage.getItem('binance_testnet') === 'true';
            setOp(currOp, true);
        }
        function closeSettings() {
            document.getElementById('settingsModal').classList.remove('active');
        }
        function saveSettings() {
            const k = document.getElementById('apiKeyInp').value.trim();
            const s = document.getElementById('apiSecretInp').value.trim();
            const t = document.getElementById('testnetCheck').checked;
            localStorage.setItem('binance_api_key', k);
            localStorage.setItem('binance_api_secret', s);
            localStorage.setItem('binance_testnet', t);
            closeSettings();
            addLog('b', '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg> Credentials saved locally.');
            if (ws && ws.readyState === WebSocket.OPEN) sendCreds(k, s, t);
        }
        function sendCreds(k, s, t) {
            ws.send(JSON.stringify({ action: 'update_credentials', apiKey: k, apiSecret: s, testnet: t }));
        }

        /* ── Bot control ─────────────────────────────────────────────── */
        let running = false, ws = null;
        function toggleBot() {
            running = !running;
            const b = document.getElementById('sb');
            if (running) {
                b.textContent = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/></svg> Stop';
                b.style.cssText = 'background:rgba(196,128,128,.2);border:1px solid rgba(196,128,128,.28);color:var(--red);';
                addLog('i', '● Initializing Python Backend...');
                if (window.aegis) window.aegis.startEngine();
                connectWS();
            } else {
                b.textContent = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start';
                b.style.cssText = '';
                addLog('i', '○ Sending shutdown signal...');
                if (window.aegis) window.aegis.stopEngine();
                if (ws) ws.close();
            }
        }
        function emergencyClose() {
            addLog('r', '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> EMERGENCY CLOSE ALL POSITIONS');
            if (ws && ws.readyState === WebSocket.OPEN)
                ws.send(JSON.stringify({ action: 'emergency_close' }));
        }

        /* ── Log helper ─────────────────────────────────────────────── */
        function addLog(type, msg) {
            const lf = document.getElementById('lf');
            if (!lf) return;
            const ts = new Date().toISOString().slice(11, 19);
            const cls = { g: 'li-g', r: 'li-r', b: 'li-b', s: 'li-s' }[type] || '';
            const row = document.createElement('div');
            row.className = 'lr';
            row.innerHTML = `<span class="lt">${ts}</span><span class="li ${cls}">${msg}</span>`;
            lf.appendChild(row);
            lf.scrollTop = lf.scrollHeight;
            if (lf.children.length > 400) lf.removeChild(lf.children[0]);
        }

        /* ── Backend log via IPC ─────────────────────────────────────── */
        if (window.aegis) {
            window.aegis.onBackendLog(logText => {
                logText.split('\n').forEach(line => {
                    const c = line.trim();
                    if (!c) return;
                    let t = 'i';
                    if (c.includes(' - ERROR - ') || c.startsWith('Traceback') || c.match(/File ".*", line \d+/)) t = 'r';
                    else if (c.includes('EXECUTE BUY')) t = 'g';
                    else if (c.includes('EXECUTE SELL')) t = 'r';
                    else if (c.includes('SIGNAL')) t = 'b';
                    else if (c.includes('HTTP/1.1" 200')) return; // suppress Telegram OK
                    addLog(t, c);
                });
            });
        }

        /* ── WebSocket (visual updates from Python engine) ───────────── */
        function connectWS() {
            ws = new WebSocket('ws://localhost:8765');
            ws.onopen = () => {
                document.getElementById('ws-ind').textContent = '↑ Engine Connected';
                const k = localStorage.getItem('binance_api_key') || '';
                const s = localStorage.getItem('binance_api_secret') || '';
                const t = localStorage.getItem('binance_testnet') === 'true';
                if (k) sendCreds(k, s, t);
            };
            ws.onclose = () => {
                document.getElementById('ws-ind').textContent = '○ WS Disconnected';
                if (running) setTimeout(connectWS, 3000);
            };
            ws.onerror = () => { };
            ws.onmessage = event => {
                const data = JSON.parse(event.data);
                if (data.type === 'log') {
                if (window.aegis) window.aegis.forwardToWidgets(data);
                    let t = 'i';
                    if (data.level === 'error') t = 'r';
                    else if (data.message.includes('EXECUTE BUY')) t = 'g';
                    else if (data.message.includes('EXECUTE SELL')) t = 'r';
                    else if (data.message.includes('SIGNAL')) t = 'b';
                    else if (data.message.includes('REJECTED')) t = 's';
                    addLog(t, data.message);
                } else if (data.type === 'signal') {
                    addLog('b', `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> ${data.direction} ${data.symbol} @ ${data.price}`);
                } else if (data.type === 'ml_eval') {
                    const p = (data.prob * 100).toFixed(1) + '%';
                    addLog(data.prob >= 0.63 ? 'g' : 's', `🧠 ML (${data.symbol}): P(Win)=${p}`);
                } else if (data.type === 'positions') {
                    renderPositions(data.data);
                } else if (data.type === 'balance_update') {
                if (window.aegis) window.aegis.forwardToWidgets(data);
                    const el = document.getElementById('equity-val');
                    if (el) el.textContent = '$' + parseFloat(data.equity).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
                }
            };
        }

        /* ── Render open positions ──────────────────────────────────── */
        function renderPositions(positions) {
            const c = document.getElementById('pos-container');
            if (!c) return;
            c.innerHTML = '<div class="lbl">Open Positions</div>';
            if (!positions || positions.length === 0) {
                c.innerHTML += '<div id="no-pos" style="font-size:11px;color:var(--muted);text-align:center;padding:14px 0">No active positions</div>';
                return;
            }
            positions.forEach(p => {
                const isL = p.side === 'LONG';
                const pnlSign = p.unrealizedPnl >= 0 ? '+' : '';
                const pnlColor = p.unrealizedPnl >= 0 ? 'var(--green)' : 'var(--red)';
                const bw = Math.min(100, Math.max(5, Math.abs(p.pnlPct) * 20));
                const bg = isL ? 'linear-gradient(90deg,#22c55e,#84cc16)' : 'linear-gradient(90deg,#ef4444,#f97316)';
                c.insertAdjacentHTML('beforeend', `
              <div class="pos-item">
                <div class="pos-top">
                  <div><span class="pos-sym">${p.symbol}</span>
                    <span class="badge ${isL ? 'badge-g' : 'badge-a'}" style="margin-left:5px;font-size:8.5px">${p.side}</span>
                    <div style="font-size:9.5px;color:var(--muted);margin-top:2px">Entry $${parseFloat(p.entryPrice).toFixed(2)}</div>
                  </div>
                  <div style="text-align:right">
                    <div class="pos-pnl" style="color:${pnlColor}">${pnlSign}$${p.unrealizedPnl.toFixed(2)}</div>
                    <div style="font-size:9.5px;color:var(--muted)">${pnlSign}${p.pnlPct.toFixed(2)}%</div>
                  </div>
                </div>
                <div class="pos-bar"><div class="pos-bar-fill" style="width:${bw}%;background:${bg}"></div></div>
              </div>`);
            });
        }

        /* ── TradingView ────────────────────────────────────────────── */
        let tvWidget = null;
        function initTV(symbol = 'BINANCE:BTCUSDT.P') {
            const el = document.getElementById('tv_chart');
            if (!el) return;
            el.innerHTML = '';
            tvWidget = new TradingView.widget({
                autosize: true, symbol, interval: '5',
                timezone: 'Etc/UTC', theme: 'dark', style: '1', locale: 'en',
                enable_publishing: false,
                backgroundColor: 'rgba(0,0,0,0)',
                gridColor: 'rgba(255,255,255,0.04)',
                hide_top_toolbar: true, hide_legend: true, save_image: false,
                container_id: 'tv_chart'
            });
        }
        window.addEventListener('DOMContentLoaded', () => initTV());

        