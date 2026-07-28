const { app, BrowserWindow, ipcMain, screen } = require('electron');
const path = require('path');

let mainWindow;

// Map of type -> BrowserWindow for floating widgets
const widgets = {};

const WIDGET_DEFAULTS = {
    portfolio: { width: 280, height: 340, offsetX: -300, offsetY: 60 },
    positions: { width: 270, height: 280, offsetX: -300, offsetY: 420 },
    logs: { width: 460, height: 230, offsetX: 20, offsetY: -250 },
    journal: { width: 380, height: 420, offsetX: 500, offsetY: 20 },
    analytics: { width: 480, height: 280 },
    aivision: { width: 320, height: 500 },
    scanner: { width: 340, height: 550 },
};

// ── Edge Magnetic Snapping & Master-Slave Dragging ────────────────────────
const SNAP_DIST = 25;
const SNAP_GAP = 12;

function applyMagneticSnap(win) {
    // 1) Drop-to-Snap logic (applies to ALL windows)
    win.on('moved', () => {
        if (win.isDestroyed()) return;
        let snapped = false;
        const b = win.getBounds();
        let nx = b.x;
        let ny = b.y;

        const targets = [];
        if (mainWindow && !mainWindow.isDestroyed() && mainWindow !== win) targets.push(mainWindow.getBounds());
        Object.values(widgets).forEach(w => {
            if (w && !w.isDestroyed() && w !== win) targets.push(w.getBounds());
        });

        for (const t of targets) {
            if (Math.abs(b.x + b.width + SNAP_GAP - t.x) < SNAP_DIST) { nx = t.x - b.width - SNAP_GAP; snapped = true; }
            else if (Math.abs(b.x - (t.x + t.width + SNAP_GAP)) < SNAP_DIST) { nx = t.x + t.width + SNAP_GAP; snapped = true; }
            else if (Math.abs(b.x - t.x) < SNAP_DIST) { nx = t.x; snapped = true; }
            else if (Math.abs(b.x + b.width - (t.x + t.width)) < SNAP_DIST) { nx = t.x + t.width - b.width; snapped = true; }

            if (Math.abs(b.y + b.height + SNAP_GAP - t.y) < SNAP_DIST) { ny = t.y - b.height - SNAP_GAP; snapped = true; }
            else if (Math.abs(b.y - (t.y + t.height + SNAP_GAP)) < SNAP_DIST) { ny = t.y + t.height + SNAP_GAP; snapped = true; }
            else if (Math.abs(b.y - t.y) < SNAP_DIST) { ny = t.y; snapped = true; }
            else if (Math.abs(b.y + b.height - (t.y + t.height)) < SNAP_DIST) { ny = t.y + t.height - b.height; snapped = true; }
        }

        if (snapped && (nx !== b.x || ny !== b.y)) {
            win.setBounds({ x: Math.round(nx), y: Math.round(ny), width: b.width, height: b.height });
        }

        // Save post-move coordinates for the Master-Slave delta calculation
        win._lastBounds = win.getBounds();
    });

    // 2) Master-Slave Drag logic (Only the MAIN window pulls others)
    if (win === mainWindow) {
        win.on('move', () => {
            if (win.isDestroyed()) return;
            const b = win.getBounds();
            if (!win._lastBounds) {
                win._lastBounds = b;
                return;
            }

            const dx = b.x - win._lastBounds.x;
            const dy = b.y - win._lastBounds.y;
            if (dx === 0 && dy === 0) return;

            Object.values(widgets).forEach(w => {
                if (!w || w.isDestroyed()) return;

                const wb = w.getBounds();
                if (!w._lastBounds) w._lastBounds = wb;

                const mb = win._lastBounds; // The master's bounds just before this tick of movement
                const txb = w._lastBounds;

                // Did the widget touch the master right before the drag?
                let attached = false;
                if (
                    Math.abs(mb.x + mb.width + SNAP_GAP - txb.x) <= 2 ||
                    Math.abs(mb.x - (txb.x + txb.width + SNAP_GAP)) <= 2 ||
                    Math.abs(mb.x - txb.x) <= 2 ||
                    Math.abs(mb.x + mb.width - (txb.x + txb.width)) <= 2
                ) attached = true;

                if (!attached && (
                    Math.abs(mb.y + mb.height + SNAP_GAP - txb.y) <= 2 ||
                    Math.abs(mb.y - (txb.y + txb.height + SNAP_GAP)) <= 2 ||
                    Math.abs(mb.y - txb.y) <= 2 ||
                    Math.abs(mb.y + mb.height - (txb.y + txb.height)) <= 2
                )) attached = true;

                // Pull the attached widget
                if (attached) {
                    w.setBounds({
                        x: wb.x + dx,
                        y: wb.y + dy,
                        width: wb.width,
                        height: wb.height
                    });
                    w._lastBounds = w.getBounds();
                }
            });

            win._lastBounds = b;
        });
    }
}

function createWindow() {
    const { width, height } = screen.getPrimaryDisplay().workAreaSize;

    mainWindow = new BrowserWindow({
        width: 1280,
        height: 780,
        x: Math.round((width - 1280) / 2),
        y: Math.round((height - 780) / 2),

        // ── Frameless transparent window ──────────────────────
        frame: false,
        transparent: true,
        backgroundColor: '#00000000',

        roundedCorners: true,
        hasShadow: true,
        show: false,
        resizable: true,
        minWidth: 900, minHeight: 600,

        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });

    // Explicitly bypass proxy for local loopback to prevent VPN/Global proxies 
    // from breaking the bridge to the Python engine (ws://127.0.0.1:8080)
    mainWindow.webContents.session.setProxy({ proxyRules: 'direct://', proxyBypassRules: '127.0.0.1, localhost' });

    // Clear cache AND storage to ensure a fresh state
    mainWindow.webContents.session.clearCache().then(() => {
        mainWindow.webContents.session.clearStorageData().then(() => {
            mainWindow.loadFile(path.join(__dirname, 'index.html'), { query: { v: Date.now() } });
        });
    });

    mainWindow.once('ready-to-show', () => mainWindow.show());
    applyMagneticSnap(mainWindow);

    // Minimize/Restore all widgets with main
    mainWindow.on('minimize', () => {
        Object.values(widgets).forEach(w => { if (!w.isDestroyed()) w.hide(); });
    });
    mainWindow.on('restore', () => {
        Object.values(widgets).forEach(w => { if (!w.isDestroyed()) w.show(); });
    });
    mainWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
        console.log(`[Renderer] ${message} (${sourceId}:${line})`);
    });

    // When main closes, close all widgets too
    mainWindow.on('closed', () => {
        Object.values(widgets).forEach(w => { if (!w.isDestroyed()) w.close(); });
        mainWindow = null;
    });
}

// ── Widget Window Factory ─────────────────────────────────────────────────
function createWidget(type) {
    if (widgets[type] && !widgets[type].isDestroyed()) {
        widgets[type].focus();
        return;
    }

    const def = WIDGET_DEFAULTS[type] || { width: 280, height: 300, offsetX: 20, offsetY: 20 };

    // Position relative to main window, or center if unavailable
    let x = 100, y = 100;
    if (mainWindow && !mainWindow.isDestroyed()) {
        const [mx, my] = mainWindow.getPosition();
        const [mw, mh] = mainWindow.getSize();
        if (type === 'logs') {
            x = mx + Math.floor(mw / 2) - Math.floor(def.width / 2);
            y = my + mh + 10;
        } else if (type === 'analytics') {
            x = mx + Math.floor(mw / 2) - Math.floor(def.width / 2);
            y = my - def.height - 10;
        } else if (type === 'aivision') {
            x = mx - def.width - 10;
            y = my + 60;
        } else if (type === 'scanner') {
            x = mx - def.width - 10;
            y = my + 60;
        } else {
            x = mx + mw + 10 + (type === 'positions' ? 0 : 0);
            y = my + (type === 'positions' ? 360 : 60);
        }
    }

    const win = new BrowserWindow({
        width: def.width,
        height: def.height,
        x, y,
        frame: false,
        transparent: true,
        backgroundColor: '#00000000',
        roundedCorners: true,
        hasShadow: true,
        resizable: true,
        movable: true,
        alwaysOnTop: true,
        skipTaskbar: true,   // don't pollute taskbar
        webPreferences: {
            preload: path.join(__dirname, 'widget-preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });

    win.loadFile(path.join(__dirname, 'widget.html'), {
        query: { type },
    });

    widgets[type] = win;
    applyMagneticSnap(win);

    win.on('closed', () => {
        delete widgets[type];
        // Tell main renderer to update its dock icon
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('widget-closed', type);
        }
    });
}

function destroyWidget(type) {
    if (widgets[type] && !widgets[type].isDestroyed()) {
        widgets[type].close();
    }
}

// ── IPC: Widget toggle (called from main renderer) ────────────────────────
ipcMain.on('widget-toggle', (_e, type) => {
    if (widgets[type] && !widgets[type].isDestroyed()) {
        destroyWidget(type);
    } else {
        createWidget(type);
    }
});

// ── IPC: Forward live data to all open widget windows ─────────────────────
ipcMain.on('widget-forward', (_e, payload) => {
    Object.entries(widgets).forEach(([, win]) => {
        if (!win.isDestroyed()) {
            win.webContents.send('widget-data', payload);
        }
    });
});

// ── IPC: Widget ready (widget tells main it loaded) ──────────────────────
ipcMain.on('widget-ready', (_e, type) => {
    // Nothing needed yet — dock state is managed by main renderer
});

// ── IPC: Widget self-close ────────────────────────────────────────────────
ipcMain.on('widget-close', (_e) => {
    // Find which widget window sent this
    Object.entries(widgets).forEach(([type, win]) => {
        if (!win.isDestroyed() && win.webContents.id === _e.sender.id) {
            destroyWidget(type);
        }
    });
});

// ── IPC: Main Window Controls ─────────────────────────────────────────────
ipcMain.on('window-minimize', () => mainWindow?.minimize());
ipcMain.on('window-maximize', () => {
    mainWindow?.isMaximized() ? mainWindow.unmaximize() : mainWindow?.maximize();
});
ipcMain.on('window-close', () => mainWindow?.close());

// ── IPC: Python Backend Execution ─────────────────────────────────────────
const { spawn, exec } = require('child_process');
let pythonProcess = null;
let arbProcess = null;

function killExistingPython(callback) {
    // Force kill any existing main.py processes to prevent port conflicts
    const cmd = process.platform === 'win32' ? 'taskkill /F /IM python.exe /T' : 'pkill -f main.py';
    exec(cmd, (err) => {
        if (callback) callback();
    });
}

ipcMain.on('start-engine', (event, config) => {
    // If we think it's running, try to kill it first for a fresh start
    if (pythonProcess) {
        event.reply('backend-log', '[SYSTEM] Cleaning up existing engine instance...\n');
        try {
            if (process.platform === 'win32') {
                exec('taskkill /pid ' + pythonProcess.pid + ' /T /F');
            } else {
                process.kill(-pythonProcess.pid); // kill process group
            }
        } catch (e) {}
        pythonProcess = null;
    }

    event.reply('backend-log', '[SYSTEM] Booting Aegis Core Engine (Rust)...\n');

    const cargoExe = 'cargo';
    const cargoArgs = [
        'run', '--release', 
        '--manifest-path', path.join(__dirname, '..', 'rust_engine', 'Cargo.toml'),
        '--bin', 'aegis_engine',
        '--color', 'never'
    ];
    const workingDir = path.join(__dirname, '..');

    logger_js = (data) => event.reply('backend-log', data.toString());

    // Strip proxy environment variables to force a clean, direct connection
    const cleanEnv = { ...process.env };
    ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'].forEach(v => delete cleanEnv[v]);
    
    // We can pass environment variables to Rust if needed
    if (config) {
        cleanEnv.PAPER_MODE = config.paperMode ? 'true' : 'false';
        cleanEnv.TRADE_AMOUNT = config.tradeAmount || '1000';
        cleanEnv.MAX_LEVERAGE = config.maxLev || '10';
        if (config.apiKey) cleanEnv.BINANCE_API_KEY = config.apiKey;
        if (config.apiSecret) cleanEnv.BINANCE_API_SECRET = config.apiSecret;
    }

    pythonProcess = spawn(cargoExe, cargoArgs, {
        cwd: workingDir,
        env: cleanEnv,
        detached: process.platform !== 'win32' // create new process group on linux/mac
    });

    pythonProcess.stdout.on('data', logger_js);
    pythonProcess.stderr.on('data', logger_js);

    pythonProcess.on('error', (err) => {
        event.reply('backend-log', `[SYSTEM] Failed to start engine: ${err.message}\n`);
        pythonProcess = null;
    });

    pythonProcess.on('close', (code) => {
        event.reply('backend-log', `[SYSTEM] Engine exited with code ${code}\n`);
        pythonProcess = null;
    });
});

ipcMain.on('stop-engine', (event) => {
    if (pythonProcess) {
        event.reply('backend-log', '[SYSTEM] Sending termination signal...\n');
        try {
            if (process.platform === 'win32') {
                exec('taskkill /pid ' + pythonProcess.pid + ' /T /F');
            } else {
                process.kill(-pythonProcess.pid); // kill process group
            }
        } catch (e) {}
        pythonProcess = null;
    } else {
        // Fallback: system-wide kill if user is stuck
        const cmd = process.platform === 'win32' ? 'taskkill /F /IM aegis_engine.exe /T' : 'pkill -f aegis_engine';
        exec(cmd, () => {
             event.reply('backend-log', '[SYSTEM] Force-stopped Aegis Rust Engine.\n');
        });
    }
});

// ── IPC: Arbitrage Engine Execution ───────────────────────────────────────
ipcMain.on('start-arb-engine', (event, config) => {
    if (arbProcess) {
        event.reply('backend-log', '[ARB SYSTEM] Cleaning up existing arb instance...\n');
        arbProcess.kill('SIGKILL');
        arbProcess = null;
    }

    event.reply('backend-log', '[ARB SYSTEM] Booting Statistical Arbitrage Core...\n');

    const pythonExe = process.platform === 'win32' ? 'python' : 'python3';
    const scriptPath = path.join(__dirname, '..', 'arb_executor.py');
    const workingDir = path.join(__dirname, '..');

    logger_js = (data) => event.reply('backend-log', data.toString());

    const cleanEnv = { ...process.env };
    ['http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'].forEach(v => delete cleanEnv[v]);
    cleanEnv.PYTHONUNBUFFERED = '1';

    if (config) {
        cleanEnv.PAPER_MODE = config.paperMode ? 'True' : 'False';
        cleanEnv.TRADE_AMOUNT = config.tradeAmount || '2000';
        cleanEnv.MAX_LEVERAGE = config.maxLev || '10';
        if (config.apiKey) cleanEnv.BINANCE_API_KEY = config.apiKey;
        if (config.apiSecret) cleanEnv.BINANCE_API_SECRET = config.apiSecret;
    }

    arbProcess = spawn(pythonExe, [scriptPath], {
        cwd: workingDir,
        env: cleanEnv
    });

    arbProcess.stdout.on('data', logger_js);
    arbProcess.stderr.on('data', logger_js);

    arbProcess.on('error', (err) => {
        event.reply('backend-log', `[ARB SYSTEM] Failed to start arb engine: ${err.message}\n`);
        arbProcess = null;
    });

    arbProcess.on('close', (code) => {
        event.reply('backend-log', `[ARB SYSTEM] Arb Engine exited with code ${code}\n`);
        arbProcess = null;
    });
});

ipcMain.on('stop-arb-engine', (event) => {
    if (arbProcess) {
        event.reply('backend-log', '[ARB SYSTEM] Sending termination signal...\n');
        arbProcess.kill();
        arbProcess = null;
    }
});

// ── App lifecycle ──────────────────────────────────────────────────────────
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});
