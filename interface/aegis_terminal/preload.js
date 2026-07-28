const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('aegis', {
    // Window controls
    minimize: () => ipcRenderer.send('window-minimize'),
    maximize: () => ipcRenderer.send('window-maximize'),
    close: () => ipcRenderer.send('window-close'),

    // Python main engine
    startEngine: (config) => ipcRenderer.send('start-engine', config),
    stopEngine: () => ipcRenderer.send('stop-engine'),

    // Python arb engine
    startArbEngine: (config) => ipcRenderer.send('start-arb-engine', config),
    stopArbEngine: () => ipcRenderer.send('stop-arb-engine'),

    onBackendLog: (cb) => ipcRenderer.on('backend-log', (_e, data) => cb(data)),

    // Widget windows (Portfolio, Positions, Logs)
    // type = 'portfolio' | 'positions' | 'logs'
    toggleWidget: (type) => ipcRenderer.send('widget-toggle', type),

    // Push live data to all open widget windows
    // payload = { type, ...data } — same shape as WebSocket messages
    forwardToWidgets: (payload) => ipcRenderer.send('widget-forward', payload),
    onWidgetClosed: (cb) => ipcRenderer.on('widget-closed', (_e, type) => cb(type))
});
