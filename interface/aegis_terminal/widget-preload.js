const { contextBridge, ipcRenderer } = require('electron');

// Bridge for floating widget windows
contextBridge.exposeInMainWorld('widget', {
    // Called on load to tell main which widget type this is
    getType: () => {
        const u = new URLSearchParams(window.location.search);
        return u.get('type') || 'portfolio';
    },

    // Listen for data pushed from the main window
    onData: (callback) => ipcRenderer.on('widget-data', (_e, payload) => callback(payload)),

    // Close this widget window
    close: () => ipcRenderer.send('widget-close', window.location.search),

    // Tell main window to update its dock icon state
    ready: (type) => ipcRenderer.send('widget-ready', type),
});
