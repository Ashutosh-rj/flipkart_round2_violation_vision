import { useState, useEffect, useRef, useCallback } from 'react';

// ── Environment-based API config ─────────────────────────────────────
// In production, set VITE_API_BASE_URL to your backend's public origin.
// Defaults to localhost for local development.
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const WS_BASE  = API_BASE.replace(/^http/, 'ws');

interface Violation {
  type: string;
  violation_type: string;
  severity: string;
  plate_number: string;
  confidence: number;
  rider_count: number;
  image_url: string;
  timestamp: string;
}


function App() {
  const [violations, setViolations] = useState<Violation[]>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [statusMsg, setStatusMsg] = useState<string>('Idle — upload a video to begin');
  const [isProcessing, setIsProcessing] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttempts = useRef(0);
  const MAX_RECONNECT_ATTEMPTS = 20;

  // ── WebSocket with auto-reconnect ────────────────────────────────
  const connectWebSocket = useCallback(() => {
    // Don't exceed max attempts
    if (reconnectAttempts.current >= MAX_RECONNECT_ATTEMPTS) {
      setStatusMsg('⚠ WebSocket: max reconnect attempts reached. Reload the page.');
      return;
    }

    const socket = new WebSocket(`${WS_BASE}/ws/alerts`);

    socket.onopen = () => {
      setWsConnected(true);
      reconnectAttempts.current = 0;           // reset on success
    };

    socket.onclose = () => {
      setWsConnected(false);
      // Exponential backoff: 1s, 2s, 4s, 8s … capped at 30s
      const delay = Math.min(1000 * 2 ** reconnectAttempts.current, 30000);
      reconnectAttempts.current += 1;
      reconnectTimer.current = setTimeout(connectWebSocket, delay);
    };

    socket.onerror = () => {
      socket.close();    // triggers onclose → reconnect
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'violation') {
        setViolations((prev) => [data, ...prev]);
      } else if (data.type === 'status') {
        setStatusMsg(data.message);
        setIsProcessing(true);
      } else if (data.type === 'complete') {
        setStatusMsg(data.message);
        setIsProcessing(false);
      } else if (data.type === 'error') {
        setStatusMsg(`⚠ Error: ${data.message}`);
        setIsProcessing(false);
      }
    };

    ws.current = socket;
  }, []);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (ws.current) ws.current.close();
    };
  }, [connectWebSocket]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setViolations([]); // Clear previous results when starting a new analysis!
      setStatusMsg('Uploading...');

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const res = await fetch(`${API_BASE}/api/upload-video`, {
          method: 'POST',
          body: formData,
        });
        if (res.ok) {
          setStatusMsg('Upload complete. Waiting for pipeline to start...');
          setIsProcessing(true);
        } else {
          setStatusMsg('Upload failed.');
        }
      } catch (error) {
        console.error('Upload failed:', error);
        setStatusMsg('Upload failed — is the backend running?');
      }
    }
  };

  const handleExportCSV = () => {
    window.open(`${API_BASE}/api/violations/export`, '_blank');
  };

  const severityColor = (severity: string) => {
    switch (severity) {
      case 'CRITICAL': return 'bg-red-600/30 text-red-300 border-red-500';
      case 'MAJOR': return 'bg-orange-600/30 text-orange-300 border-orange-500';
      default: return 'bg-yellow-600/30 text-yellow-300 border-yellow-500';
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 p-6 md:p-8">
      <header className="mb-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl md:text-4xl font-bold text-white tracking-tight">ViolationVision AI</h1>
          <p className="text-slate-400 mt-1">Bengaluru Traffic Intelligence Platform</p>
        </div>
        <div className="flex items-center gap-3">
          <button 
            onClick={handleExportCSV}
            className="px-4 py-2 rounded-lg bg-blue-600/20 text-blue-300 text-sm font-semibold hover:bg-blue-600/40 transition-colors border border-blue-500/30"
          >
            📥 Export CSV
          </button>
          <div className={`px-4 py-2 rounded-full font-semibold text-sm ${wsConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
            {wsConnected ? '● Live' : '○ Reconnecting…'}
          </div>
        </div>
      </header>

      {/* Analytics Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Total Violations</div>
          <div className="text-3xl font-bold text-white">{violations.length}</div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Triple Riding</div>
          <div className="text-3xl font-bold text-red-400">
            {violations.filter(v => v.violation_type === 'Triple Riding').length}
          </div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Avg Confidence</div>
          <div className="text-3xl font-bold text-blue-400">
            {violations.length > 0 
              ? (violations.reduce((sum, v) => sum + v.confidence, 0) / violations.length * 100).toFixed(1) + '%'
              : '—'}
          </div>
        </div>
        <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="text-slate-400 text-xs uppercase tracking-wider mb-1">Plates Read</div>
          <div className="text-3xl font-bold text-green-400">
            {violations.filter(v => v.plate_number !== 'UNREADABLE').length}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Main Panel: Upload & Status */}
        <div className="lg:col-span-2 space-y-6 min-w-0">
          <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-xl">
            <h2 className="text-xl font-semibold mb-4 text-slate-200">Video Ingestion Engine</h2>
            <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
              <input 
                type="file" 
                accept="video/mp4" 
                onChange={handleFileUpload}
                className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 mb-4 cursor-pointer"
              />
              {file ? (
                <p className="text-slate-300">Selected: {file.name}</p>
              ) : (
                <p className="text-slate-400">Upload CCTV .mp4 file to begin analysis</p>
              )}
            </div>
          </div>
          
          {/* Status Bar */}
          <div className={`rounded-xl p-4 border flex items-center gap-3 ${
            isProcessing 
              ? 'bg-blue-900/30 border-blue-500/40' 
              : statusMsg.includes('Error') 
                ? 'bg-red-900/30 border-red-500/40'
                : statusMsg.includes('complete')
                  ? 'bg-green-900/30 border-green-500/40'
                  : 'bg-slate-800 border-slate-700'
          }`}>
            {isProcessing && (
              <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
            )}
            <span className="text-sm text-slate-300 font-mono">{statusMsg}</span>
          </div>

          {/* Evidence Gallery */}
          {violations.length > 0 && (
            <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-xl">
              <h2 className="text-xl font-semibold mb-4 text-slate-200">Latest Evidence</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {violations.slice(0, 4).map((v, i) => (
                  v.image_url && (
                    <div key={i} className="relative rounded-lg overflow-hidden border border-slate-600">
                      <img 
                        src={`${API_BASE}${v.image_url}`} 
                        alt={`Evidence: ${v.violation_type}`} 
                        className="w-full h-48 object-contain bg-black/50"
                      />
                      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3">
                        <div className="flex justify-between items-end">
                          <span className="text-white text-xs font-semibold">{v.violation_type}</span>
                          <span className="text-slate-300 text-xs font-mono">{v.plate_number}</span>
                        </div>
                      </div>
                    </div>
                  )
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Side Panel: Live Alerts */}
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-xl h-fit max-h-[80vh] flex flex-col min-w-0">
          <h2 className="text-xl font-semibold mb-6 text-slate-200 flex items-center">
            <span className="w-3 h-3 rounded-full bg-red-500 mr-3 animate-pulse"></span>
            Live Violations
          </h2>
          
          <div className="flex-1 overflow-y-auto space-y-4 pr-2">
            {violations.length === 0 ? (
              <div className="text-slate-500 text-center py-10">Listening for anomalies...</div>
            ) : (
              violations.map((v, i) => (
                <div key={i} className="bg-slate-700/50 rounded-lg p-4 border-l-4 border-red-500">
                  <div className="flex justify-between items-start mb-2">
                    <span className="bg-red-500/20 text-red-300 text-xs font-bold px-2 py-1 rounded uppercase tracking-wider">{v.violation_type}</span>
                    <span className={`text-xs font-bold px-2 py-1 rounded ${severityColor(v.severity)}`}>{v.severity}</span>
                  </div>
                  <div className="font-mono text-lg text-white mb-1 font-semibold">{v.plate_number}</div>
                  <div className="flex justify-between items-end text-xs text-slate-400">
                    <span>Conf: {(v.confidence * 100).toFixed(1)}% · {v.rider_count} riders</span>
                    <span>{new Date(v.timestamp).toLocaleTimeString()}</span>
                  </div>
                  {v.image_url && (
                    <img 
                      src={`${API_BASE}${v.image_url}`} 
                      alt="Evidence" 
                      className="mt-3 rounded border border-red-500/30 w-full h-48 object-contain bg-black/50"
                    />
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
