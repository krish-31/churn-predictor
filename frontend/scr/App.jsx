import React, { useState, useMemo, useEffect } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileText,
  BarChart3,
  Users,
  AlertTriangle,
  DollarSign,
  Search,
  CheckCircle2,
  ChevronRight,
  Play,
  RefreshCcw,
  TrendingDown,
  Activity
} from "lucide-react";

const API_BASE = "http://127.0.0.1:8000";

const App = () => {
  const [activeView, setActiveView] = useState("churn"); // churn | drift

  // ---------------- CHURN STATE ----------------
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");

  // ---------------- DRIFT STATE ----------------
  const [driftFile, setDriftFile] = useState(null);
  const [driftResult, setDriftResult] = useState(null);
  const [driftLoading, setDriftLoading] = useState(false);

  // ---------------- HEALTH ----------------
  const [backendHealth, setBackendHealth] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await axios.get(`${API_BASE}/`, { timeout: 5000 });
        setBackendHealth(true);
      } catch {
        setBackendHealth(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // ---------------- HANDLERS ----------------
  const handleFileUpload = (e, setter) => {
    const f = e.target.files[0];
    if (!f) return;
    if (!f.name.endsWith(".csv") && !f.name.endsWith(".xlsx")) {
      alert("Upload CSV or XLSX only");
      return;
    }
    setter(f);
  };

  const runChurnAnalysis = async () => {
    if (!file) return alert("Upload a dataset first");
    setLoading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await axios.post(`${API_BASE}/predict-batch`, form);
      setData(res.data.data);
      setSummary(res.data.summary);
    } catch (e) {
      alert("Churn analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const runDriftAnalysis = async () => {
    if (!driftFile) return alert("Upload a dataset first");
    setDriftLoading(true);
    const form = new FormData();
    form.append("file", driftFile);
    try {
      const res = await axios.post(`${API_BASE}/drift/analyze`, form);
      setDriftResult(res.data);
    } catch (e) {
      alert("Drift analysis failed");
    } finally {
      setDriftLoading(false);
    }
  };

  const filteredData = useMemo(() => {
    if (!data) return [];
    return data.filter((d) =>
      d.customer_id.toString().toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [data, searchTerm]);

  // ---------------- UI ----------------
  return (
    <div className="min-h-screen bg-[#050505] text-gray-100 font-sans overflow-x-hidden">
      <motion.div
        animate={{ opacity: [0.1, 0.2, 0.1] }}
        transition={{ duration: 8, repeat: Infinity }}
        className="fixed inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-red-900/20 via-black to-black -z-10"
      />

      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* HEADER */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-end mb-12 gap-6">
          <div>
            <h1 className="text-5xl font-black text-red-600 italic">
              RETENTION HQ
            </h1>
            <p className="text-gray-500 uppercase tracking-[0.3em] text-[10px] mt-2">
              Enterprise ML Intelligence Console
            </p>
          </div>

          <div className="flex flex-col items-end gap-3">
            <div
              className={`flex items-center gap-2 text-xs font-bold ${backendHealth ? "text-green-500" : "text-red-500"
                }`}
            >
              <div
                className={`w-2 h-2 rounded-full animate-pulse ${backendHealth ? "bg-green-500" : "bg-red-500"
                  }`}
              />
              {backendHealth ? "ML Engine Active" : "ML Engine Offline"}
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => setActiveView("churn")}
                className={`px-5 py-2 rounded-full text-[10px] font-black uppercase tracking-widest ${activeView === "churn"
                    ? "bg-red-600 text-white"
                    : "bg-zinc-900 text-gray-400"
                  }`}
              >
                Churn Engine
              </button>
              <button
                onClick={() => setActiveView("drift")}
                className={`px-5 py-2 rounded-full text-[10px] font-black uppercase tracking-widest ${activeView === "drift"
                    ? "bg-yellow-500 text-black"
                    : "bg-zinc-900 text-gray-400"
                  }`}
              >
                Drift Detection
              </button>
            </div>
          </div>
        </header>

        <AnimatePresence mode="wait">
          {/* ================== CHURN VIEW ================== */}
          {activeView === "churn" && (
            <motion.div
              key="churn"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6 mb-10 flex items-center justify-between gap-6">
                <label className="flex items-center gap-4 cursor-pointer">
                  <Upload />
                  <span className="text-sm font-bold">
                    {file ? file.name : "Upload Dataset"}
                  </span>
                  <input
                    type="file"
                    hidden
                    onChange={(e) => handleFileUpload(e, setFile)}
                  />
                </label>

                <button
                  onClick={runChurnAnalysis}
                  disabled={loading}
                  className="px-8 py-3 bg-red-600 rounded-xl font-black uppercase text-xs"
                >
                  {loading ? "Processing..." : "Process Batch"}
                </button>
              </div>

              {!summary && (
                <div className="h-80 border border-dashed border-zinc-800 rounded-3xl flex items-center justify-center text-gray-500">
                  Upload data to start churn analysis
                </div>
              )}

              {summary && (
                <div className="text-green-400 font-bold">
                  Churn results loaded successfully
                </div>
              )}
            </motion.div>
          )}

          {/* ================== DRIFT VIEW ================== */}
          {activeView === "drift" && (
            <motion.div
              key="drift"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6 mb-10 flex items-center justify-between gap-6">
                <label className="flex items-center gap-4 cursor-pointer">
                  <Activity />
                  <span className="text-sm font-bold">
                    {driftFile ? driftFile.name : "Upload Dataset for Drift"}
                  </span>
                  <input
                    type="file"
                    hidden
                    onChange={(e) => handleFileUpload(e, setDriftFile)}
                  />
                </label>

                <button
                  onClick={runDriftAnalysis}
                  disabled={driftLoading}
                  className="px-8 py-3 bg-yellow-500 text-black rounded-xl font-black uppercase text-xs"
                >
                  {driftLoading ? "Analyzing..." : "Analyze Drift"}
                </button>
              </div>

              {!driftResult && (
                <div className="h-80 border border-dashed border-zinc-800 rounded-3xl flex items-center justify-center text-gray-500">
                  Upload recent data to detect drift
                </div>
              )}

              {driftResult && (
                <div className="space-y-6">
                  <div
                    className={`p-6 rounded-2xl font-black text-xl ${driftResult.overall_status === "Severe"
                        ? "bg-red-600/20 text-red-500"
                        : driftResult.overall_status === "Warning"
                          ? "bg-yellow-500/20 text-yellow-400"
                          : "bg-green-500/20 text-green-400"
                      }`}
                  >
                    Overall Drift Status: {driftResult.overall_status}
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(driftResult.features).map(
                      ([feature, v]) => (
                        <div
                          key={feature}
                          className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-4"
                        >
                          <div className="flex justify-between mb-2">
                            <span className="font-bold">{feature}</span>
                            <span
                              className={`text-xs font-black ${v.status === "Severe"
                                  ? "text-red-500"
                                  : v.status === "Warning"
                                    ? "text-yellow-400"
                                    : "text-green-400"
                                }`}
                            >
                              {v.status}
                            </span>
                          </div>
                          <div className="space-y-2 text-[11px] text-gray-400">
                            <div className="flex justify-between">
                              <span>PSI (Population Stability)</span>
                              <span className="font-mono">
                                <span className="text-gray-500">Recommended &lt; 0.25</span>{" "}
                                | <span className="text-white font-bold">{v.psi.toFixed(3)}</span>
                              </span>
                            </div>

                            <div className="flex justify-between">
                              <span>KS Test (p-value)</span>
                              <span className="font-mono">
                                <span className="text-gray-500">Recommended &gt; 0.05</span>{" "}
                                | <span className="text-white font-bold">{v.ks_pvalue.toFixed(3)}</span>
                              </span>
                            </div>

                            <div className="flex justify-between">
                              <span>JS Divergence</span>
                              <span className="font-mono">
                                <span className="text-gray-500">Recommended &lt; 0.10</span>{" "}
                                | <span className="text-white font-bold">{v.js_divergence.toFixed(3)}</span>
                              </span>
                            </div>
                          </div>

                        </div>
                      )
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default App;
