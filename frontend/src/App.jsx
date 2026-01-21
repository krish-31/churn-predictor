import React, { useState, useMemo, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, FileText, BarChart3, Users, 
  AlertTriangle, DollarSign, Search, CheckCircle2,
  ChevronRight, Play, RefreshCcw, TrendingDown,
  Activity, Target, Cpu, Layers, FileCode2, PieChart, Film
} from 'lucide-react';

const App = () => {
  // --- STATE MANAGEMENT ---
  const [activeTab, setActiveTab] = useState('churn'); // 'churn' | 'drift' | 'stats'
  
  // Files
  const [file, setFile] = useState(null);
  const [driftFile, setDriftFile] = useState(null); 
  const [statsFile, setStatsFile] = useState(null); 

  // Data Response
  const [data, setData] = useState(null); 
  const [summary, setSummary] = useState(null);
  const [driftMetrics, setDriftMetrics] = useState(null); 
  const [statsData, setStatsData] = useState(null); 
  const [overallDriftStatus, setOverallDriftStatus] = useState("Unknown");

  // Loading States
  const [loading, setLoading] = useState(false); 
  const [driftLoading, setDriftLoading] = useState(false); 
  const [statsLoading, setStatsLoading] = useState(false); 

  const [searchTerm, setSearchTerm] = useState("");
  // --- NEW: Filter State ---
  const [filterCategory, setFilterCategory] = useState("All"); 

  const [backendHealth, setBackendHealth] = useState(false);
  const [modelStats, setModelStats] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // --- INITIALIZATION ---
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await axios.get('https://retention-hq.onrender.com/', { timeout: 5000 });
        setBackendHealth(true);
      } catch (err) {
        setBackendHealth(false);
      }
    };

    const fetchModelStats = async () => {
      try {
        const res = await axios.get('https://retention-hq.onrender.com/model-stats');
        setModelStats(res.data);
      } catch (err) {
        console.error("Failed to fetch model stats");
      }
    };

    checkHealth();
    fetchModelStats();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // --- HANDLERS ---
  const validateAndSetFile = (selectedFile, type = 'churn') => {
    const fileName = selectedFile?.name || "";
    const isValid = fileName.endsWith('.csv') || fileName.endsWith('.xlsx');

    if (selectedFile && isValid) {
      if (type === 'drift' || type === true) setDriftFile(selectedFile);
      else if (type === 'stats') setStatsFile(selectedFile);
      else setFile(selectedFile);
    } else {
      alert("Please select a valid CSV or Excel (.xlsx) file.");
    }
  };

  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); setIsDragging(false); };
  
  const handleDrop = (e, type = 'churn') => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    validateAndSetFile(droppedFile, type);
  };

  // --- 1. RUN PREDICTION ---
  const runPrediction = async () => {
    if (!file) return alert("Please upload a dataset first.");
    setLoading(true);
    setFilterCategory("All"); // Reset filter on new run
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('https://retention-hq.onrender.com/predict-batch', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setData(response.data.data);
      setSummary(response.data.summary);
    } catch (err) {
      console.error("Prediction Error:", err);
      alert("⚠️ PROCESSING ERROR: Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  // --- 2. RUN DRIFT ANALYSIS ---
  const runDriftAnalysis = async () => {
    if (!driftFile) return alert("Please upload a dataset for drift analysis.");
    setDriftLoading(true);
    setDriftMetrics(null); 
    
    const formData = new FormData();
    formData.append('file', driftFile);

    try {
      const response = await axios.post('https://retention-hq.onrender.com/drift/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      const metrics = response.data;
      setDriftMetrics(metrics);

      const actualData = metrics.features || metrics.drift_metrics || metrics;
      
      let highCount = 0;
      let moderateCount = 0;

      if (actualData && typeof actualData === 'object') {
        Object.values(actualData).forEach(m => {
            if (m?.psi > 0.25) {
                highCount++;
            } 
            else if (m?.psi > 0.1 || m?.js_divergence > 0.1) {
                moderateCount++;
            }
        });
      }

      if (highCount >= 2 || (highCount === 1 && moderateCount >= 2)) {
          setOverallDriftStatus("Severe");
      }
      else if (highCount === 1 || moderateCount >= 3) {
          setOverallDriftStatus("Moderate");
      }
      else {
          setOverallDriftStatus("Normal");
      }

    } catch (err) {
      console.error("Drift Error:", err);
      alert("⚠️ DRIFT ERROR: Check backend logs.");
    } finally {
      setDriftLoading(false);
    }
  };

  // --- 3. RUN STATS ANALYSIS ---
  const runStatsAnalysis = async () => {
    if (!statsFile) return alert("Please upload a dataset.");
    setStatsLoading(true);
    setStatsData(null);
    const formData = new FormData();
    formData.append('file', statsFile);
    try {
      const response = await axios.post('https://retention-hq.onrender.com/stats/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setStatsData(response.data);
    } catch (err) { 
      console.error(err);
      alert("⚠️ STATS ERROR: Check backend logs."); 
    } 
    finally { setStatsLoading(false); }
  };

  // --- HELPER: Extract Columns ---
  const getDriftColumns = () => {
    if (!driftMetrics) return [];
    const columns = driftMetrics.features || driftMetrics.drift_metrics || driftMetrics;
    return Object.entries(columns).filter(([key, val]) => {
      return val && typeof val === 'object' && ('psi' in val || 'drift_status' in val);
    });
  };

  // --- FILTERING LOGIC ---
  const filteredData = useMemo(() => {
    if (!data) return [];
    return data.filter(item => {
      const matchesSearch = item.customer_id.toString().toLowerCase().includes(searchTerm.toLowerCase());
      const matchesFilter = filterCategory === "All" || item.risk_category === filterCategory;
      return matchesSearch && matchesFilter;
    });
  }, [data, searchTerm, filterCategory]);

  // --- DYNAMIC COLOR LOGIC ---
  const getHQColor = () => {
    if (activeTab === 'drift') return 'text-yellow-500';
    if (activeTab === 'stats') return 'text-blue-600';
    return 'text-red-600'; 
  };

  return (
    <div className="min-h-screen bg-[#050505] text-gray-100 font-sans selection:bg-red-500/30 overflow-x-hidden pb-20">
      <motion.div 
        animate={{ opacity: [0.1, 0.2, 0.1], scale: [1, 1.05, 1] }}
        transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
        className="fixed inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-zinc-900/40 via-black to-black -z-10"
      />

      <div className="max-w-7xl mx-auto px-6 py-12">
        
        {/* --- HEADER --- */}
        <header className="flex flex-col md:flex-row justify-between items-center mb-12 gap-4">
          <div className="flex-1">
            <h1 className="text-5xl font-black text-white italic tracking-tighter">
              RETENTION <span className={`${getHQColor()} transition-colors duration-500`}>HQ</span>
            </h1>
            <p className="text-gray-500 font-medium uppercase tracking-[0.25em] text-[10px] mt-2">Enterprise ML Intelligence Console</p>
          </div>

          <div className="flex bg-zinc-900/80 p-1 rounded-full border border-zinc-800">
            <button 
              onClick={() => setActiveTab('churn')}
              className={`px-6 py-2 rounded-full text-xs font-black uppercase tracking-widest transition-all ${activeTab === 'churn' ? 'bg-red-600 text-white shadow-lg' : 'text-gray-500 hover:text-gray-300'}`}
            >
              Churn Engine
            </button>
            <button 
              onClick={() => setActiveTab('drift')}
              className={`px-6 py-2 rounded-full text-xs font-black uppercase tracking-widest transition-all ${activeTab === 'drift' ? 'bg-yellow-500 text-black shadow-lg shadow-yellow-500/20' : 'text-gray-500 hover:text-gray-300'}`}
            >
              Drift Detection
            </button>
            <button 
              onClick={() => setActiveTab('stats')}
              className={`px-6 py-2 rounded-full text-xs font-black uppercase tracking-widest transition-all ${activeTab === 'stats' ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20' : 'text-gray-500 hover:text-gray-300'}`}
            >
              Analytics
            </button>
          </div>

          <div className="flex-1 text-right">
             <div className={`inline-flex items-center gap-2 text-xs font-bold ${backendHealth ? 'text-green-500' : 'text-red-500'}`}>
              <div className={`w-2 h-2 rounded-full animate-pulse ${backendHealth ? 'bg-green-500' : 'bg-red-500'}`} /> 
              {backendHealth ? 'System Active' : 'System Offline'}
            </div>
          </div>
        </header>

        {/* ===================================================================================== */}
        {/* PAGE 1: CHURN PREDICTION DASHBOARD                                                   */}
        {/* ===================================================================================== */}
        {activeTab === 'churn' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-12">
              <div 
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={(e) => handleDrop(e, 'churn')}
                className={`lg:col-span-3 bg-zinc-900/40 border-2 rounded-2xl p-6 backdrop-blur-xl flex flex-col md:flex-row items-center justify-between gap-6 transition-all duration-300 ${isDragging ? 'border-red-600 bg-red-600/5 scale-[1.01]' : 'border-zinc-800'}`}
              >
                <div className="flex-1 w-full">
                  <label className="flex items-center gap-4 cursor-pointer group">
                    <div className={`p-4 rounded-xl transition-all border ${isDragging ? 'bg-red-600 text-white border-red-500' : 'bg-zinc-800 text-gray-400 border-zinc-700 group-hover:border-red-500/50 group-hover:text-red-500'}`}>
                      <Upload size={24} className={isDragging ? 'animate-bounce' : ''} />
                    </div>
                    <div>
                      <p className="text-sm font-bold text-gray-200">
                        {file ? file.name : isDragging ? "Drop File Here" : "Upload or Drag & Drop Dataset"}
                      </p>
                      <p className="text-[10px] text-gray-500 uppercase font-black tracking-tighter mt-1 italic">Supports .csv and .xlsx formats</p>
                    </div>
                    <input type="file" className="hidden" onChange={(e) => validateAndSetFile(e.target.files[0], 'churn')} accept=".csv, .xlsx" />
                  </label>
                </div>
                
                <button 
                  onClick={runPrediction}
                  disabled={loading || !file}
                  className="px-8 py-4 bg-red-600 hover:bg-red-700 disabled:bg-zinc-800 disabled:text-gray-500 rounded-xl font-black text-xs uppercase tracking-[0.2em] transition-all flex items-center justify-center gap-3 active:scale-95 shadow-lg shadow-red-900/20"
                >
                  {loading ? <RefreshCcw className="animate-spin" size={18}/> : <><Play size={16} fill="currentColor"/> Process Batch</>}
                </button>
              </div>

              <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6 backdrop-blur-xl flex flex-col justify-center items-center text-center">
                <div className="flex items-center gap-2 mb-2 text-gray-500">
                  <CheckCircle2 size={14}/>
                  <span className="text-[10px] font-black uppercase tracking-widest">Schema Validation</span>
                </div>
                <button 
                  onClick={() => alert("Columns Required:\ncustomer_id, age, gender, subscription_type, watch_hours, last_login_days, region, device, monthly_fee, number_of_profiles, avg_watch_time_per_day, favorite_genre, payment_method")}
                  className="text-[10px] font-bold text-red-500 hover:text-red-400 uppercase transition-colors"
                >
                  Verify Columns
                </button>
              </div>
            </div>

            <AnimatePresence mode="wait">
              {summary ? (
                <motion.div key="results" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                  <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-12">
                    {/* RESET FILTER CARD */}
                    <div 
                      onClick={() => setFilterCategory('All')}
                      className={`bg-zinc-900/60 border p-5 rounded-2xl backdrop-blur-md cursor-pointer hover:border-blue-500/50 transition-all ${filterCategory === 'All' ? 'border-blue-500 ring-1 ring-blue-500/20' : 'border-zinc-800'}`}
                    >
                      <div className="flex justify-between items-start mb-3"><Users className="text-blue-500" size={16}/><span className="text-[8px] font-black text-gray-500 uppercase tracking-widest">Total (Reset)</span></div>
                      <div className="text-2xl font-black text-gray-100">{summary.total_customers}</div>
                    </div>
                    <div className="bg-zinc-900/60 border border-zinc-800 p-5 rounded-2xl backdrop-blur-md">
                      <div className="flex justify-between items-start mb-3"><BarChart3 className="text-purple-500" size={16}/><span className="text-[8px] font-black text-gray-500 uppercase tracking-widest">Avg Churn Risk</span></div>
                      <div className="text-2xl font-black text-gray-100">{summary.avg_risk_score}%</div>
                    </div>
                    {/* HIGH RISK FILTER CARD */}
                    <div 
                      onClick={() => setFilterCategory('High')}
                      className={`bg-zinc-900/60 border p-5 rounded-2xl backdrop-blur-md cursor-pointer hover:border-red-500/50 transition-all ${filterCategory === 'High' ? 'border-red-500 ring-1 ring-red-500/20' : 'border-zinc-800'}`}
                    >
                      <div className="flex justify-between items-start mb-3"><AlertTriangle className="text-red-500" size={16}/><span className="text-[8px] font-black text-gray-500 uppercase tracking-widest">High Risk</span></div>
                      <div className="text-2xl font-black text-gray-100">{summary.high_risk_count}</div>
                    </div>
                    <div className="bg-zinc-900/60 border border-zinc-800 p-5 rounded-2xl backdrop-blur-md">
                      <div className="flex justify-between items-start mb-3"><DollarSign className="text-yellow-500" size={16}/><span className="text-[8px] font-black text-gray-500 uppercase tracking-widest">Revenue Risk</span></div>
                      <div className="text-2xl font-black text-gray-100">${summary.revenue_at_risk.toLocaleString()}</div>
                    </div>
                    <div className="bg-zinc-900/60 border border-zinc-800 p-5 rounded-2xl backdrop-blur-md">
                      <div className="flex justify-between items-start mb-3"><Film className="text-pink-500" size={16}/><span className="text-[8px] font-black text-gray-500 uppercase tracking-widest">Max Risk Genre</span></div>
                      <div className="text-2xl font-black text-gray-100 truncate">{summary.top_churn_genre}</div>
                    </div>
                  </div>

                  {/* TABLE & SEGMENTATION */}
                  <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-12">
                    <div className="lg:col-span-1 bg-zinc-900/40 border border-zinc-800 rounded-3xl p-6 backdrop-blur-sm h-fit">
                      <h3 className="text-[10px] font-black uppercase text-gray-500 tracking-widest mb-8 flex items-center gap-2">
                        <Users size={14}/> Segmentation Breakdown
                      </h3>
                      <div className="space-y-4">
                        {[
                          { label: 'High Risk', count: summary.high_risk_count, color: 'bg-red-600', filter: 'High' },
                          { label: 'Medium Risk', count: summary.medium_risk_count, color: 'bg-yellow-600', filter: 'Medium' },
                          { label: 'Low Risk', count: summary.low_risk_count, color: 'bg-green-600', filter: 'Low' }
                        ].map((seg, i) => (
                          <div 
                            key={i} 
                            onClick={() => setFilterCategory(seg.filter)}
                            className={`cursor-pointer p-3 rounded-xl transition-all border ${filterCategory === seg.filter ? 'bg-zinc-800 border-gray-600' : 'border-transparent hover:bg-zinc-800/50'}`}
                          >
                            <div className="flex justify-between text-[10px] font-bold mb-2">
                              <span className="text-gray-400">{seg.label}</span>
                              <span className="text-gray-100">
                                {summary.total_customers > 0 ? Math.round((seg.count / summary.total_customers) * 100) : 0}% 
                                <span className="text-gray-500 ml-1">({seg.count})</span>
                              </span>
                            </div>
                            <div className="w-full bg-zinc-900 h-1.5 rounded-full overflow-hidden">
                              <motion.div initial={{ width: 0 }} animate={{ width: `${(seg.count / summary.total_customers) * 100}%` }} transition={{ duration: 1, ease: "easeOut" }} className={`${seg.color} h-full`} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="lg:col-span-3 bg-zinc-900/40 border border-zinc-800 rounded-3xl overflow-hidden backdrop-blur-md shadow-2xl">
                      <div className="p-6 border-b border-zinc-800 flex justify-between items-center bg-zinc-900/20">
                        <div className="flex items-center gap-4">
                          <h2 className="text-sm font-black uppercase tracking-tighter flex items-center gap-2">Analysis Drill-Down</h2>
                          {filterCategory !== 'All' && (
                            <span className="px-2 py-1 bg-white/10 rounded text-[9px] font-bold uppercase text-white border border-white/20">
                              Filtered: {filterCategory} Risk
                            </span>
                          )}
                        </div>
                        <div className="relative flex-1 max-w-sm ml-4">
                          <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-600" size={14}/>
                          <input placeholder="Search Customer ID..." className="w-full bg-black/40 border border-zinc-800 rounded-full py-2.5 pl-10 pr-4 text-xs outline-none focus:ring-1 focus:ring-red-600 transition-all" value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} />
                        </div>
                      </div>
                      <div className="overflow-x-auto max-h-[600px] scrollbar-hide">
                        <table className="w-full text-left">
                          <thead className="sticky top-0 bg-[#0a0a0a] z-10 border-b border-zinc-800">
                            <tr className="text-[10px] font-black text-gray-500 uppercase tracking-widest">
                              <th className="px-8 py-5">Customer ID</th>
                              <th className="px-6 py-5 text-center">Prob.</th>
                              <th className="px-6 py-5 text-center">Persona</th>
                              <th className="px-6 py-5 text-center">Category</th>
                              <th className="px-6 py-5">Recommendation</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-zinc-800/50">
                            {filteredData.map((row, idx) => (
                              <tr key={idx} className="hover:bg-white/[0.02] transition-colors">
                                {/* CUSTOMER ID COLOR CHANGED TO INDIGO/NEUTRAL */}
                                <td className="px-8 py-5 font-mono text-xs text-indigo-300 font-bold italic">{row.customer_id.toString().slice(0, 15)}...</td>
                                <td className="px-6 py-5 text-center font-black text-lg">{row.churn_probability}%</td>
                                <td className="px-6 py-5 text-center"><span className="px-2 py-1 rounded border border-blue-500/20 bg-blue-500/10 text-[9px] font-black text-blue-400 uppercase whitespace-nowrap inline-flex items-center justify-center w-28">{row.persona}</span></td>
                                <td className="px-6 py-5 text-center"><span className={`px-3 py-1 rounded text-[9px] font-black uppercase tracking-widest ${row.risk_category === 'High' ? 'bg-red-500/10 text-red-500' : row.risk_category === 'Medium' ? 'bg-yellow-500/10 text-yellow-500' : 'bg-green-500/10 text-green-500'}`}>{row.risk_category}</span></td>
                                <td className="px-6 py-5 text-xs text-gray-400 italic leading-relaxed">{row.insight}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {filteredData.length === 0 && (
                          <div className="text-center py-12 text-gray-500 text-xs italic">No customers found for this category or search term.</div>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ) : (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-96 border-2 border-dashed border-zinc-800 rounded-[2.5rem] flex flex-col items-center justify-center text-center p-6 bg-zinc-900/10">
                  <div className="p-6 bg-zinc-900 rounded-full mb-6 border border-zinc-800"><FileText size={40} className="text-zinc-700 animate-pulse"/></div>
                  <h3 className="text-sm font-bold text-gray-400 uppercase tracking-widest">Inference Engine Standby</h3>
                  <p className="text-xs text-gray-600 mt-2 max-w-xs italic leading-relaxed">Drag and drop a CSV or Excel batch file to identify at-risk subscriber segments and generate retention strategies.</p>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="mt-16 pt-12 border-t border-zinc-800">
               <div className="flex items-center justify-between mb-8">
                 <div><h2 className="text-xl font-black italic text-red-600 tracking-tighter">MODEL BENCHMARK SUITE</h2><p className="text-[10px] text-gray-500 uppercase font-bold tracking-widest mt-1">Cross-Validation Performance Results</p></div>
                 <div className="px-4 py-2 bg-red-600/10 border border-red-600/30 rounded-full text-[10px] font-black text-red-500 uppercase tracking-widest">Engine: XGBoost</div>
               </div>
               <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                 {modelStats ? Object.entries(modelStats).map(([key, stats]) => (
                   <div key={key} className={`p-6 rounded-3xl border transition-all ${key === 'xg' ? 'bg-red-600/5 border-red-500/30 ring-1 ring-red-500/20' : 'bg-zinc-900/40 border-zinc-800'}`}>
                     <div className="flex justify-between items-center mb-6">
                       <div className="flex items-center gap-2">
                         {key === 'xg' ? <Cpu size={14} className="text-red-500"/> : key === 'rf' ? <Layers size={14} className="text-gray-400"/> : <Target size={14} className="text-gray-400"/>}
                         <span className="text-[10px] font-black uppercase text-gray-400">{key === 'xg' ? 'Extreme Gradient Boosting' : key === 'rf' ? 'Random Forest' : 'Logistic Regression'}</span>
                       </div>
                     </div>
                     <div className="text-4xl font-black mb-6 tracking-tighter">{stats.accuracy}% <span className="text-[8px] text-gray-600 uppercase tracking-widest">Accuracy</span></div>
                     <div className="space-y-4">{['Precision', 'Recall', 'F1-Score'].map((label, i) => (<div key={label}><div className="flex justify-between text-[8px] font-black uppercase text-gray-600 mb-1"><span>{label}</span><span className="text-gray-100">{Object.values(stats)[i+1]}%</span></div><div className="h-1 bg-zinc-800 rounded-full overflow-hidden"><motion.div initial={{ width: 0 }} animate={{ width: `${Object.values(stats)[i+1]}%` }} className={`h-full ${key === 'xg' ? 'bg-red-500' : 'bg-zinc-600'}`} /></div></div>))}</div>
                   </div>
                 )) : <div className="col-span-3 text-center text-gray-600 italic">Syncing technical benchmarks...</div>}
               </div>
            </div>
          </motion.div>
        )}

        {/* ===================================================================================== */}
        {/* PAGE 2: DRIFT DETECTION                                                              */}
        {/* ===================================================================================== */}
        {activeTab === 'drift' && (
           <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}>
             
             {/* DRIFT UPLOAD BAR */}
             <div 
               onDragOver={handleDragOver}
               onDragLeave={handleDragLeave}
               onDrop={(e) => handleDrop(e, 'drift')}
               className={`w-full bg-zinc-900/40 border-2 rounded-2xl p-6 backdrop-blur-xl flex flex-col md:flex-row items-center justify-between gap-6 transition-all duration-300 mb-8 ${isDragging ? 'border-yellow-500 bg-yellow-500/5 scale-[1.01]' : 'border-zinc-800'}`}
             >
               <div className="flex-1 w-full">
                 <label className="flex items-center gap-4 cursor-pointer group">
                   <div className={`p-4 rounded-xl transition-all border ${isDragging ? 'bg-yellow-500 text-black border-yellow-500' : 'bg-zinc-800 text-gray-400 border-zinc-700 group-hover:border-yellow-500/50 group-hover:text-yellow-500'}`}>
                     <Activity size={24} className={isDragging ? 'animate-bounce' : ''} />
                   </div>
                   <div>
                     <p className="text-sm font-bold text-gray-200">
                       {driftFile ? driftFile.name : isDragging ? "Drop File Here" : "Upload Recent Dataset for Drift Analysis"}
                     </p>
                     <p className="text-[10px] text-gray-500 uppercase font-black tracking-tighter mt-1 italic">Supports .csv and .xlsx formats</p>
                   </div>
                   <input type="file" className="hidden" onChange={(e) => validateAndSetFile(e.target.files[0], 'drift')} accept=".csv, .xlsx" />
                 </label>
               </div>
               
               <button 
                 onClick={runDriftAnalysis}
                 disabled={driftLoading || !driftFile}
                 className="px-8 py-4 bg-yellow-500 hover:bg-yellow-400 disabled:bg-zinc-800 disabled:text-gray-500 text-black rounded-xl font-black text-xs uppercase tracking-[0.2em] transition-all flex items-center justify-center gap-3 active:scale-95 shadow-lg shadow-yellow-500/20"
               >
                 {driftLoading ? <RefreshCcw className="animate-spin" size={18}/> : "Analyze Drift"}
               </button>
             </div>

             {/* 2. OVERALL STATUS BANNER (Dynamic Color) */}
             {driftMetrics ? (
                 <div className={`w-full p-6 rounded-2xl mb-8 border transition-all ${
                     overallDriftStatus === "Severe" ? "bg-red-900/20 border-red-600/50" : 
                     overallDriftStatus === "Moderate" ? "bg-yellow-900/20 border-yellow-600/50" :
                     "bg-green-900/20 border-green-600/50"
                 }`}>
                    <h2 className={`text-xl font-black uppercase tracking-tighter ${
                        overallDriftStatus === "Severe" ? "text-red-500" : 
                        overallDriftStatus === "Moderate" ? "text-yellow-500" :
                        "text-green-500"
                    }`}>
                        Overall Drift Status: {overallDriftStatus}
                    </h2>
                 </div>
             ) : (
                 <motion.div key="empty-drift" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-96 border-2 border-dashed border-zinc-800 rounded-[2.5rem] flex flex-col items-center justify-center text-center p-6 bg-yellow-900/10">
                  <div className="p-6 bg-zinc-900 rounded-full mb-6 border border-zinc-800"><Activity size={40} className="text-yellow-500/50 animate-pulse"/></div>
                  <h3 className="text-sm font-bold text-yellow-500/50 uppercase tracking-widest">Drift Detection Standby</h3>
                  <p className="text-xs text-gray-600 mt-2 max-w-xs italic leading-relaxed">Drag and drop a recent dataset to compare against baseline and detect feature drift.</p>
                </motion.div>
             )}

             {/* 3. METRICS GRID (Granular Coloring) */}
             {driftMetrics && (
               <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pb-20">
                 {getDriftColumns().map(([feature, v]) => {
                   
                   if (typeof v !== 'object' || v === null) return null;

                   // Individual feature status logic
                   const isSevere = v.psi > 0.25;
                   const isWarning = v.psi > 0.1 || v.js_divergence > 0.1;
                   const statusLabel = isSevere ? "Drift Detected" : isWarning ? "Warning" : "Stable";
                   const statusColor = isSevere ? "text-red-500" : isWarning ? "text-yellow-500" : "text-green-500";

                   return (
                   <div key={feature} className="bg-[#0a0a0a] border border-zinc-800 p-6 rounded-2xl relative overflow-hidden group hover:border-zinc-700 transition-colors">
                     
                     <div className="flex justify-between items-start mb-6">
                       <h3 className="text-xl font-bold text-white tracking-tight">{feature}</h3>
                       <span className={`text-[10px] font-black uppercase tracking-widest px-2 py-1 rounded ${statusColor}`}>
                         {statusLabel}
                       </span>
                     </div>

                     <div className="space-y-4 text-sm font-mono">
                        <div className="flex justify-between items-center border-b border-zinc-900 pb-2">
                            <span className="text-gray-400 font-bold text-xs">PSI (Population Stability)</span>
                            <div className="text-right">
                                <span className="text-[10px] text-gray-600 mr-2 uppercase tracking-wide">Recommended &lt; 0.25 |</span>
                                <span className={`font-black ${v?.psi > 0.25 ? 'text-red-500' : 'text-white'}`}>
                                  {v?.psi !== undefined ? v.psi.toFixed(3) : "N/A"}
                                </span>
                            </div>
                        </div>

                        <div className="flex justify-between items-center border-b border-zinc-900 pb-2">
                            <span className="text-gray-400 font-bold text-xs">KS Test (p-value)</span>
                            <div className="text-right">
                                <span className="text-[10px] text-gray-600 mr-2 uppercase tracking-wide">Recommended &gt; 0.05 |</span>
                                <span className={`font-black ${v?.ks_pvalue < 0.05 ? 'text-red-500' : 'text-white'}`}>
                                  {v?.ks_pvalue !== undefined ? v.ks_pvalue.toFixed(3) : "N/A"}
                                </span>
                            </div>
                        </div>

                        <div className="flex justify-between items-center">
                            <span className="text-gray-400 font-bold text-xs">JS Divergence</span>
                            <div className="text-right">
                                <span className="text-[10px] text-gray-600 mr-2 uppercase tracking-wide">Recommended &lt; 0.10 |</span>
                                <span className={`font-black ${v?.js_divergence > 0.10 ? 'text-red-500' : 'text-white'}`}>
                                  {v?.js_divergence !== undefined ? v.js_divergence.toFixed(3) : "N/A"}
                                </span>
                            </div>
                        </div>
                     </div>
                   </div>
                 )})}
               </div>
             )}
           </motion.div>
        )}

        {/* ===================================================================================== */}
        {/* PAGE 3: STATS ANALYTICS (BLUE)                                                       */}
        {/* ===================================================================================== */}
        {activeTab === 'stats' && (
          <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="space-y-12">
            
            {/* STATS UPLOAD BAR (BLUE THEME) */}
            <div 
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={(e) => handleDrop(e, 'stats')}
                className={`w-full bg-zinc-900/40 border-2 rounded-2xl p-6 backdrop-blur-xl flex flex-col md:flex-row items-center justify-between gap-6 transition-all duration-300 ${isDragging ? 'border-blue-600 bg-blue-600/5 scale-[1.01]' : 'border-zinc-800'}`}
              >
                <div className="flex-1 w-full">
                  <label className="flex items-center gap-4 cursor-pointer group">
                    <div className={`p-4 rounded-xl transition-all border ${isDragging ? 'bg-blue-600 text-white border-blue-500' : 'bg-zinc-800 text-gray-400 border-zinc-700 group-hover:border-blue-500/50 group-hover:text-blue-500'}`}>
                      <PieChart size={24} className={isDragging ? 'animate-bounce' : ''} />
                    </div>
                    <div>
                      <p className="text-sm font-bold text-gray-200">
                        {statsFile ? statsFile.name : isDragging ? "Drop File Here" : "Upload Dataset for Analysis"}
                      </p>
                      <p className="text-[10px] text-gray-500 uppercase font-black tracking-tighter mt-1 italic">Supports .csv and .xlsx formats</p>
                    </div>
                    <input type="file" className="hidden" onChange={(e) => validateAndSetFile(e.target.files[0], 'stats')} accept=".csv, .xlsx" />
                  </label>
                </div>
                
                <button 
                  onClick={runStatsAnalysis}
                  disabled={statsLoading || !statsFile}
                  className="px-8 py-4 bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-800 disabled:text-gray-500 text-white rounded-xl font-black text-xs uppercase tracking-[0.2em] transition-all flex items-center justify-center gap-3 active:scale-95 shadow-lg shadow-blue-600/20"
                >
                  {statsLoading ? "Processing..." : "Visualize Data"}
                </button>
              </div>

             {/* 2. DASHBOARD CONTENT */}
             <AnimatePresence>
               {statsData && (
                 <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-12">
                    
                    {/* A. KEY METRICS ROW */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                       {[
                         { label: 'Total Rows', val: statsData.summary.rows, icon: Layers },
                         { label: 'Columns', val: statsData.summary.cols, icon: FileCode2 },
                         { label: 'Missing Values', val: statsData.summary.missing, icon: AlertTriangle },
                         { label: 'Duplicates', val: statsData.summary.duplicates, icon: Users }
                       ].map((item, i) => (
                         <div key={i} className="bg-[#0A0A0A] border border-zinc-800 p-6 rounded-3xl flex items-center gap-6 hover:border-blue-500/30 transition-colors">
                           <div className="p-4 bg-blue-900/10 rounded-2xl text-blue-500"><item.icon size={24}/></div>
                           <div>
                             <div className="text-[10px] font-black text-gray-500 uppercase tracking-widest">{item.label}</div>
                             <div className="text-3xl font-black text-white">{item.val}</div>
                           </div>
                         </div>
                       ))}
                    </div>

                    {/* B. CATEGORICAL (BIGGER CHARTS - 2 Per Row) */}
                    {statsData.charts.categorical.length > 0 && (
                        <div>
                            <h3 className="text-sm font-black text-blue-400 uppercase tracking-widest mb-6 flex items-center gap-2 pl-2">
                                <span className="w-1.5 h-6 bg-blue-500 rounded-full"></span> Categorical Breakdown
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                {statsData.charts.categorical.map((chartJSON, idx) => (
                                    <div key={idx} className="bg-zinc-900/30 border border-zinc-800/50 rounded-[2rem] p-6 overflow-hidden hover:border-blue-500/20 transition-all">
                                        <Plot
                                            data={chartJSON.data}
                                            layout={{...chartJSON.layout, autosize: true, height: 350}}
                                            style={{ width: '100%', height: '100%' }}
                                            config={{ displayModeBar: false }}
                                            useResizeHandler={true}
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* C. NUMERICAL (BIGGER CHARTS - 2 Per Row) */}
                    {statsData.charts.numerical.length > 0 && (
                        <div>
                            <h3 className="text-sm font-black text-blue-400 uppercase tracking-widest mb-6 flex items-center gap-2 pl-2">
                                <span className="w-1.5 h-6 bg-blue-500 rounded-full"></span> Distribution Analysis
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                {statsData.charts.numerical.map((chartJSON, idx) => (
                                    <div key={idx} className="bg-zinc-900/30 border border-zinc-800/50 rounded-[2rem] p-6 hover:border-blue-500/20 transition-all">
                                        <Plot
                                            data={chartJSON.data}
                                            layout={{...chartJSON.layout, autosize: true, height: 350}}
                                            style={{ width: '100%', height: '100%' }}
                                            config={{ displayModeBar: false }}
                                            useResizeHandler={true}
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* D. CHURN ANALYSIS (Violin Plots) */}
                    {statsData.charts.bivariate.length > 0 && (
                        <div>
                            <h3 className="text-sm font-black text-blue-400 uppercase tracking-widest mb-6 flex items-center gap-2 pl-2">
                                <span className="w-1.5 h-6 bg-blue-500 rounded-full"></span> Churn Impact
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                {statsData.charts.bivariate.map((chartJSON, idx) => (
                                    <div key={idx} className="bg-blue-900/5 border border-blue-500/20 rounded-[2rem] p-6">
                                        <Plot
                                            data={chartJSON.data}
                                            layout={{...chartJSON.layout, autosize: true, height: 350}}
                                            style={{ width: '100%', height: '100%' }}
                                            config={{ displayModeBar: false }}
                                            useResizeHandler={true}
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* E. CORRELATION (HUGE FULL WIDTH) */}
                    {statsData.charts.correlation && (
                        <div className="pb-12">
                            <h3 className="text-sm font-black text-blue-400 uppercase tracking-widest mb-6 flex items-center gap-2 pl-2">
                                <span className="w-1.5 h-6 bg-blue-500 rounded-full"></span> Feature Correlation Matrix
                            </h3>
                            <div className="bg-zinc-900/50 border border-zinc-800/80 rounded-[2.5rem] p-4 shadow-2xl">
                                <Plot
                                    data={statsData.charts.correlation.data}
                                    layout={{...statsData.charts.correlation.layout, autosize: true}}
                                    style={{ width: '100%', height: '800px' }} // HARDCODED HUGE HEIGHT
                                    config={{ displayModeBar: false }}
                                    useResizeHandler={true}
                                />
                            </div>
                        </div>
                    )}

                 </motion.div>
               )}
             </AnimatePresence>

             {/* EMPTY STATE */}
             {!statsData && !statsLoading && (
               <motion.div key="empty-stats" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-96 border-2 border-dashed border-zinc-800 rounded-[2.5rem] flex flex-col items-center justify-center text-center p-6 bg-blue-900/10">
                  <div className="p-6 bg-zinc-900 rounded-full mb-6 border border-zinc-800"><BarChart3 size={40} className="text-blue-500/50 animate-pulse"/></div>
                  <h3 className="text-sm font-bold text-blue-500/50 uppercase tracking-widest">Analytics Engine Standby</h3>
                  <p className="text-xs text-gray-600 mt-2 max-w-xs italic leading-relaxed">Drag and drop a dataset to generate automated EDA reports and visualization insights.</p>
                </motion.div>
             )}
          </motion.div>
        )}

      </div>
    </div>
  );
};

export default App;