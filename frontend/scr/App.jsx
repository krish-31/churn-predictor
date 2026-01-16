import React, { useState, useMemo, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, FileText, BarChart3, Users, 
  AlertTriangle, DollarSign, Search, CheckCircle2,
  ChevronRight, Play, RefreshCcw, TrendingDown
} from 'lucide-react';

const App = () => {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null); 
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [backendHealth, setBackendHealth] = useState(false);

  // Real-time Health Check: Ping backend every 10 seconds
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await axios.get('http://127.0.0.1:8000/', { timeout: 5000 });
        setBackendHealth(true);
      } catch (err) {
        setBackendHealth(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleFileUpload = (e) => {
    const selectedFile = e.target.files[0];
    const fileName = selectedFile?.name || "";
    const isValid = fileName.endsWith('.csv') || fileName.endsWith('.xlsx');

    if (selectedFile && isValid) {
      setFile(selectedFile);
    } else {
      alert("Please select a valid CSV or Excel (.xlsx) file.");
    }
  };

  const runAnalysis = async () => {
    if (!file) return alert("Please upload a dataset first.");
    
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict-batch', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setData(response.data.data);
      setSummary(response.data.summary);
    } catch (err) {
      console.error("Analysis Error:", err);
      if (err.response && err.response.status >= 400 && err.response.status < 500) {
        alert(`📁 DATASET ERROR:\n${err.response.data.detail}`);
      } else if (!backendHealth) {
        alert("🔴 ML ENGINE OFFLINE:\nPlease ensure the backend server is running on port 8000.");
      } else {
        alert("⚠️ PROCESSING ERROR:\nCheck your data format and try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const filteredData = useMemo(() => {
    if (!data) return [];
    return data.filter(item => 
      item.customer_id.toString().toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [data, searchTerm]);

  return (
    <div className="min-h-screen bg-[#050505] text-gray-100 font-sans selection:bg-red-500/30 overflow-x-hidden">
      {/* Cinematic Background */}
      <motion.div 
        animate={{ opacity: [0.1, 0.2, 0.1], scale: [1, 1.05, 1] }}
        transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
        className="fixed inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-red-900/20 via-black to-black -z-10"
      />

      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Header */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-end mb-12 gap-4">
          <div>
            <h1 className="text-5xl font-black text-red-600 italic tracking-tighter">RETENTION HQ</h1>
            <p className="text-gray-500 font-medium uppercase tracking-[0.25em] text-[10px] mt-2">Enterprise Batch Inference Engine</p>
          </div>
          <div className="text-left md:text-right">
            <p className="text-[10px] text-gray-600 font-bold uppercase tracking-widest">System Health</p>
            <div className={`flex items-center gap-2 text-sm font-bold mt-1 ${backendHealth ? 'text-green-500' : 'text-red-500'}`}>
              <div className={`w-2 h-2 rounded-full animate-pulse ${backendHealth ? 'bg-green-500' : 'bg-red-500'}`} /> 
              {backendHealth ? 'ML Engine Active' : 'ML Engine Offline'}
            </div>
          </div>
        </header>

        {/* Action Bar */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-12">
          <div className="lg:col-span-3 bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6 backdrop-blur-xl flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex-1 w-full">
              <label className="flex items-center gap-4 cursor-pointer group">
                <div className="p-4 bg-zinc-800 rounded-xl group-hover:bg-red-600/10 group-hover:text-red-500 transition-all border border-zinc-700 group-hover:border-red-500/50">
                  <Upload size={24} />
                </div>
                <div>
                  <p className="text-sm font-bold text-gray-200">{file ? file.name : "Upload Dataset (CSV / XLSX)"}</p>
                  <p className="text-[10px] text-gray-500 uppercase font-black tracking-tighter mt-1 italic">Supports direct Excel tool uploads</p>
                </div>
                <input type="file" className="hidden" onChange={handleFileUpload} accept=".csv, .xlsx" />
              </label>
            </div>
            <button 
              onClick={runAnalysis}
              disabled={loading || !file}
              className="w-full md:w-auto px-10 py-4 bg-red-600 hover:bg-red-700 disabled:bg-zinc-800 disabled:text-gray-500 rounded-xl font-black text-xs uppercase tracking-[0.2em] transition-all flex items-center justify-center gap-3 shadow-lg shadow-red-900/20 active:scale-95"
            >
              {loading ? <RefreshCcw className="animate-spin" size={18}/> : <><Play size={16} fill="currentColor"/> Process Batch</>}
            </button>
          </div>

          <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6 backdrop-blur-xl flex flex-col justify-center items-center text-center">
            <div className="flex items-center gap-2 mb-2 text-gray-500">
              <CheckCircle2 size={14}/>
              <span className="text-[10px] font-black uppercase tracking-widest">Multi-Format Ready</span>
            </div>
            <button 
              onClick={() => alert("Columns Required:\ncustomer_id, age, gender, subscription_type, watch_hours, last_login_days, region, device, monthly_fee, number_of_profiles, avg_watch_time_per_day, favorite_genre, payment_method")}
              className="text-[10px] font-bold text-red-500 hover:text-red-400 uppercase transition-colors"
            >
              View Expected Schema
            </button>
          </div>
        </div>

        {/* Dashboard Content */}
        <AnimatePresence mode="wait">
          {summary ? (
            <motion.div key="results" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
              
              {/* SECTION 1: Strategic Intelligence Panel */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="bg-gradient-to-br from-red-900/40 to-black border border-red-500/20 p-6 rounded-3xl backdrop-blur-md">
                  <div className="flex items-center justify-between mb-4">
                    <TrendingDown className="text-red-500" size={20} />
                    <span className="text-gray-500 text-[10px] font-black uppercase tracking-widest">Churn Hotspot</span>
                  </div>
                  <div className="text-2xl font-black text-red-500 tracking-tighter">{summary.top_churn_genre} Fans</div>
                  <p className="text-[10px] text-gray-400 mt-2 italic leading-relaxed">This genre shows the highest behavioral drift toward cancellation.</p>
                </div>
                
                <div className="bg-zinc-900/60 border border-zinc-800 p-6 rounded-3xl backdrop-blur-md">
                  <div className="flex items-center justify-between mb-4">
                    <BarChart3 className="text-blue-400" size={20} />
                    <span className="text-gray-500 text-[10px] font-black uppercase tracking-widest">Usage Floor</span>
                  </div>
                  <div className="text-2xl font-black text-blue-400 tracking-tighter">{summary.churner_avg_watch} hrs/wk</div>
                  <p className="text-[10px] text-gray-400 mt-2 italic leading-relaxed">Critical watch-time threshold identified for at-risk segments.</p>
                </div>

                <div className="bg-zinc-900/60 border border-zinc-800 p-6 rounded-3xl backdrop-blur-md">
                  <div className="flex items-center justify-between mb-4">
                    <DollarSign className="text-yellow-500" size={20} />
                    <span className="text-gray-500 text-[10px] font-black uppercase tracking-widest">Revenue Exposure</span>
                  </div>
                  <div className="text-2xl font-black text-yellow-500 tracking-tighter">${summary.revenue_at_risk.toLocaleString()}</div>
                  <p className="text-[10px] text-gray-400 mt-2 italic leading-relaxed">Estimated monthly recurring revenue lost without intervention.</p>
                </div>
              </div>

              {/* SECTION 2: Segmentation & Drill-Down */}
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-12">
                
                {/* Risk Segmentation Breakdown */}
                <div className="lg:col-span-1 bg-zinc-900/40 border border-zinc-800 rounded-3xl p-6 backdrop-blur-sm h-fit">
                   <h3 className="text-[10px] font-black uppercase text-gray-500 tracking-widest mb-8 flex items-center gap-2">
                     <Users size={14}/> Risk Distribution
                   </h3>
                   <div className="space-y-8">
                     {[
                       { label: 'High Risk', count: summary.high_risk_count, color: 'bg-red-600' },
                       { label: 'Medium Risk', count: summary.medium_risk_count, color: 'bg-yellow-600' },
                       { label: 'Low Risk', count: summary.low_risk_count, color: 'bg-green-600' }
                     ].map((seg, i) => (
                       <div key={i}>
                         <div className="flex justify-between text-[10px] font-bold mb-2">
                           <span className="text-gray-400">{seg.label}</span>
                           <span className="text-gray-100">{Math.round((seg.count/summary.total_customers)*100)}%</span>
                         </div>
                         <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                           <motion.div 
                             initial={{ width: 0 }} 
                             animate={{ width: `${(seg.count/summary.total_customers)*100}%` }}
                             transition={{ duration: 1, ease: "easeOut" }}
                             className={`${seg.color} h-full`} 
                           />
                         </div>
                         <p className="text-[9px] text-gray-600 mt-1 uppercase font-bold">{seg.count} Customers</p>
                       </div>
                     ))}
                   </div>
                   <div className="mt-10 pt-6 border-t border-zinc-800">
                      <p className="text-[10px] text-gray-500 font-bold uppercase mb-1 tracking-widest">Average Risk</p>
                      <div className="text-2xl font-black text-purple-500 tracking-tighter">{summary.avg_risk_score}%</div>
                   </div>
                </div>

                {/* Analysis Drill-Down Table */}
                <div className="lg:col-span-3 bg-zinc-900/40 border border-zinc-800 rounded-3xl overflow-hidden backdrop-blur-md shadow-2xl">
                  <div className="p-6 border-b border-zinc-800 flex flex-col md:flex-row justify-between items-center gap-4 bg-zinc-900/20">
                    <h2 className="text-sm font-black uppercase tracking-tighter flex items-center gap-2">
                      <div className="w-1.5 h-4 bg-red-600 rounded-full"/> Analysis Drill-Down
                    </h2>
                    <div className="relative flex-1 max-w-sm w-full">
                      <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-600" size={14}/>
                      <input 
                        placeholder="Search Customer ID..." 
                        className="w-full bg-black/40 border border-zinc-800 rounded-full py-2.5 pl-10 pr-4 text-xs focus:ring-1 focus:ring-red-600 transition-all outline-none"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="overflow-x-auto max-h-[600px] scrollbar-hide">
                    <table className="w-full text-left">
                      <thead className="sticky top-0 bg-[#0a0a0a] z-10 border-b border-zinc-800">
                        <tr className="text-[10px] font-black text-gray-500 uppercase tracking-widest">
                          <th className="px-8 py-5">Customer ID</th>
                          <th className="px-6 py-5 text-center">Prob.</th>
                          <th className="px-6 py-5">Risk Category</th>
                          <th className="px-6 py-5">AI Recommendation</th>
                          <th className="px-6 py-5"></th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-zinc-800/50">
                        {filteredData.map((row, idx) => (
                          <tr key={idx} className="hover:bg-white/[0.02] transition-colors group">
                            <td className="px-8 py-5 font-mono text-xs text-red-500 font-bold">
                              {row.customer_id.toString().slice(0, 15)}...
                            </td>
                            <td className="px-6 py-5 text-center">
                              <span className="text-lg font-black tracking-tighter">{row.churn_probability}%</span>
                            </td>
                            <td className="px-6 py-5">
                              <span className={`px-3 py-1 rounded text-[9px] font-black uppercase tracking-widest ${
                                row.risk_category === 'High' ? 'bg-red-500/10 text-red-500' : 
                                row.risk_category === 'Medium' ? 'bg-yellow-500/10 text-yellow-500' : 'bg-green-500/10 text-green-500'
                              }`}>
                                {row.risk_category}
                              </span>
                            </td>
                            <td className="px-6 py-5 text-xs text-gray-400 italic leading-relaxed">
                              {row.insight}
                            </td>
                            <td className="px-6 py-5 text-right">
                              <ChevronRight size={16} className="text-zinc-800 group-hover:text-red-500 transition-colors cursor-pointer"/>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-96 border-2 border-dashed border-zinc-800 rounded-[2.5rem] flex flex-col items-center justify-center text-center p-6 bg-zinc-900/10">
              <div className="p-6 bg-zinc-900 rounded-full mb-6 border border-zinc-800">
                <FileText size={40} className="text-zinc-700 animate-pulse"/>
              </div>
              <h3 className="text-sm font-bold text-gray-400 uppercase tracking-widest">Engine Standby</h3>
              <p className="text-xs text-gray-600 mt-2 max-w-xs italic leading-relaxed">
                Ingest CSV or Excel batch data to generate behavioral insights and identified at-risk subscriber segments.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default App;