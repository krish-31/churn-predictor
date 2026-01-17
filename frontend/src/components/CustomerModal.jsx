import React from 'react';
import { X, Clock, Film, DollarSign, Calendar, User, Zap } from 'lucide-react';

const CustomerModal = ({ customer, onClose }) => {
  if (!customer) return null;

  const getTheme = (risk) => {
    if (risk === 'High') return { bg: 'bg-red-600', text: 'text-red-500', border: 'border-red-200' };
    if (risk === 'Medium') return { bg: 'bg-yellow-500', text: 'text-yellow-500', border: 'border-yellow-200' };
    return { bg: 'bg-green-600', text: 'text-green-500', border: 'border-green-200' };
  };

  const theme = getTheme(customer.risk_category);

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
      <div className="bg-[#141414] border border-gray-800 rounded-2xl shadow-2xl w-full max-w-3xl overflow-hidden transform transition-all scale-100 relative">
        <div className={`${theme.bg} p-6 text-white flex justify-between items-start`}>
          <div>
            <h2 className="text-3xl font-bold flex items-center gap-3">
              <User className="w-8 h-8" />
              Customer {customer.customer_id}
            </h2>
            <p className="opacity-90 mt-1 font-medium tracking-wide uppercase text-sm">
              {customer.persona} Profile
            </p>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-white/20 rounded-full transition-colors">
            <X size={28} />
          </button>
        </div>

        <div className="p-8">
           {/* Risk Score */}
          <div className="flex flex-col md:flex-row gap-8 mb-8">
            <div className="w-full md:w-1/3 text-center border-b md:border-b-0 md:border-r border-gray-800 pb-6 md:pb-0">
              <p className="text-gray-400 text-xs font-bold uppercase tracking-widest mb-2">Churn Probability</p>
              <div className="text-6xl font-black text-white tracking-tighter">
                {customer.churn_probability}%
              </div>
              <span className={`inline-block mt-2 px-3 py-1 rounded-full text-xs font-bold uppercase bg-white/10 text-white`}>
                {customer.risk_category} Risk
              </span>
            </div>
            
            <div className="w-full md:w-2/3">
               <p className="text-gray-400 text-xs font-bold uppercase tracking-widest mb-3 flex items-center gap-2">
                 <Zap size={14} className="text-yellow-400" /> AI Strategy
               </p>
               <div className="bg-gradient-to-r from-gray-800 to-gray-900 border border-gray-700 rounded-xl p-5 shadow-inner">
                  <p className="text-gray-200 text-lg font-medium leading-relaxed italic">
                    "{customer.insight.replace("✨ AI: ", "")}"
                  </p>
               </div>
            </div>
          </div>

          <h3 className="text-gray-500 font-bold text-xs uppercase tracking-widest mb-4">Behavioral Data</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
             <div className="bg-[#1f1f1f] p-4 rounded-xl border border-gray-800"><p className="text-xl font-bold text-white">{customer.watch_hours} hrs</p></div>
             <div className="bg-[#1f1f1f] p-4 rounded-xl border border-gray-800"><p className="text-xl font-bold text-white">{customer.last_login} days</p></div>
             <div className="bg-[#1f1f1f] p-4 rounded-xl border border-gray-800"><p className="text-xl font-bold text-white">{customer.favorite_genre}</p></div>
             <div className="bg-[#1f1f1f] p-4 rounded-xl border border-gray-800"><p className="text-xl font-bold text-white">${customer.monthly_fee}</p></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CustomerModal;