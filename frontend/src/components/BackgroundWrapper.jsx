import React from 'react';

const BackgroundWrapper = ({ children }) => {
  return (
    <div className="min-h-screen w-full relative bg-[#0f0f0f] text-gray-100 overflow-x-hidden font-sans selection:bg-red-500 selection:text-white">
      {/* Top Right Red Haze */}
      <div className="fixed top-[-10%] right-[-5%] w-[600px] h-[600px] bg-red-900/20 rounded-full blur-[120px] animate-breathe pointer-events-none z-0"></div>
      
      {/* Bottom Left Blue Haze */}
      <div className="fixed bottom-[-10%] left-[-10%] w-[600px] h-[600px] bg-indigo-900/10 rounded-full blur-[100px] animate-breathe delay-1000 pointer-events-none z-0"></div>

      {/* Film Grain */}
      <div className="fixed inset-0 opacity-[0.04] pointer-events-none z-10" 
           style={{ backgroundImage: 'url("https://grainy-gradients.vercel.app/noise.svg")' }}>
      </div>

      <div className="relative z-20">
        {children}
      </div>
    </div>
  );
};

export default BackgroundWrapper;