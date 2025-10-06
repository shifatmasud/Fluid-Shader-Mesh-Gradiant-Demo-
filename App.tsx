import React from 'react';
import ShaderCanvas from './components/ShaderCanvas';

const App: React.FC = () => {
  return (
    <main className="relative w-screen h-screen bg-transparent overflow-hidden">
      <ShaderCanvas />
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none flex flex-col justify-between p-8 md:p-12">
          <header>
              <h1 className="text-2xl md:text-3xl font-bold text-white tracking-tight" style={{ textShadow: '0 1px 8px rgba(0,0,0,0.5)' }}>Fluidum</h1>
              <p className="text-sm text-gray-200 font-light tracking-wide" style={{ textShadow: '0 1px 8px rgba(0,0,0,0.5)' }}>A canvas of liquid light.</p>
          </header>
          <footer className="text-center">
              <p className="text-xs text-gray-400 font-light tracking-wide" style={{ textShadow: '0 1px 8px rgba(0,0,0,0.5)' }}>Interact with your cursor. Explore the presets.</p>
          </footer>
      </div>
    </main>
  );
};

export default App;