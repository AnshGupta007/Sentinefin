import React from "react";
import { Search, Eye, EyeOff, Sliders, Sparkles } from "lucide-react";
import { ClusterMeta } from "../types/cluster";

interface SidebarProps {
  clusters: ClusterMeta[];
  selectedClusters: number[];
  onToggleCluster: (id: number) => void;
  onSelectAll: () => void;
  onDeselectAll: () => void;
  searchQuery: string;
  onSearchChange: (q: string) => void;
  pointSize: number;
  onPointSizeChange: (s: number) => void;
  pointOpacity: number;
  onPointOpacityChange: (o: number) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  clusters,
  selectedClusters,
  onToggleCluster,
  onSelectAll,
  onDeselectAll,
  searchQuery,
  onSearchChange,
  pointSize,
  onPointSizeChange,
  pointOpacity,
  onPointOpacityChange,
}) => {
  return (
    <aside className="w-80 border-r border-gray-800 bg-[#0e1424] flex flex-col p-4 space-y-4 overflow-y-auto">
      {/* 1. SEARCH BOX */}
      <div>
        <label className="text-[11px] font-bold text-gray-400 uppercase tracking-wider block mb-1.5 font-mono">
          Filter Complaint Semantics
        </label>
        <div className="relative">
          <Search className="w-4 h-4 text-gray-500 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search keywords (e.g., dispute, fee)..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            className="w-full bg-gray-900/90 border border-gray-700/80 rounded-lg pl-9 pr-3 py-1.5 text-xs text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors"
          />
        </div>
      </div>

      {/* 2. CLUSTER VISIBILITY CONTROLS */}
      <div className="space-y-2 flex-1 flex flex-col min-h-0">
        <div className="flex items-center justify-between">
          <label className="text-[11px] font-bold text-gray-400 uppercase tracking-wider font-mono">
            Emergent Clusters ({clusters.length})
          </label>
          <div className="space-x-1.5 text-[11px] font-medium">
            <button
              onClick={onSelectAll}
              className="text-blue-400 hover:text-blue-300 transition-colors"
            >
              Select All
            </button>
            <span className="text-gray-600">|</span>
            <button
              onClick={onDeselectAll}
              className="text-gray-400 hover:text-gray-300 transition-colors"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="space-y-1.5 overflow-y-auto pr-1 flex-1 max-h-72">
          {clusters.map((c) => {
            const isSelected = selectedClusters.includes(c.id);
            return (
              <div
                key={c.id}
                onClick={() => onToggleCluster(c.id)}
                className={`flex items-start justify-between p-2 rounded-lg cursor-pointer transition-all border ${
                  isSelected
                    ? "bg-gray-800/80 border-gray-700 text-gray-200 shadow-sm"
                    : "bg-gray-900/30 border-gray-800/40 text-gray-500 opacity-50 hover:opacity-75"
                }`}
              >
                <div className="flex items-start space-x-2.5 min-w-0 flex-1">
                  <span
                    className="w-3 h-3 rounded-full flex-shrink-0 mt-0.5"
                    style={{ backgroundColor: c.color }}
                  />
                  <div className="min-w-0 flex-1">
                    <p className="text-xs font-semibold truncate leading-tight">{c.label}</p>
                    {c.keywords && c.keywords.length > 0 && (
                      <p className="text-[10px] text-gray-400 truncate mt-0.5 font-mono">
                        {c.keywords.slice(0, 3).join(", ")}
                      </p>
                    )}
                  </div>
                </div>

                <div className="flex items-center space-x-1.5 pl-2 flex-shrink-0">
                  <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-gray-900 border border-gray-700 text-gray-300">
                    {c.count}
                  </span>
                  {isSelected ? (
                    <Eye className="w-3.5 h-3.5 text-blue-400" />
                  ) : (
                    <EyeOff className="w-3.5 h-3.5 text-gray-600" />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 3. VISUAL TUNING SLIDERS */}
      <div className="pt-3 border-t border-gray-800 space-y-3">
        <label className="text-[11px] font-bold text-gray-400 uppercase tracking-wider flex items-center gap-1.5 font-mono">
          <Sliders className="w-3.5 h-3.5 text-indigo-400" /> Canvas Visual Tuning
        </label>

        <div>
          <div className="flex justify-between text-xs text-gray-400 mb-1 font-mono">
            <span>Marker Radius</span>
            <span className="text-gray-200">{pointSize}px</span>
          </div>
          <input
            type="range"
            min={2}
            max={12}
            value={pointSize}
            onChange={(e) => onPointSizeChange(Number(e.target.value))}
            className="w-full accent-blue-500 bg-gray-800 h-1.5 rounded-lg cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between text-xs text-gray-400 mb-1 font-mono">
            <span>Opacity Density</span>
            <span className="text-gray-200">{Math.round(pointOpacity * 100)}%</span>
          </div>
          <input
            type="range"
            min={0.15}
            max={1.0}
            step={0.05}
            value={pointOpacity}
            onChange={(e) => onPointOpacityChange(Number(e.target.value))}
            className="w-full accent-blue-500 bg-gray-800 h-1.5 rounded-lg cursor-pointer"
          />
        </div>
      </div>

      {/* 4. ACADEMIC HIGHLIGHT TIP */}
      <div className="p-3 bg-blue-950/30 border border-blue-800/40 rounded-xl text-[11px] text-blue-300/90 leading-relaxed">
        <div className="font-semibold flex items-center gap-1 text-blue-200 mb-1">
          <Sparkles className="w-3 h-3 text-cyan-400" /> Self-Supervised DEC
        </div>
        Student-$t$ distribution kernel ($\alpha=1$) ensures dense manifolds remain separable without overlapping centroid collapse.
      </div>
    </aside>
  );
};
