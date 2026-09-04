import React from "react";
import { Layers, Download, BarChart2, Sparkles } from "lucide-react";
import { ProjectionType } from "../types/cluster";

interface HeaderProps {
  datasetName: string;
  modelArchitecture?: string;
  projection: ProjectionType;
  onProjectionChange: (p: ProjectionType) => void;
  onOpenDistributionModal: () => void;
  activePointsCount: number;
}

export const Header: React.FC<HeaderProps> = ({
  datasetName,
  modelArchitecture,
  projection,
  onProjectionChange,
  onOpenDistributionModal,
  activePointsCount,
}) => {
  return (
    <header className="h-16 border-b border-gray-800 bg-[#0e1424]/90 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-30 shadow-md shadow-black/20">
      {/* Brand & Dataset Indicator */}
      <div className="flex items-center space-x-3.5">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-cyan-400 flex items-center justify-center shadow-lg shadow-blue-500/25 ring-1 ring-white/20">
          <Layers className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-bold tracking-tight text-white flex items-center gap-2">
              {datasetName}
            </h1>
            <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded-full bg-blue-950/80 border border-blue-700 text-blue-300">
              {projection} 2D
            </span>
            <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-gray-900 border border-gray-700 text-gray-400">
              {activePointsCount.toLocaleString()} pts
            </span>
          </div>
          <p className="text-[11px] text-gray-400 flex items-center gap-1.5 mt-0.5 font-sans">
            <Sparkles className="w-3 h-3 text-indigo-400" />
            {modelArchitecture || "Dense MiniLM-L6 (384-d) + Deep Embedded Clustering (DEC)"}
          </p>
        </div>
      </div>

      {/* Projection Controls, Distribution Analytics & Export */}
      <div className="flex items-center space-x-3">
        {/* Projection Mode Switcher */}
        <div className="flex items-center bg-gray-900/90 p-1 rounded-lg border border-gray-700/80 shadow-inner">
          {(["UMAP", "PCA", "t-SNE"] as const).map((method) => (
            <button
              key={method}
              onClick={() => onProjectionChange(method)}
              className={`px-3 py-1 text-xs font-semibold rounded-md transition-all duration-150 ${
                projection === method
                  ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-sm ring-1 ring-blue-400/40"
                  : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/60"
              }`}
            >
              {method}
            </button>
          ))}
        </div>

        {/* Analytics Distribution Modal Toggle */}
        <button
          onClick={onOpenDistributionModal}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-900 hover:bg-gray-800 border border-gray-700 text-gray-200 transition-colors shadow-sm"
          title="View Cluster Size & Silhouette Distribution"
        >
          <BarChart2 className="w-3.5 h-3.5 text-indigo-400" />
          <span>Analytics</span>
        </button>

        {/* Print / Export Brief */}
        <button
          onClick={() => window.print()}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-600 hover:bg-blue-500 text-white transition-colors shadow-sm shadow-blue-500/20"
        >
          <Download className="w-3.5 h-3.5" />
          <span>Export View</span>
        </button>
      </div>
    </header>
  );
};
