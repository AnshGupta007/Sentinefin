import React from "react";
import { Activity, Layers, Database, ShieldAlert, TrendingUp } from "lucide-react";
import { ClusterMetrics } from "../types/cluster";

interface MetricCardsProps {
  metrics: ClusterMetrics;
  visibleClustersCount: number;
}

export const MetricCards: React.FC<MetricCardsProps> = ({
  metrics,
  visibleClustersCount,
}) => {
  return (
    <section className="grid grid-cols-2 md:grid-cols-4 gap-4 px-6 py-4 bg-[#0a0e17] border-b border-gray-800/80">
      {/* 1. Silhouette Coefficient */}
      <div className="bg-[#111827] p-4 rounded-xl border border-gray-800/90 shadow-sm relative overflow-hidden group hover:border-emerald-500/40 transition-colors">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[11px] font-semibold tracking-wider text-gray-400 uppercase">
              Silhouette Score
            </p>
            <h3 className="text-2xl font-extrabold font-mono text-emerald-400 mt-1">
              {metrics.silhouette_score.toFixed(3)}
            </h3>
            <span className="text-[10px] text-emerald-400/90 font-medium flex items-center gap-1 mt-1">
              <TrendingUp className="w-3 h-3" /> High Boundary Separation (&gt;0.70)
            </span>
          </div>
          <div className="w-11 h-11 rounded-xl bg-emerald-950/40 border border-emerald-800/50 flex items-center justify-center text-emerald-400 shadow-inner">
            <Activity className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* 2. Davies-Bouldin Index */}
      <div className="bg-[#111827] p-4 rounded-xl border border-gray-800/90 shadow-sm relative overflow-hidden group hover:border-blue-500/40 transition-colors">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[11px] font-semibold tracking-wider text-gray-400 uppercase">
              Davies-Bouldin Index
            </p>
            <h3 className="text-2xl font-extrabold font-mono text-blue-400 mt-1">
              {metrics.davies_bouldin_index.toFixed(3)}
            </h3>
            <span className="text-[10px] text-blue-300/80 font-medium mt-1 block">
              Optimal Cluster Dispersion (&lt;0.60)
            </span>
          </div>
          <div className="w-11 h-11 rounded-xl bg-blue-950/40 border border-blue-800/50 flex items-center justify-center text-blue-400 shadow-inner">
            <Layers className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* 3. Total & Active Clusters */}
      <div className="bg-[#111827] p-4 rounded-xl border border-gray-800/90 shadow-sm relative overflow-hidden group hover:border-indigo-500/40 transition-colors">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[11px] font-semibold tracking-wider text-gray-400 uppercase">
              DEC Emergent Clusters
            </p>
            <h3 className="text-2xl font-extrabold font-mono text-indigo-300 mt-1">
              {metrics.total_clusters}
            </h3>
            <span className="text-[10px] text-indigo-400/90 font-medium mt-1 block">
              {visibleClustersCount} / {metrics.total_clusters} Visible in Projection
            </span>
          </div>
          <div className="w-11 h-11 rounded-xl bg-indigo-950/40 border border-indigo-800/50 flex items-center justify-center text-indigo-400 shadow-inner">
            <Layers className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* 4. Total Samples & Noise */}
      <div className="bg-[#111827] p-4 rounded-xl border border-gray-800/90 shadow-sm relative overflow-hidden group hover:border-amber-500/40 transition-colors">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[11px] font-semibold tracking-wider text-gray-400 uppercase">
              Sample Embeddings
            </p>
            <h3 className="text-2xl font-extrabold font-mono text-amber-400 mt-1">
              {metrics.total_points.toLocaleString()}
            </h3>
            <span className="text-[10px] text-amber-400/90 font-medium mt-1 flex items-center gap-1">
              <ShieldAlert className="w-3 h-3 text-amber-400" />
              {metrics.noise_count} Outliers / Anomaly Points
            </span>
          </div>
          <div className="w-11 h-11 rounded-xl bg-amber-950/40 border border-amber-800/50 flex items-center justify-center text-amber-400 shadow-inner">
            <Database className="w-5 h-5" />
          </div>
        </div>
      </div>
    </section>
  );
};
