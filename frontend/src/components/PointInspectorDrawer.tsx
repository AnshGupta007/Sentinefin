import React from "react";
import { X, Calendar, Building, AlertCircle, Hash, Compass } from "lucide-react";
import { ClusterMeta, ClusterPoint, ProjectionType } from "../types/cluster";

interface PointInspectorDrawerProps {
  point: ClusterPoint | null;
  clusters: ClusterMeta[];
  onClose: () => void;
  projection: ProjectionType;
}

export const PointInspectorDrawer: React.FC<PointInspectorDrawerProps> = ({
  point,
  clusters,
  onClose,
  projection,
}) => {
  if (!point) return null;

  const cluster = clusters.find((c) => c.id === point.cluster_id);
  const coordX = projection === "PCA" && point.pca_x !== undefined ? point.pca_x : point.x;
  const coordY = projection === "PCA" && point.pca_y !== undefined ? point.pca_y : point.y;

  return (
    <aside className="w-96 border-l border-gray-800 bg-[#0e1424] p-5 flex flex-col space-y-4 overflow-y-auto shadow-2xl z-20 transition-all duration-200">
      {/* Drawer Header */}
      <div className="flex items-center justify-between border-b border-gray-800 pb-3">
        <div className="flex items-center gap-2">
          <Compass className="w-4 h-4 text-blue-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-gray-200 font-mono">
            Latent Point Inspector
          </h3>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-lg text-gray-500 hover:text-gray-300 hover:bg-gray-800 transition-colors"
          title="Close Inspector"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Cluster Assignment Badge */}
      <div>
        <span className="text-[10px] text-gray-500 font-mono uppercase block mb-1">
          Cluster Assignment & Topic
        </span>
        <div
          className="p-3 rounded-xl border flex items-center space-x-2.5"
          style={{
            backgroundColor: `${cluster?.color || '#3b82f6'}15`,
            borderColor: `${cluster?.color || '#3b82f6'}40`,
          }}
        >
          <span
            className="w-3.5 h-3.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: cluster?.color || '#3b82f6' }}
          />
          <div>
            <h4 className="text-xs font-bold text-white leading-tight">
              {cluster?.label || `Cluster ${point.cluster_id}`}
            </h4>
            <span className="text-[10px] font-mono" style={{ color: cluster?.color || '#3b82f6' }}>
              Cluster #{point.cluster_id} (Global DEC Partition)
            </span>
          </div>
        </div>
      </div>

      {/* Coordinate & Confidence Metrics Grid */}
      <div className="grid grid-cols-2 gap-2.5">
        <div className="bg-gray-900/80 p-3 rounded-xl border border-gray-800">
          <span className="text-[10px] text-gray-500 font-mono uppercase block">
            {projection} Coordinates
          </span>
          <p className="font-mono text-xs text-blue-300 font-semibold mt-1">
            [{coordX}, {coordY}]
          </p>
        </div>

        <div className="bg-gray-900/80 p-3 rounded-xl border border-gray-800">
          <span className="text-[10px] text-gray-500 font-mono uppercase block">
            Membership Conf.
          </span>
          <p className="font-mono text-xs text-emerald-400 font-semibold mt-1">
            {(point.confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Complaint Metadata Attributes */}
      <div className="space-y-3 bg-gray-900/40 p-3.5 rounded-xl border border-gray-800/80">
        <div>
          <span className="text-[10px] text-gray-500 font-mono uppercase flex items-center gap-1">
            <Hash className="w-3 h-3 text-gray-400" /> Complaint Unique ID
          </span>
          <p className="text-xs font-mono text-gray-200 mt-0.5">{point.id}</p>
        </div>

        {point.metadata.issue && (
          <div>
            <span className="text-[10px] text-gray-500 font-mono uppercase flex items-center gap-1">
              <AlertCircle className="w-3 h-3 text-amber-400" /> Primary Issue
            </span>
            <p className="text-xs font-semibold text-gray-200 mt-0.5">
              {point.metadata.issue}
            </p>
          </div>
        )}

        {point.metadata.company && (
          <div>
            <span className="text-[10px] text-gray-500 font-mono uppercase flex items-center gap-1">
              <Building className="w-3 h-3 text-indigo-400" /> Financial Entity
            </span>
            <p className="text-xs text-gray-300 mt-0.5">{point.metadata.company}</p>
          </div>
        )}

        {point.metadata.timestamp && (
          <div>
            <span className="text-[10px] text-gray-500 font-mono uppercase flex items-center gap-1">
              <Calendar className="w-3 h-3 text-gray-400" /> Submission Timestamp
            </span>
            <p className="text-xs font-mono text-gray-400 mt-0.5">
              {point.metadata.timestamp} {point.window_id && `(Window: ${point.window_id})`}
            </p>
          </div>
        )}
      </div>

      {/* Full Contextual Text Narrative */}
      <div className="flex-1 flex flex-col min-h-0">
        <span className="text-[10px] text-gray-500 font-mono uppercase block mb-1.5">
          Consumer Narrative Excerpt
        </span>
        <div className="flex-1 bg-gray-950/80 p-3 rounded-xl border border-gray-800 text-xs text-gray-300 leading-relaxed overflow-y-auto max-h-56 font-sans">
          "{point.metadata.snippet}"
        </div>
      </div>
    </aside>
  );
};
