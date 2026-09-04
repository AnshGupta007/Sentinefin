import React from "react";
import { Database, ChevronRight } from "lucide-react";
import { ClusterMeta, ClusterPoint, ProjectionType } from "../types/cluster";

interface DataTableStripProps {
  points: ClusterPoint[];
  clusters: ClusterMeta[];
  selectedPoint: ClusterPoint | null;
  onPointSelect: (pt: ClusterPoint) => void;
  projection: ProjectionType;
}

export const DataTableStrip: React.FC<DataTableStripProps> = ({
  points,
  clusters,
  selectedPoint,
  onPointSelect,
  projection,
}) => {
  return (
    <div className="h-52 border-t border-gray-800 bg-[#0e1424] p-3 flex flex-col shadow-inner">
      <div className="flex items-center justify-between mb-2 px-1">
        <div className="flex items-center gap-2">
          <Database className="w-3.5 h-3.5 text-indigo-400" />
          <span className="text-xs font-bold uppercase text-gray-300 tracking-wider font-mono">
            Latent Point Stream
          </span>
          <span className="text-[11px] text-gray-500 font-sans">
            (Click any row or scatter node to inspect deep metadata)
          </span>
        </div>
        <span className="text-[11px] font-mono text-gray-400">
          Showing <b>{Math.min(points.length, 100)}</b> of {points.length.toLocaleString()} matching records
        </span>
      </div>

      <div className="flex-1 overflow-x-auto overflow-y-auto rounded-lg border border-gray-800/80 bg-gray-950/50">
        <table className="w-full text-left text-xs text-gray-300">
          <thead className="bg-[#111827] text-gray-400 uppercase text-[10px] font-mono border-b border-gray-800 sticky top-0 z-10">
            <tr>
              <th className="py-2 px-3 w-28">Complaint ID</th>
              <th className="py-2 px-3 w-48">Cluster Partition</th>
              <th className="py-2 px-3 w-36">{projection} Coordinates</th>
              <th className="py-2 px-3 w-28">Confidence</th>
              <th className="py-2 px-3 w-40">Company</th>
              <th className="py-2 px-3">Narrative Snippet</th>
              <th className="py-2 px-2 w-10 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/60 font-sans">
            {points.slice(0, 100).map((pt) => {
              const cluster = clusters.find((c) => c.id === pt.cluster_id);
              const isSelected = selectedPoint?.id === pt.id;
              const coordX = projection === "PCA" && pt.pca_x !== undefined ? pt.pca_x : pt.x;
              const coordY = projection === "PCA" && pt.pca_y !== undefined ? pt.pca_y : pt.y;

              return (
                <tr
                  key={pt.id}
                  onClick={() => onPointSelect(pt)}
                  className={`cursor-pointer transition-colors duration-150 ${
                    isSelected
                      ? "bg-blue-950/60 border-l-4 border-blue-500 text-white font-medium"
                      : "hover:bg-gray-800/60 text-gray-300"
                  }`}
                >
                  <td className="py-2 px-3 font-mono text-gray-400 text-[11px]">{pt.id}</td>
                  <td className="py-2 px-3">
                    <span
                      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-semibold"
                      style={{
                        backgroundColor: `${cluster?.color || '#3b82f6'}20`,
                        color: cluster?.color || '#3b82f6',
                        border: `1px solid ${cluster?.color || '#3b82f6'}40`,
                      }}
                    >
                      <span
                        className="w-1.5 h-1.5 rounded-full"
                        style={{ backgroundColor: cluster?.color || '#3b82f6' }}
                      />
                      <span className="truncate max-w-[140px]">
                        {cluster?.label || `Cluster ${pt.cluster_id}`}
                      </span>
                    </span>
                  </td>
                  <td className="py-2 px-3 font-mono text-gray-400 text-[11px]">
                    [{coordX}, {coordY}]
                  </td>
                  <td className="py-2 px-3 font-mono text-emerald-400 text-[11px]">
                    {(pt.confidence * 100).toFixed(1)}%
                  </td>
                  <td className="py-2 px-3 text-gray-300 truncate max-w-[140px]">
                    {pt.metadata.company || "CFPB Reporting"}
                  </td>
                  <td className="py-2 px-3 text-gray-400 truncate max-w-md">
                    {pt.metadata.snippet}
                  </td>
                  <td className="py-2 px-2 text-center text-gray-500 hover:text-blue-400">
                    <ChevronRight className="w-4 h-4 mx-auto" />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
