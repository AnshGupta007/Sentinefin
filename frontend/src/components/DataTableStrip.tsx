import React, { useState, useMemo } from "react";
import { Database, ChevronRight, ChevronDown, ChevronUp, FilterX } from "lucide-react";
import { ClusterMeta, ClusterPoint, ProjectionType } from "../types/cluster";

interface DataTableStripProps {
  points: ClusterPoint[];
  clusters: ClusterMeta[];
  selectedPoint: ClusterPoint | null;
  onPointSelect: (pt: ClusterPoint) => void;
  projection: ProjectionType;
  isBrushed?: boolean;
  onClearBrush?: () => void;
}

export const DataTableStrip: React.FC<DataTableStripProps> = React.memo(({
  points,
  clusters,
  selectedPoint,
  onPointSelect,
  projection,
  isBrushed,
  onClearBrush,
}) => {
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);
  const clusterMap = useMemo(() => new Map(clusters.map((c) => [c.id, c])), [clusters]);

  const getPointCoords = (pt: ClusterPoint): [number, number] => {
    if (projection === "PCA") return [pt.pca_x ?? pt.x, pt.pca_y ?? pt.y];
    if (projection === "t-SNE") return [pt.tsne_x ?? pt.x, pt.tsne_y ?? pt.y];
    return [pt.x, pt.y];
  };

  return (
    <div
      className={`border-t border-gray-800 bg-[#0e1424] flex flex-col transition-all duration-200 shadow-inner flex-shrink-0 ${
        isCollapsed ? "h-10" : "h-52"
      }`}
    >
      {/* Table Header Bar */}
      <div className="h-10 px-4 flex items-center justify-between bg-[#111827] border-b border-gray-800/80 select-none">
        <div className="flex items-center gap-2.5">
          <Database className="w-3.5 h-3.5 text-indigo-400" />
          <span className="text-xs font-bold uppercase text-gray-200 tracking-wider font-mono">
            Latent Point Stream
          </span>
          <span className="text-[11px] text-gray-400 font-sans hidden sm:inline">
            (Click any row to inspect deep narrative context)
          </span>

          {isBrushed && (
            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-blue-950/80 border border-blue-700 text-blue-300 text-[10px] font-mono">
              <span>Lasso Selection Active</span>
              {onClearBrush && (
                <button
                  onClick={onClearBrush}
                  className="hover:text-white"
                  title="Clear Lasso Selection"
                >
                  <FilterX className="w-3 h-3" />
                </button>
              )}
            </span>
          )}
        </div>

        <div className="flex items-center space-x-3">
          <span className="text-[11px] font-mono text-gray-400">
            Showing <b>{Math.min(points.length, 100)}</b> of {points.length.toLocaleString()} records
          </span>

          {/* Collapse / Expand Toggle Button */}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="p-1 rounded text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition-colors flex items-center gap-1 text-[11px] font-mono"
            title={isCollapsed ? "Expand Data Table" : "Collapse Data Table"}
          >
            {isCollapsed ? (
              <>
                <span>Expand Table</span>
                <ChevronUp className="w-3.5 h-3.5" />
              </>
            ) : (
              <>
                <span>Minimize</span>
                <ChevronDown className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </div>

      {/* Table Body (Hidden when collapsed) */}
      {!isCollapsed && (
        <div className="flex-1 overflow-x-auto overflow-y-auto p-2">
          <table className="w-full text-left text-xs text-gray-300 border-collapse">
            <thead className="bg-[#111827] text-gray-400 uppercase text-[10px] font-mono border-b border-gray-800 sticky top-0 z-10">
              <tr>
                <th className="py-2 px-3 w-28">Complaint ID</th>
                <th className="py-2 px-3 w-48">Cluster Partition</th>
                <th className="py-2 px-3 w-36">{projection} Coords</th>
                <th className="py-2 px-3 w-24">Confidence</th>
                <th className="py-2 px-3 w-48">Issue & Sub-Issue</th>
                <th className="py-2 px-3">Narrative Snippet</th>
                <th className="py-2 px-2 w-10 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/60 font-sans">
              {points.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-gray-500 font-mono text-xs">
                    No matching complaints found for the current filter criteria.
                  </td>
                </tr>
              ) : (
                points.slice(0, 100).map((pt) => {
                  const cluster = clusterMap.get(pt.cluster_id);
                  const isSelected = selectedPoint?.id === pt.id;
                  const [cx, cy] = getPointCoords(pt);

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
                      <td className="py-2 px-3 font-mono text-blue-300 text-[11px]">
                        [{cx.toFixed(3)}, {cy.toFixed(3)}]
                      </td>
                      <td className="py-2 px-3 font-mono text-emerald-400 text-[11px]">
                        {(pt.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-3 text-gray-300 truncate max-w-[160px]">
                        <span className="font-medium text-gray-200 block truncate">{pt.metadata.issue}</span>
                        {pt.metadata.sub_issue && (
                          <span className="text-[10px] text-gray-500 block truncate font-mono">
                            {pt.metadata.sub_issue}
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-3 text-gray-400 truncate max-w-md">
                        {pt.metadata.snippet}
                      </td>
                      <td className="py-2 px-2 text-center text-gray-500 hover:text-blue-400">
                        <ChevronRight className="w-4 h-4 mx-auto" />
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
});

