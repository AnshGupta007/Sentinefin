import React from "react";
import ReactECharts from "echarts-for-react";
import { X, BarChart3, Activity, Tag } from "lucide-react";
import { ClusterMeta } from "../types/cluster";

interface ClusterDistributionModalProps {
  isOpen: boolean;
  onClose: () => void;
  clusters: ClusterMeta[];
}

export const ClusterDistributionModal: React.FC<ClusterDistributionModalProps> = ({
  isOpen,
  onClose,
  clusters,
}) => {
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  if (!isOpen) return null;

  // Horizontal bar chart for ultra-clean label readability without overlap
  const sortedClusters = [...clusters].sort((a, b) => a.count - b.count);

  const barChartOption = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: "#111827",
      borderColor: "#374151",
      textStyle: { color: "#F3F4F6", fontSize: 12 },
      formatter: (params: any) => {
        const item = params[0];
        return `
          <div class="p-1">
            <div class="font-bold text-white mb-1 flex items-center gap-1.5">
              <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background-color:${item.color}"></span>
              ${item.name}
            </div>
            <div class="text-xs text-gray-300"><b>Complaint Volume:</b> ${item.value.toLocaleString()} records</div>
          </div>
        `;
      },
    },
    grid: {
      top: 20,
      right: 35,
      bottom: 25,
      left: 140,
      containLabel: true,
    },
    xAxis: {
      type: "value",
      name: "Complaint Count",
      nameTextStyle: { color: "#9CA3AF", fontSize: 10 },
      splitLine: { lineStyle: { color: "#1F2937", type: "dashed" } },
      axisLabel: { color: "#9CA3AF", fontSize: 10, fontFamily: "JetBrains Mono, monospace" },
    },
    yAxis: {
      type: "category",
      data: sortedClusters.map((c) => c.label),
      axisLabel: {
        color: "#D1D5DB",
        fontSize: 11,
        formatter: (val: string) => (val.length > 22 ? val.slice(0, 22) + "…" : val),
      },
      axisLine: { lineStyle: { color: "#4B5563" } },
    },
    series: [
      {
        name: "Cluster Size",
        type: "bar",
        barWidth: "60%",
        data: sortedClusters.map((c) => ({
          value: c.count,
          itemStyle: {
            color: c.color,
            borderRadius: [0, 4, 4, 0],
          },
        })),
      },
    ],
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-6 animate-in fade-in duration-150"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-[#0e1424] border border-gray-800 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-[#111827]">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-950/60 border border-indigo-700/50 flex items-center justify-center text-indigo-400">
              <BarChart3 className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
                Cluster Volume & Distribution Diagnostics
              </h2>
              <p className="text-xs text-gray-400">
                Evaluation of cluster support sizes and semantic representations
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* Horizontal Bar Chart */}
          <div className="bg-gray-900/60 p-4 rounded-xl border border-gray-800">
            <h3 className="text-xs font-bold text-gray-300 uppercase tracking-wider font-mono mb-2 flex items-center gap-2">
              <Activity className="w-3.5 h-3.5 text-blue-400" /> Cluster Complaint Distribution
            </h3>
            <div className="h-72 w-full">
              <ReactECharts option={barChartOption} style={{ height: "100%", width: "100%" }} />
            </div>
          </div>

          {/* Clean Keyword Badges & Details */}
          <div>
            <h3 className="text-xs font-bold text-gray-300 uppercase tracking-wider font-mono mb-3 flex items-center gap-1.5">
              <Tag className="w-3.5 h-3.5 text-indigo-400" /> Top Characteristic Financial Keywords by Cluster
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {clusters.map((c) => (
                <div
                  key={c.id}
                  className="bg-gray-900/40 p-3.5 rounded-xl border border-gray-800/90 flex items-start gap-3"
                >
                  <span
                    className="w-3.5 h-3.5 rounded-full flex-shrink-0 mt-0.5"
                    style={{ backgroundColor: c.color }}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <h4 className="text-xs font-bold text-gray-100 truncate">{c.label}</h4>
                      <span className="text-[11px] font-mono text-gray-400 flex-shrink-0 bg-gray-900 px-2 py-0.5 rounded border border-gray-800">
                        {c.count.toLocaleString()} records
                      </span>
                    </div>
                    {c.keywords && c.keywords.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mt-2">
                        {c.keywords.map((kw, i) => (
                          <span
                            key={i}
                            className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-gray-800 text-gray-200 border border-gray-700"
                          >
                            {kw}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-gray-800 bg-[#111827] flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 text-xs font-semibold rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-200 transition-colors border border-gray-700 font-mono"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
