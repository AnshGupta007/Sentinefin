import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { ClusterMeta, ClusterPoint, ProjectionType } from "../types/cluster";

interface ClusterScatterPlotProps {
  clusters: ClusterMeta[];
  points: ClusterPoint[];
  selectedClusters: number[];
  projection: ProjectionType;
  pointSize: number;
  pointOpacity: number;
  selectedPoint: ClusterPoint | null;
  onPointSelect: (pt: ClusterPoint) => void;
}

export const ClusterScatterPlot: React.FC<ClusterScatterPlotProps> = ({
  clusters,
  points,
  selectedClusters,
  projection,
  pointSize,
  pointOpacity,
  selectedPoint,
  onPointSelect,
}) => {
  // Build series mapping for ECharts
  const chartOption = useMemo(() => {
    // Generate cluster series
    const seriesList = clusters.map((cluster) => {
      const isVisible = selectedClusters.includes(cluster.id);
      if (!isVisible) {
        return {
          name: cluster.label,
          type: "scatter",
          data: [],
        };
      }

      const clusterPoints = points
        .filter((p) => p.cluster_id === cluster.id)
        .map((p) => {
          // Coordinate depending on projection
          const coordX = projection === "PCA" && p.pca_x !== undefined ? p.pca_x : p.x;
          const coordY = projection === "PCA" && p.pca_y !== undefined ? p.pca_y : p.y;
          return [coordX, coordY, p.confidence, p.metadata.title, p.id, p];
        });

      return {
        name: cluster.label,
        type: "scatter",
        symbolSize: pointSize,
        itemStyle: {
          color: cluster.color,
          opacity: pointOpacity,
          shadowBlur: 3,
          shadowColor: `${cluster.color}55`,
        },
        emphasis: {
          focus: "series",
          itemStyle: {
            borderColor: "#ffffff",
            borderWidth: 2,
            shadowBlur: 10,
            shadowColor: cluster.color,
          },
        },
        data: clusterPoints,
      };
    });

    // If there is an active selected point, add an explicit highlight ring series
    if (selectedPoint) {
      const activeX =
        projection === "PCA" && selectedPoint.pca_x !== undefined
          ? selectedPoint.pca_x
          : selectedPoint.x;
      const activeY =
        projection === "PCA" && selectedPoint.pca_y !== undefined
          ? selectedPoint.pca_y
          : selectedPoint.y;

      seriesList.push({
        name: "Inspected Point",
        type: "scatter",
        symbolSize: pointSize * 2.5,
        itemStyle: {
          color: "#ffffff",
          borderColor: "#38bdf8",
          borderWidth: 3,
          shadowBlur: 12,
          shadowColor: "#38bdf8",
        },
        data: [[activeX, activeY, selectedPoint.confidence, selectedPoint.metadata.title, selectedPoint.id, selectedPoint]],
        z: 999,
      } as any);
    }

    return {
      backgroundColor: "transparent",
      animationDuration: 600,
      grid: {
        top: 45,
        right: 35,
        bottom: 50,
        left: 55,
        containLabel: true,
      },
      tooltip: {
        trigger: "item",
        backgroundColor: "#111827",
        borderColor: "#374151",
        borderWidth: 1,
        padding: [10, 14],
        textStyle: { color: "#F3F4F6", fontSize: 12, fontFamily: "Inter, sans-serif" },
        formatter: (params: any) => {
          const raw = params.data?.[5] as ClusterPoint;
          if (!raw) return "";
          return `
            <div style="min-width: 200px; max-width: 320px;">
              <div style="font-weight: 700; color: #fff; border-bottom: 1px solid #374151; padding-bottom: 4px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background-color:${params.color || '#3b82f6'};"></span>
                <span>${params.seriesName}</span>
              </div>
              <div style="font-size: 11px; color: #9ca3af; margin-bottom: 3px;"><b>ID:</b> <span style="font-family: monospace; color: #e5e7eb;">${raw.id}</span></div>
              <div style="font-size: 11px; color: #9ca3af; margin-bottom: 3px;"><b>Coordinates:</b> <span style="font-family: monospace;">[${raw.x}, ${raw.y}]</span></div>
              <div style="font-size: 11px; color: #34d399; margin-bottom: 6px;"><b>Cluster Assignment Conf:</b> ${(raw.confidence * 100).toFixed(1)}%</div>
              <div style="font-size: 11px; color: #d1d5db; line-height: 1.4; background: rgba(0,0,0,0.25); padding: 6px; border-radius: 6px;">${raw.metadata.snippet.slice(0, 120)}...</div>
            </div>
          `;
        },
      },
      toolbox: {
        right: 25,
        top: 8,
        iconStyle: { borderColor: "#9CA3AF" },
        feature: {
          dataZoom: { title: { zoom: "Box Zoom", back: "Zoom Out" } },
          brush: { type: ["rect", "polygon", "keep", "clear"], title: { rect: "Box Select", polygon: "Lasso Select", clear: "Clear" } },
          restore: { title: "Reset View" },
          saveAsImage: { title: "Export High-Res PNG", pixelRatio: 3 },
        },
      },
      dataZoom: [
        { type: "inside", xAxisIndex: 0, yAxisIndex: 0 },
        { type: "slider", show: false },
      ],
      xAxis: {
        type: "value",
        scale: true,
        splitLine: { lineStyle: { color: "#1F2937", type: "dashed" } },
        axisLine: { lineStyle: { color: "#4B5563" } },
        axisLabel: { color: "#9CA3AF", fontSize: 11, fontFamily: "JetBrains Mono, monospace" },
        name: `${projection} Component 1`,
        nameLocation: "middle",
        nameGap: 30,
        nameTextStyle: { color: "#9CA3AF", fontSize: 12, fontWeight: 500 },
      },
      yAxis: {
        type: "value",
        scale: true,
        splitLine: { lineStyle: { color: "#1F2937", type: "dashed" } },
        axisLine: { lineStyle: { color: "#4B5563" } },
        axisLabel: { color: "#9CA3AF", fontSize: 11, fontFamily: "JetBrains Mono, monospace" },
        name: `${projection} Component 2`,
        nameLocation: "middle",
        nameGap: 35,
        nameTextStyle: { color: "#9CA3AF", fontSize: 12, fontWeight: 500 },
      },
      series: seriesList,
    };
  }, [clusters, points, selectedClusters, projection, pointSize, pointOpacity, selectedPoint]);

  const onEvents = {
    click: (params: any) => {
      if (params.data && params.data[5]) {
        onPointSelect(params.data[5] as ClusterPoint);
      }
    },
  };

  return (
    <div className="flex-1 w-full h-full relative bg-[#0b0f19]">
      {/* Active count badge */}
      <div className="absolute top-3 left-5 z-10 bg-gray-900/80 backdrop-blur-md px-3 py-1 rounded-lg border border-gray-800/90 text-xs text-gray-300 flex items-center gap-2 shadow-md">
        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span>
          Rendering <strong className="text-white">{points.length.toLocaleString()}</strong> latent coordinates
        </span>
      </div>

      <ReactECharts
        option={chartOption}
        style={{ height: "100%", width: "100%" }}
        onEvents={onEvents}
        notMerge={true}
        lazyUpdate={true}
      />
    </div>
  );
};
