import React, { useMemo, useCallback, useRef } from "react";
import ReactECharts from "echarts-for-react";
import { SearchX } from "lucide-react";
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
  onBrushSelect?: (pointIds: string[]) => void;
}

export const ClusterScatterPlot: React.FC<ClusterScatterPlotProps> = React.memo(({
  clusters,
  points,
  selectedClusters,
  projection,
  pointSize,
  pointOpacity,
  selectedPoint,
  onPointSelect,
  onBrushSelect,
}) => {
  const echartsRef = useRef<any>(null);

  // O(1) Lookup Map: map id -> ClusterPoint to avoid storing heavy metadata in ECharts series
  const pointMap = useMemo(() => {
    const map = new Map<string, ClusterPoint>();
    for (let i = 0; i < points.length; i++) {
      map.set(points[i].id, points[i]);
    }
    return map;
  }, [points]);

  // Single-pass O(N) grouping of coordinates by cluster ID
  const pointsByCluster = useMemo(() => {
    const grouped = new Map<number, Array<[number, number, number, string]>>();
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      let cx = p.x;
      let cy = p.y;
      if (projection === "PCA") {
        cx = p.pca_x ?? p.x;
        cy = p.pca_y ?? p.y;
      } else if (projection === "t-SNE") {
        cx = p.tsne_x ?? p.x;
        cy = p.tsne_y ?? p.y;
      }

      let arr = grouped.get(p.cluster_id);
      if (!arr) {
        arr = [];
        grouped.set(p.cluster_id, arr);
      }
      // Store only lightweight primitives: [x, y, confidence, id]
      arr.push([cx, cy, p.confidence, p.id]);
    }
    return grouped;
  }, [points, projection]);

  // Selected cluster set for O(1) membership checks
  const selectedClusterSet = useMemo(() => new Set(selectedClusters), [selectedClusters]);

  // Build series mapping for ECharts
  const chartOption = useMemo(() => {
    const seriesList: any[] = [];

    for (let i = 0; i < clusters.length; i++) {
      const cluster = clusters[i];
      const isVisible = selectedClusterSet.has(cluster.id);
      const data = isVisible ? (pointsByCluster.get(cluster.id) || []) : [];

      seriesList.push({
        name: cluster.label,
        type: "scatter",
        symbolSize: pointSize,
        large: data.length > 2000,
        largeThreshold: 2000,
        progressive: 2000,
        itemStyle: {
          color: cluster.color,
          opacity: pointOpacity,
        },
        emphasis: {
          focus: "series",
          itemStyle: {
            borderColor: "#ffffff",
            borderWidth: 2,
            shadowBlur: 8,
            shadowColor: cluster.color,
          },
        },
        data: data,
      });
    }

    // Selected point highlight ring (lightweight single point series)
    if (selectedPoint) {
      let activeX = selectedPoint.x;
      let activeY = selectedPoint.y;
      if (projection === "PCA") {
        activeX = selectedPoint.pca_x ?? selectedPoint.x;
        activeY = selectedPoint.pca_y ?? selectedPoint.y;
      } else if (projection === "t-SNE") {
        activeX = selectedPoint.tsne_x ?? selectedPoint.x;
        activeY = selectedPoint.tsne_y ?? selectedPoint.y;
      }

      seriesList.push({
        name: "Inspected Point",
        type: "scatter",
        symbolSize: Math.max(14, pointSize * 2.5),
        itemStyle: {
          color: "#ffffff",
          borderColor: "#38bdf8",
          borderWidth: 3,
          shadowBlur: 14,
          shadowColor: "#38bdf8",
        },
        data: [[activeX, activeY, selectedPoint.confidence, selectedPoint.id]],
        z: 999,
      });
    }

    return {
      backgroundColor: "transparent",
      animation: false, // Disabling canvas layout transition drops frame delays from 400ms to 0ms
      animationDurationUpdate: 0,
      grid: {
        top: 40,
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
          const ptId = params.data?.[3] as string;
          if (!ptId) return "";
          const raw = pointMap.get(ptId);
          if (!raw) return "";

          const cx = params.data[0];
          const cy = params.data[1];
          return `
            <div style="min-width: 220px; max-width: 320px;">
              <div style="font-weight: 700; color: #fff; border-bottom: 1px solid #374151; padding-bottom: 4px; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background-color:${params.color || '#3b82f6'};"></span>
                <span>${params.seriesName}</span>
              </div>
              <div style="font-size: 11px; color: #9ca3af; margin-bottom: 3px;"><b>Complaint ID:</b> <span style="font-family: monospace; color: #e5e7eb;">${raw.id}</span></div>
              <div style="font-size: 11px; color: #9ca3af; margin-bottom: 3px;"><b>${projection} Coords:</b> <span style="font-family: monospace; color: #93c5fd;">[${cx.toFixed(3)}, ${cy.toFixed(3)}]</span></div>
              <div style="font-size: 11px; color: #34d399; margin-bottom: 6px;"><b>Cluster Membership:</b> ${(raw.confidence * 100).toFixed(1)}%</div>
              <div style="font-size: 11px; color: #d1d5db; line-height: 1.4; background: rgba(0,0,0,0.3); padding: 6px; border-radius: 6px;">${raw.metadata.snippet.slice(0, 140)}...</div>
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
          brush: {
            type: ["rect", "polygon", "keep", "clear"],
            title: { rect: "Box Select", polygon: "Lasso Select", clear: "Clear Brush" },
          },
          restore: { title: "Reset View" },
          saveAsImage: { title: "Export High-Res PNG", pixelRatio: 2 },
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
        name: `${projection} Dimension 1`,
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
        name: `${projection} Dimension 2`,
        nameLocation: "middle",
        nameGap: 35,
        nameTextStyle: { color: "#9CA3AF", fontSize: 12, fontWeight: 500 },
      },
      series: seriesList,
    };
  }, [clusters, pointsByCluster, selectedClusterSet, projection, pointSize, pointOpacity, selectedPoint, pointMap]);

  const onChartClick = useCallback((params: any) => {
    const ptId = params.data?.[3];
    if (ptId) {
      const pt = pointMap.get(ptId);
      if (pt) onPointSelect(pt);
    }
  }, [pointMap, onPointSelect]);

  const onBrushSelected = useCallback((params: any) => {
    if (!onBrushSelect) return;
    const brushComponent = params.batch?.[0];
    if (!brushComponent) return;

    const selectedIds: string[] = [];
    brushComponent.selected?.forEach((seriesObj: any) => {
      seriesObj.dataIndex?.forEach((idx: number) => {
        const ptId = chartOption.series[seriesObj.seriesIndex]?.data?.[idx]?.[3];
        if (ptId) selectedIds.push(ptId);
      });
    });
    onBrushSelect(selectedIds);
  }, [chartOption, onBrushSelect]);

  const onEvents = useMemo(() => ({
    click: onChartClick,
    brushSelected: onBrushSelected,
  }), [onChartClick, onBrushSelected]);

  return (
    <div className="flex-1 w-full h-full relative bg-[#0b0f19]">
      {/* Active count badge */}
      <div className="absolute top-3 left-5 z-10 bg-gray-900/80 backdrop-blur-md px-3 py-1 rounded-lg border border-gray-800/90 text-xs text-gray-300 flex items-center gap-2 shadow-md">
        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span>
          Rendering <strong className="text-white">{points.length.toLocaleString()}</strong> {projection} manifold coordinates
        </span>
      </div>

      {/* Empty State Overlay */}
      {points.length === 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0b0f19]/80 backdrop-blur-sm z-20">
          <SearchX className="w-12 h-12 text-gray-600 mb-3" />
          <h3 className="text-sm font-semibold text-gray-300">No matching coordinates found</h3>
          <p className="text-xs text-gray-500 mt-1 max-w-sm text-center">
            Try adjusting your search terms or toggling cluster visibility in the sidebar.
          </p>
        </div>
      )}

      <ReactECharts
        ref={echartsRef}
        option={chartOption}
        style={{ height: "100%", width: "100%" }}
        onEvents={onEvents}
        notMerge={false} // Incremental update instead of destroying and re-creating canvas
        lazyUpdate={true}
      />
    </div>
  );
});
