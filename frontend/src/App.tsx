import React, { useState, useEffect, useMemo } from "react";
import { ClusteringDataset, ClusterPoint, ProjectionType } from "./types/cluster";
import { getMockDataset } from "./data/mockData";
import { Header } from "./components/Header";
import { MetricCards } from "./components/MetricCards";
import { Sidebar } from "./components/Sidebar";
import { ClusterScatterPlot } from "./components/ClusterScatterPlot";
import { DataTableStrip } from "./components/DataTableStrip";
import { PointInspectorDrawer } from "./components/PointInspectorDrawer";
import { ClusterDistributionModal } from "./components/ClusterDistributionModal";
import { Loader2 } from "lucide-react";

export const App: React.FC = () => {
  const [dataset, setDataset] = useState<ClusteringDataset | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [projection, setProjection] = useState<ProjectionType>("UMAP");
  const [selectedClusters, setSelectedClusters] = useState<number[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [pointSize, setPointSize] = useState<number>(6);
  const [pointOpacity, setPointOpacity] = useState<number>(0.8);
  const [selectedPoint, setSelectedPoint] = useState<ClusterPoint | null>(null);
  const [isDistributionModalOpen, setIsDistributionModalOpen] = useState<boolean>(false);

  // Fetch real exported clustering JSON from /data/clustering_data.json, or fallback to mock
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("/data/clustering_data.json");
        if (res.ok) {
          const json: ClusteringDataset = await res.json();
          setDataset(json);
          setSelectedClusters(json.clusters.map((c) => c.id));
        } else {
          console.warn("Could not load /data/clustering_data.json; loading fallback demo dataset.");
          const mock = getMockDataset();
          setDataset(mock);
          setSelectedClusters(mock.clusters.map((c) => c.id));
        }
      } catch (err) {
        console.warn("Fetch error, using fallback dataset:", err);
        const mock = getMockDataset();
        setDataset(mock);
        setSelectedClusters(mock.clusters.map((c) => c.id));
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Filter Points based on Visible Clusters and Search Query
  const filteredPoints = useMemo(() => {
    if (!dataset) return [];
    return dataset.points.filter((pt) => {
      const isVisible = selectedClusters.includes(pt.cluster_id);
      if (!isVisible) return false;

      if (!searchQuery) return true;
      const q = searchQuery.toLowerCase();
      const inTitle = pt.metadata.title?.toLowerCase().includes(q);
      const inSnippet = pt.metadata.snippet?.toLowerCase().includes(q);
      const inCompany = pt.metadata.company?.toLowerCase().includes(q);
      const inIssue = pt.metadata.issue?.toLowerCase().includes(q);
      return inTitle || inSnippet || inCompany || inIssue;
    });
  }, [dataset, selectedClusters, searchQuery]);

  // Cluster Selection Toggles
  const handleToggleCluster = (id: number) => {
    setSelectedClusters((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const handleSelectAll = () => {
    if (dataset) setSelectedClusters(dataset.clusters.map((c) => c.id));
  };

  const handleDeselectAll = () => {
    setSelectedClusters([]);
  };

  if (loading || !dataset) {
    return (
      <div className="min-h-screen bg-[#0b0f19] flex flex-col items-center justify-center text-gray-300 space-y-4">
        <Loader2 className="w-10 h-10 text-blue-500 animate-spin" />
        <p className="text-sm font-mono tracking-wider">Loading Latent Manifold Vectors & Clusters...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0b0f19] text-gray-100 flex flex-col font-sans selection:bg-blue-600 selection:text-white overflow-hidden h-screen">
      {/* 1. TOP HEADER */}
      <Header
        datasetName={dataset.dataset_name}
        modelArchitecture={dataset.model_architecture}
        projection={projection}
        onProjectionChange={setProjection}
        onOpenDistributionModal={() => setIsDistributionModalOpen(true)}
        activePointsCount={filteredPoints.length}
      />

      {/* 2. SUMMARY METRICS STRIP */}
      <MetricCards
        metrics={dataset.metrics}
        visibleClustersCount={selectedClusters.length}
      />

      {/* 3. MAIN WORKSPACE (Sidebar + Canvas + Drawer) */}
      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* Left Filter Sidebar */}
        <Sidebar
          clusters={dataset.clusters}
          selectedClusters={selectedClusters}
          onToggleCluster={handleToggleCluster}
          onSelectAll={handleSelectAll}
          onDeselectAll={handleDeselectAll}
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          pointSize={pointSize}
          onPointSizeChange={setPointSize}
          pointOpacity={pointOpacity}
          onPointOpacityChange={setPointOpacity}
        />

        {/* Center Canvas & Bottom Data Table */}
        <main className="flex-1 flex flex-col min-w-0 bg-[#0b0f19] overflow-hidden">
          {/* ECharts High-Density Scatter Canvas */}
          <ClusterScatterPlot
            clusters={dataset.clusters}
            points={filteredPoints}
            selectedClusters={selectedClusters}
            projection={projection}
            pointSize={pointSize}
            pointOpacity={pointOpacity}
            selectedPoint={selectedPoint}
            onPointSelect={setSelectedPoint}
          />

          {/* Bottom Data Table Strip */}
          <DataTableStrip
            points={filteredPoints}
            clusters={dataset.clusters}
            selectedPoint={selectedPoint}
            onPointSelect={setSelectedPoint}
            projection={projection}
          />
        </main>

        {/* Right Point Inspector Drawer */}
        <PointInspectorDrawer
          point={selectedPoint}
          clusters={dataset.clusters}
          onClose={() => setSelectedPoint(null)}
          projection={projection}
        />
      </div>

      {/* Analytics Modal */}
      <ClusterDistributionModal
        isOpen={isDistributionModalOpen}
        onClose={() => setIsDistributionModalOpen(false)}
        clusters={dataset.clusters}
      />
    </div>
  );
};

export default App;
