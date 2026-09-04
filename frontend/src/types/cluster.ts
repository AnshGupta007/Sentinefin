export type ProjectionType = "UMAP" | "PCA" | "t-SNE";

export interface ClusterPointMetadata {
  title: string;
  issue?: string;
  company?: string;
  snippet: string;
  timestamp?: string;
  [key: string]: any;
}

export interface ClusterPoint {
  id: string;
  x: number;
  y: number;
  pca_x?: number;
  pca_y?: number;
  cluster_id: number;
  confidence: number;
  window_id?: string;
  metadata: ClusterPointMetadata;
}

export interface ClusterMeta {
  id: number;
  label: string;
  color: string;
  count: number;
  keywords?: string[];
}

export interface ClusterMetrics {
  silhouette_score: number;
  davies_bouldin_index: number;
  calinski_harabasz_index: number;
  total_points: number;
  total_clusters: number;
  noise_count: number;
}

export interface ClusteringDataset {
  dataset_name: string;
  model_architecture?: string;
  projection_type: ProjectionType;
  metrics: ClusterMetrics;
  clusters: ClusterMeta[];
  points: ClusterPoint[];
}
