import { ClusteringDataset } from "../types/cluster";

const DEFAULT_PALETTE = [
  "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", 
  "#EC4899", "#06B6D4", "#F97316", "#14B8A6",
  "#E11D48", "#6366F1", "#84CC16"
];

export const getMockDataset = (): ClusteringDataset => {
  const clusters = [
    {
      id: 0,
      label: "Credit Reporting Inaccuracies & FCRA",
      color: DEFAULT_PALETTE[0],
      count: 650,
      keywords: ["dispute", "bureau", "inaccurate", "fcra", "investigation"]
    },
    {
      id: 1,
      label: "Debt Collection & Verification Notices",
      color: DEFAULT_PALETTE[1],
      count: 420,
      keywords: ["collector", "debt", "calls", "harassment", "validation"]
    },
    {
      id: 2,
      label: "Credit Card Unauthorized Charges",
      color: DEFAULT_PALETTE[2],
      count: 380,
      keywords: ["charge", "fraud", "unauthorized", "billing", "dispute"]
    },
    {
      id: 3,
      label: "Mortgage Servicing & Escrow Errors",
      color: DEFAULT_PALETTE[3],
      count: 290,
      keywords: ["mortgage", "escrow", "modification", "servicer", "foreclosure"]
    },
    {
      id: 4,
      label: "Bank Account Overdraft & Deposit Holds",
      color: DEFAULT_PALETTE[4],
      count: 260,
      keywords: ["overdraft", "deposit", "checking", "funds", "fee"]
    },
    {
      id: 5,
      label: "Money Transfers & Virtual Currencies",
      color: DEFAULT_PALETTE[5],
      count: 210,
      keywords: ["transfer", "wire", "crypto", "remittance", "delayed"]
    },
  ];

  const points = [];
  const centers = [
    { cx: -4.5, cy: 3.2 },
    { cx: 4.8, cy: 3.5 },
    { cx: 3.2, cy: -4.1 },
    { cx: -4.2, cy: -3.8 },
    { cx: 0.1, cy: 0.2 },
    { cx: 5.2, cy: -1.2 },
  ];

  for (let idx = 0; idx < clusters.length; idx++) {
    const c = clusters[idx];
    const center = centers[idx];
    for (let i = 0; i < c.count; i++) {
      const u1 = Math.random() || 0.001;
      const u2 = Math.random() || 0.001;
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
      
      const ux = Number((center.cx + z0 * 1.1).toFixed(3));
      const uy = Number((center.cy + z1 * 1.1).toFixed(3));

      points.push({
        id: `c-pt-${idx}-${i}`,
        x: ux,
        y: uy,
        pca_x: Number((ux * 0.85 + Math.sin(uy) * 0.3).toFixed(3)),
        pca_y: Number((uy * 0.85 + Math.cos(ux) * 0.3).toFixed(3)),
        cluster_id: c.id,
        confidence: Number((0.82 + Math.random() * 0.17).toFixed(2)),
        window_id: `2023-0${(i % 9) + 1}`,
        metadata: {
          title: `${c.label.split('&')[0].trim()} Case #${10000 + i}`,
          issue: `Dispute regarding ${c.keywords[0]} handling`,
          company: idx % 2 === 0 ? "Equifax Information Services LLC" : "JPMorgan Chase Bank, N.A.",
          snippet: `Consumer reports critical discrepancy in ${c.keywords.slice(0, 3).join(", ")} records. Formal dispute was filed with supporting documentation but company failed to provide adequate investigation response within statutory timeline.`,
          timestamp: `2023-0${(i % 9) + 1}-14`,
        }
      });
    }
  }

  return {
    dataset_name: "SentinelFin Latent Manifold Clustering",
    model_architecture: "all-MiniLM-L6-v2 (384-d) + Deep Embedded Clustering (DEC)",
    projection_type: "UMAP",
    metrics: {
      silhouette_score: 0.742,
      davies_bouldin_index: 0.481,
      calinski_harabasz_index: 1284.5,
      total_points: points.length,
      total_clusters: clusters.length,
      noise_count: 16,
    },
    clusters,
    points,
  };
};
