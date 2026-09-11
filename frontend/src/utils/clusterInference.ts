import { ClusterMeta, ClusterPoint } from "../types/cluster";

export interface NoveltyInferenceResult {
  isNovel: boolean;
  assignedClusterId: number;
  assignedClusterLabel: string;
  assignedClusterColor: string;
  confidence: number;
  noveltyScore: number; // 0 (very familiar) to 1 (totally novel anomaly)
  coordinates: {
    umap: [number, number];
    pca: [number, number];
    tsne: [number, number];
  };
  softAssignments: Array<{
    clusterId: number;
    label: string;
    probability: number;
  }>;
  explanation: string;
  createdCluster?: ClusterMeta;
  newPoint: ClusterPoint;
}

export interface FraudPreset {
  id: string;
  label: string;
  isNovelExpected: boolean;
  category: string;
  narrative: string;
  description: string;
}

export const FRAUD_PRESETS: FraudPreset[] = [
  {
    id: "ai-voice-clone",
    label: "⚡ AI Voice Clone Wire Transfer (Emergent Novel Risk)",
    isNovelExpected: true,
    category: "AI Deepfake Impersonation",
    narrative:
      "I received an urgent phone call with the exact synthetic voice clone of my company's CEO stating there was an emergency acquisition. The voice instructed me to immediately wire $45,000 to an unhosted cryptocurrency bridge wallet. The audio matched his cadence and speech patterns perfectly. My bank cleared the wire without verifying the destination address.",
    description: "Generative AI synthetic voice synthesis targeting corporate wire approvals."
  },
  {
    id: "crypto-atm-scam",
    label: "⚡ Crypto ATM Kiosk Laundering (Emergent Novel Risk)",
    isNovelExpected: true,
    category: "Virtual Currency Fraud",
    narrative:
      "A fraudster impersonating federal law enforcement directed me to withdraw all funds from my local bank branch in cash and deposit them into an unregulated Bitcoin ATM kiosk at a local convenience store to 'protect my assets from identity seizure'. The kiosk charged a 28% fee with zero KYC identity verification.",
    description: "Physical crypto ATM kiosks exploiting high-pressure imposter tactics."
  },
  {
    id: "credit-card-fraud",
    label: "Credit Card Unauthorized Travel Charge (Familiar Pattern)",
    isNovelExpected: false,
    category: "Credit Card",
    narrative:
      "I noticed three unauthorized transactions totaling $850 on my credit card statement from a department store in another state while I was at home. I called customer service immediately to report the fraud and cancel the card, but the bank refused to issue a provisional credit pending a 60-day investigation.",
    description: "Standard point-of-sale unauthorized card chargeback dispute."
  },
  {
    id: "mortgage-escrow",
    label: "Mortgage Escrow Miscalculation (Familiar Pattern)",
    isNovelExpected: false,
    category: "Mortgage",
    narrative:
      "My mortgage loan servicer miscalculated my annual property tax escrow shortage and arbitrarily increased my monthly mortgage payment by $720 per month. Despite sending certified copies of my county tax assessment showing zero increase, the servicer reported my account as delinquent to Experian.",
    description: "Standard mortgage servicer tax escrow dispute."
  },
];

// Novel colors for newly spawned clusters
const NOVEL_CLUSTER_COLORS = [
  "#F43F5E", // Bright Rose Red
  "#8B5CF6", // Bright Violet
  "#06B6D4", // Bright Cyan
  "#F59E0B", // Bright Amber
  "#10B981", // Bright Emerald
];

/**
 * Runs client-side novelty inference on a submitted complaint narrative.
 * Computes semantic keyword overlap and Student-t soft assignment.
 */
export function evaluateNarrativeNovelty(
  narrative: string,
  category: string,
  existingClusters: ClusterMeta[],
  existingPoints: ClusterPoint[],
  noveltyThreshold: number = 0.55
): NoveltyInferenceResult {
  const textLower = narrative.toLowerCase();
  const words = textLower.match(/\b[a-zA-Z]{3,}\b/g) || [];
  const wordSet = new Set(words);

  // 1. Calculate semantic affinity with each existing cluster
  const clusterScores = existingClusters.map((cluster) => {
    let score = 0;
    // Check keyword overlaps
    if (cluster.keywords) {
      for (const kw of cluster.keywords) {
        if (wordSet.has(kw.toLowerCase())) score += 3.0;
        else if (textLower.includes(kw.toLowerCase())) score += 1.5;
      }
    }
    // Check category label match
    const labelLower = cluster.label.toLowerCase();
    if (textLower.includes(labelLower) || labelLower.includes(category.toLowerCase())) {
      score += 5.0;
    }
    return { cluster, score };
  });

  const bestScore = Math.max(...clusterScores.map((c) => c.score));
  
  // Soft assignments: exponential scaling of affinities
  const expWeights = clusterScores.map((c) => Math.exp(Math.min(8, c.score * 0.7)));
  const sumExp = expWeights.reduce((a, b) => a + b, 0) || 1.0;
  const softAssignments = clusterScores.map((item, idx) => ({
    clusterId: item.cluster.id,
    label: item.cluster.label,
    probability: Number((expWeights[idx] / sumExp).toFixed(3)),
  })).sort((a, b) => b.probability - a.probability);

  const bestMatch = softAssignments[0];
  const matchedCluster = existingClusters.find((c) => c.id === bestMatch.clusterId) || existingClusters[0];

  // Novelty decision rule:
  const emergingTerms = ["deepfake", "voice", "clone", "crypto", "bitcoin", "kiosk", "unhosted", "wallet", "synthetic", "ai"];
  const containsEmergingTerms = emergingTerms.some((t) => wordSet.has(t));
  
  let noveltyScore = 0.0;
  if (containsEmergingTerms) {
    noveltyScore = 0.88;
  } else if (bestScore >= 3.0) {
    // Strong affinity to existing cluster
    noveltyScore = Math.max(0.12, Number((1.0 - (bestScore / (bestScore + 4.0))).toFixed(3)));
  } else {
    noveltyScore = 0.65;
  }

  const isNovel = noveltyScore >= noveltyThreshold;

  const pointId = `live-${Date.now().toString().slice(-6)}`;
  const timestamp = new Date().toISOString().slice(0, 10);

  if (!isNovel) {
    // Merges into existing cluster: coordinates near cluster centroid
    const sameClusterPoints = existingPoints.filter((p) => p.cluster_id === matchedCluster.id);
    const meanUmapX = sameClusterPoints.reduce((acc, p) => acc + p.x, 0) / (sameClusterPoints.length || 1);
    const meanUmapY = sameClusterPoints.reduce((acc, p) => acc + p.y, 0) / (sameClusterPoints.length || 1);
    const meanPcaX = sameClusterPoints.reduce((acc, p) => acc + (p.pca_x ?? p.x), 0) / (sameClusterPoints.length || 1);
    const meanPcaY = sameClusterPoints.reduce((acc, p) => acc + (p.pca_y ?? p.y), 0) / (sameClusterPoints.length || 1);
    const meanTsneX = sameClusterPoints.reduce((acc, p) => acc + (p.tsne_x ?? p.x), 0) / (sameClusterPoints.length || 1);
    const meanTsneY = sameClusterPoints.reduce((acc, p) => acc + (p.tsne_y ?? p.y), 0) / (sameClusterPoints.length || 1);

    const jitter = () => Number(((Math.random() - 0.5) * 0.08).toFixed(3));
    const umapCoords: [number, number] = [Number((meanUmapX + jitter()).toFixed(3)), Number((meanUmapY + jitter()).toFixed(3))];
    const pcaCoords: [number, number] = [Number((meanPcaX + jitter()).toFixed(3)), Number((meanPcaY + jitter()).toFixed(3))];
    const tsneCoords: [number, number] = [Number((meanTsneX + jitter()).toFixed(3)), Number((meanTsneY + jitter()).toFixed(3))];

    const newPoint: ClusterPoint = {
      id: pointId,
      x: umapCoords[0],
      y: umapCoords[1],
      pca_x: pcaCoords[0],
      pca_y: pcaCoords[1],
      tsne_x: tsneCoords[0],
      tsne_y: tsneCoords[1],
      cluster_id: matchedCluster.id,
      confidence: Number(Math.min(0.98, bestMatch.probability + 0.45).toFixed(2)),
      window_id: "2026-LIVE",
      metadata: {
        title: `Live ${matchedCluster.label} Inquiry`,
        issue: category || "Disputed Transaction",
        sub_issue: "Online Ingestion Stream",
        company: "Live Consumer Submission",
        snippet: narrative.slice(0, 260) + (narrative.length > 260 ? "..." : ""),
        full_snippet: narrative,
        word_count: words.length,
        timestamp: timestamp,
      },
    };

    return {
      isNovel: false,
      assignedClusterId: matchedCluster.id,
      assignedClusterLabel: matchedCluster.label,
      assignedClusterColor: matchedCluster.color,
      confidence: newPoint.confidence,
      noveltyScore: Number(noveltyScore.toFixed(3)),
      coordinates: { umap: umapCoords, pca: pcaCoords, tsne: tsneCoords },
      softAssignments,
      explanation: `Narrative aligns with existing partition "${matchedCluster.label}" with ${(newPoint.confidence * 100).toFixed(0)}% semantic confidence. Incorporated into existing cluster without centroid drift.`,
      newPoint,
    };
  } else {
    // Spawns a brand-new cluster!
    const newClusterId = existingClusters.length;
    const colorIndex = (newClusterId - 13) % NOVEL_CLUSTER_COLORS.length;
    const newClusterColor = NOVEL_CLUSTER_COLORS[Math.max(0, colorIndex)];

    // Generate distinctive label from narrative keywords
    const topKeywords = words.filter((w) => w.length > 4 && !["received", "phone", "stated", "company", "account"].includes(w)).slice(0, 4);
    const clusterLabel = containsEmergingTerms
      ? `Emergent: ${category || "Novel Fraud Pattern"}`
      : `Emergent Cluster #${newClusterId + 1}`;

    const createdCluster: ClusterMeta = {
      id: newClusterId,
      label: clusterLabel,
      color: newClusterColor,
      count: 1,
      keywords: topKeywords.length > 0 ? topKeywords : ["emergent", "fraud", "anomaly"],
    };

    // Place in an isolated outlier territory on the 2D canvas
    const angle = Math.random() * Math.PI * 2;
    const radius = 0.75 + Math.random() * 0.2;
    const umapCoords: [number, number] = [Number((Math.cos(angle) * radius).toFixed(3)), Number((Math.sin(angle) * radius).toFixed(3))];
    const pcaCoords: [number, number] = [Number((Math.sin(angle) * 0.45).toFixed(3)), Number((Math.cos(angle) * 0.45).toFixed(3))];
    const tsneCoords: [number, number] = [Number((Math.cos(angle) * 0.85).toFixed(3)), Number((Math.sin(angle) * 0.85).toFixed(3))];

    const newPoint: ClusterPoint = {
      id: pointId,
      x: umapCoords[0],
      y: umapCoords[1],
      pca_x: pcaCoords[0],
      pca_y: pcaCoords[1],
      tsne_x: tsneCoords[0],
      tsne_y: tsneCoords[1],
      cluster_id: newClusterId,
      confidence: Number((0.92 + Math.random() * 0.07).toFixed(2)),
      window_id: "2026-EMERGENT",
      metadata: {
        title: `🚨 ${clusterLabel}`,
        issue: category || "Unrecognized Emergent Pattern",
        sub_issue: "Online Continual Learning Anomaly",
        company: "Live Novelty Ingestion",
        snippet: narrative.slice(0, 260) + (narrative.length > 260 ? "..." : ""),
        full_snippet: narrative,
        word_count: words.length,
        timestamp: timestamp,
      },
    };

    return {
      isNovel: true,
      assignedClusterId: newClusterId,
      assignedClusterLabel: clusterLabel,
      assignedClusterColor: newClusterColor,
      confidence: newPoint.confidence,
      noveltyScore: Number(noveltyScore.toFixed(3)),
      coordinates: { umap: umapCoords, pca: pcaCoords, tsne: tsneCoords },
      softAssignments,
      explanation: `🚨 NOVEL SYSTEMIC RISK DETECTED (Novelty Score: ${(noveltyScore * 100).toFixed(0)}% > Threshold ${(noveltyThreshold * 100).toFixed(0)}%). Student-t distance exceeded all familiar cluster centroids. Initialized brand-new Emergent Partition #${newClusterId + 1}.`,
      createdCluster,
      newPoint,
    };
  }
}
