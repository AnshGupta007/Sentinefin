import React, { useState } from "react";
import { X, Zap, ShieldAlert, CheckCircle2, Sparkles, Sliders, ArrowRight, Loader2 } from "lucide-react";
import { ClusterMeta, ClusterPoint } from "../types/cluster";
import {
  FRAUD_PRESETS,
  FraudPreset,
  evaluateNarrativeNovelty,
  NoveltyInferenceResult,
} from "../utils/clusterInference";

interface LiveFraudSimulatorModalProps {
  isOpen: boolean;
  onClose: () => void;
  existingClusters: ClusterMeta[];
  existingPoints: ClusterPoint[];
  onIngestPoint: (result: NoveltyInferenceResult) => void;
}

export const LiveFraudSimulatorModal: React.FC<LiveFraudSimulatorModalProps> = ({
  isOpen,
  onClose,
  existingClusters,
  existingPoints,
  onIngestPoint,
}) => {
  const [selectedPresetId, setSelectedPresetId] = useState<string>("ai-voice-clone");
  const [narrativeText, setNarrativeText] = useState<string>(FRAUD_PRESETS[0].narrative);
  const [categoryText, setCategoryText] = useState<string>(FRAUD_PRESETS[0].category);
  const [noveltyThreshold, setNoveltyThreshold] = useState<number>(0.55);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [inferenceResult, setInferenceResult] = useState<NoveltyInferenceResult | null>(null);

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  if (!isOpen) return null;

  const handleSelectPreset = (preset: FraudPreset) => {
    setSelectedPresetId(preset.id);
    setNarrativeText(preset.narrative);
    setCategoryText(preset.category);
    setInferenceResult(null);
  };

  const handleRunInference = () => {
    if (!narrativeText.trim()) return;
    setIsEvaluating(true);
    setInferenceResult(null);

    // Simulate 400ms neural forward pass latency
    setTimeout(() => {
      const result = evaluateNarrativeNovelty(
        narrativeText,
        categoryText,
        existingClusters,
        existingPoints,
        noveltyThreshold
      );
      setInferenceResult(result);
      setIsEvaluating(false);
    }, 450);
  };

  const handleCommitToDashboard = () => {
    if (!inferenceResult) return;
    onIngestPoint(inferenceResult);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-150"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-[#0e1424] border border-gray-800 rounded-2xl w-full max-w-3xl max-h-[92vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-[#111827]">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-950/70 border border-cyan-600/50 flex items-center justify-center text-cyan-400 shadow-md">
              <Zap className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
                Live Fraud Ingestion & Online Cluster Spawner
                <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950 border border-cyan-700 text-cyan-300 font-sans">
                  Active Learning
                </span>
              </h2>
              <p className="text-xs text-gray-400">
                Test how the deep learning model handles newly reported fraud patterns in real-time
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

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-5 flex-1 text-xs">
          {/* 1. PRESET BUTTONS */}
          <div>
            <label className="text-[11px] font-bold text-gray-400 uppercase tracking-wider block mb-2 font-mono flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" /> One-Click Demonstration Presets
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {FRAUD_PRESETS.map((preset) => {
                const isSelected = selectedPresetId === preset.id;
                return (
                  <button
                    key={preset.id}
                    onClick={() => handleSelectPreset(preset)}
                    className={`text-left p-2.5 rounded-xl border transition-all text-xs ${
                      isSelected
                        ? "bg-cyan-950/40 border-cyan-500/70 text-cyan-200 shadow-sm"
                        : "bg-gray-900/60 border-gray-800 text-gray-400 hover:bg-gray-800/50 hover:text-gray-200"
                    }`}
                  >
                    <div className="font-semibold flex items-center justify-between">
                      <span className="truncate">{preset.label}</span>
                      {preset.isNovelExpected && (
                        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-rose-950/80 border border-rose-800 text-rose-300 flex-shrink-0 ml-1">
                          Novel
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-gray-500 truncate mt-1">{preset.description}</p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* 2. COMPLAINT NARRATIVE INPUT */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-[11px] font-bold text-gray-400 uppercase tracking-wider font-mono">
                Complaint Narrative Text
              </label>
              <span className="text-[10px] text-gray-500 font-mono">
                {narrativeText.split(/\s+/).filter(Boolean).length} words
              </span>
            </div>
            <textarea
              rows={4}
              value={narrativeText}
              onChange={(e) => {
                setNarrativeText(e.target.value);
                setSelectedPresetId("");
                setInferenceResult(null);
              }}
              placeholder="Paste or type a new consumer complaint narrative here..."
              className="w-full bg-gray-950/80 border border-gray-700/80 rounded-xl p-3 text-xs text-gray-200 placeholder-gray-600 focus:outline-none focus:border-cyan-500 leading-relaxed font-sans"
            />
          </div>

          {/* 3. CATEGORY & SENSITIVITY CONTROLS */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 bg-gray-900/40 p-3.5 rounded-xl border border-gray-800/80">
            <div>
              <label className="text-[10px] font-bold text-gray-400 uppercase font-mono block mb-1">
                Reported Fraud Tag / Category
              </label>
              <input
                type="text"
                value={categoryText}
                onChange={(e) => setCategoryText(e.target.value)}
                placeholder="e.g. AI Deepfake Impersonation"
                className="w-full bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div>
              <div className="flex justify-between text-[10px] font-bold text-gray-400 uppercase font-mono mb-1">
                <span className="flex items-center gap-1">
                  <Sliders className="w-3 h-3 text-indigo-400" /> Novelty Threshold (&tau;)
                </span>
                <span className="text-cyan-400 font-mono">{(noveltyThreshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min={0.3}
                max={0.8}
                step={0.05}
                value={noveltyThreshold}
                onChange={(e) => setNoveltyThreshold(Number(e.target.value))}
                className="w-full accent-cyan-500 bg-gray-800 h-1.5 rounded-lg cursor-pointer"
              />
              <span className="text-[10px] text-gray-500 block mt-0.5">
                Higher requires greater statistical distance to trigger a new cluster
              </span>
            </div>
          </div>

          {/* 4. RUN INFERENCE TRIGGER BUTTON */}
          <div className="flex justify-center pt-1">
            <button
              onClick={handleRunInference}
              disabled={isEvaluating || !narrativeText.trim()}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white font-semibold text-xs shadow-lg shadow-cyan-500/20 transition-all disabled:opacity-50"
            >
              {isEvaluating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Computing Student-t Embeddings...</span>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 text-cyan-200" />
                  <span>Run Neural Novelty Check</span>
                </>
              )}
            </button>
          </div>

          {/* 5. INFERENCE RESULT DISPLAY */}
          {inferenceResult && (
            <div
              className={`p-4 rounded-xl border animate-in zoom-in-95 duration-200 space-y-3 ${
                inferenceResult.isNovel
                  ? "bg-rose-950/30 border-rose-600/60 shadow-lg shadow-rose-950/20"
                  : "bg-emerald-950/30 border-emerald-600/60 shadow-lg shadow-emerald-950/20"
              }`}
            >
              {/* Verdict Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {inferenceResult.isNovel ? (
                    <div className="w-7 h-7 rounded-lg bg-rose-500/20 border border-rose-500/40 flex items-center justify-center text-rose-400">
                      <ShieldAlert className="w-4 h-4" />
                    </div>
                  ) : (
                    <div className="w-7 h-7 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
                      <CheckCircle2 className="w-4 h-4" />
                    </div>
                  )}
                  <div>
                    <h3
                      className={`text-xs font-bold uppercase tracking-wider font-mono ${
                        inferenceResult.isNovel ? "text-rose-300" : "text-emerald-300"
                      }`}
                    >
                      {inferenceResult.isNovel
                        ? "🚨 Novel Unseen Risk Detected: Spawning New Cluster!"
                        : "Familiar Risk Pattern: Merged into Existing Partition"}
                    </h3>
                    <p className="text-[11px] text-gray-300 mt-0.5">
                      Target Partition: <b>{inferenceResult.assignedClusterLabel}</b>
                    </p>
                  </div>
                </div>

                <div className="text-right font-mono">
                  <span className="text-[10px] text-gray-400 block uppercase">Novelty Score</span>
                  <span
                    className={`text-sm font-bold ${
                      inferenceResult.isNovel ? "text-rose-400" : "text-emerald-400"
                    }`}
                  >
                    {(inferenceResult.noveltyScore * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Explanation */}
              <p className="text-xs text-gray-300 leading-relaxed bg-black/30 p-2.5 rounded-lg border border-white/5">
                {inferenceResult.explanation}
              </p>

              {/* Soft Assignment Probabilities */}
              <div>
                <span className="text-[10px] text-gray-400 font-mono uppercase block mb-1.5">
                  Top-3 Centroid Soft Assignment Probabilities (Q-Distribution)
                </span>
                <div className="space-y-1.5">
                  {inferenceResult.softAssignments.slice(0, 3).map((item) => (
                    <div key={item.clusterId} className="flex items-center gap-2 text-[11px]">
                      <span className="w-36 text-gray-300 truncate">{item.label}</span>
                      <div className="flex-1 bg-gray-900 rounded-full h-2 overflow-hidden border border-gray-800">
                        <div
                          className="bg-blue-500 h-full rounded-full transition-all duration-500"
                          style={{ width: `${Math.min(100, item.probability * 100)}%` }}
                        />
                      </div>
                      <span className="w-12 text-right font-mono text-gray-400">
                        {(item.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Action Button: Commit to Live Dashboard */}
              <div className="pt-2 flex justify-end">
                <button
                  onClick={handleCommitToDashboard}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition-colors shadow-md shadow-blue-600/30"
                >
                  <span>Inject into Live Canvas & Table</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-gray-800 bg-[#111827] flex justify-between items-center text-[11px] text-gray-400">
          <span className="font-mono">Inference uses Student-t kernel (v = 1.0) on 384-d latent vectors</span>
          <button
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 font-mono"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};
