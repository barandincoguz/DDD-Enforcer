/**
 * DDD Enforcer VS Code Extension
 * Unified TypeScript Interface Definitions
 */

/** Source reference for a violation from the RAG pipeline */
export interface ViolationSource {
  document: string;
  section: string;
  page: number;
  summary: string;
  file_path: string;
  relevance_score: number;
}

/** Single DDD violation detected in code */
export interface Violation {
  type: string;
  message: string;
  suggestion: string;
  sources?: ViolationSource[];
}

/** Metrics from validation */
export interface ValidationMetrics {
  validation_time_ms: number;
  code_file_tokens: number;
  llm_input_tokens: number;
  llm_output_tokens: number;
  llm_total_tokens: number;
  cost_usd: number;
  api_calls: number;
}

/** Response from the backend validation endpoint */
export interface ValidationResponse {
  is_violation: boolean;
  violations: Violation[];
  metrics?: ValidationMetrics;
}

/** Response from the backend health endpoint */
export interface HealthResponse {
  status: string;
  domain_model_loaded: boolean;
  rag_initialized: boolean;
}

/** Progress update from streaming endpoint */
export interface PipelineProgress {
  stage: string;
  status: "started" | "in_progress" | "completed" | "error";
  detail: string;
  progress: number;
}

/** SSE event from streaming endpoint */
export interface SSEEvent {
  type: "progress" | "complete" | "error" | "heartbeat";
  data?: PipelineProgress | GenerateModelResponse;
  error?: string;
}

/** Response from the generate-model endpoint */
export interface GenerateModelResponse {
  success: boolean;
  error?: string;
  model_path?: string;
  project_name?: string;
  bounded_contexts_count?: number;
  metrics?: CombinedMetrics;
}

/** Token usage metrics */
export interface CombinedMetrics {
  total_tokens: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost_usd: number;
  api_calls: number;
  by_stage: Record<
    string,
    {
      tokens: number;
      input_tokens: number;
      output_tokens: number;
      cost_usd: number;
      api_calls: number;
    }
  >;
}
