/**
 * Configuration loader for RLM CLI.
 *
 * Reads rlm_config.yaml from the project root (or cwd), with sensible defaults.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

export interface RlmConfig {
	max_iterations: number;
	max_depth: 1;
	max_sub_queries: number;
	truncate_len: number;
	metadata_preview_lines: number;
	sub_model: string;  // model ID for sub-queries (empty = same as root)
	max_total_tokens: number;
	max_cost_usd: number;
	max_concurrency: number;
	min_request_interval_ms: number;
	max_retries: number;
	chunk_size_chars: number;
}

const DEFAULTS: RlmConfig = {
	max_iterations: 20,
	max_depth: 1,  // current runtime implements paper-style flat sub-calls, not nested RLM recursion
	max_sub_queries: 50,
	truncate_len: 5000,
	metadata_preview_lines: 20,
	sub_model: "",  // empty = same model as root
	max_total_tokens: 0,  // 0 = unlimited
	max_cost_usd: 0,  // 0 = unlimited
	max_concurrency: 8,
	min_request_interval_ms: 0,
	max_retries: 2,
	chunk_size_chars: 0,  // 0 = derive from the sub-model context window
};

function parseYaml(text: string): Record<string, unknown> {
	// Minimal YAML parser for flat key:value files (no nested objects, no arrays)
	const result: Record<string, unknown> = {};
	for (const line of text.split("\n")) {
		const trimmed = line.trim();
		if (!trimmed || trimmed.startsWith("#")) continue;
		const colonIdx = trimmed.indexOf(":");
		if (colonIdx === -1) continue;
		const key = trimmed.slice(0, colonIdx).trim();
		const rawVal = trimmed.slice(colonIdx + 1).trim();
		// Strip inline comments
		const val = rawVal.replace(/\s+#.*$/, "");
		// Parse number
		const num = Number(val);
		if (!isNaN(num) && val !== "") {
			result[key] = num;
		} else if (val === "true") {
			result[key] = true;
		} else if (val === "false") {
			result[key] = false;
		} else {
			// Strip quotes
			result[key] = val.replace(/^["']|["']$/g, "");
		}
	}
	return result;
}

export function loadConfig(): RlmConfig {
	// Search order: cwd, then package root
	const candidates = [
		path.resolve(process.cwd(), "rlm_config.yaml"),
		path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "rlm_config.yaml"),
	];

	for (const configPath of candidates) {
		if (fs.existsSync(configPath)) {
			try {
				const raw = fs.readFileSync(configPath, "utf-8");
				const parsed = parseYaml(raw);
				const clamp = (v: unknown, min: number, max: number, def: number) =>
					typeof v === "number" && isFinite(v) ? Math.max(min, Math.min(max, Math.round(v))) : def;
				return {
					max_iterations: clamp(parsed.max_iterations, 1, 100, DEFAULTS.max_iterations),
					max_depth: 1,
					max_sub_queries: clamp(parsed.max_sub_queries, 1, 500, DEFAULTS.max_sub_queries),
					truncate_len: clamp(parsed.truncate_len, 500, 50000, DEFAULTS.truncate_len),
					metadata_preview_lines: clamp(parsed.metadata_preview_lines, 5, 100, DEFAULTS.metadata_preview_lines),
					sub_model: typeof parsed.sub_model === "string" ? parsed.sub_model.trim() : (process.env.RLM_SUB_MODEL ?? ""),
					max_total_tokens: clamp(parsed.max_total_tokens, 0, 100000000, DEFAULTS.max_total_tokens),
					max_cost_usd: typeof parsed.max_cost_usd === "number" && isFinite(parsed.max_cost_usd)
						? Math.max(0, parsed.max_cost_usd)
						: DEFAULTS.max_cost_usd,
					max_concurrency: clamp(parsed.max_concurrency, 1, 100, DEFAULTS.max_concurrency),
					min_request_interval_ms: clamp(parsed.min_request_interval_ms, 0, 60000, DEFAULTS.min_request_interval_ms),
					max_retries: clamp(parsed.max_retries, 0, 10, DEFAULTS.max_retries),
					chunk_size_chars: clamp(parsed.chunk_size_chars, 0, 1000000, DEFAULTS.chunk_size_chars),
				};
			} catch {
				// Fall through to defaults
			}
		}
	}

	return { ...DEFAULTS, sub_model: process.env.RLM_SUB_MODEL ?? "" };
}
