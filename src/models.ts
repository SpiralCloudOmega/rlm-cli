import { getModels, getProviders, type Api, type Model } from "@mariozechner/pi-ai";

export type ModelRole = "root" | "sub";
const customApiKeys = new WeakMap<Model<Api>, string>();

function envFor(role: ModelRole, name: string): string | undefined {
	const roleValue = role === "sub" ? process.env[`RLM_SUB_${name}`] : undefined;
	return roleValue || process.env[`RLM_${name}`];
}

function positiveNumber(value: string | undefined, fallback: number): number {
	const parsed = Number(value);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function nonNegativeNumber(value: string | undefined): number {
	const parsed = Number(value);
	return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0;
}

export function isFrontierModel(modelId: string): boolean {
	const gpt = modelId.match(/^gpt-(\d+)(?:\.(\d+))?(?:-|$)/i);
	if (gpt) return Number(gpt[1]) > 5 || (Number(gpt[1]) === 5 && Number(gpt[2] || 0) >= 6);
	const claude = modelId.match(/^claude-fable-(\d+)(?:-|$)/i);
	if (claude) return Number(claude[1]) >= 5;
	const kimi = modelId.match(/^kimi-k(\d+)(?:[.-]|$)/i);
	return !!kimi && Number(kimi[1]) >= 3;
}

export function createCompatibleModel(modelId: string, role: ModelRole = "root"): Model<Api> | undefined {
	const isAnthropic = /^claude-/i.test(modelId);
	const apiKey = envFor(role, "API_KEY") || (isAnthropic ? process.env.ANTHROPIC_API_KEY : process.env.OPENAI_API_KEY);
	if (!apiKey) return undefined;

	const api = envFor(role, "API") || (isAnthropic ? "anthropic-messages" : "openai-completions");
	if (api !== "openai-completions" && api !== "anthropic-messages") {
		throw new Error(`Unsupported ${role} custom API "${api}". Use openai-completions or anthropic-messages.`);
	}
	const defaultBaseUrl = isAnthropic ? "https://api.anthropic.com" : (process.env.OPENAI_BASE_URL || "https://api.openai.com/v1");
	const baseUrl = (envFor(role, "BASE_URL") || defaultBaseUrl).replace(/\/+$/, "");
	const model: Model<Api> = {
		id: modelId,
		name: modelId,
		api,
		provider: "custom-compatible",
		baseUrl,
		reasoning: envFor(role, "REASONING") === "true",
		input: ["text"],
		cost: {
			input: nonNegativeNumber(envFor(role, "INPUT_COST_PER_M")),
			output: nonNegativeNumber(envFor(role, "OUTPUT_COST_PER_M")),
			cacheRead: nonNegativeNumber(envFor(role, "CACHE_READ_COST_PER_M")),
			cacheWrite: nonNegativeNumber(envFor(role, "CACHE_WRITE_COST_PER_M")),
		},
		contextWindow: positiveNumber(envFor(role, "CONTEXT_WINDOW"), 128000),
		maxTokens: positiveNumber(envFor(role, "MAX_OUTPUT_TOKENS"), 8192),
	};
	if (api === "openai-completions") {
		model.compat = {
			supportsStore: false,
			supportsDeveloperRole: false,
		};
	}
	customApiKeys.set(model, apiKey);
	return model;
}

export function resolveApiModel(modelId: string, role: ModelRole = "root"): { model: Model<Api>; provider: string } | undefined {
	if (!isFrontierModel(modelId)) return undefined;
	let firstMatch: { model: Model<Api>; provider: string } | undefined;
	const preferredProviders = new Set(["anthropic", "openai", "google", "openrouter"]);

	for (const provider of getProviders()) {
		for (const model of getModels(provider)) {
			if (model.id !== modelId) continue;
			const match = { model, provider };
			if (preferredProviders.has(provider)) return match;
			firstMatch ??= match;
		}
	}

	if (firstMatch) return firstMatch;
	const custom = createCompatibleModel(modelId, role);
	return custom ? { model: custom, provider: "custom-compatible" } : undefined;
}

export function getModelApiKey(model: Model<Api>): string | undefined {
	return customApiKeys.get(model);
}
