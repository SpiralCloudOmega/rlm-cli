import { afterEach, describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  createCompatibleModel,
  getModelApiKey,
  isFrontierModel,
  resolveApiModel,
} from "../dist/models.js";

const savedEnv = { ...process.env };

afterEach(() => {
  for (const key of Object.keys(process.env)) {
    if (!(key in savedEnv)) delete process.env[key];
  }
  Object.assign(process.env, savedEnv);
});

describe("frontier model resolution", () => {
  it("accepts current and newer frontier families while rejecting older models", () => {
    assert.equal(isFrontierModel("gpt-5.6-sol"), true);
    assert.equal(isFrontierModel("gpt-6"), true);
    assert.equal(isFrontierModel("claude-fable-5"), true);
    assert.equal(isFrontierModel("kimi-k3"), true);
    assert.equal(isFrontierModel("gpt-4o"), false);
    assert.equal(isFrontierModel("claude-sonnet-4-6"), false);
    assert.equal(isFrontierModel("kimi-k2.5"), false);
  });

  it("creates an OpenAI-compatible Kimi K3 sub-model from role-specific settings", () => {
    process.env.RLM_SUB_API_KEY = "test-key";
    process.env.RLM_SUB_BASE_URL = "https://example.invalid/v1/";
    process.env.RLM_SUB_CONTEXT_WINDOW = "262144";
    const model = createCompatibleModel("kimi-k3", "sub");
    assert.ok(model);
    assert.equal(model.api, "openai-completions");
    assert.equal(model.baseUrl, "https://example.invalid/v1");
    assert.equal(model.contextWindow, 262144);
    assert.equal(getModelApiKey(model), "test-key");
  });

  it("creates an Anthropic-compatible Claude Fable model", () => {
    process.env.ANTHROPIC_API_KEY = "test-key";
    const model = createCompatibleModel("claude-fable-5");
    assert.ok(model);
    assert.equal(model.api, "anthropic-messages");
    assert.equal(model.baseUrl, "https://api.anthropic.com");
  });

  it("does not resolve a model below the frontier floor", () => {
    process.env.RLM_API_KEY = "test-key";
    assert.equal(resolveApiModel("gpt-4o"), undefined);
  });
});
