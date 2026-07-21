#!/usr/bin/env tsx
/**
 * Standalone RLM CLI — run Recursive Language Model queries from the terminal.
 *
 * Usage:
 *   npx tsx src/cli.ts --file large-file.txt "What are the main themes?"
 *   npx tsx src/cli.ts --url https://example.com/big.txt "Summarize this"
 *   cat data.txt | npx tsx src/cli.ts --stdin "Count the errors"
 *
 * Environment (pick one):
 *   ANTHROPIC_API_KEY — for Anthropic models
 *   OPENAI_API_KEY    — for OpenAI models
 *   GEMINI_API_KEY    — for Google models
 *   OPENROUTER_API_KEY — for OpenRouter models
 */

import "./env.js";
import * as fs from "node:fs";

// Dynamic imports — ensures env.js has set process.env BEFORE pi-ai loads
const { PythonRepl } = await import("./repl.js");
const { runRlmLoop } = await import("./rlm.js");
const { loadConfig } = await import("./config.js");
const { resolveApiModel } = await import("./models.js");

import type { Api, Model } from "@mariozechner/pi-ai";

// ── Arg parsing ─────────────────────────────────────────────────────────────

function usage(exitCode = 1): never {
	console.error(`
rlm-cli — Recursive Language Model CLI (arXiv:2512.24601)

USAGE
  rlm run [OPTIONS] "<query>"

OPTIONS
  --model <id>     Model ID (default: RLM_MODEL from .env)
  --file <path>    Read context from a file
  --url <url>      Fetch context from a URL
  --stdin          Read context from stdin (pipe data in)
  --verbose        Show iteration progress

EXAMPLES
  rlm run "Explain recursive language models"
  rlm run --file big.txt "List all classes"
  curl -s https://example.com/large.py | rlm run --stdin "Summarize"
  rlm run --url https://raw.githubusercontent.com/.../typing.py "Count public classes"
`.trim());
	process.exit(exitCode);
}

interface CliArgs {
	modelId: string;
	file?: string;
	url?: string;
	useStdin: boolean;
	verbose: boolean;
	query: string;
}

function parseArgs(): CliArgs {
	const args = process.argv.slice(2);
	let modelId: string | undefined;
	let file: string | undefined;
	let url: string | undefined;
	let useStdin = false;
	let verbose = false;
	const positional: string[] = [];

	for (let i = 0; i < args.length; i++) {
		const arg = args[i];
		if (arg === "--model" && i + 1 < args.length) {
			modelId = args[++i];
		} else if (arg === "--file" && i + 1 < args.length) {
			file = args[++i];
		} else if (arg === "--url" && i + 1 < args.length) {
			url = args[++i];
		} else if (arg === "--stdin") {
			useStdin = true;
		} else if (arg === "--verbose") {
			verbose = true;
		} else if (arg === "--help" || arg === "-h") {
			usage(0);
		} else if (!arg.startsWith("--")) {
			positional.push(arg);
		} else {
			console.error(`Unknown option: ${arg}`);
			usage();
		}
	}

	if (!modelId) {
		modelId = process.env.RLM_MODEL || "gpt-5.6-sol";
	}

	if (positional.length === 0) {
		console.error("Error: query argument is required");
		usage();
	}

	const query = positional.join(" ");
	return { modelId, file, url, useStdin, verbose, query };
}

// ── Helpers ─────────────────────────────────────────────────────────────────

const MAX_STDIN_BYTES = 50 * 1024 * 1024; // 50MB

async function readStdin(): Promise<string> {
	const chunks: Buffer[] = [];
	let total = 0;
	for await (const chunk of process.stdin) {
		total += (chunk as Buffer).length;
		if (total > MAX_STDIN_BYTES) {
			throw new Error(`stdin exceeds ${MAX_STDIN_BYTES / 1024 / 1024}MB limit`);
		}
		chunks.push(chunk as Buffer);
	}
	return Buffer.concat(chunks).toString("utf-8");
}

const FETCH_TIMEOUT_MS = 30_000;
const MAX_RESPONSE_BYTES = 50 * 1024 * 1024; // 50MB

async function fetchUrl(url: string): Promise<string> {
	const resp = await fetch(url, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
	if (!resp.ok) {
		throw new Error(`Failed to fetch ${url}: ${resp.status} ${resp.statusText}`);
	}
	const contentLength = resp.headers.get("content-length");
	if (contentLength && parseInt(contentLength, 10) > MAX_RESPONSE_BYTES) {
		throw new Error(`Response too large (${(parseInt(contentLength, 10) / 1024 / 1024).toFixed(1)}MB). Limit is ${MAX_RESPONSE_BYTES / 1024 / 1024}MB.`);
	}
	const text = await resp.text();
	if (text.length > MAX_RESPONSE_BYTES) {
		throw new Error(`Response too large (${(text.length / 1024 / 1024).toFixed(1)}MB). Limit is ${MAX_RESPONSE_BYTES / 1024 / 1024}MB.`);
	}
	return text;
}

// ── Main ────────────────────────────────────────────────────────────────────

// Module-level so the top-level catch can tell a Ctrl-C abort apart from a real failure
const ac = new AbortController();

async function main(): Promise<void> {
	const args = parseArgs();
	const config = loadConfig();
	const resolved = resolveApiModel(args.modelId, "root");
	const model: Model<Api> | undefined = resolved?.model;

	if (!model) {
		console.error(`Error: could not resolve frontier model "${args.modelId}"`);
		console.error("Set its provider API key, or configure RLM_API_KEY and RLM_BASE_URL for an OpenAI-compatible endpoint.");
		process.exit(1);
	}
	const subModel = config.sub_model ? resolveApiModel(config.sub_model, "sub")?.model : undefined;
	if (config.sub_model && !subModel) {
		console.error(`Error: could not resolve sub-model "${config.sub_model}". Configure RLM_SUB_API_KEY and RLM_SUB_BASE_URL.`);
		process.exit(1);
	}

	// Load context
	let context: string;
	if (args.file) {
		try {
			const stat = fs.statSync(args.file);
			if (stat.isDirectory()) {
				console.error(`Error: "${args.file}" is a directory. Use the interactive mode (\`rlm\`) for directory loading.`);
				process.exit(1);
			}
		} catch (err: any) {
			console.error(`Error: could not access "${args.file}": ${err.message}`);
			process.exit(1);
		}
		console.error(`Reading context from file: ${args.file}`);
		try {
			context = fs.readFileSync(args.file, "utf-8");
		} catch (err: any) {
			console.error(`Error: could not read file "${args.file}": ${err.message}`);
			process.exit(1);
		}
	} else if (args.url) {
		console.error(`Fetching context from URL: ${args.url}`);
		context = await fetchUrl(args.url);
	} else if (args.useStdin) {
		console.error("Reading context from stdin...");
		context = await readStdin();
	} else {
		context = "";
		console.error("No context provided; running query in general-purpose mode.");
	}

	console.error(`Context loaded: ${context.length.toLocaleString()} characters`);
	console.error(`Model: ${model.id}`);
	console.error(`Query: ${args.query}`);
	console.error("---");

	// Start REPL
	const repl = new PythonRepl();

	const abortAndExit = () => {
		console.error("\nAborting...");
		ac.abort();
	};
	process.on("SIGINT", abortAndExit);
	process.on("SIGTERM", abortAndExit);

	try {
		await repl.start(ac.signal);

		const startTime = Date.now();
		const result = await runRlmLoop({
			context,
			query: args.query,
			model,
			subModel,
			repl,
			signal: ac.signal,
			onProgress: args.verbose
				? (info) => {
						const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
						console.error(
							`[${elapsed}s] Iteration ${info.iteration}/${info.maxIterations} | ` +
								`Sub-queries: ${info.subQueries} | Phase: ${info.phase}`,
						);
					}
				: undefined,
		});

		const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
		console.error("---");
		console.error(
			`Completed in ${elapsed}s | ${result.iterations} iterations | ${result.totalSubQueries} sub-queries | ${result.completed ? "success" : "incomplete"}`,
		);
		console.error("---");

		// Write the answer to stdout (not stderr) so it can be piped
		console.log(result.answer);
	} finally {
		repl.shutdown();
	}
}

main().catch((err) => {
	// A user-initiated Ctrl-C surfaces here as an abort — not a real failure
	if (ac.signal.aborted) {
		console.error("\x1b[2mInterrupted.\x1b[0m");
		process.exit(130);
	}
	console.error("Fatal error:", err);
	process.exit(1);
});
