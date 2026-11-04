# Server CLI Reference

`switchyard-server` runs the standalone Rust proxy for API clients and custom deployments.

Install the standalone binary with `cargo install --locked switchyard-server`.
It reads a native TOML deployment.

### Usage

```bash
switchyard-server --config <deployment.toml> [options]
```

| Option | Default | Purpose |
|---|---|---|
| `--config PATH` | Required | TOML file defining LLM clients, targets, and algorithm routes. |
| `--host HOST` | `0.0.0.0` | Address on which the server listens. |
| `-p, --port PORT` | `4000` | Port on which the server listens. |
| `--backlog BACKLOG` | `65535` | TCP listen backlog configured before accepting traffic. |
| `--shutdown-timeout SHUTDOWN_TIMEOUT` | `30s` | Maximum time active requests may drain during shutdown. |
| `--dry-run` | Off | Validate the deployment without binding a socket. |
| `--routing-log-file PATH` | None | Append durable per-request routing records to this JSONL file. |
| `--enable-classifier-rl-logging` | Off | Record content-bearing classifier training events. |
| `--classifier-rl-log-file PATH` | `classifier_rl_data/events.jsonl` | Append training events as JSONL; requires classifier RL logging. |
| `--classifier-rl-policy-version VERSION` | `unversioned` | Identify the classifier checkpoint and routing policy; requires classifier RL logging. |
| `--pricing-file PATH` | None | Estimate unreported call costs from a TOML or JSON price table; requires routing or classifier RL logging. |
| `--tls-cert PATH` | None | PEM certificate path; requires `--tls-key`. |
| `--tls-key PATH` | None | PEM private-key path; requires `--tls-cert`. |
| `-h, --help` | — | Print command help. |
| `-V, --version` | — | Print the server version. |

Validate a deployment, then start the proxy:

```bash
switchyard-server --config routes.toml --dry-run
switchyard-server --config routes.toml \
  --host 127.0.0.1 --port 4000
```

## Classifier training logs

```bash
switchyard-server --config routes.toml \
  --enable-classifier-rl-logging \
  --classifier-rl-log-file classifier_rl_data/events.jsonl \
  --classifier-rl-policy-version qwen-router-checkpoint-001 \
  --routing-log-file routing.jsonl
```

The Python command also accepts `switchyard serve` followed by these options.
The native server uses TOML deployments through `--config`; older Python YAML
routing bundles and `--routing-profiles` are not accepted by this fork.

Each HTTP completion request gets a generated `decision_id`. The response header
`x-switchyard-decision-id` exposes it to the evaluation harness. Routing records
carry the same ID when both loggers are enabled. Send `x-switchyard-session-id`,
`x-switchyard-task-id`, `x-switchyard-turn-id`, and `x-switchyard-trial-id` to join
turns with task outcomes. The legacy log-only session and intake-task headers
remain available as fallbacks.

Events include:

- `request_started`: normalized conversation, tools, route, and endpoint identities.
- `call_started`: actual prepared routing or completion input, target, call ID,
  phase, and candidate position. Typed decision calls retain their question schema.
- `call_completed`: classifier output, available usage, estimated or reported cost,
  duration, and errors. Streamed routing calls retain an accumulated output.
  Completion streams record their final output in `execution_completed`.
- `routing_decision`: selected target, ordered fallback candidates, and prepared
  request. `routing_outcome` retains judgments supplied by the algorithm.
- `execution_completed`: response and usage, or an explicit error, rejection,
  cancellation, or incomplete stream. `duration_ms` measures the local request
  lifecycle. Decision-only API requests use the `decision_only` status.

Every event carries `schema_version`, timestamp, run ID, policy version, and task
identity. Use a new policy version whenever checkpoint, routing configuration,
model portfolio, or evaluation-relevant settings change. Target and provider model
IDs are preserved separately in endpoint identities and responses.

`reward` and `action_probability` remain null. Classifier confidence and token
logprobs, when present in the provider response, are not routing propensities.
The evaluator must add task scores. A trainer that explores must retain its
actual action-selection probabilities. Provider-internal retries and physical
GPU costs are not separately measured by these events. Missing usage or prices
remain unknown. Supply negotiated rates with `--pricing-file`.

For cost accounting, sum `cost` from classifier/decision `call_completed` events
and final `execution_completed` events. Successful completion-call observations
do not carry a second cost. Failed candidate calls have unknown cost unless
usage is available. Response token counts may appear in both call and execution
events; do not sum every occurrence of usage. Each call ID identifies one logical
client call, whose backend may retry internally.

Logging is off by default. Enabling it persists conversation and tool content.
New files are created with owner-only permissions on Unix. HTTP metadata and
auth headers are omitted; credential fields and configured provider keys are
redacted. This is not a general detector for secrets inside arbitrary text.
Existing file permissions are retained. Rotate and retain files according to
your deployment's data policy. Failed appends emit a warning and do not fail
inference. Events are appended as complete JSON lines; they are not fsynced per
request, so a process or machine failure may lose recent writes. Logging adds
serialization, stream accumulation, and disk-write overhead when enabled.

Python hosts can pass `enable_classifier_rl_logging=True`,
`classifier_rl_log_file=...`, and `classifier_rl_policy_version=...` to
`switchyard_rust.server.Server`. `routing_log_file` and `pricing_file` are also
accepted.

## Related Documentation

- [Getting Started](getting_started.md): installation, configuration, and validation
- [`switchyard-server`](../crates/switchyard-server/README.md): complete TOML schema, TLS, and metrics
