// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in, joinable training events for classifier routing and its executions.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime};

use futures_util::StreamExt;
use parking_lot::Mutex;
use serde_json::{Value, json};
use switchyard_costs::Pricing;
use switchyard_llm_client::ClassifierRlObserver;
use switchyard_protocol::{
    LlmResponse, LlmResponseChunk, Metadata, Response, ResponseAccumulator, Usage,
};

use crate::{ServerError, ServerResult, redaction::Redactor};

#[derive(Clone)]
pub(crate) struct ClassifierRlLog {
    file: Arc<Mutex<fs::File>>,
    pub(crate) pricing: Pricing,
    policy_version: String,
    run_id: String,
}

impl ClassifierRlLog {
    pub(crate) fn new(path: &Path, policy_version: String) -> ServerResult<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|e| ServerError::new(e.to_string()))?;
        }
        let mut options = OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options
            .open(path)
            .map_err(|e| ServerError::new(e.to_string()))?;
        Ok(Self {
            file: Arc::new(Mutex::new(file)),
            pricing: Pricing::bundled().map_err(ServerError::new)?,
            policy_version,
            run_id: uuid::Uuid::now_v7().to_string(),
        })
    }

    pub(crate) fn context(&self, metadata: &Metadata, redactor: Arc<Redactor>) -> Trace {
        let header = |name: &str| {
            metadata
                .http_headers
                .as_ref()
                .and_then(|h| h.get(name))
                .and_then(|v| v.to_str().ok())
        };
        Trace {
            log: self.clone(),
            redactor,
            identity: json!({
                "decision_id": metadata.extra_metadata.as_ref().and_then(|m| m.get("classifier_rl_decision_id")),
                "request_id": metadata.correlation_id,
                "session_id": metadata.session_id.as_deref().or_else(|| header("proxy_x_session_id")),
                "task_id": metadata.task_id.as_deref().or_else(|| header("x-switchyard-intake-task")),
                "trial_id": header("x-switchyard-trial-id"),
                "turn_id": metadata.turn_id, "agent_id": metadata.agent_id,
                "parent_agent_id": metadata.parent_agent_id,
                "task_kind": metadata.task_kind,
            }),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Trace {
    log: ClassifierRlLog,
    identity: Value,
    redactor: Arc<Redactor>,
}

impl Trace {
    pub(crate) fn emit(&self, mut event: Value) {
        let Some(fields) = event.as_object_mut() else {
            return;
        };
        fields.extend(self.identity.as_object().unwrap().clone());
        fields.insert("schema_version".into(), json!(1));
        fields.insert(
            "ts".into(),
            json!(humantime::format_rfc3339_millis(SystemTime::now()).to_string()),
        );
        fields.insert("run_id".into(), json!(self.log.run_id));
        fields.insert("policy_version".into(), json!(self.log.policy_version));
        // Final execution events own successful answer cost, including streams.
        // Classifier and failed candidate calls own their individual costs.
        // This avoids billing a buffered answer again in call_completed.
        let account_cost = fields.get("event").and_then(Value::as_str)
            == Some("execution_completed")
            || (fields.get("event").and_then(Value::as_str) == Some("call_completed")
                && (fields.get("phase").and_then(Value::as_str) != Some("completion")
                    || fields.get("error").is_some_and(|error| !error.is_null())));
        if account_cost
            && let (Some(model), Some(usage)) = (
                fields.get("target").and_then(Value::as_str),
                fields.get("usage"),
            )
            && let Ok(usage) = serde_json::from_value::<Usage>(usage.clone())
        {
            let cost = self.log.pricing.cost(model, &usage);
            fields.insert("cost".into(), json!(cost));
        }
        scrub_credentials(&mut event);
        let event = self.redactor.value(event);
        let result = (|| -> std::io::Result<()> {
            let mut line = serde_json::to_vec(&event)?;
            line.push(b'\n');
            self.log.file.lock().write_all(&line)
        })();
        if let Err(error) = result {
            tracing::warn!(%error, "classifier RL log append failed");
        }
    }

    pub(crate) fn observer(&self) -> ClassifierRlObserver {
        let trace = self.clone();
        Arc::new(move |event| trace.emit(event))
    }

    pub(crate) fn guard(&self) -> TraceGuard {
        TraceGuard {
            trace: self.clone(),
            started: Instant::now(),
            finished: false,
        }
    }

    /// Preserve streaming and signed events while accumulating a training response.
    pub(crate) fn observe_response(
        &self,
        mut response: Response,
        target: String,
        mut guard: TraceGuard,
    ) -> Response {
        match response.llm_response {
            LlmResponse::Agg(agg) => {
                guard.finish(json!({"event": "execution_completed", "target": target,
                    "response": agg, "usage": agg.usage, "status": "completed", "reward": null}));
                response.llm_response = LlmResponse::Agg(agg);
            }
            LlmResponse::Stream(mut source) => {
                let wrapped = async_stream::stream! {
                    let mut accumulator = Some(ResponseAccumulator::new());
                    let mut terminal = false;
                    while let Some(item) = source.next().await {
                        let mut failed = item.is_err();
                        if let Ok(event) = &item {
                            for chunk in event.normalized() {
                                failed |= matches!(chunk, LlmResponseChunk::StreamError { .. } | LlmResponseChunk::DecodeError { .. });
                                terminal |= matches!(chunk, LlmResponseChunk::MessageStop { .. });
                                if let Some(acc) = accumulator.as_mut() { acc.push(chunk.clone()); }
                            }
                        }
                        if failed || terminal {
                            let agg = accumulator.take().map(ResponseAccumulator::finish);
                            guard.finish(json!({"event": "execution_completed", "target": target,
                                "status": if failed { "error" } else { "completed" },
                                "error": item.as_ref().err().map(ToString::to_string),
                                "usage": agg.as_ref().map(|a| &a.usage), "response": agg, "reward": null}));
                        }
                        yield item;
                    }
                    if !terminal && !guard.finished {
                        guard.finish(json!({"event": "execution_completed", "target": target,
                            "status": "incomplete", "reward": null}));
                    }
                };
                response.llm_response = LlmResponse::Stream(Box::pin(wrapped));
            }
        }
        response
    }
}

pub(crate) struct TraceGuard {
    trace: Trace,
    started: Instant,
    finished: bool,
}

impl TraceGuard {
    pub(crate) fn finish(&mut self, mut event: Value) {
        if self.finished {
            return;
        }
        self.finished = true;
        event["duration_ms"] = json!(self.started.elapsed().as_secs_f64() * 1000.0);
        self.trace.emit(event);
    }
}

impl Drop for TraceGuard {
    fn drop(&mut self) {
        self.finish(json!({"event": "execution_completed", "status": "cancelled", "reward": null}));
    }
}

fn scrub_credentials(value: &mut Value) {
    match value {
        Value::Object(fields) => {
            for (key, value) in fields {
                let key = key.to_ascii_lowercase().replace('-', "_");
                if matches!(
                    key.as_str(),
                    "authorization"
                        | "authorization_token"
                        | "api_key"
                        | "x_api_key"
                        | "x_switchyard_api_key"
                        | "access_token"
                        | "cookie"
                        | "set_cookie"
                ) {
                    *value = json!("[REDACTED]");
                } else {
                    scrub_credentials(value);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                scrub_credentials(item);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;

    fn fixture() -> (tempfile::TempDir, Trace) {
        let dir = tempfile::tempdir().unwrap();
        let log =
            ClassifierRlLog::new(&dir.path().join("events.jsonl"), "test-policy".into()).unwrap();
        let trace = log.context(
            &Metadata::default(),
            Arc::new(Redactor::new(&["provider-secret".into()])),
        );
        (dir, trace)
    }

    #[test]
    fn successful_answers_are_priced_once_and_unknown_usage_stays_unknown() {
        let (dir, mut trace) = fixture();
        trace.log.pricing = serde_json::from_value(json!({"target": {
            "input": 1, "output": 2, "cache_read": 0.1, "cache_write": 1
        }}))
        .unwrap();
        let usage = json!({"input_tokens": 10, "output_tokens": 2});
        trace.emit(json!({"event": "call_completed", "phase": "routing", "target": "target", "usage": usage, "error": null}));
        trace.emit(json!({"event": "call_completed", "phase": "completion", "target": "target", "usage": usage, "error": null}));
        trace.emit(json!({"event": "execution_completed", "target": "target", "usage": usage}));
        trace.emit(json!({"event": "call_completed", "phase": "routing", "target": "target", "usage": null, "error": "failed"}));
        let text = fs::read_to_string(dir.path().join("events.jsonl")).unwrap();
        let events: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(events.iter().filter(|e| e["cost"].is_object()).count(), 2);
        let cost: f64 = events
            .iter()
            .filter_map(|e| e["cost"]["usd"].as_f64())
            .sum();
        assert!((cost - 0.000028).abs() < 1e-12);
        assert!(events[3]["cost"].is_null());
    }

    #[test]
    fn credentials_are_redacted_and_cancelled_requests_are_explicit() {
        let (dir, trace) = fixture();
        trace.emit(json!({"event": "request_started", "request": {
            "tools": [{"authorization_token": "mcp-secret"}],
            "preservation": {"api-key": "embedded-secret"},
            "content": "provider-secret"
        }}));
        drop(trace.guard());
        let text = fs::read_to_string(dir.path().join("events.jsonl")).unwrap();
        for secret in ["provider-secret", "mcp-secret", "embedded-secret"] {
            assert!(!text.contains(secret));
        }
        let events: Vec<Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(events[1]["status"], "cancelled");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(dir.path().join("events.jsonl"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[tokio::test]
    async fn dropping_an_unpolled_stream_records_cancellation() {
        let (dir, trace) = fixture();
        let response = Response {
            llm_response: LlmResponse::Stream(Box::pin(stream::pending())),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        };
        let response = trace.observe_response(response, "target".into(), trace.guard());
        drop(response);
        let text = fs::read_to_string(dir.path().join("events.jsonl")).unwrap();
        let event: Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(event["status"], "cancelled");
    }

    #[tokio::test]
    async fn stream_errors_and_missing_terminal_events_are_not_successes() {
        for failed in [false, true] {
            let (dir, trace) = fixture();
            let chunks = if failed {
                vec![Ok(switchyard_protocol::LlmResponseStreamEvent::new(vec![
                    LlmResponseChunk::StreamError {
                        message: "provider-secret".into(),
                    },
                ]))]
            } else {
                vec![]
            };
            let response = Response {
                llm_response: LlmResponse::Stream(Box::pin(stream::iter(chunks))),
                metadata: None,
                upstream_headers: http::HeaderMap::new(),
            };
            let response = trace.observe_response(response, "target".into(), trace.guard());
            if let LlmResponse::Stream(mut source) = response.llm_response {
                while source.next().await.is_some() {}
            }
            let text = fs::read_to_string(dir.path().join("events.jsonl")).unwrap();
            let events: Vec<Value> = text
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect();
            assert_eq!(events.len(), 1);
            assert_eq!(
                events[0]["status"],
                if failed { "error" } else { "incomplete" }
            );
            assert!(!text.contains("provider-secret"));
        }
    }
}
