// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level regression coverage for the server CLI.

use std::fs;
use std::process::Command;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn dry_run_error(client_config: &str, env: Option<(&str, &str)>) -> TestResult<String> {
    let directory = tempfile::tempdir()?;
    let config = directory.path().join("routes.toml");
    fs::write(
        &config,
        format!(
            r#"
schema_version = 1

[llm_clients.invalid]
format = "openai_chat"
{client_config}

[targets.invalid]
id = "upstream-model"
llm_client = "invalid"

[routes.invalid]
id = "test-route"
type = "passthrough"
target = "invalid"
"#
        ),
    )?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_switchyard-server"));
    command.args(["--config", config.to_string_lossy().as_ref(), "--dry-run"]);
    if let Some((name, value)) = env {
        command.env(name, value);
    }
    let output = command.output()?;
    assert!(!output.status.success());
    Ok(String::from_utf8(output.stderr)?)
}

#[test]
fn dry_run_rejects_invalid_base_url() -> TestResult {
    let stderr = dry_run_error("base_url = \"not a url\"", None)?;
    assert!(
        stderr.contains("base_url must be an absolute HTTP(S) URL"),
        "{stderr}"
    );
    Ok(())
}

// A judge adds no routing choice when every completion category names the same target.
#[test]
fn dry_run_warns_when_custom_classifier_has_one_target() -> TestResult {
    let directory = tempfile::tempdir()?;
    let config = directory.path().join("routes.toml");
    for (targets, warning_count) in [("[\"answer\"]", 1), ("[\"answer\", \"second\"]", 0)] {
        fs::write(
            &config,
            format!(
                r#"
schema_version = 1
[llm_clients.upstream]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
[targets]
judge = {{ id = "judge", llm_client = "upstream" }}
answer = {{ id = "answer", llm_client = "upstream" }}
second = {{ id = "second", llm_client = "upstream" }}
[routes.custom]
id = "custom"
type = "llm_classifier"
mode = "custom"
models = {{ judge = ["judge"], any = {targets}, efficient = ["answer"] }}
default_target = "efficient"
prompt = "Choose efficient."
response_schema = '{{"type":"object","properties":{{"target":{{"type":"string"}}}},"required":["target"],"additionalProperties":false}}'
policy = {{ type = "target_selector", selector = "/target" }}
"#
            ),
        )?;
        let output = Command::new(env!("CARGO_BIN_EXE_switchyard-server"))
            .args(["--config", config.to_string_lossy().as_ref(), "--dry-run"])
            .env("RUST_LOG", "warn")
            .output()?;
        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout)?;
        let stderr = String::from_utf8(output.stderr)?;
        assert!(stdout.contains("server OK: custom"));
        let logs = format!("{stdout}{stderr}");
        assert_eq!(
            logs.matches("only one routing target").count(),
            warning_count,
            "{logs}"
        );
    }
    Ok(())
}

// Dry-run rejects unsendable headers before startup without exposing credentials.
#[test]
fn dry_run_rejects_unsendable_configured_headers() -> TestResult {
    const INVALID_KEY_ENV: &str = "SWITCHYARD_CLI_TEST_INVALID_HEADER_KEY";
    const INVALID_KEY: &str = "canary\nsecret";
    let cases = [
        (
            "base_url = \"https://example.test/v1\"\n\
             extra_headers = { \"bad header\" = \"value\" }"
                .to_string(),
            None,
            "invalid HTTP header name \"bad header\"",
        ),
        (
            format!("base_url = \"https://example.test/v1\"\napi_key_env = \"{INVALID_KEY_ENV}\""),
            Some((INVALID_KEY_ENV, INVALID_KEY)),
            "api_key cannot be encoded as an HTTP header",
        ),
    ];

    for (client_config, env, expected) in cases {
        let stderr = dry_run_error(&client_config, env)?;
        assert!(stderr.contains(expected), "{stderr}");
        assert!(!stderr.contains(INVALID_KEY), "API key leaked in: {stderr}");
    }
    Ok(())
}

#[test]
fn classifier_rl_flags_are_opt_in_and_accept_independent_pricing() -> TestResult {
    let directory = tempfile::tempdir()?;
    let config = directory.path().join("routes.toml");
    let trace = directory.path().join("events.jsonl");
    let prices = directory.path().join("prices.json");
    fs::write(
        &prices,
        r#"{"model":{"input":1,"output":2,"cache_read":0.1,"cache_write":1}}"#,
    )?;
    fs::write(
        &config,
        r#"
schema_version = 1
[llm_clients.local]
format = "openai_chat"
base_url = "http://127.0.0.1:1/v1"
[targets.model]
id = "model"
llm_client = "local"
[routes.direct]
id = "direct"
type = "passthrough"
target = "model"
"#,
    )?;
    let command = || {
        let mut c = Command::new(env!("CARGO_BIN_EXE_switchyard-server"));
        c.arg("--config")
            .arg(&config)
            .arg("--dry-run")
            .current_dir(directory.path());
        c
    };
    assert!(command().output()?.status.success());
    assert!(!directory.path().join("classifier_rl_data").exists());
    let output = command()
        .arg("--classifier-rl-log-file")
        .arg(&trace)
        .output()?;
    assert!(!output.status.success());
    assert!(!trace.exists());
    let output = command()
        .arg("--enable-classifier-rl-logging")
        .arg("--classifier-rl-log-file")
        .arg(&trace)
        .arg("--classifier-rl-policy-version")
        .arg("qwen-test")
        .arg("--pricing-file")
        .arg(&prices)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(fs::read_to_string(&trace)?, "");
    Ok(())
}
