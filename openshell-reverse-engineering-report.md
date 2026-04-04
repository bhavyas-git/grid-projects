# Project 4 Reverse Engineering Report: OpenShell

- **Project Name:** OpenShell
- **Repository:** [https://github.com/NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell)
- **Project Category:** AI agent runtime, sandboxing, policy enforcement, gateway orchestration

## 1. Project Overview and Key Components

### Repository Analysis Summary

OpenShell is a security-first runtime for autonomous AI agents. Its core design goal is not just to run an agent in Docker, but to create a layered execution environment where the agent process, its credentials, its network traffic, and its control plane are all constrained by explicit policy. The repository implements that design as a multi-crate Rust workspace with a gateway (`openshell-server`), sandbox runtime (`openshell-sandbox`), YAML/protobuf policy layer (`openshell-policy`, `proto/`), bootstrap/deployment tooling (`openshell-bootstrap`, `deploy/`, `tasks/`), a CLI (`openshell-cli`), a TUI (`openshell-tui`), a Python SDK (`python/openshell/`), and a large architecture/doc/test surface that explains and validates the system.

The strongest architectural theme across the repository is separation of concerns:

- The **gateway** is a control plane, persistence layer, and authentication boundary.
- The **sandbox** is where enforcement happens: proxying, OPA/Rego evaluation, Landlock, seccomp, network namespaces, SSH mediation, and credential placeholder rewriting.
- The **policy representation** is a typed schema expressed in protobuf and YAML, separate from any one enforcement backend.
- The **provider system** manages secrets as named bundles and resolves them at runtime rather than baking them into pod specs or image layers.
- The **documentation** is unusually rich for an early-stage project and directly maps to implementation details, making the repository closer to a reference architecture than a code dump.

### High-Level Repository Structure

The repository contains **515 files**. Top-level distribution:

| Top-level area | File count | Primary role |
|---|---:|---|
| `crates/` | 228 | Rust workspace implementation |
| `docs/` | 61 | User-facing documentation and Sphinx extensions |
| `e2e/` | 38 | End-to-end validation |
| `deploy/` | 26 | Docker, Helm, Kubernetes manifests |
| `.github/` | 24 | CI, issue/PR templates, contribution gates |
| `tasks/` | 24 | `mise` task definitions and release/build scripts |
| repository root | 23 | project metadata, onboarding, policy, testing |
| `examples/` | 20 | example policies and deployment patterns |
| `.agents/` | 19 | agent workflow skills |
| `architecture/` | 16 | architecture specifications |
| `scripts/` | 11 | utilities |
| `proto/` | 5 | gRPC contracts |
| `python/` | 5 | Python SDK |
| remaining dirs | 15 | IDE/agent metadata, RFCs |

Dominant file types:

| Extension | Count | Interpretation |
|---|---:|---|
| `.rs` | 202 | Rust is the primary implementation language |
| `.md` | 91 | strong architectural/documentation discipline |
| `.py` | 45 | Python SDK, docs tooling, e2e tests, scripts |
| `.toml` | 30 | Rust workspace and `mise` task orchestration |
| `.json` | 26 | schemas, configs, OCSF artifacts |
| `.sh` | 22 | bootstrap/build/test helpers |
| `.yml` / `.yaml` | 39 | CI, Helm, policies, manifests |

### Core Runtime Components

| Component | Key files | Role |
|---|---|---|
| CLI | `crates/openshell-cli/src/main.rs`, `run.rs` | User entry point for sandbox, provider, policy, and gateway operations |
| Gateway | `crates/openshell-server/src/lib.rs`, `grpc.rs`, `sandbox/mod.rs` | Control plane, auth boundary, persistence, sandbox lifecycle |
| Sandbox | `crates/openshell-sandbox/src/lib.rs`, `proxy.rs`, `opa.rs`, `ssh.rs` | Runtime enforcement inside each sandbox |
| Policy model | `crates/openshell-policy/src/lib.rs`, `proto/sandbox.proto` | Canonical schema and YAML/proto conversion |
| Bootstrap | `crates/openshell-bootstrap/src/lib.rs`, `pki.rs` | Cluster deployment, PKI generation, image/bootstrap logistics |
| Provider system | `crates/openshell-providers/src/providers/*.rs`, `architecture/sandbox-providers.md` | Credential discovery and normalization |
| Inference router | `crates/openshell-router/src/*.rs`, `architecture/inference-routing.md` | Privacy-aware model routing |
| TUI | `crates/openshell-tui/src/ui/*.rs` | Monitoring and management terminal UI |

### Repository-Wide Architectural Pattern

The repository repeatedly uses the same pattern:

1. Model policy or state as explicit typed data.
2. Keep the gateway authoritative for storage and distribution.
3. Push the minimum necessary data into the sandbox at runtime.
4. Enforce locally inside the sandbox using kernel and proxy mechanisms.
5. Preserve operator ergonomics with hot-reload where safe and immutability where required.

That pattern is visible in:

- policy updates (`architecture/security-policy.md`, `crates/openshell-server/src/grpc.rs`, `crates/openshell-sandbox/src/lib.rs`)
- credentials (`architecture/sandbox-providers.md`, `crates/openshell-server/src/grpc.rs`, `crates/openshell-sandbox/src/secrets.rs`)
- transport security (`architecture/gateway-security.md`, `crates/openshell-bootstrap/src/pki.rs`, `crates/openshell-server/src/tls.rs`)
- network isolation (`architecture/sandbox.md`, `crates/openshell-sandbox/src/proxy.rs`, `crates/openshell-sandbox/src/sandbox/linux/netns.rs`, `seccomp.rs`)

## 2. Deep Reasoning Questions and Analysis

### Q1. Why does OpenShell use layered security with multiple enforcement points rather than relying on a single security boundary?

**Short answer:** because the repository is built around the assumption that different attacks target different layers, and no single mechanism can simultaneously protect filesystem access, syscall behavior, network egress, credential handling, and control-plane trust.

**Proof from the repository**

- `README.md` explicitly defines four protection layers: filesystem, network, process, and inference.
- `architecture/sandbox.md` describes combined use of Landlock, seccomp, network namespaces, HTTP CONNECT proxying, OPA/Rego evaluation, L7 inspection, SSH mediation, and credential rewriting.
- `architecture/system-architecture.md` shows these layers placed at different points: gateway mTLS, sandbox supervisor proxy, OPA policy evaluation, agent-process kernel isolation.
- `crates/openshell-sandbox/src/sandbox/linux/landlock.rs` enforces path allowlists.
- `crates/openshell-sandbox/src/sandbox/linux/seccomp.rs` restricts socket domains and dangerous syscall behavior.
- `crates/openshell-sandbox/src/sandbox/linux/netns.rs` isolates the network path into a veth pair and namespace.
- `crates/openshell-sandbox/src/proxy.rs` evaluates every outbound flow through OPA and optional L7 checks.
- `crates/openshell-sandbox/src/secrets.rs` prevents raw provider secrets from being directly exposed to child processes.
- `architecture/gateway-security.md` adds mTLS and gateway-side authentication controls before sandbox logic is even reached.

**Why one layer would be insufficient**

- Landlock cannot stop credential exfiltration if the process is allowed to make arbitrary outbound HTTP calls.
- A proxy alone cannot stop an agent from reading arbitrary host files if filesystem isolation is absent.
- mTLS to the gateway does nothing to protect runtime behavior once the agent process is already inside a sandbox.
- Seccomp cannot express "allow `GET /repos/**` but deny `POST /repos/**/issues`."
- L7 rules cannot protect against a binary that is replaced after startup unless process identity is also checked; OpenShell adds TOFU binary hashing in `crates/openshell-sandbox/src/identity.rs`.

**Concrete attacker-model mapping**

| Threat | Enforcement point |
|---|---|
| Agent reads sensitive files | Landlock filesystem policy |
| Agent opens raw sockets / bypass channels | seccomp + network namespace |
| Agent uses allowed TCP but wrong API path/method | proxy + L7 policy |
| Agent steals another provider's credentials | provider scoping + placeholder indirection |
| Unauthenticated client hits control plane | gateway TLS/mTLS/auth |
| Binary swapped after first approval | `BinaryIdentityCache` TOFU check |

**Conclusion**

OpenShell is explicitly designed for defense in depth because its threat model is multi-dimensional. The codebase assumes partial failure of individual controls and compensates by placing additional controls at different trust boundaries.

### Q2. What problem does OpenShell solve by supporting hot-reloading of policies without restarting orchestrated agents?

**Short answer:** it lets operators change network and inference permissions during a running session without killing the agent process, losing agent state, or forcing a full sandbox restart.

**Proof from the repository**

- `README.md` states that network and inference sections are hot-reloadable while filesystem and process sections are locked.
- `architecture/security-policy.md` and `architecture/sandbox.md` document a poll-based live-update path for dynamic policy domains.
- `crates/openshell-sandbox/src/lib.rs` implements `run_policy_poll_loop()`, which repeatedly polls the gateway and reloads OPA only when the policy hash changes.
- `crates/openshell-server/src/grpc.rs` implements deterministic policy hashing, versioning, revision storage, and `ReportPolicyStatus`.
- `crates/openshell-cli/src/run.rs` has UX for unchanged policies (`"Policy unchanged"`), which only makes sense in an idempotent live-update system.

**Operational problem being solved**

Without hot reload, every policy change would imply:

1. stop the sandbox,
2. restart the sandbox,
3. restart or reconnect the agent,
4. lose in-memory context, transient workflows, shell sessions, and possibly user trust.

For autonomous agents, that is expensive because the valuable thing is often the ongoing stateful session: open files, partially-completed work, shell history, tool state, live SSH session, or an inference workflow in progress.

**Concrete example from repository behavior**

The quickstart in `README.md` shows a workflow where a sandbox starts with blocked GitHub access, then `openshell policy set ... --wait` enables allowed requests without recreating the sandbox. That is a user-facing demonstration of why hot reload exists.

**Deeper design implication**

OpenShell deliberately limits hot reload to policy parts enforced by the proxy and OPA engine, because those can be atomically swapped at runtime. That gives fast operator response for network changes while preserving correctness for kernel-bound controls that are inherently one-way.

**Code evidence**

Snippet 1 is the sandbox-side poll loop from `crates/openshell-sandbox/src/lib.rs`:

```rust
async fn run_policy_poll_loop(
    endpoint: &str,
    sandbox_id: &str,
    opa_engine: &Arc<OpaEngine>,
    interval_secs: u64,
) -> Result<()> {
    // Reuse a cached client so policy polling does not pay a full reconnect cost each time.
    let client = CachedOpenShellClient::connect(endpoint).await?;
    // Track the last effective configuration and policy we have already applied.
    let mut current_config_revision: u64 = 0;
    let mut current_policy_hash = String::new();
    ...
    // If the effective config fingerprint is unchanged, there is nothing to do.
    if result.config_revision == current_config_revision {
        continue;
    }

    // Settings can change without the policy payload changing, so compare hashes separately.
    let policy_changed = result.policy_hash != current_policy_hash;
    ...
    if policy_changed {
        // Reload only if the gateway actually returned a policy payload.
        let Some(policy) = result.policy.as_ref() else { ... };
        match opa_engine.reload_from_proto(policy) {
            Ok(()) => { ... }
            Err(e) => {
                // Fail closed for the new revision but keep the previous working engine active.
                warn!(
                    version = result.version,
                    error = %e,
                    "Policy reload failed, keeping last-known-good policy"
                );
            }
        }
    }
}
```

What this snippet shows:

- the sandbox keeps a persistent connection context with `CachedOpenShellClient`,
- it compares `config_revision` so it does not reload on every poll,
- it separately checks `policy_hash` so settings-only changes do not force OPA replacement,
- it calls `reload_from_proto(policy)` only when the policy actually changed,
- on failure it explicitly keeps the last-known-good policy rather than forcing a restart.

Snippet 2 is the server-side deterministic hash from `crates/openshell-server/src/grpc.rs`:

```rust
fn deterministic_policy_hash(policy: &ProtoSandboxPolicy) -> String {
    let mut hasher = Sha256::new();
    // Include scalar top-level fields first.
    hasher.update(policy.version.to_le_bytes());
    if let Some(fs) = &policy.filesystem {
        hasher.update(fs.encode_to_vec());
    }
    if let Some(ll) = &policy.landlock {
        hasher.update(ll.encode_to_vec());
    }
    if let Some(p) = &policy.process {
        hasher.update(p.encode_to_vec());
    }
    // Sort map entries so logically identical policies hash the same way every time.
    let mut entries: Vec<_> = policy.network_policies.iter().collect();
    entries.sort_by_key(|(k, _)| k.as_str());
    for (key, value) in entries {
        // Hash both the key and the encoded rule contents.
        hasher.update(key.as_bytes());
        hasher.update(value.encode_to_vec());
    }
    hex::encode(hasher.finalize())
}
```

Why this code matters:

- `network_policies` is a map, so sorting by key avoids nondeterministic hashing,
- stable hashing is what makes idempotent hot reload practical,
- without this, equivalent policies could appear different and trigger unnecessary reloads.

### Q3. How does OpenShell's credential swapping mechanism prevent agents from accessing credentials outside their authorization scope?

**Short answer:** credentials are attached to the sandbox by provider name, fetched at runtime, replaced with placeholders in child-process environments, and only resolved back to real values inside the supervisor-controlled outbound proxy path.

**Proof from the repository**

- `architecture/sandbox-providers.md` states that providers are validated at sandbox creation but credentials are fetched at runtime and are not embedded in pod specs.
- `crates/openshell-server/src/grpc.rs` `resolve_provider_environment()` fetches only the providers named in `spec.providers`; first provider wins on duplicate keys and invalid env names are skipped.
- `crates/openshell-sandbox/src/secrets.rs` defines placeholder grammar `openshell:resolve:env:*` and `SecretResolver::from_provider_env()`, which converts real env values into placeholders for children while keeping a supervisor-only map of placeholder-to-secret.
- `crates/openshell-sandbox/src/secrets.rs` validates secrets to reject CR/LF/NUL injection and rejects unsafe path uses.
- `crates/openshell-sandbox/src/proxy.rs` uses `SecretResolver` during request rewriting.
- `architecture/system-architecture.md` states credentials are stripped/replaced by the sandbox-side routing path.

**Why this constrains authorization scope**

The agent can only receive placeholders for providers attached to its own sandbox. It never receives:

- credentials from unrelated providers,
- a global credential store,
- raw provider database access,
- raw gateway secret files,
- another sandbox's provider bundle.

So even if the agent inspects its environment, it sees placeholder tokens rather than reusable API keys.

**Examples from tests**

- `crates/openshell-sandbox/src/secrets.rs` test `provider_env_is_replaced_with_placeholders()` proves raw secrets are replaced before reaching child env.
- `rewrites_bearer_placeholder_header_values()` proves the supervisor can reconstruct `Authorization: Bearer ...` only in the outbound request path.
- `full_round_trip_child_env_to_rewritten_headers()` proves placeholders do not leak to the upstream request once rewritten.
- path and unresolved-placeholder tests prove fail-closed behavior for malformed or unknown placeholders.

**Important architectural nuance**

This is not merely masking for logs. It changes where secret knowledge exists:

- child processes know placeholder identifiers,
- the supervisor knows real values,
- the gateway knows stored provider records,
- policy determines whether a request carrying those credentials is even allowed.

That is a stronger compartmentalization model than "inject all env vars and hope the application behaves."

**Code evidence**

Snippet 1 is gateway-side provider resolution from `crates/openshell-server/src/grpc.rs`:

```rust
async fn resolve_provider_environment(
    store: &crate::persistence::Store,
    provider_names: &[String],
) -> Result<std::collections::HashMap<String, String>, Status> {
    let mut env = std::collections::HashMap::new();

    for name in provider_names {
        // Only providers explicitly attached to this sandbox are even considered.
        let provider = store
            .get_message_by_name::<Provider>(name)
            .await
            ...
            .ok_or_else(|| Status::failed_precondition(format!("provider '{name}' not found")))?;

        for (key, value) in &provider.credentials {
            if is_valid_env_key(key) {
                // First provider wins on duplicate keys; later providers cannot silently override it.
                env.entry(key.clone()).or_insert_with(|| value.clone());
            } else {
                // Reject malformed env keys instead of injecting something ambiguous or unsafe.
                warn!(provider_name = %name, key = %key, ...);
            }
        }
    }

    Ok(env)
}
```

What each part means:

- `provider_names` is the sandbox-scoped allowlist of which providers are even eligible,
- the gateway fetches only those providers from persistent storage,
- only `provider.credentials` are injected, not arbitrary provider config,
- `env.entry(...).or_insert_with(...)` implements the “first provider wins” rule,
- invalid environment keys are skipped instead of being injected unsafely.

Snippet 2 is sandbox-side placeholder conversion from `crates/openshell-sandbox/src/secrets.rs`:

```rust
impl SecretResolver {
    pub(crate) fn from_provider_env(
        provider_env: HashMap<String, String>,
    ) -> (HashMap<String, String>, Option<Self>) {
        if provider_env.is_empty() {
            return (HashMap::new(), None);
        }

        // This map is what child processes will actually receive.
        let mut child_env = HashMap::with_capacity(provider_env.len());
        // This map stays supervisor-side and keeps the real secrets.
        let mut by_placeholder = HashMap::with_capacity(provider_env.len());

        for (key, value) in provider_env {
            // Convert KEY -> openshell:resolve:env:KEY
            let placeholder = placeholder_for_env_key(&key);
            // Child sees placeholder, not the raw secret.
            child_env.insert(key, placeholder.clone());
            // Supervisor retains the placeholder -> real value mapping.
            by_placeholder.insert(placeholder, value);
        }

        (child_env, Some(Self { by_placeholder }))
    }
}
```

Why this snippet is the core of the mechanism:

- `provider_env` starts with real secrets,
- `child_env` becomes the environment passed to the child process,
- `child_env` stores placeholders instead of real values,
- `by_placeholder` remains supervisor-private and keeps the actual credential mapping.

Snippet 3 is the test proving that transformation from the same file `crates/openshell-sandbox/src/secrets.rs`:

```rust
#[test]
fn provider_env_is_replaced_with_placeholders() {
    let (child_env, resolver) = SecretResolver::from_provider_env(
        [("ANTHROPIC_API_KEY".to_string(), "sk-test".to_string())]
            .into_iter()
            .collect(),
    );

    // The child-visible env var is a placeholder token, not the API key itself.
    assert_eq!(
        child_env.get("ANTHROPIC_API_KEY"),
        Some(&"openshell:resolve:env:ANTHROPIC_API_KEY".to_string())
    );
    // The supervisor-side resolver can still recover the original secret when needed.
    assert_eq!(
        resolver
            .as_ref()
            .and_then(|resolver| resolver
                .resolve_placeholder("openshell:resolve:env:ANTHROPIC_API_KEY")),
        Some("sk-test")
    );
}
```

This test is useful because it proves both halves of the design at once:

- the child sees only the placeholder,
- the supervisor can still resolve it back to the real secret when needed.

### Q4. Why does OpenShell containerize policy execution rather than running policies in-process with the orchestrator?

**Short answer:** enforcement belongs with the sandboxed workload because the gateway is the control plane, while the sandbox pod has the only correct vantage point for process identity, network namespace routing, outbound traffic inspection, and credential rewriting.

**Proof from the repository**

- `architecture/system-architecture.md` places OPA, proxy, SSH server, TLS MITM cache, and inference router inside the sandbox pod supervisor, not the gateway.
- `architecture/gateway.md` describes the gateway as the central control plane, persistence, and lifecycle manager.
- `architecture/sandbox.md` describes the sandbox binary as the enforcement runtime.
- `crates/openshell-sandbox/src/proxy.rs` resolves per-connection process identity and evaluates policy close to the actual network origin.
- `crates/openshell-sandbox/src/procfs.rs` and `identity.rs` depend on local `/proc` visibility and sandbox runtime context.

**Why orchestrator-only enforcement would be weaker**

- The gateway cannot reliably infer which binary inside the sandbox opened a given outbound connection; the sandbox proxy can inspect `/proc` and bind policy to binary identity.
- The gateway is not in the child process pre-exec path, so it cannot apply Landlock or seccomp.
- Credential placeholder swapping only makes sense inside the sandbox supervisor that spawns children and sees outbound requests.
- Network namespace isolation is local to the sandbox pod.

**Security and reliability benefit**

Containerized enforcement localizes blast radius. If one sandbox policy engine fails, the failure is limited to that sandbox. The gateway remains a coordinator instead of becoming a giant privileged inline enforcement choke point for all agents.

### Q5. What architectural advantage does OpenShell gain by separating policy definitions from policy enforcement mechanisms?

**Short answer:** it can keep a stable, typed policy contract while changing or combining multiple enforcement backends underneath it.

**Proof from the repository**

- `proto/sandbox.proto` defines the typed policy contract.
- `crates/openshell-policy/src/lib.rs` is the canonical YAML<->proto conversion layer.
- `crates/openshell-sandbox/src/opa.rs` consumes typed policy as OPA data.
- `crates/openshell-sandbox/src/policy.rs` converts proto policy into runtime-specific structures and forces proxy mode semantics.
- `architecture/security-policy.md` maps schema fields to enforcement points.

**Architectural advantages**

1. **Transport independence**
   - policies can be stored and transmitted as protobuf through gRPC.
   - operators can author them as YAML.

2. **Backend flexibility**
   - filesystem fields map to Landlock,
   - process fields map to privilege dropping and seccomp,
   - network fields map to OPA and proxy logic,
   - future backends could change without rewriting the policy authoring model.

3. **Round-trip fidelity**
   - `crates/openshell-policy/src/lib.rs` explicitly states its serde types are the single canonical representation, reducing divergence between parse and render paths.

4. **Validation before enforcement**
   - L7 config validation and preset expansion happen before the policy reaches active enforcement.

5. **Explainability**
   - architecture docs can describe fields once and trace them to multiple enforcement layers.

**Why this matters in practice**

If the project later changes from one Rego rule set to another, or introduces additional kernel restrictions, the user-facing policy language can remain stable. That lowers migration cost and reduces operator confusion.

**Code evidence**

Snippet 1 is the canonical YAML-side schema from `crates/openshell-policy/src/lib.rs`:

```rust
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyFile {
    version: u32,
    // Human-authored filesystem intent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    filesystem_policy: Option<FilesystemDef>,
    // Landlock compatibility setting, still expressed as data rather than enforcement code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    landlock: Option<LandlockDef>,
    // Runtime user/group intent for the sandboxed process.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    process: Option<ProcessDef>,
    // Declarative network rules keyed by policy name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    network_policies: BTreeMap<String, NetworkPolicyRuleDef>,
}
```

What this means:

- `PolicyFile` is the human-authorable schema surface,
- `deny_unknown_fields` makes the YAML strict and predictable,
- the top-level fields are semantic categories, not enforcement implementation details.

Snippet 2 is the protobuf contract from `proto/sandbox.proto`:

```proto
message SandboxPolicy {
  // Version metadata for policy revision tracking.
  uint32 version = 1;
  // Declarative filesystem rules.
  FilesystemPolicy filesystem = 2;
  // Declarative Landlock mode.
  LandlockPolicy landlock = 3;
  // Declarative process identity rules.
  ProcessPolicy process = 4;
  // Declarative network policy map; still data, not enforcement logic.
  map<string, NetworkPolicyRule> network_policies = 5;
}
```

Why this snippet matters:

- this is the transport and storage contract used across the system,
- it mirrors the same semantic policy domains seen in the YAML schema,
- the schema says *what* should be controlled, not *how* the runtime will enforce it.

Snippet 3 is the enforcement-side conversion into the OPA engine from `crates/openshell-sandbox/src/opa.rs`:

```rust
pub fn from_proto(proto: &ProtoSandboxPolicy) -> Result<Self> {
    // Convert the typed protobuf policy into generic JSON data for OPA/Rego.
    let data_json_str = proto_to_opa_data_json(proto);
    let mut data: serde_json::Value = serde_json::from_str(&data_json_str)
        .map_err(|e| miette::miette!("internal: failed to parse proto JSON: {e}"))?;

    // Validate L7 semantics before loading anything into the live engine.
    let (errors, warnings) = crate::l7::validate_l7_policies(&data);
    ...
    // Expand shorthand presets into explicit rule sets.
    crate::l7::expand_access_presets(&mut data);

    // Load the enforcement backend only after schema data is validated and normalized.
    let mut engine = regorus::Engine::new();
    engine.add_policy("policy.rego".into(), BAKED_POLICY_RULES.into())?;
    engine.add_data_json(&data.to_string())?;
    Ok(Self { engine: Mutex::new(engine) })
}
```

This snippet shows the architectural separation directly:

- the policy is defined as typed data first,
- then validated and normalized,
- then loaded into a specific enforcement backend,
- meaning the policy contract can stay stable even if enforcement details evolve.

### Q6. Why does OpenShell require explicit agent authorization declarations rather than inferring permissions from runtime behavior?

**Short answer:** the repository consistently prefers explicit allowlists over emergent behavior because runtime inference is ambiguous, unsafe, and difficult to audit.

**Proof from the repository**

- `proto/sandbox.proto` models explicit `network_policies`, `binaries`, `endpoints`, `rules`, `access`, `allowed_ips`, `filesystem`, `process`.
- `README.md` describes declarative YAML policies rather than behavioral learning.
- `crates/openshell-sandbox/src/opa.rs` evaluates explicit host/port/path/method/binary inputs against explicit rules.
- `crates/openshell-sandbox/src/grpc.rs` validates policy safety before sandbox creation.
- `architecture/security-policy.md` documents deny-by-default proxy behavior and immutable static fields.

**Why inference from behavior would be weaker**

- Observed runtime behavior can include probing, mistakes, or adversarial exploration; auto-approving that would convert attacks into permissions.
- The same tool binary may issue very different requests depending on prompt injection or compromised dependencies.
- Explicit declarations make review possible before use; inferred permissions are retrospective and harder to justify.

**Example**

A `curl` process calling `api.github.com` once does not mean it should later be allowed to `POST /repos/.../issues`. OpenShell requires explicit L7 declarations or access presets so authorization is intentional.

**Repository hint toward controlled recommendation, not inference**

OpenShell does have `denial_aggregator.rs` and `mechanistic_mapper.rs` to generate policy recommendations from denied traffic, but those are proposals, not automatic authorization. That distinction is important: the system can assist humans without silently widening access.

### Q7. How does OpenShell's containerized orchestration prevent policy failures from cascading to other agents?

**Short answer:** each sandbox gets its own pod, supervisor, proxy, OPA engine, network namespace, TLS cache, and policy state, while the gateway distributes configuration but does not centralize the actual enforcement path for all traffic.

**Proof from the repository**

- `architecture/system-architecture.md` shows one sandbox pod per sandbox with separate supervisor and agent process.
- `crates/openshell-server/src/sandbox/mod.rs` creates per-sandbox pod specs and injects sandbox-specific environment.
- `crates/openshell-sandbox/src/lib.rs` starts per-sandbox proxy, policy engine, SSH server, and poll loop.
- `architecture/security-policy.md` stores policy revisions per sandbox and only updates sandbox-specific `current_policy_version`.

**Containment benefits**

- A malformed hot-reload only affects the sandbox polling that policy.
- `run_policy_poll_loop()` keeps last-known-good policy on reload failure instead of globally breaking traffic.
- Provider environment resolution is scoped to one sandbox's `spec.providers`.
- Denial logs and watch streams are keyed by `sandbox_id`.

**Why cascade is reduced**

If enforcement were centralized, one bad policy parse or global proxy failure could cut off every agent. In OpenShell, the per-sandbox runtime architecture localizes many failure modes to the affected sandbox instance.

### Q8. What problem does OpenShell solve by compartmentalizing credentials rather than storing them globally?

**Short answer:** it reduces blast radius, avoids accidental cross-agent secret reuse, and keeps credentials aligned to the sandbox that actually needs them.

**Proof from the repository**

- `architecture/sandbox-providers.md` describes providers as first-class entities referenced by sandbox `spec.providers`.
- `crates/openshell-server/src/grpc.rs` resolves only listed providers into environment variables.
- `create_sandbox()` validates referenced provider names exist but explicitly does not inject credentials into pod specs at creation time.
- `architecture/gateway.md` notes `GetSandboxProviderEnvironment` is a sandbox-runtime delivery RPC.

**Problems solved**

1. **Cross-sandbox leakage**
   - a sandbox without provider `X` never receives provider `X` placeholders.

2. **Overprivileged runtime env**
   - the child env contains only the credentials mapped from attached providers.

3. **Persistence leakage**
   - credentials are not serialized into pod specs, image layers, or sandbox env snapshots.

4. **Operational sprawl**
   - users manage named providers once, but attach only what a sandbox needs.

**Concrete example**

A sandbox created for GitHub automation can attach a GitHub provider without also inheriting Anthropic, GitLab, or Outlook credentials. The repository’s provider architecture is designed precisely to avoid a monolithic secret bucket.

### Q9. Why does OpenShell's policy hot-reloading require ensuring in-flight operations complete before applying new policies?

**Short answer:** because the project aims for atomic policy transitions without corrupting active request handling, and because partially-applied changes would create inconsistent authorization decisions.

**Proof from the repository**

- `crates/openshell-sandbox/src/lib.rs` reloads by constructing a new engine and replacing it only on success; otherwise it keeps the previous engine.
- `architecture/security-policy.md` emphasizes last-known-good behavior and atomic replacement semantics.
- revision/status flow (`pending`, `loaded`, `failed`, `superseded`) in `architecture/security-policy.md` shows the design assumes policy versions transition cleanly, not mid-request.

**Why this matters**

If a policy changed halfway through request evaluation:

- the CONNECT decision could be made under one policy while L7 request validation runs under another,
- logs and operator status could point to the wrong revision,
- a request could be partially authorized with one credential/rewrite context and partially denied under another.

The repository avoids that by:

- using a complete new OPA engine build before replacement,
- only updating status after success/failure is known,
- retaining last-known-good state on error.

**Practical interpretation**

OpenShell does not need to freeze the whole sandbox; it needs to ensure the policy object being used for evaluation is swapped coherently. Its versioned polling model is built around that requirement.

**Code evidence**

Snippet 1 is the atomic engine swap from `crates/openshell-sandbox/src/opa.rs`:

```rust
pub fn reload_from_proto(&self, proto: &ProtoSandboxPolicy) -> Result<()> {
    // Build and validate a complete replacement engine first.
    let new = Self::from_proto(proto)?;
    let new_engine = new
        .engine
        .into_inner()
        .map_err(|_| miette::miette!("lock poisoned on new engine"))?;
    // Only lock the live engine for the final swap.
    let mut engine = self
        .engine
        .lock()
        .map_err(|_| miette::miette!("OPA engine lock poisoned"))?;
    // Atomic replacement: future evaluations see either the old engine or the new one.
    *engine = new_engine;
    Ok(())
}
```

Why this snippet matters:

- the new policy engine is fully built first,
- only then is the live engine replaced,
- the critical section is just the final assignment,
- that minimizes the chance of mixed old/new evaluation state.

Snippet 2 is the last-known-good behavior from `crates/openshell-sandbox/src/lib.rs`:

```rust
match opa_engine.reload_from_proto(policy) {
    Ok(()) => {
        // Success means all future requests use the newly loaded policy.
        info!(policy_hash = %result.policy_hash, "Policy reloaded successfully");
    }
    Err(e) => {
        // Failure means the previous policy remains active instead of leaving the sandbox in limbo.
        warn!(
            version = result.version,
            error = %e,
            "Policy reload failed, keeping last-known-good policy"
        );
    }
}
```

This is the operational guarantee:

- policy replacement is all-or-nothing for future evaluations,
- a broken update does not half-apply,
- active sandboxes stay on a coherent previous state if reload fails.

### Q10. How does OpenShell's multi-layer security approach address different attacker models better than single-layer enforcement?

**Short answer:** it maps different attacker capabilities to different defenses, so compromise of one assumption does not automatically collapse the entire security model.

**Attacker models and repository responses**

| Attacker model | Example | OpenShell response |
|---|---|---|
| Prompt-injected agent | Agent is tricked into exfiltrating data | deny-by-default network policy, L7 method/path rules |
| Malicious tool/plugin | Tool opens raw sockets or bypasses proxy env | seccomp + network namespace + bypass monitor |
| Compromised binary after startup | Executable swapped on disk | `BinaryIdentityCache` TOFU hash verification |
| Secret-stealing process | Reads env and forwards API key | placeholder envs + supervisor-only secret resolution |
| Unauthorized client | External process hits gateway API | mTLS / gateway auth |
| SSRF / internal network abuse | Agent resolves external host to private IP | `allowed_ips` logic and proxy SSRF protections |
| Policy rollout mistake | Bad live update pushed | versioned revisions + last-known-good rollback behavior |

**Proof from code and docs**

- mTLS coverage: `architecture/gateway-security.md`, `e2e/python/test_security_tls.py`
- filesystem/process/network/inference layering: `README.md`, `architecture/sandbox.md`, `architecture/security-policy.md`
- credential compartmentalization: `architecture/sandbox-providers.md`, `secrets.rs`
- SSRF and internal IP defense: `proxy.rs`, `mechanistic_mapper.rs`, `proto/sandbox.proto`
- live-update safety: `crates/openshell-server/src/grpc.rs`, `crates/openshell-sandbox/src/lib.rs`

**Why this is stronger than single-layer enforcement**

Single-layer systems typically fail catastrophically when that one layer is bypassed. OpenShell instead assumes attackers may:

- control prompts,
- misuse allowed binaries,
- tamper with files,
- exploit transport gaps,
- abuse credentials,
- race policy updates.

Its architecture is stronger because it distributes enforcement across transport, orchestration, kernel, process identity, outbound request semantics, and secret-handling boundaries.

**Code evidence**

Snippet 1 is the mTLS test intent from `e2e/python/test_security_tls.py`:

```python
"""E2e tests for server mTLS enforcement.

# Only clients with the provisioned cluster-issued client cert should succeed.
These tests verify that the OpenShell server correctly requires valid client
certificates signed by the cluster CA.  Only callers presenting the provisioned
mTLS client cert should be able to reach the OpenShell gRPC API; all other
# Missing, rogue, or plaintext clients are expected to fail.
connection attempts must be rejected.
"""
```

Why this snippet matters:

- it shows the repository explicitly tests the “unauthorized client” attacker model,
- transport authentication is treated as a first-class security boundary, not just a deployment detail.

Snippet 2 is the SSRF/internal-IP defense from `crates/openshell-sandbox/src/proxy.rs`:

```rust
/// This is a defense-in-depth check to prevent SSRF via the CONNECT proxy.
fn is_internal_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            // Reject loopback, RFC1918 private ranges, link-local, and unspecified IPv4.
            v4.is_loopback() || v4.is_private() || v4.is_link_local() || v4.is_unspecified()
        }
        IpAddr::V6(v6) => {
            // Reject loopback/unspecified IPv6 before checking more specific internal ranges.
            if v6.is_loopback() || v6.is_unspecified() {
                return true;
            }
            ...
        }
    }
}
```

This snippet is important because it demonstrates a more nuanced attacker model:

- not just “is this hostname allowed,”
- but also “does this resolve to loopback/private/internal space and therefore look like SSRF or lateral movement?”

Snippet 3 is the direct-proxy-bypass detection from `crates/openshell-sandbox/src/bypass_monitor.rs`:

```rust
warn!(
    // Destination details of the attempted direct connection.
    dst_addr = %event.dst_addr,
    dst_port = event.dst_port,
    proto = %event.proto,
    // Process identity metadata used for attribution.
    binary = %binary,
    binary_pid = %binary_pid,
    ancestors = %ancestors,
    // Structured classification of the event as a bypass attempt.
    action = "reject",
    reason = "direct connection bypassed HTTP CONNECT proxy",
    hint = hint,
    "BYPASS_DETECT",
);
```

What this adds to the report:

- OpenShell is not only filtering authorized traffic,
- it is also trying to catch processes that attempt to route *around* the intended enforcement path,
- which is exactly why the attacker-model table includes malicious tooling and bypass attempts as separate cases.

## 3. Findings and Conclusion

OpenShell is best understood as a policy-distributed agent runtime rather than a simple sandbox launcher. The repository demonstrates a coherent security architecture where:

- the gateway is authoritative for identity, lifecycle, storage, and policy distribution,
- each sandbox is a local enforcement island,
- secrets are compartmentalized and reconstructed only at the outbound edge,
- hot reload is intentionally limited to domains that can be changed safely,
- the system is documented with enough specificity that architecture and code mostly match.

The most important reverse-engineering conclusion is that OpenShell is not relying on one "sandbox" trick. It composes multiple narrow controls, each defending against a different class of failure. That is why the design repeatedly chooses explicit declarations, runtime scoping, local enforcement, and immutable startup-only controls for kernel-bound restrictions.

## Appendix A. Repository-Wide Coverage Notes

This report was built after enumerating all **515 files** in the repository. The analysis directly inspected:

- root project metadata and onboarding docs
- architecture specifications
- workspace `Cargo.toml` and key crate entry points
- gateway security, policy, sandbox, provider, and inference architecture docs
- key gateway source (`grpc.rs`, `sandbox/mod.rs`, `tls.rs`)
- key sandbox source (`lib.rs`, `proxy.rs`, `opa.rs`, `secrets.rs`, `identity.rs`)
- provider and policy conversion code
- representative deployment, proto, test, example, and documentation files

Not every file is equally semantically important. For example:

- OCSF schema JSON files are static schema payloads rather than novel runtime logic.
- docs extension Python files support documentation generation rather than sandbox security behavior.
- Helm/templates and task scripts operationalize the architecture already described in the Rust/docs layers.

Even so, the repository-wide inventory informed the report’s conclusions about scope, project maturity, and where enforcement versus documentation versus operational support live.

## Appendix B. Clean Repository Structure Diagram

```text
OpenShell-main/
├── .agents/
│   └── skills/                       # agent workflows used to build, triage, debug, and review
├── .claude/                          # Claude-specific local agent metadata
├── .github/
│   ├── ISSUE_TEMPLATE/              # bug and feature intake templates
│   ├── workflows/                   # CI, release, docs, vouch, and test automation
│   └── CODEOWNERS
├── .opencode/                        # OpenCode-specific local agent metadata
├── architecture/
│   ├── gateway.md                   # gateway/control-plane architecture
│   ├── gateway-security.md          # mTLS and transport security model
│   ├── sandbox.md                   # sandbox runtime and enforcement design
│   ├── sandbox-providers.md         # provider and credential architecture
│   ├── security-policy.md           # policy language and enforcement mapping
│   ├── system-architecture.md       # end-to-end system diagram
│   └── ...                          # additional architecture references
├── crates/
│   ├── openshell-bootstrap/         # cluster bootstrap, PKI, image/runtime prep
│   ├── openshell-cli/               # main CLI
│   ├── openshell-core/              # shared types, config, generated proto bindings
│   ├── openshell-ocsf/              # OCSF event/schema support
│   ├── openshell-policy/            # YAML <-> protobuf policy conversion
│   ├── openshell-providers/         # provider discovery and normalization
│   ├── openshell-router/            # inference routing layer
│   ├── openshell-sandbox/           # sandbox runtime, proxy, OPA, SSH, kernel controls
│   ├── openshell-server/            # gateway server, persistence, sandbox orchestration
│   └── openshell-tui/               # terminal UI
├── deploy/
│   ├── docker/                      # Dockerfiles and cluster container scripts
│   ├── helm/                        # Helm chart for OpenShell deployment
│   ├── kube/                        # Kubernetes manifests
│   └── sbom/                        # SBOM tooling
├── docs/
│   ├── about/                       # overview, architecture, release notes
│   ├── get-started/                 # quickstart docs
│   ├── inference/                   # inference configuration docs
│   ├── reference/                   # policy schema, auth, support matrix
│   ├── sandboxes/                   # gateway/provider/sandbox management docs
│   ├── tutorials/                   # guided walkthroughs
│   ├── _ext/                        # custom Sphinx extensions and search assets
│   └── conf.py
├── e2e/
│   ├── install/                     # shell installer validation
│   ├── python/                      # Python end-to-end tests
│   └── rust/                        # Rust end-to-end harness and tests
├── examples/
│   ├── bring-your-own-container/    # BYOC example
│   ├── local-inference/             # local inference routing example
│   ├── policy-advisor/              # policy recommendation example
│   ├── private-ip-routing/          # private-IP routing example
│   └── sandbox-policy-quickstart/   # quickstart policy example
├── proto/
│   ├── datamodel.proto              # core data model
│   ├── inference.proto              # inference service contracts
│   ├── openshell.proto              # main gateway API
│   ├── sandbox.proto                # sandbox policy/settings contracts
│   └── test.proto
├── python/
│   └── openshell/                   # Python SDK and tests
├── rfc/                             # RFC templates and design notes
├── scripts/                         # helper utilities and binaries
├── tasks/                           # mise task definitions and task scripts
├── AGENTS.md                        # agent instructions
├── CONTRIBUTING.md                  # contributor workflow
├── Cargo.toml                       # Rust workspace manifest
├── README.md                        # top-level project overview
├── SECURITY.md                      # vulnerability reporting policy
├── install.sh                       # installation script
├── mise.toml                        # task runner configuration
├── pyproject.toml                   # Python project metadata
└── uv.lock                          # Python lockfile
```
