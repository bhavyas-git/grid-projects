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

## Appendix B. Full Repository File Manifest

```text
./.DS_Store
./.agents/skills/build-from-issue/SKILL.md
./.agents/skills/create-github-issue/SKILL.md
./.agents/skills/create-github-pr/SKILL.md
./.agents/skills/create-spike/SKILL.md
./.agents/skills/debug-inference/SKILL.md
./.agents/skills/debug-openshell-cluster/SKILL.md
./.agents/skills/fix-security-issue/SKILL.md
./.agents/skills/generate-sandbox-policy/SKILL.md
./.agents/skills/generate-sandbox-policy/examples.md
./.agents/skills/openshell-cli/SKILL.md
./.agents/skills/openshell-cli/cli-reference.md
./.agents/skills/review-github-pr/SKILL.md
./.agents/skills/review-security-issue/SKILL.md
./.agents/skills/sbom/SKILL.md
./.agents/skills/sync-agent-infra/SKILL.md
./.agents/skills/triage-issue/SKILL.md
./.agents/skills/tui-development/SKILL.md
./.agents/skills/update-docs/SKILL.md
./.agents/skills/watch-github-actions/SKILL.md
./.claude/README.md
./.claude/agent-memory/arch-doc-writer/MEMORY.md
./.claude/agent-memory/principal-engineer-reviewer/MEMORY.md
./.claude/agents/arch-doc-writer.md
./.claude/agents/principal-engineer-reviewer.md
./.dockerignore
./.env.example
./.gitattributes
./.github/CODEOWNERS
./.github/DISCUSSION_TEMPLATE/vouch-request.yml
./.github/ISSUE_TEMPLATE/bug_report.yml
./.github/ISSUE_TEMPLATE/config.yml
./.github/ISSUE_TEMPLATE/feature_request.yml
./.github/PULL_REQUEST_TEMPLATE.md
./.github/VOUCHED.td
./.github/actions/setup-buildx/action.yml
./.github/workflows/branch-checks.yml
./.github/workflows/branch-e2e.yml
./.github/workflows/ci-image.yml
./.github/workflows/dco.yml
./.github/workflows/docker-build.yml
./.github/workflows/docs-build.yml
./.github/workflows/docs-preview-pr.yml
./.github/workflows/e2e-test.yml
./.github/workflows/issue-triage.yml
./.github/workflows/release-auto-tag.yml
./.github/workflows/release-canary.yml
./.github/workflows/release-dev.yml
./.github/workflows/release-tag.yml
./.github/workflows/test-install.yml
./.github/workflows/vouch-check.yml
./.github/workflows/vouch-command.yml
./.gitignore
./.idea/.gitignore
./.idea/OpenShell-main.iml
./.idea/inspectionProfiles/profiles_settings.xml
./.idea/misc.xml
./.idea/modules.xml
./.idea/workspace.xml
./.opencode/agents/arch-doc-writer.md
./.opencode/agents/principal-engineer-reviewer.md
./.python-version
./AGENTS.md
./CLAUDE.md
./CONTRIBUTING.md
./Cargo.lock
./Cargo.toml
./DCO
./LICENSE
./README.md
./SECURITY.md
./STYLEGUIDE.md
./TESTING.md
./THIRD-PARTY-NOTICES
./about.toml
./architecture/README.md
./architecture/build-containers.md
./architecture/gateway-deploy-connect.md
./architecture/gateway-security.md
./architecture/gateway-settings.md
./architecture/gateway-single-node.md
./architecture/gateway.md
./architecture/inference-routing.md
./architecture/policy-advisor.md
./architecture/plans/openshell-reverse-engineering-report.md
./architecture/sandbox-connect.md
./architecture/sandbox-custom-containers.md
./architecture/sandbox-providers.md
./architecture/sandbox.md
./architecture/security-policy.md
./architecture/system-architecture.md
./architecture/tui.md
./crates/openshell-bootstrap/Cargo.toml
./crates/openshell-bootstrap/src/build.rs
./crates/openshell-bootstrap/src/constants.rs
./crates/openshell-bootstrap/src/docker.rs
./crates/openshell-bootstrap/src/edge_token.rs
./crates/openshell-bootstrap/src/errors.rs
./crates/openshell-bootstrap/src/image.rs
./crates/openshell-bootstrap/src/lib.rs
./crates/openshell-bootstrap/src/metadata.rs
./crates/openshell-bootstrap/src/mtls.rs
./crates/openshell-bootstrap/src/pki.rs
./crates/openshell-bootstrap/src/paths.rs
./crates/openshell-bootstrap/src/push.rs
./crates/openshell-bootstrap/src/runtime.rs
./crates/openshell-cli/Cargo.toml
./crates/openshell-cli/src/auth.rs
./crates/openshell-cli/src/bootstrap.rs
./crates/openshell-cli/src/completers.rs
./crates/openshell-cli/src/doctor_llm_prompt.md
./crates/openshell-cli/src/edge_tunnel.rs
./crates/openshell-cli/src/lib.rs
./crates/openshell-cli/src/main.rs
./crates/openshell-cli/src/run.rs
./crates/openshell-cli/src/ssh.rs
./crates/openshell-cli/src/tls.rs
./crates/openshell-cli/tests/ensure_providers_integration.rs
./crates/openshell-cli/tests/mtls_integration.rs
./crates/openshell-cli/tests/provider_commands_integration.rs
./crates/openshell-cli/tests/sandbox_create_lifecycle_integration.rs
./crates/openshell-cli/tests/sandbox_name_fallback_integration.rs
./crates/openshell-core/Cargo.toml
./crates/openshell-core/build.rs
./crates/openshell-core/src/config.rs
./crates/openshell-core/src/error.rs
./crates/openshell-core/src/forward.rs
./crates/openshell-core/src/inference.rs
./crates/openshell-core/src/lib.rs
./crates/openshell-core/src/paths.rs
./crates/openshell-core/src/proto/mod.rs
./crates/openshell-core/src/proto/openshell.datamodel.v1.rs
./crates/openshell-core/src/proto/openshell.sandbox.v1.rs
./crates/openshell-core/src/proto/openshell.test.v1.rs
./crates/openshell-core/src/proto/openshell.v1.rs
./crates/openshell-core/src/settings.rs
./crates/openshell-ocsf/Cargo.toml
./crates/openshell-ocsf/schemas/ocsf/README.md
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/VERSION
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/classes/application_lifecycle.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/classes/base_event.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/classes/detection_finding.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/classes/device_config_state_change.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/classes/http_activity.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/classes/network_activity.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/classes/process_activity.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/classes/ssh_activity.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/actor.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/attack.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/connection_info.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/container.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/device.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/evidences.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/finding_info.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/firewall_rule.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/http_request.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/http_response.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/metadata.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/network_endpoint.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/network_proxy.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/process.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/product.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/remediation.json
./crates/openshell-ocsf/schemas/ocsf/v1.7.0/objects/url.json
./crates/openshell-ocsf/src/builders/base.rs
./crates/openshell-ocsf/src/builders/config.rs
./crates/openshell-ocsf/src/builders/finding.rs
./crates/openshell-ocsf/src/builders/http.rs
./crates/openshell-ocsf/src/builders/lifecycle.rs
./crates/openshell-ocsf/src/builders/mod.rs
./crates/openshell-ocsf/src/builders/network.rs
./crates/openshell-ocsf/src/builders/process.rs
./crates/openshell-ocsf/src/builders/ssh.rs
./crates/openshell-ocsf/src/enums/action.rs
./crates/openshell-ocsf/src/enums/activity.rs
./crates/openshell-ocsf/src/enums/auth.rs
./crates/openshell-ocsf/src/enums/disposition.rs
./crates/openshell-ocsf/src/enums/http_method.rs
./crates/openshell-ocsf/src/enums/launch.rs
./crates/openshell-ocsf/src/enums/mod.rs
./crates/openshell-ocsf/src/enums/security.rs
./crates/openshell-ocsf/src/enums/severity.rs
./crates/openshell-ocsf/src/enums/status.rs
./crates/openshell-ocsf/src/events/app_lifecycle.rs
./crates/openshell-ocsf/src/events/base_event.rs
./crates/openshell-ocsf/src/events/config_state_change.rs
./crates/openshell-ocsf/src/events/detection_finding.rs
./crates/openshell-ocsf/src/events/http_activity.rs
./crates/openshell-ocsf/src/events/mod.rs
./crates/openshell-ocsf/src/events/network_activity.rs
./crates/openshell-ocsf/src/events/process_activity.rs
./crates/openshell-ocsf/src/events/serde_helpers.rs
./crates/openshell-ocsf/src/events/ssh_activity.rs
./crates/openshell-ocsf/src/format/jsonl.rs
./crates/openshell-ocsf/src/format/mod.rs
./crates/openshell-ocsf/src/format/shorthand.rs
./crates/openshell-ocsf/src/lib.rs
./crates/openshell-ocsf/src/objects/attack.rs
./crates/openshell-ocsf/src/objects/connection.rs
./crates/openshell-ocsf/src/objects/container.rs
./crates/openshell-ocsf/src/objects/device.rs
./crates/openshell-ocsf/src/objects/endpoint.rs
./crates/openshell-ocsf/src/objects/finding.rs
./crates/openshell-ocsf/src/objects/firewall_rule.rs
./crates/openshell-ocsf/src/objects/http.rs
./crates/openshell-ocsf/src/objects/metadata.rs
./crates/openshell-ocsf/src/objects/mod.rs
./crates/openshell-ocsf/src/objects/process.rs
./crates/openshell-ocsf/src/tracing_layers/event_bridge.rs
./crates/openshell-ocsf/src/tracing_layers/jsonl_layer.rs
./crates/openshell-ocsf/src/tracing_layers/mod.rs
./crates/openshell-ocsf/src/tracing_layers/shorthand_layer.rs
./crates/openshell-ocsf/src/validation/mod.rs
./crates/openshell-ocsf/src/validation/schema.rs
./crates/openshell-policy/Cargo.toml
./crates/openshell-policy/src/lib.rs
./crates/openshell-providers/Cargo.toml
./crates/openshell-providers/src/context.rs
./crates/openshell-providers/src/discovery.rs
./crates/openshell-providers/src/lib.rs
./crates/openshell-providers/src/providers/anthropic.rs
./crates/openshell-providers/src/providers/claude.rs
./crates/openshell-providers/src/providers/codex.rs
./crates/openshell-providers/src/providers/copilot.rs
./crates/openshell-providers/src/providers/generic.rs
./crates/openshell-providers/src/providers/github.rs
./crates/openshell-providers/src/providers/gitlab.rs
./crates/openshell-providers/src/providers/mod.rs
./crates/openshell-providers/src/providers/nvidia.rs
./crates/openshell-providers/src/providers/openai.rs
./crates/openshell-providers/src/providers/opencode.rs
./crates/openshell-providers/src/providers/outlook.rs
./crates/openshell-providers/src/test_helpers.rs
./crates/openshell-router/Cargo.toml
./crates/openshell-router/README.md
./crates/openshell-router/src/backend.rs
./crates/openshell-router/src/config.rs
./crates/openshell-router/src/lib.rs
./crates/openshell-router/src/mock.rs
./crates/openshell-router/tests/backend_integration.rs
./crates/openshell-sandbox/Cargo.toml
./crates/openshell-sandbox/data/sandbox-policy.rego
./crates/openshell-sandbox/src/bypass_monitor.rs
./crates/openshell-sandbox/src/child_env.rs
./crates/openshell-sandbox/src/denial_aggregator.rs
./crates/openshell-sandbox/src/grpc_client.rs
./crates/openshell-sandbox/src/identity.rs
./crates/openshell-sandbox/src/l7/inference.rs
./crates/openshell-sandbox/src/l7/mod.rs
./crates/openshell-sandbox/src/l7/provider.rs
./crates/openshell-sandbox/src/l7/relay.rs
./crates/openshell-sandbox/src/l7/rest.rs
./crates/openshell-sandbox/src/l7/tls.rs
./crates/openshell-sandbox/src/lib.rs
./crates/openshell-sandbox/src/log_push.rs
./crates/openshell-sandbox/src/main.rs
./crates/openshell-sandbox/src/mechanistic_mapper.rs
./crates/openshell-sandbox/src/opa.rs
./crates/openshell-sandbox/src/policy.rs
./crates/openshell-sandbox/src/process.rs
./crates/openshell-sandbox/src/procfs.rs
./crates/openshell-sandbox/src/proxy.rs
./crates/openshell-sandbox/src/sandbox/linux/landlock.rs
./crates/openshell-sandbox/src/sandbox/linux/mod.rs
./crates/openshell-sandbox/src/sandbox/linux/netns.rs
./crates/openshell-sandbox/src/sandbox/linux/seccomp.rs
./crates/openshell-sandbox/src/sandbox/mod.rs
./crates/openshell-sandbox/src/secrets.rs
./crates/openshell-sandbox/src/ssh.rs
./crates/openshell-sandbox/testdata/sandbox-policy.yaml
./crates/openshell-sandbox/tests/system_inference.rs
./crates/openshell-server/Cargo.toml
./crates/openshell-server/migrations/postgres/001_create_objects.sql
./crates/openshell-server/migrations/postgres/002_create_sandbox_policies.sql
./crates/openshell-server/migrations/postgres/003_create_policy_recommendations.sql
./crates/openshell-server/migrations/sqlite/001_create_objects.sql
./crates/openshell-server/migrations/sqlite/002_create_sandbox_policies.sql
./crates/openshell-server/migrations/sqlite/003_create_policy_recommendations.sql
./crates/openshell-server/src/auth.rs
./crates/openshell-server/src/grpc.rs
./crates/openshell-server/src/http.rs
./crates/openshell-server/src/inference.rs
./crates/openshell-server/src/lib.rs
./crates/openshell-server/src/main.rs
./crates/openshell-server/src/multiplex.rs
./crates/openshell-server/src/persistence/mod.rs
./crates/openshell-server/src/persistence/postgres.rs
./crates/openshell-server/src/persistence/sqlite.rs
./crates/openshell-server/src/persistence/tests.rs
./crates/openshell-server/src/sandbox/mod.rs
./crates/openshell-server/src/sandbox_index.rs
./crates/openshell-server/src/sandbox_watch.rs
./crates/openshell-server/src/ssh_tunnel.rs
./crates/openshell-server/src/tls.rs
./crates/openshell-server/src/tracing_bus.rs
./crates/openshell-server/src/ws_tunnel.rs
./crates/openshell-server/tests/auth_endpoint_integration.rs
./crates/openshell-server/tests/edge_tunnel_auth.rs
./crates/openshell-server/tests/multiplex_integration.rs
./crates/openshell-server/tests/multiplex_tls_integration.rs
./crates/openshell-server/tests/ws_tunnel_integration.rs
./crates/openshell-tui/Cargo.toml
./crates/openshell-tui/src/app.rs
./crates/openshell-tui/src/clipboard.rs
./crates/openshell-tui/src/event.rs
./crates/openshell-tui/src/lib.rs
./crates/openshell-tui/src/theme.rs
./crates/openshell-tui/src/ui/create_provider.rs
./crates/openshell-tui/src/ui/create_sandbox.rs
./crates/openshell-tui/src/ui/dashboard.rs
./crates/openshell-tui/src/ui/global_settings.rs
./crates/openshell-tui/src/ui/mod.rs
./crates/openshell-tui/src/ui/providers.rs
./crates/openshell-tui/src/ui/sandbox_detail.rs
./crates/openshell-tui/src/ui/sandbox_draft.rs
./crates/openshell-tui/src/ui/sandbox_logs.rs
./crates/openshell-tui/src/ui/sandbox_policy.rs
./crates/openshell-tui/src/ui/sandbox_settings.rs
./crates/openshell-tui/src/ui/sandboxes.rs
./crates/openshell-tui/src/ui/splash.rs
./deploy/docker/.dockerignore
./deploy/docker/Dockerfile.ci
./deploy/docker/Dockerfile.cli-macos
./deploy/docker/Dockerfile.images
./deploy/docker/Dockerfile.python-wheels
./deploy/docker/Dockerfile.python-wheels-macos
./deploy/docker/cluster-entrypoint.sh
./deploy/docker/cluster-healthcheck.sh
./deploy/docker/cross-build.sh
./deploy/helm/openshell/.helmignore
./deploy/helm/openshell/Chart.yaml
./deploy/helm/openshell/templates/_helpers.tpl
./deploy/helm/openshell/templates/clusterrole.yaml
./deploy/helm/openshell/templates/clusterrolebinding.yaml
./deploy/helm/openshell/templates/networkpolicy.yaml
./deploy/helm/openshell/templates/role.yaml
./deploy/helm/openshell/templates/rolebinding.yaml
./deploy/helm/openshell/templates/service.yaml
./deploy/helm/openshell/templates/serviceaccount.yaml
./deploy/helm/openshell/templates/statefulset.yaml
./deploy/helm/openshell/values.yaml
./deploy/kube/gpu-manifests/nvidia-device-plugin-helmchart.yaml
./deploy/kube/manifests/agent-sandbox.yaml
./deploy/kube/manifests/openshell-helmchart.yaml
./deploy/sbom/resolve_licenses.py
./deploy/sbom/sbom_to_csv.py
./docs/CONTRIBUTING.md
./docs/_ext/json_output/README.md
./docs/_ext/json_output/__init__.py
./docs/_ext/json_output/config.py
./docs/_ext/json_output/content/__init__.py
./docs/_ext/json_output/content/extractor.py
./docs/_ext/json_output/content/metadata.py
./docs/_ext/json_output/content/structured.py
./docs/_ext/json_output/content/text.py
./docs/_ext/json_output/core/__init__.py
./docs/_ext/json_output/core/builder.py
./docs/_ext/json_output/core/document_discovery.py
./docs/_ext/json_output/core/global_metadata.py
./docs/_ext/json_output/core/hierarchy_builder.py
./docs/_ext/json_output/core/json_formatter.py
./docs/_ext/json_output/core/json_writer.py
./docs/_ext/json_output/processing/__init__.py
./docs/_ext/json_output/processing/cache.py
./docs/_ext/json_output/processing/processor.py
./docs/_ext/json_output/utils.py
./docs/_ext/policy_table.py
./docs/_ext/search_assets/__init__.py
./docs/_ext/search_assets/enhanced-search.css
./docs/_ext/search_assets/main.js
./docs/_ext/search_assets/modules/DocumentLoader.js
./docs/_ext/search_assets/modules/EventHandler.js
./docs/_ext/search_assets/modules/ResultRenderer.js
./docs/_ext/search_assets/modules/SearchEngine.js
./docs/_ext/search_assets/modules/SearchInterface.js
./docs/_ext/search_assets/modules/SearchPageManager.js
./docs/_ext/search_assets/modules/Utils.js
./docs/_ext/search_assets/templates/search.html
./docs/_templates/layout.html
./docs/about/architecture.md
./docs/about/architecture.svg
./docs/about/overview.md
./docs/about/release-notes.md
./docs/about/supported-agents.md
./docs/assets/openshell-terminal.png
./docs/conf.py
./docs/get-started/quickstart.md
./docs/index.md
./docs/inference/configure.md
./docs/inference/index.md
./docs/project.json
./docs/reference/default-policy.md
./docs/reference/gateway-auth.md
./docs/reference/policy-schema.md
./docs/reference/support-matrix.md
./docs/resources/license.md
./docs/sandboxes/community-sandboxes.md
./docs/sandboxes/index.md
./docs/sandboxes/manage-gateways.md
./docs/sandboxes/manage-providers.md
./docs/sandboxes/manage-sandboxes.md
./docs/sandboxes/policies.md
./docs/tutorials/first-network-policy.md
./docs/tutorials/github-sandbox.md
./docs/tutorials/index.md
./docs/tutorials/inference-ollama.md
./docs/tutorials/local-inference-lmstudio.md
./e2e/install/bash_test.sh
./e2e/install/fish_test.fish
./e2e/install/helpers.sh
./e2e/install/sh_test.sh
./e2e/install/zsh_test.sh
./e2e/python/conftest.py
./e2e/python/test_inference_routing.py
./e2e/python/test_policy_validation.py
./e2e/python/test_sandbox_api.py
./e2e/python/test_sandbox_exec_python.py
./e2e/python/test_sandbox_gpu.py
./e2e/python/test_sandbox_policy.py
./e2e/python/test_sandbox_providers.py
./e2e/python/test_sandbox_venv.py
./e2e/python/test_security_tls.py
./e2e/rust/Cargo.lock
./e2e/rust/Cargo.toml
./e2e/rust/src/harness/binary.rs
./e2e/rust/src/harness/mod.rs
./e2e/rust/src/harness/output.rs
./e2e/rust/src/harness/port.rs
./e2e/rust/src/harness/sandbox.rs
./e2e/rust/src/lib.rs
./e2e/rust/tests/cf_auth_smoke.rs
./e2e/rust/tests/cli_smoke.rs
./e2e/rust/tests/community_image.rs
./e2e/rust/tests/custom_image.rs
./e2e/rust/tests/docker_preflight.rs
./e2e/rust/tests/edge_tunnel_e2e.rs
./e2e/rust/tests/forward_proxy_l7_bypass.rs
./e2e/rust/tests/host_gateway_alias.rs
./e2e/rust/tests/no_proxy.rs
./e2e/rust/tests/port_forward.rs
./e2e/rust/tests/provider_auto_create.rs
./e2e/rust/tests/sandbox_lifecycle.rs
./e2e/rust/tests/settings_management.rs
./e2e/rust/tests/sync.rs
./e2e/rust/tests/upload_create.rs
./examples/bring-your-own-container/Dockerfile
./examples/bring-your-own-container/README.md
./examples/bring-your-own-container/app.py
./examples/gateway-deploy-connect.md
./examples/local-inference/README.md
./examples/local-inference/inference.py
./examples/local-inference/routes.yaml
./examples/local-inference/sandbox-policy.yaml
./examples/openclaw.md
./examples/policy-advisor/README.md
./examples/policy-advisor/ctf.py
./examples/policy-advisor/sandbox-policy.yaml
./examples/private-ip-routing/Dockerfile
./examples/private-ip-routing/README.md
./examples/private-ip-routing/server.py
./examples/sandbox-policy-quickstart/README.md
./examples/sandbox-policy-quickstart/demo.sh
./examples/sandbox-policy-quickstart/policy.yaml
./examples/sync-files.md
./examples/vscode-remote-sandbox.md
./install.sh
./mise.toml
./proto/datamodel.proto
./proto/inference.proto
./proto/openshell.proto
./proto/sandbox.proto
./proto/test.proto
./pyproject.toml
./python/openshell/__init__.py
./python/openshell/_proto/__init__.py
./python/openshell/openshell_test.py
./python/openshell/sandbox.py
./python/openshell/sandbox_test.py
./rfc/0000-template.md
./rfc/README.md
./scripts/bin/k9s
./scripts/bin/kubectl
./scripts/bin/openshell
./scripts/build-benchmark/README.md
./scripts/build-benchmark/cluster-deploy-fast-test.sh
./scripts/docker-cleanup.sh
./scripts/generate_third_party_notices.py
./scripts/remote-deploy.sh
./scripts/smoke-test-network-policy.sh
./scripts/test-release-tag.sh
./scripts/update_license_headers.py
./tasks/ci.toml
./tasks/cluster.toml
./tasks/docker.toml
./tasks/docs.toml
./tasks/helm.toml
./tasks/license.toml
./tasks/notices.toml
./tasks/publish.toml
./tasks/python.toml
./tasks/rust.toml
./tasks/sandbox.toml
./tasks/sbom.toml
./tasks/scripts/cluster-bootstrap.sh
./tasks/scripts/cluster-deploy-fast.sh
./tasks/scripts/cluster-push-component.sh
./tasks/scripts/cluster.sh
./tasks/scripts/docker-build-ci.sh
./tasks/scripts/docker-build-image.sh
./tasks/scripts/docker-publish-multiarch.sh
./tasks/scripts/release.py
./tasks/scripts/sandbox.sh
./tasks/term.toml
./tasks/test.toml
./tasks/version.toml
./uv.lock
```
