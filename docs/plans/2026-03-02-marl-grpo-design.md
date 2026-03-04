# MARL Inference-First Design (Reboot Baseline)

> Date: 2026-03-02 (reboot-cleaned)
> Status: Seed design for new brainstorming cycle

---

## 1. Reboot Objective

The reboot scope is intentionally narrow:

1. Build a non-invasive inference-side module that can launch and run MAS workflows through an AgentPipe/AgentLoop.
2. Keep MAS orchestration logic external and decoupled; do not modify MAS core source code.
3. Accept `prompt`, `mas_config`, and `llm_endpoint` as inputs; output structured trajectories.

---

## 2. Hard Boundaries

1. No invasive edits to external MAS repositories.
2. No coupling to training/runtime control-plane in this phase.
3. All agent LLM calls must pass through a monitorable gateway path.
4. The module must remain framework-agnostic at the boundary (LangGraph/AutoGen/custom MAS compatible via adapter).

---

## 3. Target Module Contract

### 3.1 Inputs

1. `prompt`: user task payload.
2. `mas_config`: agent roles, topology mode, step/turn constraints, stop conditions.
3. `llm_endpoint`: model endpoint/base URL/port and model mapping.

### 3.2 Outputs

1. `trajectory`: ordered event/action records with run/episode/agent/turn identity.
2. `capture_report`: accepted/rejected counts and reason distribution.

---

## 4. High-Level Architecture

1. `AgentPipe Launcher`: starts one MAS run instance from normalized config.
2. `AgentLoop Runner`: executes the multi-agent loop and emits action boundaries.
3. `Model Monitor Gateway`: mandatory path for LLM request/response capture.
4. `Trajectory Builder`: transforms monitored events into canonical trajectory records.
5. `Store Writer`: persists trajectory and reject streams.

---

## 5. Non-Goals for This Baseline

1. Training ingestion and optimization pipeline.
2. Runtime scheduler/process supervisor for RL training.
3. Branch coordinator and online tree execution control-plane.

---

## 6. Next Brainstorming Starting Point

Future design work should decide:

1. AgentPipe/AgentLoop lifecycle API.
2. Adapter contract for heterogeneous MAS frameworks.
3. Canonical trajectory schema and rejection vocabulary for inference-only capture.
