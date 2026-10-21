# Engineering Acceptance Checklist — RealTimeVoiceChanger

## 1. Before Implementation

* [ ] Read applicable project instructions.
* [ ] Inspect Git status and preserve existing changes.
* [ ] Inspect relevant implementation and call sites.
* [ ] Identify the actual problem and root cause.
* [ ] Identify backward-compatibility and environment risks.
* [ ] Define the tests needed for the change.

## 2. Code Quality

* [ ] Reuse existing functionality where appropriate.
* [ ] Follow established architecture and coding conventions.
* [ ] Validate inputs and handle errors explicitly.
* [ ] Avoid unnecessary dependencies and unrelated refactoring.
* [ ] Preserve public interfaces or document justified changes.
* [ ] Inspect the final diff for unintended modifications.

## 3. Training and Models

* [ ] Verify the intended training interpreter and dependencies.
* [ ] Validate dataset metadata and checksums.
* [ ] Preserve the held-out partition.
* [ ] Verify pretrained and auxiliary assets.
* [ ] Run the smallest useful smoke test before expensive training.
* [ ] Record experiment configuration and actual results.
* [ ] Verify checkpoint and Faiss index compatibility.
* [ ] Evaluate the model using held-out audio.
* [ ] Test the model through the actual inference adapter.

## 4. Real-Time Audio

* [ ] Keep real-time callbacks non-blocking and predictable.
* [ ] Verify queue and buffer behavior.
* [ ] Test startup, shutdown, and repeated start/stop.
* [ ] Verify worker exception handling.
* [ ] Test sample continuity and accounting.
* [ ] Measure inference RTF separately from end-to-end latency.
* [ ] Run relevant streaming regression tests.
* [ ] Record dropouts and memory measurements when available.
* [ ] Distinguish simulated tests from physical audio-device tests.

## 5. Backend–GUI Contract

* [ ] Reuse existing backend interfaces where possible.
* [ ] Define configuration and validation behavior.
* [ ] Define service lifecycle and status semantics.
* [ ] Expose device information and relevant metrics.
* [ ] Provide actionable errors.
* [ ] Add headless integration tests.
* [ ] Document the actual interface for Gemini Flash 3.8.
* [ ] Confirm that no GUI implementation was added.

## 6. Documentation and Verification

* [ ] Update affected documentation.
* [ ] Run focused tests.
* [ ] Run relevant regression tests when feasible.
* [ ] Record exact commands and actual results.
* [ ] Mark unavailable tests as BLOCKED or NOT RUN.
* [ ] Record known limitations and remaining risks.
* [ ] Confirm that no unrelated user files were overwritten.

## 7. Completion Decision

Mark a task COMPLETE only when its predefined acceptance criteria are satisfied and supported by evidence.

Use:

* PASS — verified successfully.
* FAIL — verification exposed a problem.
* BLOCKED — could not verify because of an identified prerequisite.
* NOT RUN — verification has not been attempted.
* NOT APPLICABLE — the check does not apply to the task.

A passing unit test does not prove hardware compatibility. A successful training run does not prove audio quality. A successful simulated streaming run does not prove live device routing.

Every completion report must distinguish implemented code from verified behavior.
