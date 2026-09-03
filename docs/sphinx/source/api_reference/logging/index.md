# Training Logging — moved to `uni_rl`

The W&B / TensorBoard bridges and structured training loggers moved out of
the `unilab` package into the independently released **uni_rl** package
(issue #1480): `uni_rl.logging` hosts `OnPolicyLogger`, `OffPolicyLogger`,
and the trace-event recorder.
