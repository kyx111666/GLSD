# EXP-7A Python review

The script replays only archived canonical selections and frozen curves, verifies each replayed P2 list against its saved counterpart, and aborts before diagnostic output when the parent gates fail. All width, translation, peak-centered, and joint intervals are isolated enumeration-based diagnostic oracles; none is passed to P1/P2, matching, F1, Recognition, or STRS. Closed-interval IoU consistently uses `+1` endpoints. Bootstrap resamples outer subjects (N=10,000, seed=100), not individual events.
