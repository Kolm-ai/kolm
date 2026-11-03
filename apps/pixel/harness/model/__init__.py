"""The recovered layer: a conditioned residual UNet trained on pairs the degradation sampler renders on the
fly from clean tiles (harness.data.degrade), with the target state's metric vector as the conditioning.
`python -m harness model train|apply|eval-demo`. The chart metrics on the synthetic scene are the judge."""
