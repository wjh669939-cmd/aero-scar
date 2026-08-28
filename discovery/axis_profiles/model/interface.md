# Interface contract (CONTRACT-TIER: changes require freeze process + coordinator sign-off)

- editable paths: models/AirFM/fusion/**, models/AirFM/encoders/**, UnifiedSeries2Vec.encode assembly in unified_model.py;
- parameter budget cap enforced by validator (param_budget_counter);
- I/O tensor shapes must be preserved (io_shape_check).
