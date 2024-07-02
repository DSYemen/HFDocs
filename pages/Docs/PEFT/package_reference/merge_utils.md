# دمج النماذج

يوفر PEFT العديد من المرافق الداخلية لـ [دمج وحدات LoRA](../developer_guides/model_merging) مع طريقتي TIES و DARE.

[[autodoc]] utils.merge_utils.prune

[[autodoc]] utils.merge_utils.calculate_majority_sign_mask

[[autodoc]] utils.merge_utils.disjoint_merge

[[autodoc]] utils.merge_utils.task_arithmetic

[[autodoc]] utils.merge_utils.ties

[[autodoc]] utils.merge_utils.dare_linear

[[autodoc]] utils.merge_utils.dare_ties