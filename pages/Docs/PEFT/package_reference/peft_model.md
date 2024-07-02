# النماذج

[`PeftModel`] هي فئة النموذج الأساسي لتحديد نموذج Transformer الأساسي وتكوينه لتطبيق طريقة PEFT عليه. يحتوي نموذج "PeftModel" الأساسي على طرق لتحميل النماذج وحفظها من Hub.

## PeftModel

[[autodoc]] PeftModel

- الكل

## PeftModelForSequenceClassification

نموذج `PeftModel` لمهام تصنيف التسلسل.

[[autodoc]] PeftModelForSequenceClassification

- الكل

## PeftModelForTokenClassification

نموذج `PeftModel` لمهام تصنيف الرموز.

[[autodoc]] PeftModelForTokenClassification

- الكل

## PeftModelForCausalLM

نموذج `PeftModel` لنمذجة اللغة السببية.

[[autodoc]] PeftModelForCausalLM

- الكل

## PeftModelForSeq2SeqLM

نموذج `PeftModel` لنمذجة اللغة التسلسلية.

[[autodoc]] PeftModelForSeq2SeqLM

- الكل

## PeftModelForQuestionAnswering

نموذج `PeftModel` للإجابة على الأسئلة.

[[autodoc]] PeftModelForQuestionAnswering

- الكل

## PeftModelForFeatureExtraction

نموذج `PeftModel` لاستخراج الميزات/المتجهات المدمجة من نماذج Transformer.

[[autodoc]] PeftModelForFeatureExtraction

- الكل

## PeftMixedModel

نموذج `PeftModel` لمزج أنواع مختلفة من المحولات (مثل LoRA وLoHa).

[[autodoc]] PeftMixedModel

- الكل

## المرافق

[[autodoc]] utils.cast_mixed_precision_params

[[autodoc]] get_peft_model

[[autodoc]] inject_adapter_in_model

[[autodoc]] utils.get_peft_model_state_dict

[[autodoc]] utils.prepare_model_for_kbit_training

[[autodoc]] get_layer_status

[[autodoc]] get_model_status