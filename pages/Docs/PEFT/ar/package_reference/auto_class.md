# AutoPeftModels

تحمّل فئة `AutoPeftModel` نموذج PEFT المناسب لنوع المهمة من خلال استنتاجه تلقائيًا من ملف التكوين. تم تصميمها لتحميل نموذج PEFT بسرعة وسهولة في سطر واحد من التعليمات البرمجية دون القلق بشأن فئة النموذج المحددة التي تحتاجها أو التحميل اليدوي لـ [`PeftConfig`].

## AutoPeftModel

[[autodoc]] auto.AutoPeftModel

- from_pretrained

## AutoPeftModelForCausalLM

[[autodoc]] auto.AutoPeftModelForCausalLM

## AutoPeftModelForSeq2SeqLM

[[autodoc]] auto.AutoPeftModelForSeq2SeqLM

## AutoPeftModelForSequenceClassification

[[autodoc]] auto.AutoPeftModelForSequenceClassification

## AutoPeftModelForTokenClassification

[[autodoc]] auto.AutoPeftModelForTokenClassification

## AutoPeftModelForQuestionAnswering

[[autodoc]] auto.AutoPeftModelForQuestionAnswering

## AutoPeftModelForFeatureExtraction

[[autodoc]] auto.AutoPeftModelForFeatureExtraction