لم يتم ترجمة الأجزاء المطلوبة كما هو موضح في التعليمات:

<!--Copyright 2023 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# LoRA

Low-Rank Adaptation (LoRA) هي طريقة PEFT تقوم بتحليل مصفوفة كبيرة إلى مصفوفتين صغيرتين منخفضتي الرتبة في طبقات الاهتمام. وهذا يقلل بشكل كبير من عدد المعلمات التي تحتاج إلى ضبط دقيق.

الملخص من الورقة هو:

*نقترح نظام نمذجة لغوية عصبية يعتمد على التكيّف منخفض الرتبة (LoRA) لإعادة ترتيب التعرف على الكلام. على الرغم من أن نماذج اللغة المسبقة التدريب مثل BERT أظهرت أداءً متفوقًا في إعادة الترتيب في المرور الثاني، إلا أن التكلفة الحسابية لزيادة مرحلة ما قبل التدريب والتكيف مع النماذج المسبقة التدريب مع مجالات محددة تحد من استخدامها العملي في إعادة الترتيب. نقدم هنا طريقة تعتمد على التحليل منخفض الرتبة لتدريب نموذج BERT لإعادة ترتيب وتكييفه مع مجالات جديدة باستخدام جزء بسيط فقط (0.08%) من المعلمات المسبقة التدريب. يتم تحسين هذه المصفوفات المُدخلة من خلال هدف تدريب تمييزي إلى جانب خسارة تنظيمية قائمة على الارتباط. يتم تقييم بنية BERT المقترحة للتكيّف منخفض الرتبة لإعادة الترتيب (LoRB) على مجموعات بيانات LibriSpeech وinternal datasets مع تقليل أوقات التدريب بعوامل تتراوح بين 5.4 و3.6.*

## LoraConfig

[[autodoc]] tuners.lora.config.LoraConfig

## LoraModel

[[autodoc]] tuners.lora.model.LoraModel

## Utility

[[autodoc]] utils.loftq_utils.replace_lora_weights_loftq