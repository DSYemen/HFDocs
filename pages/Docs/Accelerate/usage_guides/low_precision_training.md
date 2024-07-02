لم يتم ترجمة الأجزاء المحددة من النص كما هو مطلوب:

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

# أساليب التدريب منخفض الدقة

🤗 يوفر Accelerate تكاملات لتدريب أساليب الدقة المنخفضة باستخدام أجهزة الأجهزة المدعومة المحددة من خلال حزم `TransformersEngine` و`MS-AMP`. ستساعدك هذه الوثيقة على فهم الأجهزة المدعومة، وكيفية تكوين [`Accelerator`] للاستفادة من أساليب الدقة المنخفضة، وما يمكن توقعه أثناء التدريب.

## ما يعنيه التدريب على FP8

لاستكشاف المزيد من التفاصيل الدقيقة في التدريب على FP8 باستخدام PyTorch و🤗 Accelerate، راجع [concept_guide](../concept_guides/low_precision_training.md) حول سبب صعوبة ذلك. ولكن باختصار، بدلاً من التدريب في BF16، يمكن تنفيذ بعض جوانب (أو كلها) لتدريب النموذج باستخدام 8 بت بدلاً من 16. يكمن التحدي في القيام بذلك دون تقليل الأداء النهائي.

يتم تمكين هذا فقط على أجهزة NVIDIA محددة، وهي:

- أي شيء بعد سلسلة بطاقات الرسومات 3000 للمستهلكين (مثل 4090)
- هندسات GPU المستندة إلى Hopper (مثل "H100" و"H200")

ستؤدي النتيجة إلى بعض المكاسب في الذاكرة المستخدمة (نظرًا لأننا قللنا الذاكرة اللازمة إلى النصف لبعض أجزاء التدريب) ويجب أن نشاهد أيضًا زيادة في الإنتاجية للنماذج الأكبر التي يمكنها استبدال طبقات معينة بوحدات FP8.

## تكوين المعجل

يتم حاليًا دعم خادمين خلفيين مختلفين لـ FP8 (`TransformersEngine` و`MS-AMP`)، لكل منهما قدرات وتكوينات مختلفة.

لاستخدام أي منهما، يتم استخدام نفس واجهة برمجة التطبيقات الأساسية. ما عليك سوى تمرير `mixed_precision="fp8"` إلى [`Accelerator`]، أثناء `accelerate config` عند مطالبتك بالدقة المختلطة، أو كجزء من ملف `config.yaml` في مفتاح `mixed_precision`:

```python
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="fp8")
```

افتراضيًا، إذا كان `MS-AMP` متاحًا في بيئتك، فسيستخدم 🤗 Accelerate تلقائيًا كخلفية. لتحديده بنفسك (وتخصيص أجزاء أخرى من إعداد الدقة المختلطة FP8)، يمكنك استخدام [`utils.FP8RecipeKwargs`]:

```python
from accelerate import Accelerator
from accelerate.utils import FP8RecipeKwargs
kwargs = [FP8RecipeKwargs(backend="msamp")]
# أو لتحديد الخلفية كـ `TransformersEngine` حتى إذا تم تثبيت MS-AMP
# kwargs = [FP8RecipeKwargs(backend="te")]
accelerator = Accelerator(mixed_precision="fp8", kwarg_handlers=kwargs)
```

## تكوين MS-AMP

من بين الاثنين، `MS-AMP` هو الأسهل عادةً في التكوين لأنه يحتوي على حجة واحدة فقط: مستوى التحسين.

يتم حاليًا دعم مستويين من التحسين في تكامل 🤗 Accelerate، `"O1"` و`"O2"` (باستخدام الحرف "o"، وليس الصفر).

- سيقوم `"O1"` بتحويل تدرجات الأوزان واتصالات `all_reduce` إلى 8 بت، بينما يتم تنفيذ الباقي في 16 بت. يقلل هذا من استخدام ذاكرة GPU العامة ويُسرع نطاقات التردد.
- سيقوم `"O2"` أيضًا بتحويل حالات محسن الترتيب الأول إلى 8 بت، بينما تكون حالات الترتيب الثاني في FP16. (يتم حاليًا دعم محسن "Adam" فقط). تحاول هذه الطريقة جاهدة تقليل تدهور الدقة النهائية وستوفر أكبر قدر ممكن من الذاكرة.

لتحديد مستوى التحسين، مرره إلى `FP8KwargsHandler` عن طريق تعيين حجة `optimization_level`:

```python
from accelerate import Accelerator
from accelerate.utils import FP8RecipeKwargs
kwargs = [FP8RecipeKwargs(backend="msamp", optimization_level="O2")]
accelerator = Accelerator(mixed_precision="fp8", kwarg_handlers=kwargs)
```

## تكوين TransformersEngine

يحتوي TransformersEngine على الكثير من التخصيصات المتاحة لكيفية ونوع حسابات FP8 التي يتم إجراؤها. تتوفر قائمة كاملة بالحجج المدعومة ومعانيها في [وثائق NVIDIA](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html)، ولكنها مذكورة أيضًا كجزء من docstring [`FP8KwargsHandler`] لراحتك.

يحاول 🤗 Accelerate تعيين الإعدادات الافتراضية المعقولة، ولكن يمكن أن يؤدي استكشاف المعلمات المختلفة وتعديلها بنفسك إلى أداء أفضل.

لاستخدامه، حدد `backend="te"` وقم بتعديل أي من الحجج التي تريدها كجزء من برنامج التعامل مع الحجج الخاص بك:

```python
from accelerate import Accelerator
from accelerate.utils import FP8RecipeKwargs
kwargs = [FP8RecipeKwargs(backend="te", ...)]
accelerator = Accelerator(mixed_precision="fp8", kwarg_handlers=kwargs)
```

## قراءات إضافية

لمعرفة المزيد حول التدريب في FP8، يرجى الاطلاع على الموارد التالية:

- [دليل المفاهيم الخاص بنا](../concept_guides/low_precision_training.md) الذي يتناول بالتفصيل كل من TransformersEngine وMS-AMP
- [وثائق `transformers-engine`](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html)
- [وثائق `MS-AMP`](https://azure.github.io/MS-AMP/docs/)