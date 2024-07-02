# تنسيق نقطة تفتيش PEFT

يصف هذا المستند كيفية تنظيم ملفات نقاط تفتيش PEFT وكيفية التحويل بين تنسيق PEFT والتنسيقات الأخرى.

## ملفات PEFT

تقوم طرق PEFT (ضبط دقيق فعال للمعلمات) بتحديث مجموعة فرعية صغيرة فقط من معلمات النموذج بدلاً من تحديثها جميعًا. وهذا أمر جيد لأن ملفات نقاط التفتيش يمكن أن تكون أصغر بكثير بشكل عام من ملفات النموذج الأصلي، كما أنها أسهل في التخزين والمشاركة. ومع ذلك، هذا يعني أيضًا أنه لتحميل نموذج PEFT، تحتاج إلى توفر النموذج الأصلي أيضًا.

عند استدعاء [`~PeftModel.save_pretrained`] على نموذج PEFT، يحفظ نموذج PEFT ثلاثة ملفات، موصوفة أدناه:

1. `adapter_model.safetensors` أو `adapter_model.bin`

بشكل افتراضي، يتم حفظ النموذج بتنسيق `safetensors`، وهو بديل آمن لتنسيق `bin`، المعروف أنه عرضة لـ [ثغرات أمنية](https://huggingface.co/docs/hub/security-pickle) لأنه يستخدم أداة pickle تحت الغطاء. كلا التنسيقين يخزنان نفس `state_dict`، ويمكن استبدالهما.

يحتوي `state_dict` فقط على معلمات وحدة المحول، وليس النموذج الأساسي. ولتوضيح الفرق في الحجم، يتطلب نموذج BERT العادي حوالي 420 ميجابايت من مساحة القرص، في حين أن محول IA³ الموجود أعلى نموذج BERT هذا يتطلب حوالي 260 كيلوبايت فقط.

2. `adapter_config.json`

يحتوي ملف `adapter_config.json` على تكوين وحدة المحول، والذي يعد ضروريًا لتحميل النموذج. وفيما يلي مثال على ملف `adapter_config.json` لمحول IA³ مع إعدادات قياسية مطبقة على نموذج BERT:

```json
{
"auto_mapping": {
"base_model_class": "BertModel",
"parent_library": "transformers.models.bert.modeling_bert"
},
"base_model_name_or_path": "bert-base-uncased",
"fan_in_fan_out": false,
"feedforward_modules": [
"output.dense"
],
"inference_mode": true,
"init_ia3_weights": true,
"modules_to_save": null,
"peft_type": "IA3",
"revision": null,
"target_modules": [
"key",
"value",
"output.dense"
],
"task_type": null
}
```

يحتوي ملف التكوين على:

- نوع وحدة المحول المخزنة، `"peft_type": "IA3"`
- معلومات حول النموذج الأساسي مثل `"base_model_name_or_path": "bert-base-uncased"`
- مراجعة النموذج (إن وجدت)، `"revision": null`

إذا لم يكن النموذج الأساسي نموذج محولين مسبقًا، فستكون الإدخالان الأخيران `null`. وبخلاف ذلك، ترتبط جميع الإعدادات بوحدة IA³ المحددة التي تم استخدامها لضبط نموذج الدقة.

3. `README.md`

يعد ملف `README.md` الذي تم إنشاؤه بطاقة نموذج لنموذج PEFT ويحتوي على بعض الإدخالات المعبأة مسبقًا. والغرض من ذلك هو تسهيل مشاركة النموذج مع الآخرين وتوفير بعض المعلومات الأساسية حول النموذج. هذا الملف غير مطلوب لتحميل النموذج.

## التحويل إلى تنسيق PEFT

عند التحويل من تنسيق آخر إلى تنسيق PEFT، نحتاج إلى كل من ملف `adapter_model.safetensors` (أو `adapter_model.bin`) وملف `adapter_config.json`.

### adapter_model

بالنسبة لأوزان النموذج، من المهم استخدام الخريطة الصحيحة من اسم المعلمة إلى القيمة لـ PEFT لتحميل الملف. إن الحصول على هذه الخريطة الصحيحة هو ممارسة في التحقق من تفاصيل التنفيذ، حيث لا يوجد تنسيق متفق عليه بشكل عام لمحولات PEFT.

لحسن الحظ، فإن معرفة هذه الخريطة ليست معقدة للغاية للحالات الأساسية الشائعة. دعنا نلقي نظرة على مثال ملموس، وهو [`LoraLayer`](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py):

```python
# إظهار جزء فقط من الكود

class LoraLayer(BaseTunerLayer):
# جميع أسماء الطبقات التي قد تحتوي على أوزان محول (قابلة للتدريب)
adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A"، "lora_embedding_B")
# جميع أسماء المعلمات الأخرى التي قد تحتوي على معلمات ذات صلة بالمحول
other_param_names = ("r"، "lora_alpha"، "scaling"، "lora_dropout")

def __init__(self، base_layer: nn.Module، ** kwargs) -> None:
self.base_layer = base_layer
self.r = {}
self.lora_alpha = {}
self.scaling = {}
self.lora_dropout = nn.ModuleDict({})
self.lora_A = nn.ModuleDict({})
self.lora_B = nn.ModuleDict({})
# لطبقة التضمين
self.lora_embedding_A = nn.ParameterDict({})
self.lora_embedding_B = nn.ParameterDict({})
# قم بتعطيل علامة الوزن المندمج
self._disable_adapters = False
self.merged_adapters = []
self.use_dora: dict[str، bool] = {}
self.lora_magnitude_vector: Optional[torch.nn.ParameterDict] = None # لـ DoRA
self._caches: dict[str، أي] = {}
self.kwargs = kwargs
```

في كود `__init__` المستخدم من قبل جميع فئات `LoraLayer` في PEFT، هناك مجموعة من المعلمات المستخدمة لتهيئة النموذج، ولكن القليل منها فقط له صلة بملف نقطة التفتيش: `lora_A`، `lora_B`، `lora_embedding_A`، و`lora_embedding_B`. يتم سرد هذه المعلمات في سمة فئة `adapter_layer_names` وتحتوي على المعلمات القابلة للتعلم، لذلك يجب تضمينها في ملف نقطة التفتيش. جميع المعلمات الأخرى، مثل الرتبة `r`، مشتقة من `adapter_config.json` ويجب تضمينها هناك (ما لم يتم استخدام القيمة الافتراضية).

دعنا نتحقق من `state_dict` لنموذج PEFT LoRA المطبق على BERT. عند طباعة أول خمسة مفاتيح باستخدام إعدادات LoRA الافتراضية (المفاتيح المتبقية هي نفسها، فقط مع أرقام طبقة مختلفة)، نحصل على ما يلي:

- `base_model.model.encoder.layer.0.attention.self.query.lora_A.weight`
- `base_model.model.encoder.layer.0.attention.self.query.lora_B.weight`
- `base_model.model.encoder.layer.0.attention.self.value.lora_A.weight`
- `base_model.model.encoder.layer.0.attention.self.value.lora_B.weight`
- `base_model.model.encoder.layer.1.attention.self.query.lora_A.weight`
- إلخ.

دعنا نقسم هذا:

- بشكل افتراضي، بالنسبة لنماذج BERT، يتم تطبيق LoRA على طبقات `query` و`value` لوحدة الاهتمام. هذا هو السبب في أنك ترى `attention.self.query` و`attention.self.value` في أسماء المفاتيح لكل طبقة.
- تقوم LoRA بتفسخ الأوزان إلى مصفوفتين منخفضتي الرتبة، `lora_A` و`lora_B`. هذا هو المكان الذي يأتي منه `lora_A` و`lora_B` في أسماء المفاتيح.
- يتم تنفيذ مصفوفات LoRA هذه كطبقات `nn.Linear`، لذلك يتم تخزين المعلمات في سمة `.weight` (`lora_A.weight`، `lora_B.weight`).
- بشكل افتراضي، لا يتم تطبيق LoRA على طبقة تضمين BERT، لذا لا توجد إدخالات لـ `lora_A_embedding` و`lora_B_embedding`.
- تبدأ مفاتيح `state_dict` دائمًا بـ `"base_model.model."`. والسبب هو أنه في PEFT، نقوم بتغليف النموذج الأساسي داخل نموذج محدد لـ tuner (`LoraModel` في هذه الحالة)، والذي يتم تغليفه بدوره في نموذج PEFT عام (`PeftModel`). ولهذا السبب، يتم إضافة هذين البادئة إلى المفاتيح. عند التحويل إلى تنسيق PEFT، يلزم إضافة هذه البادئات.

<Tip>
هذه النقطة الأخيرة غير صحيحة لتقنيات الضبط الدقيق للبادئة مثل الضبط الدقيق للفوارة. هناك، يتم تخزين التعليقات الإضافية مباشرة في `state_dict` دون إضافة أي بادئات إلى المفاتيح.
</Tip>

عند فحص أسماء المعلمات في النموذج المحمل، قد تفاجأ بأن مظهرها مختلف قليلاً، على سبيل المثال `base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight`. الفرق هو الجزء *`.default`* في الجزء الثاني من القطاع الأخير. يوجد هذا الجزء لأن PEFT يسمح بشكل عام بإضافة عدة محولات في نفس الوقت (باستخدام `nn.ModuleDict` أو `nn.ParameterDict` لتخزينها). على سبيل المثال، إذا قمت بإضافة محول آخر يسمى "other"، فسيكون المفتاح لهذا المحول هو `base_model.model.encoder.layer.0.attention.self.query.lora_A.other.weight`.

عند استدعاء [`~PeftModel.save_pretrained`]، تتم إزالة اسم المحول من المفاتيح. والسبب هو أن اسم المحول ليس جزءًا مهمًا من بنية النموذج؛ إنه مجرد اسم عشوائي. عند تحميل المحول، يمكنك اختيار اسم مختلف تمامًا، وسيظل النموذج يعمل بنفس الطريقة. هذا هو السبب في أن اسم المحول غير مخزن في ملف نقطة التفتيش.

<Tip>
إذا قمت باستدعاء `save_pretrained("some/path")` وكان اسم المحول ليس `"default"`، فسيتم تخزين المحول في دليل فرعي يحمل نفس اسم المحول. لذا إذا كان الاسم هو "other"، فسيتم تخزينه داخل `some/path/other`.
</Tip>

في بعض الظروف، قد يصبح تحديد القيم التي يجب إضافتها إلى ملف نقطة التفتيش أكثر تعقيدًا بعض الشيء. على سبيل المثال، في PEFT، يتم تنفيذ DoRA كحالة خاصة من LoRA. إذا كنت تريد تحويل نموذج DoRA إلى PEFT، فيجب عليك إنشاء نقطة تفتيش LoRA مع إدخالات إضافية لـ DoRA. يمكنك رؤية ذلك في `__init__` لكود `LoraLayer` السابق:

```python
self.lora_magnitude_vector: Optional[torch.nn.ParameterDict] = None # لـ DoRA
```

يشير هذا إلى وجود معلمة اختيارية إضافية لكل طبقة لـ DoRA.

### adapter_config

يحتوي ملف `adapter_config.json` على جميع المعلومات الأخرى اللازمة لتحميل نموذج PEFT. دعنا نتحقق من هذا الملف لنموذج LoRA المطبق على BERT:

```json
{
"alpha_pattern": {},
"auto_mapping": {
"base_model_class": "BertModel"،
"parent_library": "transformers.models.bert.modeling_bert"
}،
"base_model_name_or_path": "bert-base-uncased"،
"bias": "none"،
"fan_in_fan_out": false،
"inference_mode": true،
"init_lora_weights": true،
"layer_replication": null،
"layers_pattern": null،
"layers_to_transform": null،
"loftq_config": {}،
"lora_alpha": 8،
"lora_dropout": 0.0،
"megatron_config": null،
"megatron_core": "megatron.core"،
"modules_to_save": null،
"peft_type": "LORA"،
"r": 8،
"rank_pattern": {}،
"revision": null،
"target_modules": [
"query"،
"value"
]،
"task_type": null،
"use_dora": false،
"use_rslora": false
}
```

يحتوي هذا على الكثير من الإدخالات، وفي النظرة الأولى، قد يبدو ساحقًا معرفة جميع القيم الصحيحة التي يجب وضعها هناك. ومع ذلك، فإن معظم الإدخالات غير ضرورية لتحميل النموذج. إما لأنها تستخدم القيم الافتراضية ولا تحتاج إلى الإضافة، أو لأنها تؤثر فقط على تهيئة أوزان LoRA، والتي لا علاقة لها بتحميل النموذج. إذا وجدت أنك لا تعرف ما يفعله معلم محدد، على سبيل المثال `"use_rslora"`، فلا تضيفه، ويجب أن تكون على ما يرام. لاحظ أيضًا أنه مع إضافة المزيد من الخيارات، سيحصل هذا الملف على المزيد من الإدخالات في المستقبل، ولكنه يجب أن يكون متوافقًا مع الإصدارات السابقة.

كحد أدنى، يجب عليك تضمين الإدخالات التالية:

```json
{
"target_modules": ["query"، "value"]،
"peft_type": "LORA"
}
```

ومع ذلك، يوصى بإضافة أكبر عدد ممكن من الإدخالات، مثل الرتبة `r` أو `base_model_name_or_path` (إذا كان نموذج المحولين). يمكن أن تساعد هذه المعلومات الآخرين على فهم النموذج بشكل أفضل ومشاركته بسهولة أكبر. للتحقق من المفاتيح والقيم المتوقعة، راجع ملف [config.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py) (كمثال، هذا هو ملف التكوين لـ LoRA) في رمز مصدر PEFT.

## تخزين النماذج

في بعض الظروف، قد ترغب في تخزين نموذج PEFT بالكامل، بما في ذلك أوزان القاعدة. قد يكون هذا ضروريًا إذا كان النموذج الأساسي، على سبيل المثال، غير متوفر للمستخدمين الذين يحاولون تحميل نموذج PEFT. يمكنك دمج الأوزان أولاً أو تحويلها إلى نموذج محول.

### دمج الأوزان

أبسط طريقة لتخزين نموذج PEFT بالكامل هي دمج أوزان المحول في أوزان القاعدة:

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained(...)
```

ومع ذلك، هناك بعض العيوب لهذا النهج:

- بمجرد استدعاء [`~LoraModel.merge_and_unload`]، تحصل على نموذج أساسي بدون أي وظائف محددة لـ PEFT. وهذا يعني أنه لا يمكنك استخدام أي من طرق PEFT المحددة بعد الآن.
- لا يمكنك إلغاء دمج الأوزان أو تحميل عدة محولات في نفس الوقت أو تعطيل المحول، وما إلى ذلك.
- لا تدعم جميع طرق PEFT دمج الأوزان.
- قد تسمح بعض طرق PEFT بالدمج بشكل عام، ولكن ليس مع إعدادات محددة (على سبيل المثال، عند استخدام تقنيات التكميم معينة).
- سيكون النموذج بالكامل أكبر بكثير من نموذج PEFT، حيث سيتضمن جميع أوزان القاعدة أيضًا.

ولكن يجب أن يكون الاستدلال باستخدام نموذج مدمج أسرع قليلاً.
### التحويل إلى نموذج Transformers

هناك طريقة أخرى لحفظ النموذج بالكامل، بافتراض أن النموذج الأساسي هو نموذج Transformers، وهي استخدام هذا النهج غير التقليدي لإدراج أوزان PEFT مباشرة في النموذج الأساسي وحفظه، والذي يعمل فقط إذا "خدعت" مكتبة Transformers وجعلتها تعتقد أن نموذج PEFT ليس نموذج PEFT. هذه الطريقة تعمل فقط مع LoRA لأن المهايئات الأخرى غير مطبقة في Transformers.

```python
model = ...  # نموذج PEFT
...
# بعد الانتهاء من تدريب النموذج، قم بحفظه في موقع مؤقت
model.save_pretrained(<temp_location>)
# الآن قم بتحميل هذا النموذج مباشرة إلى نموذج Transformers، بدون غلاف PEFT
# يتم حقن أوزان PEFT مباشرة في النموذج الأساسي
model_loaded = AutoModel.from_pretrained(<temp_location>)
# الآن اجعل النموذج المحمل يعتقد أنه ليس نموذج PEFT
model_loaded._hf_peft_config_loaded = False
# الآن عند حفظه، سيحفظ النموذج بالكامل
model_loaded.save_pretrained(<final_location>)
# أو قم بتحميله إلى Hugging Face Hub
model_loaded.push_to_hub(<final_location>)
```