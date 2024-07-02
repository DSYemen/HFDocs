# حقن المحول (Adapter injection)

مع PEFT، يمكنك حقن محولات قابلة للتدريب في أي وحدة `torch`، مما يتيح لك استخدام طرق المحول دون الاعتماد على فئات النمذجة في PEFT. حاليًا، يدعم PEFT حقن [LoRA](../conceptual_guides/adapter#low-rank-adaptation-lora) و [AdaLoRA](../conceptual_guides/adapter#adaptive-low-rank-adaptation-adalora) و [IA3](../conceptual_guides/ia3) في النماذج لأن هذه المحولات تعديل في المكان (inplace modification) للنماذج كافٍ لضبط دقة النماذج.

تحقق من الجدول أدناه لمعرفة متى يجب حقن المحولات.

| المزايا | العيوب |
|---|---|
| يتم تعديل النموذج في المكان، مع الاحتفاظ بجميع السمات والطرق الأصلية | كتابة دالات المنفعة `from_pretrained` و `save_pretrained` يدويًا من Hugging Face لحفظ المحولات وتحميلها |
| يعمل مع أي وحدة `torch` ووضعية | لا يعمل مع أي من طرق المنفعة التي توفرها `PeftModel` مثل تعطيل المحولات ودمجها |

لحقن المحول، استخدم طريقة [`inject_adapter_in_model`]. تأخذ هذه الطريقة 3 وسائط، وهي تكوين PEFT، والنمذجة، واسم المحول الاختياري. يمكنك أيضًا ربط عدة محولات بالنموذج إذا استدعيت [`inject_adapter_in_model`] عدة مرات بأسماء محولات مختلفة.

على سبيل المثال، لضخ محولات LoRA في الوحدة الفرعية "linear" للوحدة النمطية "DummyModel":

```python
import torch
from peft import inject_adapter_in_model, LoraConfig

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 10)
        self.linear = torch.nn.Linear(10, 10)
        self.lm_head = torch.nn.Linear(10, 10)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        x = self.lm_head(x)
        return x

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    target_modules=["linear"],
)

model = DummyModel()
model = inject_adapter_in_model(lora_config, model)

dummy_inputs = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
dummy_outputs = model(dummy_inputs)
```

اطبع النموذج للتأكد من حقن المحولات بشكل صحيح.

```bash
DummyModel(
    (embedding): Embedding(10, 10)
    (linear): Linear(
        in_features=10, out_features=10, bias=True
        (lora_dropout): ModuleDict(
            (default): Dropout(p=0.1, inplace=False)
        )
        (lora_A): ModuleDict(
            (default): Linear(in_features=10, out_features=64, bias=False)
        )
        (lora_B): ModuleDict(
            (default): Linear(in_features=64, out_features=10, bias=False)
        )
        (lora_embedding_A): ParameterDict()
        (lora_embedding_B): ParameterDict()
    )
    (lm_head): Linear(in_features=10, out_features=10, bias=True)
)
```

لحفظ المحول فقط، استخدم دالة [`get_peft_model_state_dict`]:

```python
from peft import get_peft_model_state_dict

peft_state_dict = get_peft_model_state_dict(model)
print(peft_state_dict)
```

وبخلاف ذلك، فإن `model.state_dict()` تعيد القاموس الكامل لحالة النموذج.