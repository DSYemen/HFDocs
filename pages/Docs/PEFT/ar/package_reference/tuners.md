# Tuners

تعد أداة الضبط (أو المحول) وحدة نمطية يمكن توصيلها بـ `torch.nn.Module`. تعتبر [BaseTuner] فئة أساسية لأدوات الضبط الأخرى، وتوفر طرقًا وسمات مشتركة لإعداد تكوين المحول واستبدال الوحدة النمطية المستهدفة بوحدة نمطية للمحول. [BaseTunerLayer] هي فئة أساسية لطبقات المحول. فهو يوفر طرقًا وسمات لإدارة المحولات مثل تنشيط وتعطيل المحولات.

## BaseTuner

[[autodoc]] tuners.tuners_utils.BaseTuner

## BaseTunerLayer

[[autodoc]] tuners.tuners_utils.BaseTunerLayer