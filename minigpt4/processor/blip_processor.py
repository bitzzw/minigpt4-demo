from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from minigpt4.common.registry import registry
    

@registry.register_processor("blip2_image_eval")
class Blip2ImageEvalProcessor:
    def __init__(self, image_size=224):
        super().__init__()
        
        self.normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls):
        return cls(image_size=224)
