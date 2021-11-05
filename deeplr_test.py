from io import BytesIO
from PIL import Image

import datasets

def bytes_to_pil(example_batch):
    example_batch['img'] = [
        Image.open(BytesIO(b)) for b in example_batch.pop('img_bytes')
    ]
    return example_batch

ds = datasets.load_dataset('nateraw/fairface')
ds = ds.with_transform(bytes_to_pil)

