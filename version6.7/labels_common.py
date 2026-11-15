"""
labels_common.py  --  Shared pieces of version6.7: the image-type classes, the text prompts, and BiomedCLIP.

BiomedCLIP (Microsoft, MIT licence; Zhang et al. 2023, "BiomedCLIP: a multimodal biomedical foundation model
pretrained from fifteen million scientific image-text pairs") is used AS-IS: no training, no fine-tuning.
It scores how well an image matches a short text such as "a western blot"; the best-matching class is the
suggested label. Every suggestion on Recod.ai is then checked by a person (review_labels.py).

Classes follow BioFors (Blot/Gel, Microscopy, Macroscopy, FACS) plus "Histology" (stained tissue sections) and
"Other" (charts, diagrams, text, unclear). BioFors folds histology into Microscopy, so for any BioFors comparison
Histology is merged back into Microscopy (TO_BIOFORS).
"""

import os
import numpy as np

CLASSES = ["Blot/Gel", "Microscopy", "Histology", "Macroscopy", "FACS", "Other"]
# keys 1-5 unchanged from the first review round; Histology added as 6
KEYS = {"1": "Blot/Gel", "2": "Microscopy", "3": "Macroscopy", "4": "FACS", "5": "Other", "6": "Histology"}
BIOFORS_CLASSES = ["Blot/Gel", "Microscopy", "Macroscopy", "FACS"]
# BioFors has no Histology class: its Microscopy includes stained tissue. For BioFors comparisons Histology -> Microscopy.
TO_BIOFORS = {"Histology": "Microscopy"}

# Several plain descriptions per class. Written once, generic: nothing here names a specific dataset's content.
DESCRIPTIONS = {
    "Blot/Gel": ["a western blot", "a gel electrophoresis image", "protein bands on a western blot membrane",
                 "an agarose gel with DNA bands", "a northern or southern blot"],
    "Microscopy": ["a fluorescence microscopy image", "a light microscopy image of cells",
                   "an electron microscopy image", "an immunofluorescence image of cells",
                   "a confocal microscopy image", "a phase contrast image of cultured cells"],
    "Histology": ["a histology image of stained tissue", "an H&E stained tissue section",
                  "an immunohistochemistry image of a tissue section", "a pathology slide of tissue",
                  "a microscope image of a stained tissue slice"],
    "Macroscopy": ["a photograph of an animal", "a photograph of an organ or tissue sample",
                   "a photograph of a plant or plant part", "a photograph of a laboratory specimen",
                   "a photograph of a mouse or rat"],
    "FACS": ["a flow cytometry scatter plot", "a FACS dot plot with gates", "a flow cytometry density plot",
             "a flow cytometry histogram"],
    "Other": ["a bar chart", "a line graph", "a schematic diagram", "a table of text", "a page of text"],
}
# Wrappers around each description. Which one works best is chosen on a small tuning split of BioFors only.
TEMPLATES = ["{}", "this is {}", "a figure panel showing {}", "a biomedical image: {}"]

MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class Encoder:
    """Wraps BiomedCLIP. encode_images(list of BGR numpy images) and encode_texts(list of str) return
    L2-normalised float32 arrays; logit_scale is the model's temperature."""

    def __init__(self, threads=None, batch=32):
        self.batch = batch
        if os.environ.get("V67_FAKE_ENCODER") == "1":            # code tests only; never for real labels
            self.fake = True
            self.logit_scale = 50.0
            return
        self.fake = False
        try:
            import torch
            from open_clip import create_model_from_pretrained, get_tokenizer
        except ImportError as e:
            raise SystemExit(f"BiomedCLIP needs extra packages ({e}). Run:  python check_setup.py  for the install commands.")
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        print("loading BiomedCLIP (first time: downloads ~800 MB from Hugging Face, then cached) ...")
        self.model, self.preprocess = create_model_from_pretrained(MODEL_ID)
        self.tokenizer = get_tokenizer(MODEL_ID)
        self.model.eval()
        self.logit_scale = float(self.model.logit_scale.exp().item())

    def encode_texts(self, texts):
        if self.fake:
            rng = np.random.default_rng(abs(hash(tuple(texts))) % 2**32)
            v = rng.normal(size=(len(texts), 6)).astype(np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        with self.torch.no_grad():
            t = self.tokenizer(texts, context_length=256)
            v = self.model.encode_text(t).float().numpy()
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def encode_images(self, bgr_images):
        if self.fake:
            v = np.array([[im[..., c].mean() for c in range(3)] + [im[..., c].std() for c in range(3)]
                          for im in bgr_images], dtype=np.float32) + 1e-3
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        from PIL import Image
        out = []
        for i in range(0, len(bgr_images), self.batch):
            chunk = [self.preprocess(Image.fromarray(im[..., ::-1].copy())) for im in bgr_images[i:i + self.batch]]
            with self.torch.no_grad():
                v = self.model.encode_image(self.torch.stack(chunk)).float().numpy()
            out.append(v)
        v = np.vstack(out)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def class_embeddings(enc, template_ids):
    """One embedding per class: mean of its descriptions wrapped in the given templates, re-normalised."""
    E = []
    for c in CLASSES:
        texts = [TEMPLATES[t].format(d) for t in template_ids for d in DESCRIPTIONS[c]]
        m = enc.encode_texts(texts).mean(axis=0)
        E.append(m / np.linalg.norm(m))
    return np.array(E, dtype=np.float32)


def probabilities(img_emb, cls_emb, logit_scale):
    z = logit_scale * img_emb @ cls_emb.T
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def read_bgr(path):
    import cv2
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return im
