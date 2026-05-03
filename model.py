import torch
import torch.nn as nn
import timm
from transformers import DistilBertModel, DistilBertTokenizer

# dimensions
IMG_FEAT_DIM  = 384   # vit_small output
TEXT_FEAT_DIM = 768   # distilbert hidden size
FUSED_DIM     = 256   # common projection dim
NUM_CLASSES   = 3


class ImageEncoder(nn.Module):
    def __init__(self, dropoutRate=0.1):
        super().__init__()
        # vit_small_patch16 with 4-channel input and 240x240 patches
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            in_chans=4,
            img_size=240,
            num_classes=0,   # remove classification head, return CLS token
        )
        self.dropout = nn.Dropout(dropoutRate)

    def forward(self, x):
        # x: (B, 4, 240, 240)
        feats = self.vit(x)           # (B, 384)
        return self.dropout(feats)


class TextEncoder(nn.Module):
    def __init__(self, dropoutRate=0.1):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(dropoutRate)

    def forward(self, inputIds, attentionMask):
        # inputIds, attentionMask: (B, seqLen)
        out = self.bert(input_ids=inputIds, attention_mask=attentionMask)
        # use CLS token representation
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        return self.dropout(cls)


class ReliabilityGatedFusion(nn.Module):
    # learned scalar gate: alpha * img + (1-alpha) * txt
    def __init__(self, dropoutRate=0.1):
        super().__init__()
        self.imgProj  = nn.Linear(IMG_FEAT_DIM,  FUSED_DIM)
        self.txtProj  = nn.Linear(TEXT_FEAT_DIM, FUSED_DIM)
        # gate takes concatenation of both projections
        self.gate = nn.Sequential(
            nn.Linear(FUSED_DIM * 2, FUSED_DIM),
            nn.ReLU(),
            nn.Linear(FUSED_DIM, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropoutRate)
        self.norm = nn.LayerNorm(FUSED_DIM)

    def forward(self, imgFeats, txtFeats):
        imgProj = self.imgProj(imgFeats)   # (B, 256)
        txtProj = self.txtProj(txtFeats)   # (B, 256)
        concat  = torch.cat([imgProj, txtProj], dim=-1)  # (B, 512)
        alpha   = self.gate(concat)        # (B, 1)  in [0,1]
        fused   = alpha * imgProj + (1 - alpha) * txtProj
        return self.norm(self.dropout(fused)), alpha


class SeverityClassifier(nn.Module):
    def __init__(self, dropoutRate=0.3):
        super().__init__()
        self.imgEncoder  = ImageEncoder(dropoutRate=0.1)
        self.txtEncoder  = TextEncoder(dropoutRate=0.1)
        self.fusion      = ReliabilityGatedFusion(dropoutRate=0.1)
        # MC dropout is applied in the classification head
        self.head = nn.Sequential(
            nn.Dropout(dropoutRate),
            nn.Linear(FUSED_DIM, 128),
            nn.ReLU(),
            nn.Dropout(dropoutRate),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, image, inputIds, attentionMask):
        imgFeats = self.imgEncoder(image)
        txtFeats = self.txtEncoder(inputIds, attentionMask)
        fused, alpha = self.fusion(imgFeats, txtFeats)
        logits = self.head(fused)
        return logits, alpha

    def enableMCDropout(self):
        # set all dropout layers to train mode for MC sampling
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predictWithUncertainty(self, image, inputIds, attentionMask, nSamples=20):
        # run nSamples stochastic forward passes
        self.eval()
        self.enableMCDropout()
        probs = []
        with torch.no_grad():
            for _ in range(nSamples):
                logits, _ = self.forward(image, inputIds, attentionMask)
                probs.append(torch.softmax(logits, dim=-1))
        probs = torch.stack(probs, dim=0)       # (nSamples, B, C)
        meanProbs = probs.mean(dim=0)           # (B, C)
        uncertainty = probs.var(dim=0).sum(-1)  # (B,) — sum of per-class variance
        return meanProbs, uncertainty


# quick sanity check
if __name__ == "__main__":
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = SeverityClassifier()

    batchSize = 2
    images = torch.randn(batchSize, 4, 240, 240)
    texts  = ["68 year old male, WHO grade 2 Oligodendroglioma, IDH mut",
              "45 year old female, WHO grade 4 Glioblastoma, IDH wt"]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

    logits, alpha = model(images, enc["input_ids"], enc["attention_mask"])
    print("logits shape:", logits.shape)    # (2, 3)
    print("alpha shape:", alpha.shape)      # (2, 1)

    meanProbs, unc = model.predictWithUncertainty(images, enc["input_ids"], enc["attention_mask"])
    print("mean probs:", meanProbs)
    print("uncertainty:", unc)
