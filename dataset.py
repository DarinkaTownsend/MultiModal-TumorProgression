import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset


class GliomaSeverityDataset(Dataset):
    # loads preprocessed 2D axial slices with clinical text
    # for tumor severity classification (3-class: low/mid/high)

    def __init__(self, manifestPath, split="TRAIN", modalities=None, transform=None):
        self.transform = transform
        self.modalities = modalities or ["flair", "t1", "t1ce", "t2"]
        self.samples = []

        with open(manifestPath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                self.samples.append(row)

        print("Loaded %d %s samples (%d unique patients)" % (
            len(self.samples), split,
            len(set(s["subjectId"] for s in self.samples))
        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # load MRI modalities for time1 and stack as channels
        channels = []
        for mod in self.modalities:
            key = "time1_%s" % mod
            sliceData = np.load(sample[key])
            channels.append(sliceData)

        # (C, H, W) — 4 modalities as channels
        image = np.stack(channels, axis=0)

        # normalize each channel independently to [0, 1]
        for c in range(image.shape[0]):
            ch = image[c]
            chMin = ch.min()
            chMax = ch.max()
            if chMax - chMin > 1e-8:
                image[c] = (ch - chMin) / (chMax - chMin)
            else:
                image[c] = 0.0

        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        # severity label
        label = int(sample["severity"])

        # clinical text string for text encoder
        clinicalText = sample["clinicalText"]

        # seg mask for time1 (can be used for auxiliary supervision)
        segPath = sample.get("time1_seg", None)
        seg = None
        if segPath and os.path.exists(segPath):
            seg = torch.from_numpy(np.load(segPath)).long()

        return {
            "image": image,
            "label": label,
            "clinicalText": clinicalText,
            "seg": seg,
            "subjectId": int(sample["subjectId"]),
            "sliceIdx": int(sample["sliceIdx"]),
        }


# quick test
if __name__ == "__main__":
    import sys
    manifestPath = sys.argv[1] if len(sys.argv) > 1 else "processed/manifest.csv"

    trainSet = GliomaSeverityDataset(manifestPath, split="TRAIN")
    testSet = GliomaSeverityDataset(manifestPath, split="TEST")

    if len(trainSet) > 0:
        sample = trainSet[0]
        print("\nSample:")
        print("  Image shape: %s" % str(sample["image"].shape))
        print("  Label: %d" % sample["label"])
        print("  Clinical text: %s" % sample["clinicalText"])
        print("  Seg shape: %s" % str(sample["seg"].shape if sample["seg"] is not None else None))
        print("  Subject: %d, Slice: %d" % (sample["subjectId"], sample["sliceIdx"]))

        # class distribution
        from collections import Counter
        labelCounts = Counter(int(s["severity"]) for s in trainSet.samples)
        print("\nTrain distribution:")
        for k in sorted(labelCounts.keys()):
            print("  Class %d: %d slices" % (k, labelCounts[k]))

